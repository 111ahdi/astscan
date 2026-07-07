"""C++ 静态分析工具 — 基于 tree-sitter + LLM 的内存泄漏检测。

用法:
    python analyzer.py [target_file] [--collect-only]

选项:
    target_file      要分析的 C++ 文件路径（默认 test.cpp）
    --collect-only   仅收集并打印分配点，不调用 AI
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskID
from rich.text import Text
from tree_sitter import Language, Node, Parser

import tree_sitter_cpp

# ── 加载 .env 中的环境变量 ────────────────────────────────────────────────────
load_dotenv()

# ── Rich 控制台实例 ───────────────────────────────────────────────────────────
console = Console()

# ── 最大并发 API 调用数 ────────────────────────────────────────────────────────
MAX_WORKERS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryOp:
    """单次内存操作（new 或 delete）。"""
    row: int            # 1-based 行号
    var_name: str       # 变量名，匿名时为 "?"
    op_type: str        # "new" | "delete" | "delete[]"
    op_text: str        # 原始表达式文本，如 "new int[10]" / "delete[] p"


@dataclass
class VariableTrace:
    """同一变量在函数内的所有内存操作轨迹。"""
    var_name: str
    allocs: list[MemoryOp] = field(default_factory=list)   # new 操作
    deallocs: list[MemoryOp] = field(default_factory=list)  # delete 操作

    @property
    def is_leaked(self) -> bool:
        """有 new 但无对应 delete → 泄漏嫌疑。"""
        return len(self.allocs) > 0 and len(self.deallocs) == 0

    @property
    def alloc_count(self) -> int:
        return len(self.allocs)

    @property
    def dealloc_count(self) -> int:
        return len(self.deallocs)


@dataclass
class FunctionAnalysis:
    """一个函数内的完整内存操作分析数据。"""
    func_name: str
    func_code: str
    func_id: str                                # "函数名:起始行"
    variables: dict[str, VariableTrace] = field(default_factory=dict)

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    def _get_or_create(self, var_name: str) -> VariableTrace:
        """获取或创建变量的操作轨迹。"""
        if var_name not in self.variables:
            self.variables[var_name] = VariableTrace(var_name=var_name)
        return self.variables[var_name]

    def add_new(self, row: int, var_name: str, op_text: str) -> None:
        self._get_or_create(var_name).allocs.append(
            MemoryOp(row=row, var_name=var_name, op_type="new", op_text=op_text)
        )

    def add_delete(self, row: int, var_name: str, op_text: str) -> None:
        op_type = "delete[]" if "[" in op_text else "delete"
        self._get_or_create(var_name).deallocs.append(
            MemoryOp(row=row, var_name=var_name, op_type=op_type, op_text=op_text)
        )

    # ── 查询方法 ──────────────────────────────────────────────────────────

    @property
    def all_ops(self) -> list[MemoryOp]:
        """返回所有内存操作（按行号排序）。"""
        ops: list[MemoryOp] = []
        for v in self.variables.values():
            ops.extend(v.allocs)
            ops.extend(v.deallocs)
        return sorted(ops, key=lambda o: o.row)

    @property
    def new_ops(self) -> list[MemoryOp]:
        """返回所有 new 操作（按行号排序）。"""
        return [op for op in self.all_ops if op.op_type == "new"]

    @property
    def leaked_vars(self) -> list[VariableTrace]:
        """返回存在泄漏嫌疑的变量列表。"""
        return [v for v in self.variables.values() if v.is_leaked]

    @property
    def safe_vars(self) -> list[VariableTrace]:
        """返回正确释放的变量列表。"""
        return [v for v in self.variables.values()
                if v.alloc_count > 0 and v.dealloc_count > 0]

    @property
    def orphan_deletes(self) -> list[MemoryOp]:
        """返回没有对应 new 的 delete（wild pointer / 重复释放）。"""
        result: list[MemoryOp] = []
        for v in self.variables.values():
            if v.alloc_count == 0:
                result.extend(v.deallocs)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# AST 辅助工具
# ═══════════════════════════════════════════════════════════════════════════════

def _find_identifier(node: Node) -> str:
    """在当前节点的子树中递归查找第一个 identifier，返回其源码文本。

    找不到时返回 "?"。
    """
    if node.type == "identifier":
        return node.text.decode() if node.text else "?"
    for child in node.children:
        result = _find_identifier(child)
        if result != "?":
            return result
    return "?"


def _get_function_name(func_node: Node) -> str:
    """从 function_definition 节点中提取函数名。"""
    for child in func_node.children:
        if child.type == "function_declarator":
            return _find_identifier(child)
    return "(匿名函数)"


def get_enclosing_function(node: Node) -> tuple[str | None, str, int]:
    """从 node 出发向上遍历父节点，找到所在的 function_definition。

    返回:
        (func_code, func_name, func_start_row) — func_start_row 为 1-based 行号。
        未找到时返回 (None, "(全局作用域)", 0)。
    """
    current = node
    while current is not None:
        if current.type == "function_definition":
            code = current.text.decode() if current.text else None
            name = _get_function_name(current)
            start_row = current.start_point.row + 1
            return code, name, start_row
        current = current.parent
    return None, "(全局作用域)", 0


# ═══════════════════════════════════════════════════════════════════════════════
# AST 收集 — 按函数遍历，收集 new 和 delete 操作
# ═══════════════════════════════════════════════════════════════════════════════

def collect_memory_ops(
    node: Node,
    ops: list[tuple[str, MemoryOp]] | None = None,
) -> list[tuple[str, MemoryOp]]:
    """递归遍历 AST，收集所有 new_expression 和 delete_expression。

    返回:
        (func_id, MemoryOp) 列表，func_id 用于后续按函数分组。
    """
    if ops is None:
        ops = []

    if node.type in ("new_expression", "delete_expression"):
        row = node.start_point.row + 1
        op_text = node.text.decode() if node.text else ""

        parent = node.parent
        var_name = _find_identifier(parent) if parent else "?"

        func_code, func_name, func_start_row = get_enclosing_function(node)
        if func_code is None:
            func_code = "(无法定位所在函数)"

        func_id = f"{func_name}:{func_start_row}" if func_name != "(全局作用域)" else "(全局作用域)"

        if node.type == "new_expression":
            op_type = "new"
        else:
            op_type = "delete[]" if "[" in op_text else "delete"

        ops.append((
            func_id,
            MemoryOp(row=row, var_name=var_name, op_type=op_type, op_text=op_text),
        ))

    # 不加 else：new 内部可能嵌套 new / delete
    for child in node.children:
        collect_memory_ops(child, ops)

    return ops


def build_function_analyses(
    ops: list[tuple[str, MemoryOp]],
) -> list[FunctionAnalysis]:
    """将收集到的操作按 func_id 分组，构建 FunctionAnalysis 列表。

    需要同时提供各函数的 func_code。为此重新从 AST 获取 —
    这里采用一个简化方案：调用方传入完整的树和 ops。
    """
    # 按 func_id 分组
    ops.sort(key=lambda x: x[0])
    grouped: dict[str, list[MemoryOp]] = {}
    for func_id, op in ops:
        grouped.setdefault(func_id, []).append(op)

    results: list[FunctionAnalysis] = []
    for func_id, op_list in grouped.items():
        # 从 func_id 提取 func_name（格式: "func_name:start_row"）
        parts = func_id.rsplit(":", 1)
        func_name = parts[0]

        # func_code 需要从 AST 重新获取 — 这里用第一个 op 来追溯
        # 实际场景中可改进为收集阶段一并存储，此处保持简洁
        func_code = "(函数源码将在分析时获取)"

        fa = FunctionAnalysis(
            func_name=func_name,
            func_code=func_code,
            func_id=func_id,
        )
        for op in op_list:
            if op.op_type == "new":
                fa.add_new(op.row, op.var_name, op.op_text)
            else:
                fa.add_delete(op.row, op.var_name, op.op_text)

        results.append(fa)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(fa: FunctionAnalysis) -> str:
    """为一个函数生成聚合分析 prompt，包含 new 和 delete 的完整信息。"""
    lines: list[str] = []

    # ── 列出所有 new 操作 ──────────────────────────────────────────────────
    for op in fa.new_ops:
        if op.var_name != "?":
            lines.append(f"- 第 {op.row} 行: **new** 分配，指针 **{op.var_name}** (`{op.op_text}`)")
        else:
            lines.append(f"- 第 {op.row} 行: **new** 分配，(未绑定变量) (`{op.op_text}`)")

    # ── 列出所有 delete 操作 ───────────────────────────────────────────────
    for op in fa.all_ops:
        if op.op_type in ("delete", "delete[]"):
            lines.append(f"- 第 {op.row} 行: **{op.op_type}** 释放，变量 **{op.var_name}** (`{op.op_text}`)")

    # ── 汇总变量状态 ────────────────────────────────────────────────────────
    named_all = [op.var_name for op in fa.new_ops if op.var_name != "?"]

    if named_all:
        main_task = f"逐一判断指针 {'、'.join(named_all)} 是否存在内存泄漏。"
    else:
        main_task = "判断这些分配是否存在内存泄漏。"

    # ── 构建完整 prompt ────────────────────────────────────────────────────
    return f"""\
你是一个严格的 C++ 静态分析引擎。
在下面的 C++ 代码块中，发现了以下内存操作：

{chr(10).join(lines)}

请分析该代码块，{main_task}

输出要求：
1. 先给出分析结论，按指针在代码中的声明顺序逐一说明。
   - 有泄漏的指针用 **指针名** 加粗强调。
   - 分配操作（如 new int）和缺少的释放调用（如 delete）用行内代码块 \`包裹\`。
   - 每个指针只写一行，不要另起一行单独写"p2 泄漏。"之类的冗余总结。
2. 如果存在泄漏，用 ```diff 代码块给出修复后的完整代码：
   - 新增行前加 +
   - 删除行前加 -
   - 未改动的行不加任何前缀。

待分析代码：
{fa.func_code}"""


# ═══════════════════════════════════════════════════════════════════════════════
# AI 调用
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_with_ai(prompt: str, client: OpenAI) -> str:
    """将 prompt 发送给 DeepSeek API，返回模型生成的 Markdown 文本。

    发生任何错误时打印警告并返回空字符串。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        content = response.choices[0].message.content
        return content if content else ""
    except Exception as exc:
        console.print(f"[yellow]警告: API 调用失败 — {exc}[/yellow]")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 输出渲染
# ═══════════════════════════════════════════════════════════════════════════════

def _has_actual_leak(plain_text: str) -> bool:
    """判断文本行是否表示真实的内存泄漏（排除否定表述）。"""
    if "泄漏" not in plain_text:
        return False
    for negation in (
        "无泄漏", "没有泄漏", "未泄漏", "不存在泄漏",
        "未发生泄漏", "无内存泄漏",
    ):
        if negation in plain_text:
            return False
    return True


def _render_markdown_with_leak_highlight(md_text: str) -> Text:
    """渲染 Markdown，然后将含真实泄漏的整行标红加粗。

    先用 Rich Markdown 完整渲染保留所有样式，再按行拆分，
    对含"泄漏"的行调用 stylize("bold red") 覆盖样式。
    """
    segments = list(Markdown(md_text).__rich_console__(console, console.options))
    rendered = Text()
    for seg_text, style, control in segments:
        if not control:
            rendered.append(seg_text, style=style)

    plain = rendered.plain
    line_start = 0
    for i, ch in enumerate(plain):
        if ch == "\n" or i == len(plain) - 1:
            line_end = i + 1 if ch == "\n" else i + 1
            line_text = plain[line_start:line_end]
            if _has_actual_leak(line_text):
                rendered.stylize("bold red", line_start, line_end)
            line_start = i + 1

    return rendered


def _render_diff_block(diff_text: str) -> Panel:
    """将 diff 文本渲染为带颜色背景的 Panel（仿 Claude Code diff 风格）。"""
    lines = diff_text.split("\n")
    styled_lines: list[str] = []

    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            styled_lines.append(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            styled_lines.append(f"[bold cyan]{line}[/bold cyan]")
        elif line.startswith("+"):
            styled_lines.append(f"[white on green]{line}[/white on green]")
        elif line.startswith("-"):
            styled_lines.append(f"[white on red]{line}[/white on red]")
        else:
            styled_lines.append(line)

    return Panel(Text.from_markup("\n".join(styled_lines)), border_style="green")


def render_analysis(raw_text: str) -> Panel:
    """将 AI 返回的 Markdown 渲染为带颜色高亮的 Panel。"""
    diff_pattern = re.compile(r"```diff\n(.*?)```", re.DOTALL)
    parts: list = []
    last_end = 0

    for match in diff_pattern.finditer(raw_text):
        if match.start() > last_end:
            parts.append(_render_markdown_with_leak_highlight(
                raw_text[last_end:match.start()]
            ))

        parts.append(_render_diff_block(match.group(1)))
        last_end = match.end()

    if last_end < len(raw_text):
        parts.append(_render_markdown_with_leak_highlight(raw_text[last_end:]))

    body = Group(*parts) if parts else Markdown(raw_text)
    return Panel(body, title="代码分析结果", border_style="yellow")


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="C++ 静态分析工具 — 基于 tree-sitter + LLM 的内存泄漏检测",
    )
    parser.add_argument(
        "target_file",
        nargs="?",
        default="test.cpp",
        help="要分析的 C++ 文件路径（默认 test.cpp）",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="仅收集并打印分配点，不调用 AI",
    )
    return parser.parse_args()


def main() -> None:
    """扫描目标 C++ 文件，按函数分组后逐函数提交 LLM 分析。"""
    args = _parse_args()

    # ── 1. 环境检查 ───────────────────────────────────────────────────────────
    if not args.collect_only:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            console.print(
                "[red]错误: 未设置 DEEPSEEK_API_KEY 环境变量。"
                "请在 .env 文件中配置 API Key。[/red]"
            )
            sys.exit(1)

    # ── 2. 确定目标文件 ───────────────────────────────────────────────────────
    target_file = Path(args.target_file)
    if not target_file.exists():
        console.print(f"[red]错误: 找不到文件 {target_file}[/red]")
        sys.exit(1)

    # ── 3. 初始化 C++ 解析器 ──────────────────────────────────────────────────
    lang = Language(tree_sitter_cpp.language())
    parser = Parser(lang)

    # ── 4. 读取并解析源码 ─────────────────────────────────────────────────────
    code = target_file.read_bytes()
    tree = parser.parse(code)

    # ── 5. 收集所有 new / delete 操作 ─────────────────────────────────────────
    console.print(f"\n[bold]正在扫描: {target_file}[/bold]\n")
    raw_ops = collect_memory_ops(tree.root_node)

    if not raw_ops:
        console.print("[yellow]未发现任何 new / delete 表达式。[/yellow]")
        return

    # ── 6. 填充 func_code 并构建 FunctionAnalysis ─────────────────────────────
    # 重新遍历 AST 获取各函数的源码（比在收集阶段缓存更清晰）
    _func_code_cache: dict[str, str] = {}

    def _cache_func_codes(node: Node) -> None:
        if node.type == "function_definition":
            name = _get_function_name(node)
            start_row = node.start_point.row + 1
            fid = f"{name}:{start_row}"
            if fid not in _func_code_cache:
                _func_code_cache[fid] = node.text.decode() if node.text else ""
        for child in node.children:
            _cache_func_codes(child)

    _cache_func_codes(tree.root_node)

    # 构建 FunctionAnalysis 列表
    raw_ops.sort(key=lambda x: x[0])
    analyses: list[FunctionAnalysis] = []
    current_fa: FunctionAnalysis | None = None
    current_fid: str | None = None

    for func_id, op in raw_ops:
        if func_id != current_fid:
            parts = func_id.rsplit(":", 1)
            func_name = parts[0]
            func_code = _func_code_cache.get(func_id, "(无法定位所在函数)")
            current_fa = FunctionAnalysis(
                func_name=func_name,
                func_code=func_code,
                func_id=func_id,
            )
            analyses.append(current_fa)
            current_fid = func_id

        assert current_fa is not None
        if op.op_type == "new":
            current_fa.add_new(op.row, op.var_name, op.op_text)
        else:
            current_fa.add_delete(op.row, op.var_name, op.op_text)

    # ── 7. 打印摘要 ───────────────────────────────────────────────────────────
    total_new = sum(len(fa.new_ops) for fa in analyses)
    total_delete = sum(
        len([op for op in fa.all_ops if op.op_type in ("delete", "delete[]")])
        for fa in analyses
    )
    console.print(
        f"[dim]发现 {total_new} 个 new / {total_delete} 个 delete，"
        f"分布在 {len(analyses)} 个函数中。[/dim]\n"
    )

    # ── 8. 仅收集模式 ─────────────────────────────────────────────────────────
    if args.collect_only:
        for fa in analyses:
            console.print(f"[bold]函数: {fa.func_name}[/bold]")
            for op in fa.all_ops:
                label = {"new": "[cyan]NEW [/cyan]", "delete": "[green]DEL [/green]", "delete[]": "[green]DEL[][/green]"}[op.op_type]
                console.print(f"  {label} 第 {op.row} 行: {op.var_name} ({op.op_text})")
            # 泄漏快速判断
            for v in fa.leaked_vars:
                console.print(f"  [yellow]  -> {v.var_name}: 可能泄漏（{v.alloc_count} alloc / {v.dealloc_count} dealloc）[/yellow]")
            for op in fa.orphan_deletes:
                console.print(f"  [red]  -> {op.var_name}: 无对应 new 的释放！[/red]")
            console.print()
        return

    # ── 9. 调用 AI 分析 ───────────────────────────────────────────────────────
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    # 构建 prompt 列表（按函数顺序）
    prompt_list: list[tuple[str, str]] = []
    for fa in analyses:
        prompt_list.append((fa.func_id, build_prompt(fa)))

    total = len(prompt_list)
    results: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task(
            f"[dim]正在调用 AI 分析 {total} 个函数…[/dim]", total=total
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(analyze_with_ai, prompt, client): func_id
                for func_id, prompt in prompt_list
            }

            for future in as_completed(future_map):
                func_id = future_map[future]
                results[func_id] = future.result()
                progress.update(task, advance=1)

    # ── 10. 按原顺序渲染结果 ──────────────────────────────────────────────────
    for func_id, _prompt in prompt_list:
        result = results.get(func_id, "")
        if not result:
            # 找到对应函数名
            fa_name = next((fa.func_name for fa in analyses if fa.func_id == func_id), func_id)
            console.print(
                f"[yellow]警告: 函数 {fa_name} 分析失败，已跳过。[/yellow]"
            )
            continue

        panel = render_analysis(result)
        console.print(panel)
        console.print()


if __name__ == "__main__":
    main()
