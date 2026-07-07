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
from dataclasses import dataclass
from itertools import groupby
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
class AllocationInfo:
    """一次内存分配操作的完整元信息。"""
    row: int               # 1-based 行号
    var_name: str          # 指针变量名，未知时为 "?"
    alloc_text: str        # 分配表达式源码，如 "new int"
    func_name: str         # 所在函数名，未知时为 "(未知函数)"
    func_code: str         # 所在函数的完整源码
    func_id: str           # 唯一函数标识符 "函数名:起始行号"


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
    # 备选：lambda / 匿名函数等情况
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
# 核心分析逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def collect_allocations(
    node: Node,
    allocations: list[AllocationInfo] | None = None,
) -> list[AllocationInfo]:
    """递归遍历 AST，收集所有 new_expression 的元信息。"""
    if allocations is None:
        allocations = []

    if node.type == "new_expression":
        row = node.start_point.row + 1
        alloc_text = node.text.decode() if node.text else ""

        parent = node.parent
        var_name = _find_identifier(parent) if parent else "?"

        func_code, func_name, func_start_row = get_enclosing_function(node)
        if func_code is None:
            func_code = "(无法定位所在函数)"

        # 构造唯一函数标识：函数名:函数起始行
        func_id = f"{func_name}:{func_start_row}" if func_name != "(全局作用域)" else "(全局作用域)"

        allocations.append(AllocationInfo(
            row=row,
            var_name=var_name,
            alloc_text=alloc_text,
            func_name=func_name,
            func_code=func_code,
            func_id=func_id,
        ))

    # 不加 else：new 内部可能嵌套 new
    for child in node.children:
        collect_allocations(child, allocations)

    return allocations


def build_prompt(func_code: str, items: list[AllocationInfo]) -> str:
    """为一个函数内的所有分配点生成一条聚合分析 prompt。"""
    # 分离有变量名和匿名的分配
    named = [a for a in items if a.var_name != "?"]
    anonymous = [a for a in items if a.var_name == "?"]

    lines: list[str] = []
    for a in items:
        if a.var_name != "?":
            lines.append(
                f"- 第 {a.row} 行: 指针 **{a.var_name}**, 分配 `{a.alloc_text}`"
            )
        else:
            lines.append(
                f"- 第 {a.row} 行: (未绑定变量) 分配 `{a.alloc_text}`"
            )

    pointer_names = "、".join(a.var_name for a in named)

    # 构建提示文案
    if named:
        main_task = f"逐一判断指针 {pointer_names} 是否存在内存泄漏。"
    else:
        main_task = "判断这些分配是否存在内存泄漏。"

    if anonymous:
        anonymous_note = (
            "\n注意：存在未绑定变量的内存分配（如作为构造函数参数），"
            "请根据上下文判断是否需要关注。"
        )
    else:
        anonymous_note = ""

    return f"""\
你是一个严格的 C++ 静态分析引擎。
在下面的 C++ 代码块中，发现了以下内存分配操作：

{chr(10).join(lines)}
{anonymous_note}
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
{func_code}"""


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
    # ── 第一步：全段 Markdown 渲染 ────────────────────────────────────────────
    segments = list(Markdown(md_text).__rich_console__(console, console.options))
    rendered = Text()
    for seg_text, style, control in segments:
        if not control:
            rendered.append(seg_text, style=style)

    # ── 第二步：按换行定位每一行，标红泄漏行 ──────────────────────────────────
    plain = rendered.plain
    line_start = 0
    for i, ch in enumerate(plain):
        if ch == "\n" or i == len(plain) - 1:
            line_end = i + 1 if ch == "\n" else i + 1
            line_text = plain[line_start:line_end]
            if _has_actual_leak(line_text):
                rendered.stylize("bold red", line_start, line_end)
            line_start = i + 1  # 跳过换行符

    return rendered


def _render_diff_block(diff_text: str) -> Panel:
    """将 diff 文本渲染为带颜色背景的 Panel（仿 Claude Code diff 风格）。

    + 开头的行 → 绿底白字（新增）
    - 开头的行 → 红底白字（删除）
    @@ 开头的行 → 青色加粗（hunk header）
    +++ / ---   → 加粗（文件头）
    其余行       → 保持原样（上下文）
    """
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
    """将 AI 返回的 Markdown 渲染为带颜色高亮的 Panel。

    1. 提取 ```diff 代码块，渲染为绿/红底 diff 面板。
    2. 其余文本用 Markdown 渲染后将泄漏行标红。
    3. 组装并包裹 Panel。
    """
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

    # ── 5. 收集所有分配点 ─────────────────────────────────────────────────────
    console.print(f"\n[bold]正在扫描: {target_file}[/bold]\n")
    allocations = collect_allocations(tree.root_node)

    if not allocations:
        console.print("[yellow]未发现任何 new 表达式。[/yellow]")
        return

    # ── 6. 按函数分组 ─────────────────────────────────────────────────────────
    allocations.sort(key=lambda a: a.func_id)
    groups: list[tuple[str, list[AllocationInfo]]] = [
        (func_id, list(items))
        for func_id, items in groupby(allocations, key=lambda a: a.func_id)
    ]

    console.print(
        f"[dim]发现 {len(allocations)} 个分配点，"
        f"分布在 {len(groups)} 个函数中。[/dim]\n"
    )

    # ── 7. 仅收集模式 ─────────────────────────────────────────────────────────
    if args.collect_only:
        for func_id, items in groups:
            func_name = items[0].func_name
            console.print(f"[bold]函数: {func_name}[/bold] ({len(items)} 个分配点)")
            for a in items:
                console.print(
                    f"  第 {a.row} 行: {a.var_name} -> {a.alloc_text}"
                )
            console.print()
        return

    # ── 8. 调用 AI 分析 ───────────────────────────────────────────────────────
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    # 构建每个函数的 prompt
    prompts: list[tuple[str, str, list[AllocationInfo]]] = []
    for func_id, items in groups:
        func_code = items[0].func_code
        prompt = build_prompt(func_code, items)
        prompts.append((func_id, prompt, items))

    # 并发调用
    results: dict[str, str] = {}
    total = len(prompts)

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
                executor.submit(analyze_with_ai, prompt, client): (func_id, prompt)
                for func_id, prompt, _ in prompts
            }

            for future in as_completed(future_map):
                func_id, _ = future_map[future]
                results[func_id] = future.result()
                progress.update(task, advance=1)

    # ── 9. 按原顺序渲染结果 ──────────────────────────────────────────────────
    for func_id, prompt, items in prompts:
        result = results.get(func_id, "")
        if not result:
            console.print(
                f"[yellow]警告: 函数 {items[0].func_name} 分析失败，已跳过。[/yellow]"
            )
            continue

        panel = render_analysis(result)
        console.print(panel)
        console.print()


if __name__ == "__main__":
    main()
