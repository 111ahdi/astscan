"""C++ 静态分析工具 — 基于 tree-sitter + LLM 的内存泄漏检测。

用法:
    python analyzer.py [target_file]

默认分析 test.cpp，也可指定其他 C++ 文件路径。
"""

import os
import re
import sys
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from tree_sitter import Language, Node, Parser

import tree_sitter_cpp

# ── 加载 .env 中的环境变量（API Key 等）───────────────────────────────────────
load_dotenv()

# ── Rich 控制台实例（整个模块复用同一个）───────────────────────────────────────
console = Console()


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


def get_enclosing_function(node: Node) -> str | None:
    """从 node 出发向上遍历父节点，找到所在的 function_definition 并返回源码。

    若一直遍历到根节点仍未找到，返回 None。
    """
    current = node
    while current is not None:
        if current.type == "function_definition":
            return current.text.decode() if current.text else None
        current = current.parent
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 核心分析逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def collect_allocations(node: Node, allocations: list[dict] | None = None) -> list[dict]:
    """递归遍历 AST，收集所有 new_expression 的元信息。

    参数:
        node:        当前遍历的 AST 节点。
        allocations: 收集结果的列表（递归时复用，外部调用无需传入）。

    返回:
        字典列表，每项包含 row / var_name / alloc_text / func_code。
    """
    if allocations is None:
        allocations = []

    if node.type == "new_expression":
        row = node.start_point.row + 1  # 1-based 行号
        alloc_text = node.text.decode() if node.text else ""

        parent = node.parent
        var_name = _find_identifier(parent) if parent else "?"

        func_code = get_enclosing_function(node) or "(无法定位所在函数)"

        allocations.append({
            "row": row,
            "var_name": var_name,
            "alloc_text": alloc_text,
            "func_code": func_code,
        })

    # 不加 else：new 内部可能嵌套 new，如 new MyClass(new InnerClass())
    for child in node.children:
        collect_allocations(child, allocations)

    return allocations


def build_prompt(func_code: str, items: list[dict]) -> str:
    """为一个函数内的所有分配点生成一条聚合分析 prompt。"""
    allocation_lines = "\n".join(
        f"- 第 {a['row']} 行: 指针 **{a['var_name']}**, 分配 `{a['alloc_text']}`"
        for a in items
    )

    pointer_names = "、".join(
        a["var_name"] for a in items if a["var_name"] != "?"
    )

    return f"""\
你是一个严格的 C++ 静态分析引擎。
在下面的 C++ 代码块中，发现了以下内存分配操作：

{allocation_lines}

请分析该代码块，逐一判断指针 {pointer_names} 是否存在内存泄漏。

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


def analyze_with_ai(prompt: str) -> str:
    """将 prompt 发送给 DeepSeek API，返回模型生成的 Markdown 文本。"""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# 输出渲染 — diff 绿/红高亮 + 泄漏行标红
# ═══════════════════════════════════════════════════════════════════════════════

def _has_actual_leak(line: str) -> bool:
    """判断行是否表示真实的内存泄漏（排除"无泄漏""没有泄漏"等否定表述）。"""
    if "泄漏" not in line:
        return False
    # 排除否定表述
    for negation in ("无泄漏", "没有泄漏", "未泄漏", "不存在泄漏", "未发生泄漏", "无内存泄漏"):
        if negation in line:
            return False
    return True


def _render_markdown_with_leak_highlight(md_text: str) -> Text:
    """渲染 Markdown，将实际存在泄漏的整行文字标红。

    策略：
    1. 按段落（\\n\\n）拆分原文。
    2. 含"泄漏"的段落：逐行判断是否为真实泄漏，是则整行红色加粗。
    3. 不含"泄漏"的段落：直接用 Markdown 渲染。
    """
    paragraphs = md_text.split("\n\n")
    result = Text()

    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue

        if "泄漏" in para:
            lines = para.split("\n")
            styled_lines: list[str] = []
            for line in lines:
                if _has_actual_leak(line):
                    # 泄漏行：去 ** 标记，整行红色加粗
                    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
                    styled_lines.append(f"[bold red]{plain}[/bold red]")
                else:
                    # 非泄漏行：将 **text** 转为 rich 加粗标记
                    converted = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", line)
                    styled_lines.append(converted)
            rendered = Text.from_markup("\n".join(styled_lines))
        else:
            segments = list(Markdown(para).__rich_console__(console, console.options))
            rendered = Text()
            for seg_text, style, control in segments:
                if not control:
                    rendered.append(seg_text, style=style)

        result.append_text(rendered)

        if i < len(paragraphs) - 1:
            result.append("\n\n")

    return result


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

    处理逻辑：
    1. 提取 ```diff 代码块，渲染为绿/红底 diff 面板。
    2. 其余文本使用标准 Markdown 渲染（由 AI 自行控制加粗 / 行内代码等）。
    3. 组装所有部分并包裹 Panel。

    参数:
        raw_text: AI 返回的原始 Markdown 文本。
    """
    diff_pattern = re.compile(r"```diff\n(.*?)```", re.DOTALL)
    parts: list = []
    last_end = 0

    for match in diff_pattern.finditer(raw_text):
        if match.start() > last_end:
            parts.append(
                _render_markdown_with_leak_highlight(raw_text[last_end:match.start()])
            )

        parts.append(_render_diff_block(match.group(1)))
        last_end = match.end()

    if last_end < len(raw_text):
        parts.append(_render_markdown_with_leak_highlight(raw_text[last_end:]))

    body = Group(*parts) if parts else Markdown(raw_text)
    return Panel(body, title="代码分析结果", border_style="yellow")


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """扫描目标 C++ 文件，按函数分组后逐函数提交 LLM 分析。"""
    # ── 1. 确定目标文件 ───────────────────────────────────────────────────────
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.cpp")
    if not target_file.exists():
        console.print(f"[red]错误: 找不到文件 {target_file}[/red]")
        return

    # ── 2. 初始化 C++ 解析器 ──────────────────────────────────────────────────
    lang = Language(tree_sitter_cpp.language())
    parser = Parser(lang)

    # ── 3. 读取并解析源码 ─────────────────────────────────────────────────────
    code = target_file.read_bytes()
    tree = parser.parse(code)

    # ── 4. 收集所有分配点，按所在函数分组 ──────────────────────────────────────
    console.print(f"\n[bold]正在扫描: {target_file}[/bold]\n")
    allocations = collect_allocations(tree.root_node)

    if not allocations:
        console.print("[yellow]未发现任何 new 表达式。[/yellow]")
        return

    allocations.sort(key=lambda a: a["func_code"])
    groups = [
        (func_code, list(items))
        for func_code, items in groupby(allocations, key=lambda a: a["func_code"])
    ]

    console.print(
        f"[dim]发现 {len(allocations)} 个分配点，"
        f"分布在 {len(groups)} 个函数中。[/dim]\n"
    )

    # ── 5. 逐函数提交 LLM 分析，并渲染彩色结果 ────────────────────────────────
    for i, (func_code, items) in enumerate(groups, start=1):
        console.print(
            f"[dim]... 正在分析函数 {i}/{len(groups)} "
            f"（包含 {len(items)} 个分配点）…[/dim]"
        )

        prompt = build_prompt(func_code, items)
        result = analyze_with_ai(prompt)

        panel = render_analysis(result)
        console.print(panel)
        console.print()


if __name__ == "__main__":
    main()
