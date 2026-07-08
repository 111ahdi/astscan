"""C++ 静态分析工具 — 基于 tree-sitter + LLM 的内存泄漏检测。

用法:
    python analyzer.py [target_file] [--collect-only]

选项:
    target_path      要分析的 C++ 文件或项目文件夹路径（默认 test.cpp）
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

    @property
    def has_mismatch(self) -> bool:
        """检查是否发生了 new/delete 与 new[]/delete[] 混用。"""
        if not self.allocs or not self.deallocs:
            return False
        alloc_is_array = "[]" in self.allocs[0].op_type
        dealloc_is_array = "[]" in self.deallocs[0].op_type
        return alloc_is_array != dealloc_is_array


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

    def add_new(self, row: int, var_name: str, op_text: str, op_type: str = "new") -> None:
        self._get_or_create(var_name).allocs.append(
            MemoryOp(row=row, var_name=var_name, op_type=op_type, op_text=op_text)
        )

    def add_delete(self, row: int, var_name: str, op_text: str,
                   op_type: str | None = None) -> None:
        if op_type is None:
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
        return [op for op in self.all_ops if op.op_type in ("new", "new[]")]

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


@dataclass
class ClassInfo:
    """类级别信息，用于跨成员函数追踪成员变量的分配/释放配对。"""
    class_name: str
    member_vars: set[str] = field(default_factory=set)    # 成员变量名集合
    method_ids: list[str] = field(default_factory=list)    # 归属该类的 func_id 列表
    ctor_ids: list[str] = field(default_factory=list)      # 构造函数 func_id
    dtor_ids: list[str] = field(default_factory=list)      # 析构函数 func_id


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
    """从 function_definition 节点中提取函数名。

    注意：function_declarator 可能被 pointer_declarator / reference_declarator 包裹
    （如 int* func() 或 int& func()），需要递归查找。
    同时处理析构函数名（~ClassName）。
    """
    def _find_dtor_name(node: Node) -> Node | None:
        """在子树中递归查找 destructor_name 节点。"""
        if node.type == "destructor_name":
            return node
        for child in node.children:
            result = _find_dtor_name(child)
            if result:
                return result
        return None

    def _search(node: Node) -> str:
        if node.type == "function_declarator":
            # 优先检测析构函数（~ClassName）
            dtor = _find_dtor_name(node)
            if dtor:
                ident = _find_identifier(dtor)
                return f"~{ident}" if ident != "?" else "?"
            return _find_identifier(node)
        for child in node.children:
            result = _search(child)
            if result != "?":
                return result
        return "?"
    return _search(func_node)


def _resolve_method_class(func_node: Node) -> tuple[str | None, str, bool, bool]:
    """判断函数是否属于某个类。

    返回:
        (class_name, short_func_name, is_constructor, is_destructor)
        class_name 为 None 表示自由函数。

    支持两种形式：
    1. 行内定义 — function_definition 直接嵌套在 class_specifier 内
    2. 行外定义 — 如 Player::Player()、Player::~Player()
    """
    # 情况 1：行内成员函数（父级链包含 class_specifier / struct_specifier）
    cursor = func_node.parent
    while cursor is not None:
        if cursor.type in ("class_specifier", "struct_specifier"):
            name_node = cursor.child_by_field_name("name")
            class_name = name_node.text.decode() if name_node and name_node.text else "?"
            short_name = _get_function_name(func_node)
            is_dtor = short_name.startswith("~")
            is_ctor = (short_name == class_name)
            return class_name, short_name, is_ctor, is_dtor
        cursor = cursor.parent

    # 情况 2：行外定义 — 通过 function_declarator 中的 :: 识别
    for child in func_node.children:
        if child.type == "function_declarator":
            decl_text = child.text.decode() if child.text else ""
            paren = decl_text.find("(")
            name_part = decl_text[:paren] if paren != -1 else decl_text.strip()
            if "::" in name_part:
                parts = name_part.split("::")
                class_name = parts[0].strip()
                short_name = parts[1].strip() if len(parts) > 1 else ""
                is_dtor = short_name.startswith("~")
                is_ctor = (short_name == class_name)
                return class_name, short_name, is_ctor, is_dtor
            break  # 只检查第一个 function_declarator

    return None, _get_function_name(func_node), False, False


def _collect_callees(node: Node, result: set[str]) -> None:
    """递归遍历子树，收集所有 call_expression 中被调用的函数名。"""
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            name = _find_identifier(func_node)
            if name != "?":
                result.add(name)
    for child in node.children:
        _collect_callees(child, result)


def _collect_class_definitions(tree_root: Node) -> dict[str, ClassInfo]:
    """遍历 AST 收集所有 class_specifier，提取成员变量声明。"""
    classes: dict[str, ClassInfo] = {}

    def _extract_fields(body_node: Node, info: ClassInfo) -> None:
        """从类体中递归提取 field_declaration 的变量名。

        跳过函数声明（function_declarator），只收集真正的成员变量。
        """
        def _find_field_name(decl_node: Node) -> str:
            """在 declarator 子树中递归查找 field_identifier。"""
            if decl_node.type in ("field_identifier", "identifier"):
                return decl_node.text.decode() if decl_node.text else "?"
            for c in decl_node.children:
                result = _find_field_name(c)
                if result and result != "?":
                    return result
            return ""

        for child in body_node.children:
            if child.type == "field_declaration":
                # 跳过函数声明（构造/析构/成员函数）
                if any(c.type == "function_declarator"
                       for c in child.children):
                    pass  # 函数声明，不提取
                else:
                    decl = child.child_by_field_name("declarator")
                    if decl:
                        name = _find_field_name(decl)
                        if name:
                            info.member_vars.add(name)
            if child.child_count > 0:
                _extract_fields(child, info)

    def _scan(node: Node) -> None:
        if node.type in ("class_specifier", "struct_specifier"):
            name_node = node.child_by_field_name("name")
            class_name = name_node.text.decode() if name_node and name_node.text else None
            if class_name:
                info = ClassInfo(class_name=class_name)
                body = node.child_by_field_name("body")
                if body:
                    _extract_fields(body, info)
                classes[class_name] = info
        for child in node.children:
            _scan(child)

    _scan(tree_root)
    return classes


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
    file_path: str = "",
) -> list[tuple[str, MemoryOp]]:
    """递归遍历 AST，收集所有 new_expression 和 delete_expression。

    参数:
        node:      AST 根节点。
        ops:       收集结果的列表（递归复用）。
        file_path: 源文件路径，用于构造唯一 func_id 避免跨文件冲突。

    返回:
        (func_id, MemoryOp) 列表，func_id = "文件路径:函数名:起始行"。
    """
    if ops is None:
        ops = []

    if node.type in ("new_expression", "delete_expression"):
        row = node.start_point.row + 1
        op_text = node.text.decode() if node.text else ""

        parent = node.parent
        var_name = _find_identifier(parent) if parent else "?"

        _, func_name, func_start_row = get_enclosing_function(node)

        # func_id 包含文件路径，确保跨文件唯一
        scope = file_path if file_path else "(全局作用域)"
        func_id = f"{scope}:{func_name}:{func_start_row}" if func_name != "(全局作用域)" else scope

        if node.type == "new_expression":
            op_type = "new[]" if "[" in op_text else "new"
        else:
            op_type = "delete[]" if "[" in op_text else "delete"

        ops.append((
            func_id,
            MemoryOp(row=row, var_name=var_name, op_type=op_type, op_text=op_text),
        ))

    # 不加 else：new 内部可能嵌套 new / delete
    for child in node.children:
        collect_memory_ops(child, ops, file_path=file_path)

    return ops


# ═══════════════════════════════════════════════════════════════════════════════
# 类级别跨函数合并
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_class_level_traces(
    analyses: list[FunctionAnalysis],
    class_defs: dict[str, ClassInfo],
) -> None:
    """对每个类，将成员变量的 alloc/dealloc 跨成员函数配对。

    若成员变量 data 在构造函数中 new，在析构函数中 delete，
    则在构造函数中补充"虚拟 dealloc"，在析构函数中补充"虚拟 alloc"，
    使各自的 VariableTrace 恢复平衡（is_leaked=False, orphan_deletes 消失）。
    """
    fa_map: dict[str, FunctionAnalysis] = {fa.func_id: fa for fa in analyses}

    for _cls_name, info in class_defs.items():
        if not info.method_ids:
            continue

        for var_name in info.member_vars:
            all_allocs: list[tuple[str, MemoryOp]] = []     # (func_id, MemoryOp)
            all_deallocs: list[tuple[str, MemoryOp]] = []

            for fid in info.method_ids:
                fa = fa_map.get(fid)
                if not fa or var_name not in fa.variables:
                    continue
                trace = fa.variables[var_name]
                for op in trace.allocs:
                    all_allocs.append((fid, op))
                for op in trace.deallocs:
                    all_deallocs.append((fid, op))

            # 仅当跨函数同时存在 alloc 和 dealloc 时才处理
            if not all_allocs or not all_deallocs:
                continue

            # ── 配对规则：至少一端必须是 ctor 或 dtor ──
            # ctor 分配但 dtor 未释放 → 真正的泄漏，不做跨函数配对
            alloc_funcs = {fid for fid, _ in all_allocs}
            dealloc_funcs = {fid for fid, _ in all_deallocs}
            has_ctor_alloc = bool(alloc_funcs & set(info.ctor_ids))
            has_dtor_dealloc = bool(dealloc_funcs & set(info.dtor_ids))

            if has_ctor_alloc and not has_dtor_dealloc:
                # 构造函数分配了，但析构函数没释放 → 潜在泄漏，不配对
                continue
            if not has_ctor_alloc and not has_dtor_dealloc:
                # 两端都是普通成员函数 → 不确定性太高，不配对
                continue

            # 在 alloc 所在函数中补充虚拟 dealloc（类型匹配）
            for fid, alloc_op in all_allocs:
                fa = fa_map.get(fid)
                if fa:
                    matched_del = "delete[]" if alloc_op.op_type == "new[]" else "delete"
                    fa.add_delete(alloc_op.row, var_name,
                                  f"(跨函数: 由其他成员函数 {matched_del} 释放)",
                                  op_type=matched_del)

            # 在 dealloc 所在函数中补充虚拟 alloc（类型匹配）
            for fid, dealloc_op in all_deallocs:
                fa = fa_map.get(fid)
                if fa:
                    matched_new = "new[]" if dealloc_op.op_type == "delete[]" else "new"
                    fa.add_new(dealloc_op.row, var_name,
                               f"(跨函数: 由其他成员函数 {matched_new} 分配)",
                               op_type=matched_new)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    fa: FunctionAnalysis,
    global_functions: dict[str, str] | None = None,
    func_callees: dict[str, set[str]] | None = None,
    func_class_map: dict[str, str] | None = None,
    class_defs: dict[str, ClassInfo] | None = None,
    func_code_cache: dict[str, str] | None = None,
) -> str:
    """为一个函数生成聚合分析 prompt。

    参数:
        fa:              当前函数的分析数据。
        global_functions: 纯函数名 → 函数源码（跨文件全局映射）。
        func_callees:     func_id → 该函数内部调用的函数名集合。
        func_class_map:   func_id → 类名（用于跨函数分析）。
        class_defs:       类名 → ClassInfo（成员变量和方法列表）。
        func_code_cache:  func_id → 函数源码。
    """
    lines: list[str] = []

    # ── 列出所有 new 操作 ──────────────────────────────────────────────────
    for op in fa.new_ops:
        if op.op_text.startswith("(跨函数:"):
            continue  # 虚拟操作（跨函数配对），不展示给 AI
        kind = "new[]" if op.op_type == "new[]" else "new"
        if op.var_name != "?":
            lines.append(f"- 第 {op.row} 行: **{kind}** 分配，指针 **{op.var_name}** (`{op.op_text}`)")
        else:
            lines.append(f"- 第 {op.row} 行: **{kind}** 分配，(未绑定变量) (`{op.op_text}`)")

    # ── 列出所有 delete 操作 ───────────────────────────────────────────────
    for op in fa.all_ops:
        if op.op_type in ("delete", "delete[]"):
            if op.op_text.startswith("(跨函数:"):
                continue  # 虚拟操作（跨函数配对），不展示给 AI
            lines.append(f"- 第 {op.row} 行: **{op.op_type}** 释放，变量 **{op.var_name}** (`{op.op_text}`)")

    # ── 汇总变量状态 ────────────────────────────────────────────────────────
    named_all = [op.var_name for op in fa.new_ops if op.var_name != "?"]

    if named_all:
        main_task = f"逐一判断指针 {'、'.join(named_all)} 是否存在内存泄漏。"
    else:
        main_task = "判断这些分配是否存在内存泄漏。"

    # ── 关联函数参考 ────────────────────────────────────────────────────────
    reference_block = ""
    if global_functions and func_callees:
        callees = func_callees.get(fa.func_id, set())
        related = [fn for fn in callees if fn in global_functions]
        if related:
            reference_lines: list[str] = []
            for fn in related:
                reference_lines.append(f"\n### {fn}\n```cpp\n{global_functions[fn]}\n```")
            reference_block = "\n## 关联函数的源码（供参考）\n" + "\n".join(reference_lines)

    # ── 同类成员函数参考 ───────────────────────────────────────────────────
    class_block = ""
    if func_class_map and class_defs and func_code_cache:
        cls_name = func_class_map.get(fa.func_id)
        if cls_name:
            info = class_defs.get(cls_name)
            if info:
                peers = [fid for fid in info.method_ids
                         if fid != fa.func_id and fid in func_code_cache]
                if peers:
                    peer_lines: list[str] = []
                    for fid in peers:
                        parts = fid.rsplit(":", 2)
                        peer_name = parts[-2] if len(parts) >= 2 else fid
                        peer_lines.append(
                            f"\n### {peer_name} (同类 {cls_name})\n```cpp\n{func_code_cache[fid]}\n```"
                        )
                    class_block = (
                        f"\n## 同类 {cls_name} 的其他成员函数（供跨函数分析参考）\n"
                        + "\n".join(peer_lines)
                    )

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
{fa.func_code}{reference_block}{class_block}"""


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
        "target_path",
        nargs="?",
        default="test.cpp",
        help="要分析的 C++ 文件或项目文件夹路径（默认 test.cpp）",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="仅收集并打印分配点，不调用 AI",
    )
    return parser.parse_args()


def resolve_targets(target_path: str) -> list[Path]:
    """将用户输入的路径解析为待分析文件列表。

    - 文件 → 返回只含该文件的列表
    - 目录 → 递归搜集所有 .cpp / .h / .hpp 文件
    - 不存在 → 报错退出
    """
    path = Path(target_path)
    if not path.exists():
        console.print(f"[red]错误: 路径不存在 — {path}[/red]")
        sys.exit(1)
    if path.is_file():
        return [path]

    extensions = ("*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp", "*.hh", "*.hxx")
    files: list[Path] = []
    for ext in extensions:
        files.extend(path.rglob(ext))
    return sorted(files)


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

    # ── 2. 解析目标文件列表 ───────────────────────────────────────────────────
    source_files = resolve_targets(args.target_path)
    console.print(f"[dim]共发现 {len(source_files)} 个源文件。[/dim]")

    # ── 3. 初始化 C++ 解析器（只创建一次）─────────────────────────────────────
    lang = Language(tree_sitter_cpp.language())
    parser = Parser(lang)

    # ── 4. 跨文件收集所有 new / delete 操作 ───────────────────────────────────
    all_raw_ops: list[tuple[str, MemoryOp]] = []
    func_code_cache: dict[str, str] = {}          # func_id → 函数源码
    func_callees: dict[str, set[str]] = {}        # func_id → 该函数内部调用的函数名集合
    global_functions: dict[str, str] = {}          # 纯函数名 → 函数源码（同名后者覆盖）
    global_classes: dict[str, ClassInfo] = {}       # 类名 → 类信息（成员变量、方法列表）
    func_class_map: dict[str, str] = {}             # func_id → 所属类名

    def _cache_functions(tree_root: Node, file_path_str: str) -> None:
        """遍历 AST，缓存函数源码、被调用函数名，更新全局函数/类映射。"""
        for node in tree_root.children:
            _cache_functions(node, file_path_str)  # 先递归子节点
            if node.type == "function_definition":
                name = _get_function_name(node)
                start_row = node.start_point.row + 1
                fid = f"{file_path_str}:{name}:{start_row}"
                func_code = node.text.decode() if node.text else ""

                if fid not in func_code_cache:
                    func_code_cache[fid] = func_code
                # 记录当前函数内调用过哪些函数
                callees: set[str] = set()
                _collect_callees(node, callees)
                func_callees[fid] = callees
                # 纯函数名 → 源码（方便跨文件按函数名检索）
                if name not in ("(匿名函数)", "(全局作用域)"):
                    global_functions[name] = func_code

                # ── 新增：检测类归属 ──
                cls_name, _short, is_ctor, is_dtor = _resolve_method_class(node)
                if cls_name and cls_name in global_classes:
                    func_class_map[fid] = cls_name
                    global_classes[cls_name].method_ids.append(fid)
                    if is_ctor:
                        global_classes[cls_name].ctor_ids.append(fid)
                    if is_dtor:
                        global_classes[cls_name].dtor_ids.append(fid)

    # ── 第一趟：收集所有类定义（需先于函数缓存，因 .cpp 可能先于 .h 扫描）───
    for file_path in source_files:
        code = file_path.read_bytes()
        tree = parser.parse(code)
        file_classes = _collect_class_definitions(tree.root_node)
        for cls_name, info in file_classes.items():
            if cls_name not in global_classes:
                global_classes[cls_name] = info
            else:
                global_classes[cls_name].member_vars.update(info.member_vars)

    # ── 第二趟：收集内存操作 + 缓存函数信息 ──────────────────────────────────
    for file_path in source_files:
        console.print(f"[dim]  扫描: {file_path}[/dim]")

        code = file_path.read_bytes()
        tree = parser.parse(code)

        file_ops = collect_memory_ops(tree.root_node, file_path=str(file_path))
        all_raw_ops.extend(file_ops)

        _cache_functions(tree.root_node, str(file_path))

    if not all_raw_ops:
        console.print("[yellow]未发现任何 new / delete 表达式。[/yellow]")
        return

    # ── 5. 构建 FunctionAnalysis 列表 ─────────────────────────────────────────
    all_raw_ops.sort(key=lambda x: x[0])
    analyses: list[FunctionAnalysis] = []
    current_fa: FunctionAnalysis | None = None
    current_fid: str | None = None

    for func_id, op in all_raw_ops:
        if func_id != current_fid:
            # func_id = "文件路径:函数名:起始行"
            parts = func_id.rsplit(":", 2)       # 从右侧拆两次: 文件名, 函数名, 行号
            func_name = parts[-2] if len(parts) >= 2 else func_id
            func_code = func_code_cache.get(func_id, "(无法定位所在函数)")
            current_fa = FunctionAnalysis(
                func_name=func_name,
                func_code=func_code,
                func_id=func_id,
            )
            analyses.append(current_fa)
            current_fid = func_id

        assert current_fa is not None
        if op.op_type in ("new", "new[]"):
            current_fa.add_new(op.row, op.var_name, op.op_text, op.op_type)
        else:
            current_fa.add_delete(op.row, op.var_name, op.op_text)

    # ── 5.5 类级别跨函数合并（成员变量跨 ctor/dtor 配对）────────────────────
    _merge_class_level_traces(analyses, global_classes)

    # ── 6. 打印摘要 ───────────────────────────────────────────────────────────
    total_new = sum(len(fa.new_ops) for fa in analyses)
    total_delete = sum(
        len([op for op in fa.all_ops if op.op_type in ("delete", "delete[]")])
        for fa in analyses
    )
    console.print(
        f"\n[bold]分析完毕:[/bold] "
        f"[dim]{len(source_files)} 个文件, {len(analyses)} 个函数, "
        f"{total_new} 个 new / {total_delete} 个 delete。[/dim]\n"
    )

    # ── 7. 仅收集模式 ─────────────────────────────────────────────────────────
    if args.collect_only:
        for fa in analyses:
            console.print(f"[bold]函数: {fa.func_name}[/bold]")
            for op in fa.all_ops:
                label = {
                    "new": "[cyan]NEW  [/cyan]", "new[]": "[cyan]NEW[][/cyan]",
                    "delete": "[green]DEL  [/green]", "delete[]": "[green]DEL[][/green]",
                }.get(op.op_type, f"[dim]{op.op_type:7}[/dim]")
                console.print(f"  {label} 第 {op.row} 行: {op.var_name} ({op.op_text})")
            for v in fa.leaked_vars:
                console.print(f"  [yellow]  -> {v.var_name}: 可能泄漏（{v.alloc_count} alloc / {v.dealloc_count} dealloc）[/yellow]")
            for v in fa.variables.values():
                if v.has_mismatch and v.var_name != "?":
                    console.print(
                        f"  [red]  -> {v.var_name}: 交叉释放！（{v.allocs[0].op_type} / {v.deallocs[0].op_type}）[/red]"
                    )
            for op in fa.orphan_deletes:
                console.print(f"  [red]  -> {op.var_name}: 无对应 new 的释放！[/red]")
            console.print()
        return

    # ── 8. 静态判定：三层分流 ──────────────────────────────────────────────────
    safe_funcs: list[FunctionAnalysis] = []       # 内存闭环，无泄漏
    mismatch_funcs: list[tuple[FunctionAnalysis, list[VariableTrace]]] = []  # 交叉释放
    needs_ai: list[FunctionAnalysis] = []          # 疑似泄漏，需 AI 深度分析

    for fa in analyses:
        has_allocs = len(fa.new_ops) > 0
        has_mismatches = any(v.has_mismatch for v in fa.variables.values())
        has_leaks = len(fa.leaked_vars) > 0 or len(fa.orphan_deletes) > 0

        # 交叉释放 → 本地报致命错误（可与其他层级并存）
        if has_mismatches:
            mismatched_vars = [v for v in fa.variables.values() if v.has_mismatch]
            mismatch_funcs.append((fa, mismatched_vars))

        if not has_allocs and not has_leaks:
            # 没有任何内存操作 → 跳过（理论上不会出现）
            continue

        if not has_mismatches and not has_leaks:
            # 内存完全闭环 → 安全
            safe_funcs.append(fa)
        elif has_leaks:
            # 存在泄漏嫌疑 → 交给 AI（即使同时有 mismatch，leak 部分仍需 AI）
            needs_ai.append(fa)

    # ── 打印安全函数 ──────────────────────────────────────────────────────────
    for fa in safe_funcs:
        # 检测是否依赖跨函数配对（含虚拟 op）
        has_cross = any(
            op.op_text.startswith("(跨函数:")
            for v in fa.variables.values()
            for op in v.allocs + v.deallocs
        )
        tag = " [dim](跨函数配对确认)[/dim]" if has_cross else ""
        real_allocs = len([op for op in fa.new_ops
                           if not op.op_text.startswith("(跨函数:")])
        real_deallocs = sum(
            len([op for op in v.deallocs if not op.op_text.startswith("(跨函数:")])
            for v in fa.variables.values()
        )
        console.print(
            f"[green][AST 静态确认安全] 函数 {fa.func_name} 内存闭环，无泄漏。{tag}"
            f"（{real_allocs} alloc / {real_deallocs} dealloc）[/green]"
        )
    if safe_funcs:
        console.print()

    # ── 打印致命错误（交叉释放）─────────────────────────────────────────────────
    for fa, mismatched in mismatch_funcs:
        for v in mismatched:
            alloc_op = v.allocs[0]
            dealloc_op = v.deallocs[0]
            console.print(
                f"[red][致命错误] 函数 {fa.func_name} 指针 {v.var_name} "
                f"分配为 {alloc_op.op_type}（{alloc_op.op_text}），"
                f"却使用 {dealloc_op.op_type}（{dealloc_op.op_text}）释放！"
                f"（第 {alloc_op.row} 行分配，第 {dealloc_op.row} 行释放）[/red]"
            )
    if mismatch_funcs:
        console.print()

    # ── 9. 仅疑难杂症调用 AI 分析 ────────────────────────────────────────────
    if not needs_ai:
        if mismatch_funcs:
            console.print("[yellow]所有问题已在本地判定（交叉释放），无需 AI 分析。[/yellow]")
        else:
            console.print("[green]所有函数均为静态安全，无需 AI 分析。[/green]")
        return

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    prompt_list: list[tuple[str, str]] = []
    for fa in needs_ai:
        prompt_list.append(
            (fa.func_id, build_prompt(fa, global_functions, func_callees,
                                      func_class_map, global_classes, func_code_cache))
        )

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

    # ── 10. 按原顺序渲染 AI 结果（安全/纯交叉释放函数已本地处理）──────────────
    skip_ids = {fa.func_id for fa in safe_funcs}
    skip_ids.update(fa.func_id for fa, _ in mismatch_funcs if fa not in needs_ai)
    for fa in analyses:
        if fa.func_id in skip_ids:
            continue

        result = results.get(fa.func_id, "")
        if not result:
            console.print(
                f"[yellow]警告: 函数 {fa.func_name} 分析失败，已跳过。[/yellow]"
            )
            continue

        panel = render_analysis(result)
        console.print(panel)
        console.print()


if __name__ == "__main__":
    main()
