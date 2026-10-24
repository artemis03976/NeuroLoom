"""Static import (PRD 11.2–11.3): PyTorch source -> NeuroLoom IR without running it.

Level L1 turns `__init__` assignments into layers, module instances, repeats and
nn.Sequential containers, and the `forward` data flow into edges; parameter
values that refer to constructor arguments stay expressions. Level L3 keeps
what L1 cannot structure as Python: statements become code-block nodes, and a
class that cannot be structured at all becomes an opaque module with its source.

Nothing in the user's file is executed. torch is imported only to read layer
constructor signatures, and is optional for that.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import importlib
import inspect
import json
import re
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .i18n import t
from .importer import BATCH_PROBE, INPUT, OUTPUT, ImportFailure, InputSpec

CODE = "neuroloom.Code"
NN_MODULE = "torch.nn.Module"
SEQUENTIAL = "torch.nn.Sequential"
MODULE_LIST = "torch.nn.ModuleList"

# Bindings the generated code creates itself; imports that only create these are not kept.
# dtype= of a Parameter / Buffer initial value.
STATE_DTYPES = {
    "torch.float32": "float32",
    "torch.float": "float32",
    "torch.float64": "float64",
    "torch.double": "float64",
    "torch.float16": "float16",
    "torch.half": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int64": "int64",
    "torch.long": "int64",
    "torch.int32": "int32",
    "torch.int": "int32",
    "torch.bool": "bool",
}
# Imports the generated code writes itself (OrderedDict: for nn.Sequential with named children).
STANDARD_BINDINGS = {"torch": "torch", "nn": "torch.nn", "F": "torch.nn.functional", "OrderedDict": "collections.OrderedDict"}
# Names expressions may use besides hyperparameters and constants (the TypeScript evaluator knows them).
EXPR_BUILTINS = {"int", "float", "abs", "min", "max", "round", "len", "bool", "sum", "tuple", "list", "math", "True", "False", "None"}
BINOPS = {ast.Add: "operator.add", ast.Sub: "operator.sub", ast.Mult: "operator.mul", ast.Div: "operator.truediv", ast.MatMult: "operator.matmul"}


class Unsupported(Exception):
    """A construct level L1 cannot structure; it falls back to L3 and the reason goes into the report."""


class Opaque(Exception):
    """A class that cannot be structured at all; it is kept as source."""


# ---------------------------------------------------------------- catalog


@dataclass
class _Catalog:
    layers: dict[str, dict]
    functions: dict[str, dict]
    methods: dict[str, dict]


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    manifest = json.loads(Path(__file__).with_name("specs.json").read_text(encoding="utf-8"))
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise RuntimeError("specs.json is out of date; run `npm run specs`")
    layers, functions, methods = {}, {}, {}
    for spec in manifest["specs"]:
        if spec.get("layerClass"):
            layers[spec["layerClass"]] = spec
        python = spec.get("python") or {}
        for name in python.get("functions", []):
            functions[name] = spec
        for name in python.get("methods", []):
            methods[name] = spec
    return _Catalog(layers, functions, methods)


_NO_DEFAULT = object()


@lru_cache(maxsize=None)
def _ctor_signature(qualname: str, spec_params: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Positional parameter order and defaults of a torch layer; the spec's order when torch is missing."""
    with contextlib.suppress(Exception):
        module, _, attr = qualname.rpartition(".")
        cls = getattr(importlib.import_module(module), attr)
        params = list(inspect.signature(cls).parameters.values())
        if any(p.kind == p.VAR_POSITIONAL for p in params):  # nn.LSTM(*args, **kwargs): the spec follows the documented order
            return spec_params, {}
        names = tuple(p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        defaults = {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}
        return names, defaults
    return spec_params, {}


def _literal(value: Any) -> Any:
    """A Python literal as IR JSON: tuples become lists."""
    if isinstance(value, (tuple, list)):
        return [_literal(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(value)


def _normalize(value: Any, param: dict) -> Any:
    """The canonical literal for a parameter, as NeuroLoom writes it: uniform tuples collapse (kernel_size (3, 3) -> 3)."""
    if param["type"] == "ints" and isinstance(value, list) and value:
        if param.get("arity") is not None and all(v == value[0] for v in value):
            return value[0]
        if param.get("arity") is None and len(value) == 1:
            return value[0]
    if param["type"] == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _outputs(spec: dict, params: dict) -> list[dict]:
    dynamic = spec.get("dynamicOutputs")
    if dynamic:
        count = params.get(dynamic["param"], {}).get("literal")
        if not isinstance(count, int):
            raise Unsupported(t(f"{spec['type']} 的输出个数不是常数", f"the number of outputs of {spec['type']} is not a constant"))
        return [{"name": f"{dynamic['prefix']}{i}", "path": [i]} for i in range(count)]
    return spec["outputs"]


# ---------------------------------------------------------------- the file


class _File:
    def __init__(self, source: str, tree: ast.Module):
        self.source = source
        self.lines = source.splitlines()
        self.tree = tree
        self.aliases: dict[str, str] = {}
        # Names bound at module level other than by imports: constants, helper functions.
        self.globals: set[str] = set()
        for stmt in tree.body:
            for local, qualified in _bindings(stmt):
                self.aliases[local] = qualified
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.globals.add(stmt.name)
            elif not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.globals |= _stored_names(stmt)
        # nn.Module subclasses, directly or through another class in the file.
        self.classes: dict[str, ast.ClassDef] = {}
        defs = [s for s in tree.body if isinstance(s, ast.ClassDef)]
        changed = True
        while changed:
            changed = False
            for cls in defs:
                if cls.name in self.classes:
                    continue
                if any(self.qualname(b) == NN_MODULE or self.qualname(b) in self.classes for b in cls.bases):
                    self.classes[cls.name] = cls
                    changed = True

    def qualname(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualname(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def segment(self, node: ast.AST) -> str:
        text = ast.get_source_segment(self.source, node)
        return text if text is not None and "\n" not in text else ast.unparse(node)

    def statements(self, stmts: list[ast.stmt]) -> str:
        """Source lines of consecutive statements, dedented, with their comments."""
        start = min(getattr(s, "lineno", 1) for s in stmts)
        for s in stmts:
            for d in getattr(s, "decorator_list", []):
                start = min(start, d.lineno)
        end = max(s.end_lineno or s.lineno for s in stmts)
        return textwrap.dedent("\n".join(self.lines[start - 1 : end])).strip("\n")


def _bindings(stmt: ast.stmt) -> list[tuple[str, str]]:
    """Names an import statement binds, with the qualified names they refer to."""
    if isinstance(stmt, ast.Import):
        return [(a.asname, a.name) if a.asname else (a.name.split(".")[0], a.name.split(".")[0]) for a in stmt.names]
    if isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
        return [(a.asname or a.name, f"{stmt.module}.{a.name}") for a in stmt.names if a.name != "*"]
    return []


def _is_standard_import(stmt: ast.stmt) -> bool:
    binds = _bindings(stmt)
    return bool(binds) and all(STANDARD_BINDINGS.get(local) == qualified for local, qualified in binds)


def _is_main_guard(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and "__name__" in ast.unparse(stmt.test) and "__main__" in ast.unparse(stmt.test)


def _loaded_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _stored_names(node: ast.AST) -> set[str]:
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update(local for local, _ in _bindings(n))
    return names


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    """Names every run of the statements binds: plain assignments at the top level, not ones under if/for/try."""
    names: set[str] = set()
    for stmt in stmts:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value else []
        for target in targets:
            names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return names


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_super_init(stmt: ast.stmt) -> bool:
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    func = stmt.value.func
    if not (isinstance(func, ast.Attribute) and func.attr == "__init__"):
        return False
    target = func.value
    return (isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == "super") or ast.unparse(target) in (
        "nn.Module",
        "torch.nn.Module",
    )


def _self_attr(node: ast.AST) -> str | None:
    """`self.name` -> "name"."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


# ---------------------------------------------------------------- one class


@dataclass
class _Child:
    """What a parent needs to know about a converted module."""

    module_id: str
    hyperparams: list[str]
    inputs: list[str]  # interface input node ids, in forward() order
    input_names: list[str]
    outputs: list[str]
    call_style: bool = False  # opaque modules whose instances pass arguments verbatim
    optional: list[str] = field(default_factory=list)  # forward() arguments with defaults, made ports when a call passes them


@dataclass
class _Value:
    """A value in forward(): one output of a node, or a node's tuple result (possibly partly indexed)."""

    node: str
    port: str | None = None
    path: list[int] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)

    @property
    def is_tensor(self) -> bool:
        return self.port is not None


class _Module:
    """Builds the IR of one module definition."""

    def __init__(self, module_id: str, class_name: str):
        self.mod: dict[str, Any] = {
            "id": module_id,
            "className": class_name,
            "hyperparams": [],
            "nodes": {},
            "edges": [],
            "layout": {},
            "interface": {"inputs": [], "outputs": []},
        }
        self.count = 0
        self.reached: list[str] = []  # node ids in the order they are added or first used

    def add(self, kind: str, type_: str, name: str, params: dict | None = None, **extra: Any) -> str:
        self.count += 1
        node_id = f"n{self.count}"
        self.mod["nodes"][node_id] = {"id": node_id, "kind": kind, "type": type_, "name": name, "params": params or {}, **extra}
        self.reached.append(node_id)
        return node_id

    def link(self, source: _Value, node_id: str, port: str) -> None:
        if not source.is_tensor:
            raise Unsupported(t("把整个元组结果作为输入", "a whole tuple result used as an input"))
        self.mod["edges"].append(
            {"id": f"e{len(self.mod['edges']) + 1}", "source": {"node": source.node, "port": source.port}, "target": {"node": node_id, "port": port}}
        )

    def taken(self) -> set[str]:
        return {n["name"] for n in self.mod["nodes"].values()}

    def unique(self, base: str, style: str = "fx") -> str:
        taken = self.taken()
        if style != "numbered" and base not in taken:
            return base
        i = 1
        while (f"{base}_{i}" if style == "fx" else f"{base}{i}") in taken:
            i += 1
        return f"{base}_{i}" if style == "fx" else f"{base}{i}"


class _ClassConverter:
    def __init__(self, importer: "_Importer", cls: ast.ClassDef):
        self.imp = importer
        self.file = importer.file
        self.cls = cls
        self.m = _Module(cls.name, cls.name)
        self.hp: list[str] = []
        self.locals: dict[str, ast.expr] = {}
        self.local_stmts: dict[str, list[ast.stmt]] = {}  # every assignment, in order (n = 32; n = f(n))
        self.attrs: dict[str, str] = {}  # self.<attr> -> node id
        self.state_attrs: set[str] = set()  # attrs that are Parameter / Buffer nodes, read as self.<attr> in forward()
        self.repeat_style: dict[str, str] = {}  # repeat attr -> "list" (for loop) or "sequential" (called)
        self.constants: dict[str, str] = {}
        self.plain_constants: set[str] = set()
        self.stored: set[str] = set()
        self.used: set[str] = set()  # node ids already called in forward()
        self.values: dict[str, _Value] = {}
        self.notes: list[str] = []
        self.code_blocks = 0

    # .................................................... class level

    def convert(self) -> _Child:
        cls = self.cls
        if cls.decorator_list or cls.keywords or len(cls.bases) != 1 or self.file.qualname(cls.bases[0]) != NN_MODULE:
            raise Opaque(
                t("类有装饰器、多个基类或继承自其他自定义模块", "the class has decorators or several base classes, or inherits from another custom module")
            )
        methods = {s.name: s for s in cls.body if isinstance(s, ast.FunctionDef)}
        forward = methods.get("forward")
        if forward is None:
            raise Opaque(t("没有 forward 方法", "no forward method"))
        init = methods.get("__init__")
        members = [s for s in cls.body if not (isinstance(s, ast.FunctionDef) and s.name in ("__init__", "forward"))]
        members = [s for s in members if not (s is cls.body[0] and _is_docstring(s))]
        if init:
            self._signature(init)
            self._init_body(init.body)
        self.m.reached = []
        self._forward(forward)
        # Nodes in the order forward() reaches them, then those it never uses: the
        # order breaks ties between parallel branches, so it must follow the code.
        nodes = self.m.mod["nodes"]
        order = list(dict.fromkeys(n for n in self.m.reached if n in nodes))
        self.m.mod["nodes"] = {n: nodes[n] for n in order + [n for n in nodes if n not in order]}
        if members:
            self.m.mod["members"] = self.file.statements(members)
        if self.constants:
            self.m.mod["constants"] = [{"name": k, "expr": v} for k, v in self.constants.items()]
        return _Child(
            cls.name,
            self.hp,
            self.m.mod["interface"]["inputs"],
            [self.m.mod["nodes"][i]["name"] for i in self.m.mod["interface"]["inputs"]],
            self.m.mod["interface"]["outputs"],
        )

    def _signature(self, init: ast.FunctionDef) -> None:
        args = init.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
            raise Opaque(t("__init__ 使用了 *args、**kwargs 或仅限关键字的参数", "__init__ uses *args, **kwargs or keyword-only arguments"))
        params = args.args[1:]
        defaults = [None] * (len(params) - len(args.defaults)) + list(args.defaults)
        for p, d in zip(params, defaults):
            hp: dict[str, Any] = {"name": p.arg}
            if p.annotation is not None:
                hp["annotation"] = self.file.segment(p.annotation)
            if d is not None:
                hp["default"] = self._literal_or_expr(d, raw=True)
            self.m.mod["hyperparams"].append(hp)
            self.hp.append(p.arg)

    # .................................................... expressions

    def _literal_or_expr(self, node: ast.expr, *, raw: bool = False, forward: bool = False) -> dict:
        with contextlib.suppress(ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return {"literal": _literal(ast.literal_eval(node))}
        return {"expr": self.file.segment(node) if raw else self._expr(node, forward=forward)}

    def _expr(self, node: ast.expr, *, forward: bool = False) -> str:
        """IR expression text: __init__ locals substituted, `self.<constant>` written as its name."""
        substitute = False
        for n in ast.walk(node):
            attr = _self_attr(n)
            if attr is not None:
                if attr not in self.constants and attr not in self.stored:
                    raise Unsupported(t(f"self.{attr} 不是常量", f"self.{attr} is not a constant"))
                if forward and attr in self.constants and attr not in self.plain_constants:
                    raise Unsupported(t(f"self.{attr} 不是数值常量", f"self.{attr} is not a numeric constant"))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id != "self":
                if n.id in EXPR_BUILTINS or n.id in self.file.globals or n.id in self.file.aliases:
                    continue
                if forward:
                    segment = self.file.segment(node)
                    raise Unsupported(t(f"参数依赖运行时的值（{segment}）", f"argument depends on run-time values ({segment})"))
                if n.id in self.locals:
                    substitute = True
                elif n.id not in self.hp and not hasattr(builtins, n.id):
                    raise Unsupported(t(f"无法解析名称 {n.id}", f"cannot resolve name {n.id}"))
            elif isinstance(n, ast.Attribute):
                # The editor evaluates names and math.*, not attributes of objects (config.n_embd, block.expansion).
                root = n
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and root.id == "math"):
                    if forward:  # x.size(0), x.shape[1]
                        segment = self.file.segment(node)
                        raise Unsupported(t(f"参数依赖运行时的值（{segment}）", f"argument depends on run-time values ({segment})"))
                    segment = self.file.segment(n)
                    raise Unsupported(t(f"属性 {segment} 无法在编辑器中求值", f"the attribute {segment} cannot be evaluated in the editor"))
            elif isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.comprehension)):
                raise Unsupported(t("表达式过于复杂", "expression too complex"))
        if substitute:
            return ast.unparse(_Rewrite(self.locals).visit(copy.deepcopy(node)))
        text = ast.get_source_segment(self.file.source, node)
        if text is None or "\n" in text:
            return ast.unparse(_Rewrite({}).visit(copy.deepcopy(node)))
        # Replace `self.name` spans with `name`, keeping everything else as written.
        spans = []
        for n in ast.walk(node):
            if _self_attr(n) is not None:
                spans.append((n.col_offset - node.col_offset, n.value.end_col_offset - node.col_offset + 1))
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + text[end:]
        return text

    # .................................................... __init__

    def _init_body(self, body: list[ast.stmt]) -> None:
        init_code: list[ast.stmt] = []
        unresolved: set[str] = set()  # names the moved statements bind
        for stmt in body:
            if _is_super_init(stmt) or _is_docstring(stmt):
                continue
            try:
                if _loaded_names(stmt) & unresolved:
                    raise Unsupported(t("依赖前面无法解析的语句", "depends on an earlier statement that could not be resolved"))
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    attr = _self_attr(target)
                    if attr is not None:
                        self._attribute(attr, stmt.value)
                        continue
                    if isinstance(target, ast.Name) and target.id not in self.hp:
                        self._expr(stmt.value)  # must be expressible
                        self.locals[target.id] = _Rewrite(self.locals).visit(copy.deepcopy(stmt.value))
                        self.local_stmts.setdefault(target.id, []).append(stmt)
                        continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and _self_attr(stmt.value.func) == "register_buffer":
                    self._buffer(stmt.value)
                    continue
                raise Unsupported(type(stmt).__name__)
            except Unsupported:
                init_code.append(stmt)
                unresolved |= _stored_names(stmt)
        if init_code:
            # Locals were inlined into the structured statements; the moved ones may still need them.
            needed: list[ast.stmt] = []
            want = set().union(*(_loaded_names(s) for s in init_code))
            grew = True
            while grew:
                grew = False
                for name, stmts in self.local_stmts.items():
                    for stmt in stmts if name in want else []:
                        if stmt not in needed:
                            needed.append(stmt)
                            want |= _loaded_names(stmt)
                            grew = True
            ordered = [s for s in body if s in needed or s in init_code]
            self.m.mod["initCode"] = "\n".join(self.file.statements([s]) for s in ordered)
            self.notes.append(
                t(
                    "__init__ 中的部分语句原样保留（如权重初始化、注册缓冲区）",
                    "some __init__ statements kept as written (such as weight initialization or buffer registration)",
                )
            )

    def _tensor_init(self, call: ast.expr) -> dict | None:
        """`torch.zeros(1, 1, d, dtype=torch.int64)` as Parameter / Buffer params, or None for other initial values."""
        if not isinstance(call, ast.Call):
            return None
        init = {f"torch.{name}": name for name in ("zeros", "ones", "randn", "rand", "empty")}.get(self.file.qualname(call.func) or "")
        if init is None:
            return None
        params: dict[str, Any] = {"init": {"literal": init}}
        for kw in call.keywords:
            dtype = STATE_DTYPES.get(self.file.qualname(kw.value) or "") if kw.arg == "dtype" else None
            if dtype is None:
                return None
            params["dtype"] = {"literal": dtype}
        sizes = call.args[0].elts if len(call.args) == 1 and isinstance(call.args[0], (ast.Tuple, ast.List)) else call.args
        if all(isinstance(s, ast.Constant) and type(s.value) is int for s in sizes):
            params["shape"] = {"literal": [s.value for s in sizes]}
        elif len(call.args) == 1 and not isinstance(call.args[0], (ast.Tuple, ast.List)):
            params["shape"] = {"expr": f"({self._expr(call.args[0])},)"}  # torch.ones(dim): one dimension
        else:
            parts = [self._expr(s) for s in sizes]
            params["shape"] = {"expr": f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"}
        return params

    def _state(self, attr: str, type_: str, params: dict) -> None:
        self.attrs[attr] = self.m.add("layer", type_, attr, params)
        self.state_attrs.add(attr)

    def _parameter(self, attr: str, call: ast.Call) -> bool:
        """`self.cls_token = nn.Parameter(torch.zeros(1, 1, d))`: a Parameter node."""
        params = self._tensor_init(call.args[0]) if call.args else None
        grad = call.args[1] if len(call.args) == 2 else next((k.value for k in call.keywords if k.arg == "requires_grad"), None)
        if params is None or len(call.args) > 2 or any(k.arg != "requires_grad" for k in call.keywords):
            return False
        if grad is not None:
            if not (isinstance(grad, ast.Constant) and isinstance(grad.value, bool)):
                return False
            if not grad.value:
                params["requires_grad"] = {"literal": False}
        self._state(attr, "neuroloom.Parameter", params)
        return True

    def _buffer(self, call: ast.Call) -> None:
        """`self.register_buffer("index", torch.zeros(49, dtype=torch.int64))`: a Buffer node."""
        name = call.args[0] if call.args else None
        params = self._tensor_init(call.args[1]) if len(call.args) >= 2 else None
        persistent = call.args[2] if len(call.args) == 3 else next((k.value for k in call.keywords if k.arg == "persistent"), None)
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)) or params is None or len(call.args) > 3 or any(k.arg != "persistent" for k in call.keywords):
            raise Unsupported(t("register_buffer 的写法", "this form of register_buffer"))
        if persistent is not None:
            if not (isinstance(persistent, ast.Constant) and isinstance(persistent.value, bool)):
                raise Unsupported(t("register_buffer 的写法", "this form of register_buffer"))
            if not persistent.value:
                params["persistent"] = {"literal": False}
        if name.value in self.attrs or name.value in self.constants or name.value in self.stored:
            raise Unsupported(t(f"self.{name.value} 被重复赋值", f"self.{name.value} is assigned more than once"))
        self._state(name.value, "neuroloom.Buffer", params)

    def _attribute(self, attr: str, value: ast.expr) -> None:
        if attr in self.attrs or attr in self.constants or attr in self.stored:
            raise Unsupported(t(f"self.{attr} 被重复赋值", f"self.{attr} is assigned more than once"))
        if isinstance(value, ast.Call) and self.file.qualname(value.func) in ("torch.nn.Parameter", "torch.nn.parameter.Parameter"):
            if self._parameter(attr, value):
                return
            raise Unsupported(t("nn.Parameter 的初始值", "the initial value of this nn.Parameter"))
        if isinstance(value, ast.Call):
            q = self.file.qualname(value.func)
            catalog = self.imp.catalog
            if q in catalog.layers:
                self.attrs[attr] = self.m.add("layer", catalog.layers[q]["type"], attr, self._layer_params(catalog.layers[q], q, value))
                return
            if q in self.file.classes or q == SEQUENTIAL or q == MODULE_LIST or self._external(q, value.func):
                self._module_attr(attr, value)
                return
        if isinstance(value, ast.Name) and value.id == attr and attr in self.hp and attr not in self.locals:
            self.stored.add(attr)
            return
        if attr in self.hp:
            raise Unsupported(t(f"常量 self.{attr} 与超参数同名", f"constant self.{attr} has the same name as a hyperparameter"))
        text = self._expr(value)
        self.constants[attr] = text
        if _plain(value):
            self.plain_constants.add(attr)

    def _external(self, q: str | None, func: ast.expr) -> bool:
        """A module class from elsewhere: capitalized by convention, and not a torch function."""
        name = q.rsplit(".", 1)[-1] if q else ""
        if not name[:1].isupper() or q in (SEQUENTIAL, MODULE_LIST, "torch.nn.ModuleDict", "torch.nn.Parameter", "torch.nn.parameter.Parameter"):
            return False
        return not q.startswith("torch.") or q.startswith("torch.nn.")

    def _layer_params(self, spec: dict, qualname: str, call: ast.Call) -> dict:
        spec_params = {p["name"]: p for p in spec["params"]}
        names, defaults = _ctor_signature(qualname, tuple(spec_params))
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            raise Unsupported(t("构造参数使用了 * 或 ** 展开", "constructor arguments use * or ** unpacking"))
        bound: dict[str, ast.expr] = {}
        for i, arg in enumerate(call.args):
            if i >= len(names):
                raise Unsupported(t(f"{qualname} 的参数过多", f"too many arguments to {qualname}"))
            bound[names[i]] = arg
        for kw in call.keywords:
            bound[kw.arg] = kw.value
        params: dict[str, dict] = {}
        for name, node in bound.items():
            if name not in spec_params:
                # torch accepts it but NeuroLoom does not model it: fine only at its default.
                with contextlib.suppress(Exception):
                    if name in defaults and ast.literal_eval(node) == defaults[name]:
                        continue
                raise Unsupported(t(f"{qualname} 的参数 {name}", f"argument {name} of {qualname}"))
            params[name] = self._param(node, spec_params[name])
        for p in spec["params"]:
            if p["name"] not in params and "default" not in p:
                raise Unsupported(t(f"{qualname} 缺少参数 {p['name']}", f"{qualname} is missing argument {p['name']}"))
        return params

    def _param(self, node: ast.expr, param: dict, *, forward: bool = False) -> dict:
        value = self._literal_or_expr(node, forward=forward)
        if "literal" in value:
            value["literal"] = _normalize(value["literal"], param)
        return value

    # Module-valued attributes: instances, nn.Sequential, nn.ModuleList repeats.

    def _module_attr(self, attr: str, call: ast.Call) -> None:
        q = self.file.qualname(call.func)
        if q == MODULE_LIST:
            if len(call.args) != 1 or call.keywords:
                raise Unsupported(t("nn.ModuleList 的写法", "this form of nn.ModuleList"))
            elt, count, first = _repeat_parts(call.args[0])
            self._repeat(attr, elt, count, first, "list")
            return
        if q == SEQUENTIAL and len(call.args) == 1 and isinstance(call.args[0], ast.Starred):
            elt, count, first = _repeat_parts(call.args[0].value)
            self._repeat(attr, elt, count, first, "sequential")
            return
        kind, type_, params = self._ctor(call, attr)
        self.attrs[attr] = self.m.add(kind, type_, attr, params)

    def _ctor(self, call: ast.Call, name: str) -> tuple[str, str, dict]:
        """(kind, type, params) of a module constructor call, in this class's scope."""
        q = self.file.qualname(call.func)
        catalog = self.imp.catalog
        if q in catalog.layers:
            return "layer", catalog.layers[q]["type"], self._layer_params(catalog.layers[q], q, call)
        if q == SEQUENTIAL:
            return "module", self._sequential(call, name), {}
        if q in self.file.classes:
            child = self.imp.child(q)
            return "module", child.module_id, self._instance_params(child, call)
        if self._external(q, call.func):
            child = self.imp.external(self.file.segment(call.func))
            return "module", child.module_id, self._instance_params(child, call)
        func = self.file.segment(call.func)
        raise Unsupported(t(f"{func}() 不是可识别的模块", f"{func}() is not a recognized module"))

    def _instance_params(self, child: _Child, call: ast.Call) -> dict:
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            raise Unsupported(t("构造参数使用了 * 或 ** 展开", "constructor arguments use * or ** unpacking"))
        params: dict[str, dict] = {}
        if child.call_style:
            for i, arg in enumerate(call.args):
                params[f"#{i}"] = self._literal_or_expr(arg)
            for kw in call.keywords:
                params[kw.arg] = self._literal_or_expr(kw.value)
            self.imp.note_args(child, params)
            return params
        if len(call.args) > len(child.hyperparams):
            raise Unsupported(t("构造参数过多", "too many constructor arguments"))
        for name, arg in zip(child.hyperparams, call.args):
            params[name] = self._literal_or_expr(arg)
        for kw in call.keywords:
            if kw.arg not in child.hyperparams:
                raise Unsupported(t(f"没有名为 {kw.arg} 的构造参数", f"no constructor argument named {kw.arg}"))
            params[kw.arg] = self._literal_or_expr(kw.value)
        return params

    def _sequential(self, call: ast.Call, name: str) -> str:
        """
        An nn.Sequential(...) of module constructors becomes an inline sequential
        module: children numbered 0, 1, …, or named by nn.Sequential(OrderedDict([("conv0", ...), ...])).
        """
        children = [(str(i), arg) for i, arg in enumerate(call.args)]
        if len(call.args) == 1 and isinstance(call.args[0], ast.Call) and self.file.qualname(call.args[0].func) == "collections.OrderedDict":
            pairs = call.args[0].args[0] if len(call.args[0].args) == 1 and not call.args[0].keywords else None
            items = pairs.elts if isinstance(pairs, (ast.List, ast.Tuple)) else []
            named = [e.elts for e in items if isinstance(e, ast.Tuple) and len(e.elts) == 2 and isinstance(e.elts[0], ast.Constant) and isinstance(e.elts[0].value, str)]
            if not items or len(named) != len(items):
                raise Unsupported(t("nn.Sequential 的写法", "this form of nn.Sequential"))
            children = [(key.value, value) for key, value in named]
        if call.keywords or not children or any(not isinstance(arg, ast.Call) for _, arg in children):
            raise Unsupported(t("nn.Sequential 的写法", "this form of nn.Sequential"))
        seq = _Module(self.imp.unique_id(f"{self.cls.name}_{name}"), f"{self.cls.name}_{name}")
        seq.mod["sequential"] = True
        prev = seq.add("input", INPUT, "x", {"shape": {"literal": ""}, "dtype": {"literal": "float32"}})
        seq.mod["interface"]["inputs"].append(prev)
        prev_port = "out"
        for child_name, arg in children:
            kind, type_, params = self._ctor(arg, f"{name}_{child_name}")
            node = seq.add(kind, type_, child_name, params)
            seq.link(_Value(prev, prev_port), node, self._first_input(kind, type_))
            prev, prev_port = node, self._first_output(kind, type_)
        out = seq.add("output", OUTPUT, "output")
        seq.mod["interface"]["outputs"].append(out)
        seq.link(_Value(prev, prev_port), out, "in")
        self.imp.add_module(seq.mod)
        return seq.mod["id"]

    def _first_input(self, kind: str, type_: str) -> str:
        if kind == "module":
            child = self.imp.module_info(type_)
            if not child.inputs:
                self.imp.ensure_ports(child, 1, 1)
            return child.inputs[0]
        spec = self.imp.layer_spec(type_)
        return spec["inputs"][0]["name"]

    def _first_output(self, kind: str, type_: str) -> str:
        if kind == "module":
            return self.imp.module_info(type_).outputs[0]
        spec = self.imp.layer_spec(type_)
        return spec["outputs"][0]["name"]

    def _repeat(self, attr: str, elt: ast.Call, count: ast.expr, first: ast.Call | None, style: str) -> None:
        kind, type_, params = self._ctor(elt, attr)
        if first is not None:
            kind1, type1, params1 = self._ctor(first, attr)
            if (kind1, type1) != (kind, type_):
                raise Unsupported(t("重复块的第一份与其余份不是同一个类", "the first repeated block is not of the same class as the others"))
        repeat: dict[str, Any] = {}
        # [first] + [rest for _ in range(n)] has n + 1 copies; `range(depth - 1)` gives back `depth`.
        if isinstance(count, ast.Constant) and isinstance(count.value, int) and not isinstance(count.value, bool):
            repeat["count"] = {"literal": count.value + (1 if first is not None else 0)}
        elif first is not None and isinstance(count, ast.BinOp) and isinstance(count.op, ast.Sub) and _is_one(count.right):
            repeat["count"] = {"expr": self._expr(count.left)}
        elif first is not None:
            text = self._expr(count)
            if not isinstance(count, (ast.Name, ast.Attribute, ast.Call, ast.Constant)):
                text = f"({text})"
            repeat["count"] = {"expr": f"{text} + 1"}
        else:
            repeat["count"] = {"expr": self._expr(count)}
        if first is not None:
            overrides = {k: v for k, v in params1.items() if params.get(k) != v}
            for k in params:
                if k not in params1:
                    raise Unsupported(t("重复块的第一份缺少其余份的参数", "the first repeated block lacks arguments the others have"))
            if overrides:
                repeat["first"] = overrides
        node_id = self.m.add(kind, type_, attr, params, repeat=repeat)
        self.attrs[attr] = node_id
        self.repeat_style[attr] = style

    # .................................................... forward()

    def _forward(self, fn: ast.FunctionDef) -> None:
        args = fn.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults:
            raise Opaque(t("forward 有默认参数或可变参数", "forward has default or variadic arguments"))
        for a in args.args[1:]:
            node = self.m.add("input", INPUT, a.arg, {"shape": {"literal": ""}, "dtype": {"literal": "float32"}})
            self.m.mod["interface"]["inputs"].append(node)
            self.values[a.arg] = _Value(node, "out")
        body = [s for i, s in enumerate(fn.body) if not (i == 0 and _is_docstring(s))]
        for i, stmt in enumerate(body):
            if any(isinstance(n, ast.Return) for n in ast.walk(stmt)) and not (i == len(body) - 1 and isinstance(stmt, ast.Return)):
                raise Opaque(t("forward 中途返回", "forward returns early"))
        later = [set().union(*(_loaded_names(s) for s in body[i + 1 :])) for i in range(len(body))]

        pending: list[int] = []
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Return):
                self._flush(body, pending, later)
                pending = []
                self._return(stmt)
                break
            snapshot = self._snapshot()
            try:
                self._flush(body, pending, later)
                self._statement(stmt)
                pending = []
            except Unsupported as e:
                self._restore(snapshot)
                if not pending:
                    self.notes.append(t(f"第 {stmt.lineno} 行保留为代码块：{e}", f"line {stmt.lineno} kept as a code block: {e}"))
                pending.append(i)
        self._flush(body, pending, later)

    def _snapshot(self) -> tuple:
        return (copy.deepcopy(self.m.mod), self.m.count, len(self.m.reached), dict(self.values), set(self.used), self.code_blocks, self.imp.snapshot())

    def _restore(self, s: tuple) -> None:
        self.m.mod, self.m.count, reached, self.values, self.used, self.code_blocks, imp = s
        del self.m.reached[reached:]
        self.imp.restore(imp)

    def _flush(self, body: list[ast.stmt], pending: list[int], later: list[set[str]]) -> None:
        """Turns consecutive statements L1 could not handle into one code block."""
        if not pending:
            return
        stmts = [body[i] for i in pending]
        read = set().union(*(_loaded_names(s) for s in stmts))
        written = set().union(*(_stored_names(s) for s in stmts))
        # A name the block may leave as it was (`if ...: identity = f(x)`, `out += y`) comes in too.
        inputs = [n for n in sorted(read | (written - _assigned_names(stmts))) if n in self.values]
        outputs = sorted(written & later[pending[-1]])
        node = self.m.add("code", CODE, self.m.unique("code"), {}, code={"source": self.file.statements(stmts), "inputs": inputs, "outputs": outputs})
        for name in inputs:
            self.m.link(self.values[name], node, name)
        for name in outputs:
            self.values[name] = _Value(node, name)
        self.code_blocks += 1

    def _statement(self, stmt: ast.stmt) -> None:
        if _is_docstring(stmt):
            return
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                if isinstance(stmt.value, ast.Name):  # an alias: y = x
                    self.values[target.id] = self._value(stmt.value)
                    return
                value = self._lower(stmt.value, hint=target.id if target.id not in self.values else None)
                if not value.is_tensor:
                    raise Unsupported(t("把元组结果赋给一个变量", "a tuple result assigned to one variable"))
                self.values[target.id] = value
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                value = self._lower(stmt.value)
                if value.is_tensor:
                    raise Unsupported(t("解包单个张量", "unpacking a single tensor"))
                self._unpack(target, value)
                return
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            binop = ast.BinOp(left=ast.Name(stmt.target.id, ast.Load()), op=stmt.op, right=stmt.value)
            ast.copy_location(binop, stmt)
            self.values[stmt.target.id] = self._lower(binop)
            return
        if isinstance(stmt, ast.For):
            self._for(stmt)
            return
        raise Unsupported(type(stmt).__name__)

    def _unpack(self, target: ast.Tuple | ast.List, value: _Value, path: list[int] | None = None) -> None:
        path = path or []
        for i, elt in enumerate(target.elts):
            sub = [*path, i]
            if isinstance(elt, (ast.Tuple, ast.List)):
                self._unpack(elt, value, sub)
            elif isinstance(elt, ast.Name):
                match = [o for o in value.outputs if o.get("path") == [*value.path, *sub]]
                if not match:
                    nested = any(o.get("path", [])[: len(value.path) + len(sub)] == [*value.path, *sub] for o in value.outputs)
                    if nested and elt.id == "_":
                        continue  # `out, _ = self.lstm(x)` discards the whole (h, c) tuple
                    if nested:
                        raise Unsupported(t("把元组结果赋给一个变量", "a tuple result assigned to one variable"))
                    raise Unsupported(t("解包的元素个数不符", "the number of unpacked elements does not match"))
                if elt.id != "_":
                    self.values[elt.id] = _Value(value.node, match[0]["name"])
            else:
                raise Unsupported(t("解包目标", "this unpacking target"))
        if not any(o.get("path", [])[: len(value.path)] == value.path for o in value.outputs):
            raise Unsupported(t("解包的元素个数不符", "the number of unpacked elements does not match"))

    def _for(self, stmt: ast.For) -> None:
        """for block in self.blocks: x = block(x, ...)  ->  a call of the repeat node."""
        attr = _self_attr(stmt.iter)
        if (
            attr is None
            or self.repeat_style.get(attr) != "list"
            or stmt.orelse
            or not isinstance(stmt.target, ast.Name)
            or len(stmt.body) != 1
            or not isinstance(stmt.body[0], ast.Assign)
            or len(stmt.body[0].targets) != 1
            or not isinstance(stmt.body[0].targets[0], ast.Name)
        ):
            raise Unsupported(t("for 循环", "for loop"))
        assign = stmt.body[0]
        var = assign.targets[0].id
        call = assign.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == stmt.target.id):
            raise Unsupported(t("for 循环", "for loop"))
        if not call.args or not (isinstance(call.args[0], ast.Name) and call.args[0].id == var) or var not in self.values:
            raise Unsupported(t("for 循环没有把上一次的输出传给下一次", "the for loop does not pass each output on to the next iteration"))
        self.values[var] = self._call_node(self.attrs[attr], call.args, call.keywords)

    def _return(self, stmt: ast.Return) -> None:
        if stmt.value is None:
            return
        elements = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
        for i, elt in enumerate(elements):
            # Output names must not clash with the model's own variables (`output = F.log_softmax(...)`).
            taken = self.m.taken() | set(self.values)
            name = next(n for n in ["output", *(f"output_{k}" for k in range(1, 100))] if n not in taken)
            snapshot = self._snapshot()
            try:
                value = self._lower(elt)
                if not value.is_tensor:
                    raise Unsupported(t("返回整个元组结果", "returning a whole tuple result"))
            except Unsupported as e:
                self._restore(snapshot)
                self.notes.append(t(f"第 {elt.lineno} 行的返回值保留为代码块：{e}", f"the return value on line {elt.lineno} kept as a code block: {e}"))
                read = [n for n in sorted(_loaded_names(elt)) if n in self.values]
                code = {"source": f"{name} = {self.file.segment(elt)}", "inputs": read, "outputs": [name]}
                node = self.m.add("code", CODE, self.m.unique("code"), {}, code=code)
                for n in read:
                    self.m.link(self.values[n], node, n)
                self.code_blocks += 1
                value = _Value(node, name)
            out = self.m.add("output", OUTPUT, name)
            self.m.mod["interface"]["outputs"].append(out)
            self.m.link(value, out, "in")

    # .................................................... expressions in forward()

    def _value(self, node: ast.Name) -> _Value:
        value = self.values.get(node.id)
        if value is None:
            raise Unsupported(t(f"变量 {node.id}", f"variable {node.id}"))
        return value

    def _tensor(self, node: ast.expr) -> _Value:
        value = self._lower(node)
        if not value.is_tensor:
            raise Unsupported(t("把整个元组结果作为输入", "a whole tuple result used as an input"))
        return value

    def _lower(self, node: ast.expr, hint: str | None = None) -> _Value:
        if isinstance(node, ast.Name):
            return self._value(node)
        if _self_attr(node) in self.state_attrs:  # self.cls_token: a Parameter / Buffer node
            self.m.reached.append(self.attrs[_self_attr(node)])
            return _Value(self.attrs[_self_attr(node)], "out")
        if isinstance(node, ast.Call):
            attr = _self_attr(node.func)
            if attr is not None:
                if attr not in self.attrs or (attr in self.repeat_style and self.repeat_style[attr] != "sequential"):
                    raise Unsupported(f"self.{attr}(...)")
                return self._call_node(self.attrs[attr], node.args, node.keywords)
            q = self.file.qualname(node.func)
            catalog = self.imp.catalog
            if q in catalog.functions and not (isinstance(node.func, ast.Name) and node.func.id in self.values):
                return self._op(catalog.functions[q], node.args, node.keywords, None, hint)
            if isinstance(node.func, ast.Attribute) and node.func.attr in catalog.methods and q not in catalog.functions:
                base = node.func.value
                if not (isinstance(base, ast.Name) and base.id in self.file.aliases):
                    receiver = self._tensor(base)
                    return self._op(catalog.methods[node.func.attr], node.args, node.keywords, receiver, hint)
            raise Unsupported(f"{self.file.segment(node.func)}()")
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            spec = self.imp.catalog.functions[BINOPS[type(node.op)]]
            left, right = node.left, node.right
            # `2 * x` puts the constant first; addition and multiplication commute.
            if isinstance(node.op, (ast.Add, ast.Mult)) and _is_number(left) and not _is_number(right):
                left, right = right, left
            return self._op(spec, [left, right], [], None, hint)
        if isinstance(node, ast.Subscript):
            base = self._lower(node.value)
            index = node.slice
            if base.is_tensor or not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
                raise Unsupported(t("张量切片或索引（如 x[:, 0]）", "tensor slicing or indexing (such as x[:, 0])"))
            path = [*base.path, index.value]
            match = [o for o in base.outputs if o.get("path") == path]
            if match:
                return _Value(base.node, match[0]["name"])
            if any(o.get("path", [])[: len(path)] == path for o in base.outputs):
                return _Value(base.node, None, path, base.outputs)
            raise Unsupported(t("元组下标超出范围", "tuple index out of range"))
        raise Unsupported(type(node).__name__)

    def _call_node(self, node_id: str, args: list[ast.expr], keywords: list[ast.keyword]) -> _Value:
        """A call of a layer, module instance or repeat: one node, one call."""
        if node_id in self.used:
            node = self.m.mod["nodes"][node_id]
            spec = self.imp.layer_spec(node["type"]) if node["kind"] == "layer" else None
            if not spec or not spec.get("stateless") or node.get("repeat"):
                raise Unsupported(t("同一个层被调用了多次（共享权重）", "the same layer is called more than once (shared weights)"))
            # A layer without parameters (self.relu used twice) is the same as a second copy.
            node_id = self.m.add("layer", node["type"], self.m.unique(node["name"]), copy.deepcopy(node["params"]))
        node = self.m.mod["nodes"][node_id]
        if any(isinstance(a, ast.Starred) for a in args) or any(k.arg is None for k in keywords):
            raise Unsupported(t("调用使用了 * 或 ** 展开", "the call uses * or ** unpacking"))
        if node["kind"] == "module":
            child = self.imp.module_info(node["type"])
            if child.call_style and not child.inputs:
                self.imp.ensure_ports(child, len(args) + len(keywords), 1)
            if child.optional:
                self.imp.optional_ports(child, len(args), [k.arg for k in keywords])
            ports = list(zip(child.inputs, child.input_names))
            outputs = [{"name": o, "path": [i]} if len(child.outputs) > 1 else {"name": o} for i, o in enumerate(child.outputs)]
        else:
            spec = self.imp.layer_spec(node["type"])
            ports = [(p["name"], p["name"]) for p in spec["inputs"]]
            outputs = _outputs(spec, node["params"])
        if len(args) > len(ports):
            raise Unsupported(t(f"{node['name']} 的参数过多", f"too many arguments to {node['name']}"))
        links = [(ports[i][0], self._tensor(a)) for i, a in enumerate(args)]
        for kw in keywords:
            port = next((p for p, label in ports if label == kw.arg), None)
            if port is None:
                raise Unsupported(t(f"{node['name']} 的参数 {kw.arg}", f"argument {kw.arg} of {node['name']}"))
            links.append((port, self._tensor(kw.value)))
        for port, value in links:
            self.m.link(value, node_id, port)
        self.used.add(node_id)
        self.m.reached.append(node_id)
        if len(outputs) == 1 and not outputs[0].get("path"):
            return _Value(node_id, outputs[0]["name"])
        return _Value(node_id, None, [], outputs)

    def _op(self, spec: dict, args: list[ast.expr], keywords: list[ast.keyword], receiver: _Value | None, hint: str | None) -> _Value:
        signature = spec["python"]["args"]
        params_by_name = {p["name"]: p for p in spec["params"]}
        positional: list[Any] = ([receiver] if receiver is not None else []) + list(args)
        if any(isinstance(a, ast.Starred) for a in args) or any(k.arg is None for k in keywords):
            raise Unsupported(t("调用使用了 * 或 ** 展开", "the call uses * or ** unpacking"))
        bound: dict[str, Any] = {}
        for i, value in enumerate(positional):
            if i >= len(signature):
                raise Unsupported(t(f"{spec['type']} 的参数过多", f"too many arguments to {spec['type']}"))
            entry = signature[i]
            if "param" in entry and params_by_name[entry["param"]].get("spread") and not isinstance(value, (ast.Tuple, ast.List)):
                bound[entry["name"]] = ast.Tuple(elts=positional[i:], ctx=ast.Load())  # x.permute(0, 2, 1)
                break
            bound[entry["name"]] = value
        names = {e["name"] for e in signature}
        for kw in keywords:
            if kw.arg not in names:
                raise Unsupported(t(f"{spec['type']} 的参数 {kw.arg}", f"argument {kw.arg} of {spec['type']}"))
            bound[kw.arg] = kw.value

        params: dict[str, dict] = {}
        links: list[tuple[str, _Value]] = []
        for entry in signature:
            if entry["name"] not in bound:
                continue
            value = bound[entry["name"]]
            if "ports" in entry:
                if not isinstance(value, (ast.List, ast.Tuple)):
                    raise Unsupported(t(f"{spec['type']} 的 {entry['name']} 必须是张量列表", f"{entry['name']} of {spec['type']} must be a list of tensors"))
                links += [(f"{entry['ports']}{i}", self._tensor(v)) for i, v in enumerate(value.elts)]
            elif "port" in entry:
                if isinstance(value, _Value):
                    links.append((entry["port"], value))
                elif entry.get("scalarParam") and _is_scalar(value) and _self_attr(value) not in self.state_attrs:
                    params[entry["scalarParam"]] = self._param(value, params_by_name[entry["scalarParam"]], forward=True)
                else:
                    links.append((entry["port"], self._tensor(value)))
            else:
                params[entry["param"]] = self._param(value, params_by_name[entry["param"]], forward=True)
        for p in spec["params"]:
            if p["name"] not in params:
                if "default" not in p:
                    raise Unsupported(t(f"{spec['type']} 缺少参数 {p['name']}", f"{spec['type']} is missing argument {p['name']}"))
                params[p["name"]] = {"literal": p["default"]}

        name = hint if hint and hint not in self.m.taken() and hint not in self.values else self.m.unique(spec["defaultName"])
        node_id = self.m.add("op", spec["type"], name, params)
        for port, value in links:
            self.m.link(value, node_id, port)
        outputs = _outputs(spec, params)
        if len(outputs) == 1 and not outputs[0].get("path"):
            return _Value(node_id, outputs[0]["name"])
        return _Value(node_id, None, [], outputs)


class _Rewrite(ast.NodeTransformer):
    """Substitutes __init__ locals and turns `self.name` into `name`."""

    def __init__(self, locals_: dict[str, ast.expr]):
        self.locals = locals_

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.locals:
            return copy.deepcopy(self.locals[node.id])
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        attr = _self_attr(node)
        if attr is not None:
            return ast.copy_location(ast.Name(attr, ast.Load()), node)
        self.generic_visit(node)
        return node


def _is_one(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1 and not isinstance(node.value, bool)


def _is_number(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _is_scalar(node: Any) -> bool:
    """A constant operand (`x * 0.5`, `x * self.scale`) rather than a tensor."""
    if not isinstance(node, ast.expr):
        return False
    return _is_number(node) or (_self_attr(node) is not None) or (
        isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)) and not _loaded_names(node) - EXPR_BUILTINS - {"self"}
    )


def _plain(node: ast.expr) -> bool:
    """A constant the editor can evaluate: numbers, names, arithmetic and a few builtins."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            if name not in EXPR_BUILTINS and not name.startswith("math."):
                return False
        elif not isinstance(
            n,
            (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Tuple, ast.List, ast.Attribute, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.Subscript, ast.keyword),
        ):
            return False
    return True


def _repeat_parts(node: ast.expr) -> tuple[ast.Call, ast.expr, ast.Call | None]:
    """`[X(...) for _ in range(n)]`, or `[X(first)] + [X(...) for _ in range(n)]` -> (element, n, first)."""
    first = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and isinstance(node.left, ast.List) and len(node.left.elts) == 1:
        first = node.left.elts[0]
        node = node.right
        if not isinstance(first, ast.Call):
            raise Unsupported(t("重复块的写法", "this form of repeated block"))
    if not isinstance(node, (ast.ListComp, ast.GeneratorExp)) or len(node.generators) != 1:
        raise Unsupported(t("nn.ModuleList 只支持 [X(...) for _ in range(n)] 的写法", "nn.ModuleList is supported only in the form [X(...) for _ in range(n)]"))
    gen = node.generators[0]
    call = gen.iter
    if (
        gen.ifs
        or gen.is_async
        or not isinstance(gen.target, ast.Name)
        or not isinstance(node.elt, ast.Call)
        or not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range" and len(call.args) == 1 and not call.keywords)
    ):
        raise Unsupported(t("nn.ModuleList 只支持 [X(...) for _ in range(n)] 的写法", "nn.ModuleList is supported only in the form [X(...) for _ in range(n)]"))
    if gen.target.id in _loaded_names(node.elt):
        raise Unsupported(t("每一份重复块的参数不同", "the repeated blocks have different arguments"))
    return node.elt, call.args[0], first


# ---------------------------------------------------------------- the file as a project


class _Importer:
    def __init__(self, file: _File):
        self.file = file
        self.catalog = _catalog()
        self.modules: dict[str, dict] = {}
        self.children: dict[str, _Child] = {}
        self.report: list[dict] = []
        self.warnings: list[str] = []
        self.converting: list[str] = []

    def unique_id(self, base: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", base) or "Module"
        if base not in self.modules and base not in self.file.classes:
            return base
        i = 2
        while f"{base}_{i}" in self.modules or f"{base}_{i}" in self.file.classes:
            i += 1
        return f"{base}_{i}"

    def add_module(self, mod: dict) -> None:
        self.modules[mod["id"]] = mod

    def snapshot(self) -> tuple:
        return (set(self.modules), {k: copy.deepcopy(v) for k, v in self.modules.items() if self.children.get(k) and self.children[k].call_style}, copy.deepcopy({k: v for k, v in self.children.items() if v.call_style}))

    def restore(self, s: tuple) -> None:
        ids, externals, children = s
        for k in list(self.modules):
            if k not in ids:
                del self.modules[k]
                self.children.pop(k, None)
        self.modules.update(externals)
        self.children.update(children)

    def layer_spec(self, type_: str) -> dict | None:
        """The manifest entry of a layer type ("torch.nn.ReLU")."""
        return next((s for s in self.catalog.layers.values() if s["type"] == type_), None)

    def module_info(self, module_id: str) -> _Child:
        if module_id in self.children:
            return self.children[module_id]
        mod = self.modules[module_id]  # a sequential container
        return _Child(module_id, [], mod["interface"]["inputs"], ["x"], mod["interface"]["outputs"])

    def child(self, class_name: str) -> _Child:
        if class_name not in self.children:
            if class_name in self.converting:
                raise Unsupported(t(f"{class_name} 递归地使用了自己", f"{class_name} uses itself recursively"))
            self.convert(class_name)
        return self.children[class_name]

    def convert(self, class_name: str) -> None:
        cls = self.file.classes[class_name]
        self.converting.append(class_name)
        conv = _ClassConverter(self, cls)
        try:
            info = conv.convert()
            self.modules[class_name] = conv.m.mod
            self.children[class_name] = info
            level = "L1" if conv.code_blocks == 0 and "initCode" not in conv.m.mod else "L1+L3"
            self.report.append({"className": class_name, "level": level, "codeBlocks": conv.code_blocks, "notes": conv.notes})
        except Opaque as e:
            self.opaque(cls, str(e))
        finally:
            self.converting.remove(class_name)

    def opaque(self, cls: ast.ClassDef, reason: str) -> None:
        """Keeps a class as source; its forward() signature gives the ports."""
        m = _Module(cls.name, cls.name)
        m.mod["opaque"] = {"source": self.file.statements([cls])}
        methods = {s.name: s for s in cls.body if isinstance(s, ast.FunctionDef)}
        forward = methods.get("forward") or self.inherited_forward(cls)
        init = methods.get("__init__")
        call_style = True
        if init and not (init.args.vararg or init.args.kwarg or init.args.kwonlyargs or init.args.posonlyargs):
            call_style = False
            params = init.args.args[1:]
            defaults = [None] * (len(params) - len(init.args.defaults)) + list(init.args.defaults)
            for p, d in zip(params, defaults):
                hp: dict[str, Any] = {"name": p.arg}
                if d is not None:
                    try:
                        hp["default"] = {"literal": _literal(ast.literal_eval(d))}
                    except (ValueError, TypeError, SyntaxError):
                        hp["default"] = {"expr": self.file.segment(d)}
                m.mod["hyperparams"].append(hp)
        elif not init:
            call_style = False
        if call_style:
            m.mod["opaque"]["verbatimArgs"] = True
        info = _Child(cls.name, [h["name"] for h in m.mod["hyperparams"]], [], [], [], call_style)
        if forward:
            args = forward.args.args[1:]
            required = len(args) - len(forward.args.defaults)
            for a in args[:required]:
                node = m.add("input", INPUT, a.arg, {"shape": {"literal": ""}, "dtype": {"literal": "float32"}})
                info.inputs.append(node)
                info.input_names.append(a.arg)
            info.optional = [a.arg for a in args[required:]]
            last = forward.body[-1] if forward.body else None
            count = len(last.value.elts) if isinstance(last, ast.Return) and isinstance(last.value, ast.Tuple) else 1
            for i in range(count):
                info.outputs.append(m.add("output", OUTPUT, "output" if i == 0 else f"output_{i}"))
        m.mod["interface"] = {"inputs": info.inputs, "outputs": info.outputs}
        self.modules[cls.name] = m.mod
        self.children[cls.name] = info
        note = t(f"整个类保留为源代码：{reason}", f"whole class kept as source code: {reason}")
        self.report.append({"className": cls.name, "level": "L3", "codeBlocks": 0, "notes": [note]})

    def inherited_forward(self, cls: ast.ClassDef, seen: frozenset[str] = frozenset()) -> ast.FunctionDef | None:
        """forward() of the nearest base class in the file that defines one (class ResNet152(ResNet) with only __init__)."""
        for base in cls.bases:
            parent = self.file.classes.get(self.file.qualname(base) or "")
            if parent is None or parent.name in seen:
                continue
            own = next((s for s in parent.body if isinstance(s, ast.FunctionDef) and s.name == "forward"), None)
            found = own or self.inherited_forward(parent, seen | {parent.name})
            if found:
                return found
        return None

    def external(self, written: str) -> _Child:
        """A module class from another package (timm.layers.DropPath): a call-style opaque module without source."""
        module_id = "ext_" + re.sub(r"[^A-Za-z0-9_]", "_", written)
        if module_id not in self.children:
            m = _Module(module_id, written)
            m.mod["opaque"] = {}
            self.modules[module_id] = m.mod
            self.children[module_id] = _Child(module_id, [], [], [], [], call_style=True)
            self.warnings.append(
                t(
                    f"{written} 来自其他库，按外部模块导入（形状需在运行后得知）",
                    f"{written} comes from another library and is imported as an external module (its shapes are known only after running)",
                )
            )
        return self.children[module_id]

    def note_args(self, child: _Child, params: dict) -> None:
        """Call-style modules: every argument name seen becomes a hyperparameter, positional ones first."""
        mod = self.modules[child.module_id]
        names = [h["name"] for h in mod["hyperparams"]]
        for key in params:
            if key not in names:
                names.append(key)
        names.sort(key=lambda n: (not n.startswith("#"), int(n[1:]) if n.startswith("#") else 0))
        mod["hyperparams"] = [{"name": n} for n in names]
        child.hyperparams = names

    def optional_ports(self, child: _Child, positional: int, keywords: list[str | None]) -> None:
        """Arguments with defaults become ports once a call passes them, in forward() order."""
        wanted = [name for i, name in enumerate(child.optional) if len(child.inputs) + i < positional or name in keywords]
        if not wanted:
            return
        upto = child.optional.index(wanted[-1]) + 1
        if len(wanted) < upto:
            raise Unsupported(t(f"{child.module_id} 的可选参数跳过了前面的参数", f"optional arguments of {child.module_id} skip earlier ones"))
        mod = self.modules[child.module_id]
        m = _Module(child.module_id, mod["className"])
        m.mod = mod
        m.count = len(mod["nodes"])
        for name in child.optional[:upto]:
            node = m.add("input", INPUT, name, {"shape": {"literal": ""}, "dtype": {"literal": "float32"}})
            child.inputs.append(node)
            child.input_names.append(name)
        del child.optional[:upto]
        mod["interface"]["inputs"] = child.inputs

    def ensure_ports(self, child: _Child, inputs: int, outputs: int) -> None:
        mod = self.modules[child.module_id]
        m = _Module(child.module_id, mod["className"])
        m.mod = mod
        m.count = len(mod["nodes"])
        for i in range(inputs):
            name = "x" if i == 0 else f"x{i}"
            child.inputs.append(m.add("input", INPUT, name, {"shape": {"literal": ""}, "dtype": {"literal": "float32"}}))
            child.input_names.append(name)
        for i in range(outputs):
            child.outputs.append(m.add("output", OUTPUT, "output" if i == 0 else f"output_{i}"))
        mod["interface"] = {"inputs": child.inputs, "outputs": child.outputs}


def _pick_entry(file: _File, class_name: str | None) -> str:
    names = list(file.classes)
    if not names:
        raise ImportFailure(t("代码中没有找到 nn.Module 的子类", "no nn.Module subclass found in the code"))
    if class_name:
        if class_name not in file.classes:
            available = ", ".join(names)
            raise ImportFailure(t(f"找不到类 {class_name}；可选：{available}", f"class {class_name} not found; available: {available}"))
        return class_name
    used = set()
    for cls in file.classes.values():
        for n in ast.walk(cls):
            if isinstance(n, ast.Call):
                q = file.qualname(n.func)
                if q in file.classes and q != cls.name:
                    used.add(q)
    candidates = [n for n in names if n not in used]
    # Model files usually define the top-level model last.
    return (candidates or names)[-1]


def _preamble(file: _File, warnings: list[str]) -> str:
    """Top-level code other than the standard imports and the module classes, verbatim."""
    text = ""
    previous: ast.stmt | None = None  # the last statement kept, when it was also the last one seen
    last: ast.stmt | None = None  # the last statement kept
    seen_class = False
    for stmt in file.tree.body:
        module_class = isinstance(stmt, ast.ClassDef) and stmt.name in file.classes
        drop = _is_standard_import(stmt) or module_class
        if _is_main_guard(stmt):
            warnings.append(t("已忽略 if __name__ == '__main__' 中的测试代码", "ignored the test code under if __name__ == '__main__'"))
            drop = True
        elif not drop and seen_class and _loaded_names(stmt) & set(file.classes):
            warnings.append(
                t(f"第 {stmt.lineno} 行的顶层代码引用了模型类，已忽略", f"ignored the top-level code on line {stmt.lineno}, which refers to a model class")
            )
            drop = True
        seen_class |= module_class
        if drop:
            previous = None
            continue
        body = file.statements([stmt])
        if text and previous is not None:
            # Neighbours in the original keep the blank lines and comments between them.
            start = min([stmt.lineno, *(d.lineno for d in getattr(stmt, "decorator_list", []))])
            gap = file.lines[(previous.end_lineno or previous.lineno) : start - 1]
            text += "\n" + "".join(f"{line}\n" for line in gap) + body
        elif text:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                text += "\n\n\n" + body
            else:
                # A blank line after the imports, as the original most likely had one.
                after_imports = isinstance(last, (ast.Import, ast.ImportFrom)) and not isinstance(stmt, (ast.Import, ast.ImportFrom))
                text += ("\n\n" if after_imports else "\n") + body
        else:
            text = body
        previous = last = stmt
    return text


def _guess_input(module: dict, input_id: str, warnings: list[str]) -> InputSpec:
    """An input shape from the layer that consumes the input, with symbols where the code says nothing."""
    first = next((e for e in module["edges"] if e["source"]["node"] == input_id), None)
    node = module["nodes"].get(first["target"]["node"]) if first else None
    name = module["nodes"][input_id]["name"]
    if node and node["kind"] == "layer":
        p = node["params"]

        def value(key: str, symbol: str) -> int | str:
            v = p.get(key, {}).get("literal")
            return v if isinstance(v, int) and not isinstance(v, bool) else symbol

        layer = node["type"].rsplit(".", 1)[-1]
        dims: list[int | str] | None = None
        dtype = "float32"
        if layer == "Linear":
            dims = ["B", value("in_features", "D")]
        elif layer == "Embedding":
            dims, dtype = ["B", "T"], "int64"
        elif layer == "Conv1d":
            dims = ["B", value("in_channels", "C"), "L"]
        elif layer == "Conv2d":
            dims = ["B", value("in_channels", "C"), "H", "W"]
        elif layer == "Conv3d":
            dims = ["B", value("in_channels", "C"), "D", "H", "W"]
        elif layer in ("LSTM", "GRU", "RNN"):
            size = value("input_size", "D")
            dims = ["B", "T", size] if p.get("batch_first", {}).get("literal") else ["T", "B", size]
        if dims is not None:
            text = ", ".join(str(d) for d in dims)
            warnings.append(
                t(
                    f"输入 {name} 的形状按 {node['name']} 推断为 [{text}]，可在输入节点上修改",
                    f"shape of input {name} inferred from {node['name']} as [{text}]; you can change it on the input node",
                )
            )
            return InputSpec(dims, dtype)
    warnings.append(t(f"无法推断输入 {name} 的形状，请在输入节点上填写", f"cannot infer the shape of input {name}; fill it in on the input node"))
    return InputSpec(["B"])


def import_static(
    source: str,
    class_name: str | None = None,
    inputs: dict[str, InputSpec] | None = None,
    filename: str = "<model.py>",
) -> dict[str, Any]:
    try:
        tree = ast.parse(source, filename)
    except SyntaxError as e:
        raise ImportFailure(t(f"语法错误（第 {e.lineno} 行）：{e.msg}", f"syntax error (line {e.lineno}): {e.msg}")) from e
    file = _File(source, tree)
    entry = _pick_entry(file, class_name)
    imp = _Importer(file)
    # Dependencies are converted on demand, so a parent always knows its children's interfaces.
    for name in file.classes:
        if name not in imp.children:
            imp.convert(name)

    # Modules in class order, sequential containers after the class that owns them.
    order = list(file.classes) + [k for k in imp.modules if k not in file.classes]
    modules = {k: imp.modules[k] for k in order}

    entry_mod = modules[entry]
    inputs = dict(inputs or {})
    for input_id in entry_mod["interface"]["inputs"]:
        node = entry_mod["nodes"][input_id]
        spec = inputs.get(node["name"]) or _guess_input(entry_mod, input_id, imp.warnings)
        node["params"]["shape"] = {"literal": ", ".join(str(d) for d in spec.dims)}
        node["params"]["dtype"] = {"literal": spec.dtype}

    project: dict[str, Any] = {"schemaVersion": SCHEMA_VERSION, "name": entry, "entry": entry, "modules": modules}
    preamble = _preamble(file, imp.warnings)
    if preamble:
        project["preamble"] = preamble
    report = sorted(imp.report, key=lambda r: list(file.classes).index(r["className"]))
    return {"project": project, "report": report, "warnings": imp.warnings, "observedShapes": {}, "batchProbe": BATCH_PROBE}
