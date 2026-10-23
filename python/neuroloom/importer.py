"""Dynamic import: PyTorch source -> NeuroLoom IR, via torch.fx (PRD section 11, level L2).

M0 scope: flat graphs of Linear, ReLU and F.relu. User-defined submodules are
traced through (flattened); keeping them as module nodes is M2 work, and so is
the code-block fallback for unsupported operations.
"""

from __future__ import annotations

import contextlib
import keyword
import sys
from typing import Any

from . import SCHEMA_VERSION

# Symbolic dimensions (e.g. B) are given this size when shapes are probed on the meta device.
BATCH_PROBE = 2

INPUT = "neuroloom.Input"
OUTPUT = "neuroloom.Output"
LINEAR = "torch.nn.Linear"
RELU = "torch.nn.ReLU"
RELU_FN = "torch.nn.functional.relu"


class ImportFailure(Exception):
    """An import that cannot produce a graph; the message is shown to the user."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


Dims = list[int | str]


def parse_dims(text: str) -> Dims:
    """"B,784" -> ["B", 784]."""
    dims: Dims = []
    for token in text.replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if not token:
            continue
        dims.append(int(token) if token.isdigit() else token)
    if not dims:
        raise ImportFailure(f"无法解析形状：{text!r}")
    return dims


def _shape_text(dims: Dims) -> str:
    return ", ".join(str(d) for d in dims)


def _pick_class(namespace: dict[str, Any], module_name: str, class_name: str | None):
    from torch import nn

    candidates = [
        v
        for v in namespace.values()
        if isinstance(v, type) and issubclass(v, nn.Module) and v.__module__ == module_name
    ]
    if not candidates:
        raise ImportFailure("代码中没有找到 nn.Module 的子类")
    if class_name:
        for cls in candidates:
            if cls.__name__ == class_name:
                return cls
        names = ", ".join(c.__name__ for c in candidates)
        raise ImportFailure(f"找不到类 {class_name}；可选：{names}")
    # Default to the last class defined: model files usually define the top-level model last.
    return candidates[-1]


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.ir_of: dict[Any, str] = {}  # fx.Node -> IR node id
        self.warnings: list[str] = []

    def add(self, fx_node, kind: str, type_: str, name: str, params: dict[str, Any]) -> str:
        node_id = f"n{len(self.nodes) + 1}"
        self.nodes[node_id] = {
            "id": node_id,
            "kind": kind,
            "type": type_,
            "name": name,
            "params": {k: {"literal": v} for k, v in params.items()},
        }
        if fx_node is not None:
            self.ir_of[fx_node] = node_id
        return node_id

    def link(self, producer, consumer_id: str) -> None:
        if producer not in self.ir_of:
            return  # the producer was unsupported and is already reported
        self.edges.append(
            {
                "id": f"e{len(self.edges) + 1}",
                "source": {"node": self.ir_of[producer], "port": "out"},
                "target": {"node": consumer_id, "port": "in"},
            }
        )


def _module_name(target: str, builder: _Builder) -> str:
    name = target.replace(".", "_")
    if name != target:
        builder.warnings.append(f"子模块 {target} 被展平为 {name}，state_dict 键名会变化（M2 将保留层级）")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ImportFailure(f"子模块名 {target} 无法作为变量名")
    return name


def import_source(
    source: str,
    class_name: str | None = None,
    input_shapes: dict[str, Dims] | None = None,
    filename: str = "<model.py>",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import fx, nn

    input_shapes = input_shapes or {}
    module_name = "neuroloom_user_code"
    namespace: dict[str, Any] = {"__name__": module_name}

    # User code must not write to stdout: in the CLI that is where the result goes.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            exec(compile(source, filename, "exec"), namespace)
        except SyntaxError as e:
            raise ImportFailure(f"语法错误（第 {e.lineno} 行）：{e.msg}") from e
        cls = _pick_class(namespace, module_name, class_name)
        try:
            with torch.device("meta"):
                model = cls()
        except TypeError as e:
            raise ImportFailure(f"无法用默认参数实例化 {cls.__name__}：{e}") from e

        try:
            graph = fx.Tracer().trace(model)
        except Exception as e:  # fx raises many types; all mean "cannot trace"
            raise ImportFailure(f"torch.fx 追踪失败：{e}", {"exception": type(e).__name__}) from e
    gm = fx.GraphModule(model, graph)

    b = _Builder()
    unsupported: list[str] = []
    placeholders = []

    for node in graph.nodes:
        if node.op == "placeholder":
            node_id = b.add(node, "input", INPUT, str(node.target), {})
            placeholders.append((node, node_id))
        elif node.op == "call_module":
            sub = gm.get_submodule(node.target)
            name = _module_name(str(node.target), b)
            if type(sub) is nn.Linear:
                params = {"in_features": sub.in_features, "out_features": sub.out_features, "bias": sub.bias is not None}
                node_id = b.add(node, "layer", LINEAR, name, params)
            elif type(sub) is nn.ReLU:
                node_id = b.add(node, "layer", RELU, name, {"inplace": sub.inplace})
            else:
                unsupported.append(f"{type(sub).__module__}.{type(sub).__name__}（self.{node.target}）")
                continue
            b.link(node.args[0], node_id)
        elif node.op == "call_function" and node.target in (F.relu, torch.relu):
            inplace = bool(node.kwargs.get("inplace", node.args[1] if len(node.args) > 1 else False))
            node_id = b.add(node, "op", RELU_FN, node.name, {"inplace": inplace})
            b.link(node.args[0], node_id)
        elif node.op == "call_method" and node.target == "relu":
            node_id = b.add(node, "op", RELU_FN, node.name, {"inplace": False})
            b.link(node.args[0], node_id)
        elif node.op == "output":
            results = node.args[0]
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            for i, value in enumerate(results):
                if not isinstance(value, fx.Node):
                    unsupported.append(f"返回值中的非张量 {value!r}")
                    continue
                out_id = b.add(None, "output", OUTPUT, "output" if i == 0 else f"output_{i}", {})
                b.link(value, out_id)
        else:
            target = getattr(node.target, "__name__", node.target)
            unsupported.append(f"{node.op} {target}")

    if unsupported:
        raise ImportFailure(
            "M0 尚不支持以下操作：" + "；".join(unsupported),
            {"unsupported": unsupported},
        )

    # Input shapes: given explicitly, or inferred from a Linear that consumes the input directly.
    for node, node_id in placeholders:
        dims = input_shapes.get(str(node.target))
        if dims is None:
            first = next(iter(node.users), None)
            sub = gm.get_submodule(first.target) if first is not None and first.op == "call_module" else None
            if type(sub) is nn.Linear:
                dims = ["B", sub.in_features]
                b.warnings.append(f"输入 {node.target} 的形状由 {first.target} 推断为 [{_shape_text(dims)}]")
            else:
                raise ImportFailure(f"请提供输入 {node.target} 的形状，例如 {node.target}=B,3,224,224")
        b.nodes[node_id]["params"]["shape"] = {"literal": _shape_text(dims)}

    observed = _probe_shapes(gm, placeholders, b)

    module = {"id": "main", "className": cls.__name__, "nodes": b.nodes, "edges": b.edges, "layout": {}}
    project = {"schemaVersion": SCHEMA_VERSION, "name": cls.__name__, "entry": "main", "modules": {"main": module}}
    return {"project": project, "observedShapes": observed, "batchProbe": BATCH_PROBE, "warnings": b.warnings}


def _probe_shapes(gm, placeholders, b: _Builder) -> dict[str, list[list[int]]]:
    """Runs the traced graph once on the meta device and records every output shape."""
    import torch
    from torch.fx.passes.shape_prop import ShapeProp

    examples = []
    for _, node_id in placeholders:
        dims = parse_dims(b.nodes[node_id]["params"]["shape"]["literal"])
        examples.append(torch.empty([d if isinstance(d, int) else BATCH_PROBE for d in dims], device="meta"))
    try:
        ShapeProp(gm).propagate(*examples)
    except Exception as e:
        raise ImportFailure(f"按声明的输入形状运行模型失败：{e}") from e

    observed: dict[str, list[list[int]]] = {}
    for fx_node, node_id in b.ir_of.items():
        meta = fx_node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            observed[node_id] = [list(meta.shape)]
    return observed
