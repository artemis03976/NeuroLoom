"""Dynamic import: PyTorch source -> NeuroLoom IR, via torch.fx (PRD section 11, level L2).

Which layers and ops are understood, and how their arguments map to node
parameters, comes from specs.json, exported from the TypeScript NodeSpecs
(`npm run specs`). User-defined submodules are traced through (flattened);
keeping them as module nodes is M2 work, as is the code-block fallback for
unsupported operations.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import keyword
import operator
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

# Symbolic dimensions (e.g. B) are given this size when shapes are probed.
BATCH_PROBE = 2

INPUT = "neuroloom.Input"
OUTPUT = "neuroloom.Output"
DTYPES = ("float32", "int64")


class ImportFailure(Exception):
    """An import that cannot produce a graph; the message is shown to the user."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


Dims = list[int | str]


@dataclass
class InputSpec:
    dims: Dims
    dtype: str = "float32"


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


def parse_input(text: str) -> InputSpec:
    """"B,784" or "B,16:int64"."""
    dims, _, dtype = text.partition(":")
    dtype = dtype.strip() or "float32"
    if dtype not in DTYPES:
        raise ImportFailure(f"不支持的数据类型 {dtype}（可用：{'、'.join(DTYPES)}）")
    return InputSpec(parse_dims(dims), dtype)


def _shape_text(dims: Dims) -> str:
    return ", ".join(str(d) for d in dims)


# ---------------------------------------------------------------- catalog


def _resolve(qualname: str) -> Any:
    module, _, attr = qualname.rpartition(".")
    return getattr(importlib.import_module(module), attr)


@dataclass
class _Catalog:
    layers: dict[type, dict]
    functions: dict[Any, dict]
    methods: dict[str, dict]


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    manifest = json.loads(Path(__file__).with_name("specs.json").read_text(encoding="utf-8"))
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise RuntimeError("specs.json is out of date; run `npm run specs`")
    layers: dict[type, dict] = {}
    functions: dict[Any, dict] = {}
    methods: dict[str, dict] = {}
    for spec in manifest["specs"]:
        if spec.get("layerClass"):
            with contextlib.suppress(ImportError, AttributeError):  # e.g. a class newer than this torch
                layers[_resolve(spec["layerClass"])] = spec
        python = spec.get("python") or {}
        for name in python.get("functions", []):
            with contextlib.suppress(ImportError, AttributeError):
                functions[_resolve(name)] = spec
        for name in python.get("methods", []):
            methods[name] = spec
    return _Catalog(layers, functions, methods)


def _outputs(spec: dict, params: dict) -> list[dict]:
    dynamic = spec.get("dynamicOutputs")
    if dynamic:
        return [{"name": f"{dynamic['prefix']}{i}", "path": [i]} for i in range(int(params[dynamic["param"]]))]
    return spec["outputs"]


# ---------------------------------------------------------------- values


def _plain(value: Any) -> Any:
    import torch

    if isinstance(value, (tuple, list, torch.Size)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ImportFailure(f"无法表示的参数值 {value!r}")


def _normalize(value: Any, param: dict) -> Any:
    """Python value -> the canonical IR literal for this parameter."""
    if param["type"] == "bool" and not isinstance(value, bool):
        return value is not None  # e.g. a bias Parameter, or None when disabled
    value = _plain(value)
    if param["type"] == "ints" and isinstance(value, list) and value:
        if param.get("arity") is not None and all(v == value[0] for v in value):
            return value[0]
        if param.get("arity") is None and len(value) == 1:
            return value[0]
    return value


def _expand(value: Any, n: int) -> list:
    return list(value) if isinstance(value, (tuple, list)) else [value] * n


def _pool_params(module) -> dict:
    # stride=None means "same as kernel_size"; torch stores the resolved value.
    k, s = module.kernel_size, module.stride
    n = max(len(k) if isinstance(k, tuple) else 1, len(s) if isinstance(s, tuple) else 1)
    return {"stride": None if _expand(k, n) == _expand(s, n) else s}


def _transformer_layer_params(module) -> dict:
    import torch.nn.functional as F
    from torch import nn

    gelu = module.activation is F.gelu or isinstance(module.activation, nn.GELU)
    return {
        "d_model": module.self_attn.embed_dim,
        "nhead": module.self_attn.num_heads,
        "dim_feedforward": module.linear1.out_features,
        "dropout": module.dropout.p,
        "activation": "gelu" if gelu else "relu",
        "layer_norm_eps": module.norm1.eps,
        "batch_first": module.self_attn.batch_first,
        "norm_first": module.norm_first,
        "bias": module.linear1.bias is not None,
    }


# Layers whose constructor arguments are not all stored as same-named attributes.
_CUSTOM_LAYER_PARAMS = {
    "torch.nn.MultiheadAttention": lambda m: {"bias": m.in_proj_bias is not None},
    "torch.nn.TransformerEncoderLayer": _transformer_layer_params,
    "torch.nn.TransformerDecoderLayer": _transformer_layer_params,
    **{f"torch.nn.{kind}Pool{n}d": _pool_params for kind in ("Max", "Avg") for n in (1, 2, 3)},
}


def _layer_params(spec: dict, module) -> dict:
    custom = _CUSTOM_LAYER_PARAMS.get(spec["type"])
    known = custom(module) if custom else {}
    params = {}
    for p in spec["params"]:
        name = p["name"]
        if name in known:
            value = known[name]
        elif hasattr(module, name):
            value = getattr(module, name)
        else:
            raise ImportFailure(f"无法读取 {type(module).__name__}.{name}")
        params[name] = _normalize(value, p)
    return params


def _describe(node) -> str:
    target = node.target
    if node.op == "call_method":
        return f".{target}()"
    return f"{getattr(target, '__module__', '') or ''}.{getattr(target, '__name__', target)}()".lstrip(".")


def _defaults(fn) -> dict[str, Any]:
    try:
        return {
            name: p.default
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
    except (TypeError, ValueError):  # builtins without a signature
        return {}


def _op_params(spec: dict, node) -> tuple[dict, list[tuple[str, Any]]]:
    """Maps a call's arguments to parameters and (port, producer) pairs, following the spec's Python signature."""
    from torch import fx

    signature = spec["python"]["args"]
    params_by_name = {p["name"]: p for p in spec["params"]}
    args = list(node.args)
    # `2 * x` puts the constant first; addition and multiplication commute.
    if spec["type"] in ("torch.add", "torch.mul") and len(args) == 2 and not isinstance(args[0], fx.Node):
        args.reverse()

    bound: dict[str, Any] = {}
    for i, value in enumerate(args):
        if i >= len(signature):
            raise ImportFailure(f"{_describe(node)} 的参数过多")
        entry = signature[i]
        spread = "param" in entry and params_by_name[entry["param"]].get("spread")
        if spread and not isinstance(value, (tuple, list)):
            bound[entry["name"]] = [value, *args[i + 1 :]]  # x.permute(0, 2, 1)
            break
        bound[entry["name"]] = value
    defaults = _defaults(node.target) if node.op == "call_function" else {}
    for name, value in node.kwargs.items():
        if not any(e["name"] == name for e in signature):
            # fx records some functions' defaults explicitly (F.softmax adds _stacklevel=3, dtype=None).
            if name in defaults and defaults[name] == value:
                continue
            raise ImportFailure(f"{_describe(node)} 的参数 {name} 暂不支持")
        bound[name] = value

    params: dict[str, Any] = {}
    edges: list[tuple[str, Any]] = []
    for entry in signature:
        name = entry["name"]
        value = bound.get(name)
        if "ports" in entry:
            if not isinstance(value, (tuple, list)) or not all(isinstance(v, fx.Node) for v in value):
                raise ImportFailure(f"{_describe(node)} 的 {name} 必须是张量列表")
            edges += [(f"{entry['ports']}{i}", v) for i, v in enumerate(value)]
        elif "port" in entry:
            if isinstance(value, fx.Node):
                edges.append((entry["port"], value))
            elif entry.get("scalarParam") and isinstance(value, (int, float)) and not isinstance(value, bool):
                params[entry["scalarParam"]] = float(value)
            else:
                raise ImportFailure(f"{_describe(node)} 的参数 {name} 必须是张量")
        elif name in bound:
            if isinstance(value, fx.Node) or (isinstance(value, (tuple, list)) and any(isinstance(v, fx.Node) for v in value)):
                raise ImportFailure(f"{_describe(node)} 的参数 {name} 依赖运行时的值（如 x.size(0)），M2 起支持")
            params[entry["param"]] = _normalize(value, params_by_name[entry["param"]])

    for p in spec["params"]:
        if p["name"] not in params:
            if "default" not in p:
                raise ImportFailure(f"{_describe(node)} 缺少参数 {p['name']}")
            params[p["name"]] = p["default"]
    return params, edges


# ---------------------------------------------------------------- graph


@dataclass
class _Graph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    value_of: dict[Any, tuple[str, str]] = field(default_factory=dict)  # fx node -> (IR id, port)
    tuple_of: dict[Any, tuple[str, list[int], list[dict]]] = field(default_factory=dict)  # fx node -> (IR id, path, outputs)
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def add(self, kind: str, type_: str, name: str, params: dict) -> str:
        node_id = f"n{len(self.nodes) + 1}"
        self.nodes[node_id] = {
            "id": node_id,
            "kind": kind,
            "type": type_,
            "name": name,
            "params": {k: {"literal": v} for k, v in params.items()},
        }
        return node_id

    def produces(self, fx_node, node_id: str, outputs: list[dict]) -> None:
        if len(outputs) == 1 and not outputs[0].get("path"):
            self.value_of[fx_node] = (node_id, outputs[0]["name"])
        else:
            self.tuple_of[fx_node] = (node_id, [], outputs)

    def link(self, producer, node_id: str, port: str) -> None:
        source = self.value_of.get(producer)
        if source is None:
            if producer in self.tuple_of:
                self.unsupported.append("把整个元组结果作为输入")
            return  # otherwise the producer was unsupported and is already reported
        self.edges.append(
            {
                "id": f"e{len(self.edges) + 1}",
                "source": {"node": source[0], "port": source[1]},
                "target": {"node": node_id, "port": port},
            }
        )


def _module_name(target: str, graph: _Graph) -> str:
    name = target.replace(".", "_")
    if name != target:
        graph.warnings.append(f"子模块 {target} 被展平为 {name}，state_dict 键名会变化（M2 将保留层级）")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ImportFailure(f"子模块名 {target} 无法作为变量名")
    return name


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
        raise ImportFailure(f"找不到类 {class_name}；可选：{', '.join(c.__name__ for c in candidates)}")
    # Default to the last class defined: model files usually define the top-level model last.
    return candidates[-1]


def import_source(
    source: str,
    class_name: str | None = None,
    inputs: dict[str, InputSpec] | None = None,
    filename: str = "<model.py>",
) -> dict[str, Any]:
    import torch
    from torch import fx

    inputs = dict(inputs or {})
    catalog = _catalog()
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

    g = _Graph()
    placeholders = []
    for node in graph.nodes:
        try:
            _add_node(g, gm, node, catalog, placeholders)
        except ImportFailure as e:
            g.unsupported.append(str(e))

    if g.unsupported:
        raise ImportFailure("暂不支持以下内容：" + "；".join(dict.fromkeys(g.unsupported)), {"unsupported": g.unsupported})

    for node, node_id in placeholders:
        spec = inputs.get(str(node.target)) or _guess_input(node, gm, g)
        g.nodes[node_id]["params"]["shape"] = {"literal": _shape_text(spec.dims)}
        g.nodes[node_id]["params"]["dtype"] = {"literal": spec.dtype}
        inputs[str(node.target)] = spec

    observed = _probe_shapes(gm, cls, graph, placeholders, inputs, g)
    interface = {
        "inputs": [i for i, n in g.nodes.items() if n["kind"] == "input"],
        "outputs": [i for i, n in g.nodes.items() if n["kind"] == "output"],
    }
    module = {
        "id": "main",
        "className": cls.__name__,
        "nodes": g.nodes,
        "edges": g.edges,
        "layout": {},
        "interface": interface,
    }
    project = {"schemaVersion": SCHEMA_VERSION, "name": cls.__name__, "entry": "main", "modules": {"main": module}}
    return {"project": project, "observedShapes": observed, "batchProbe": BATCH_PROBE, "warnings": g.warnings}


def _add_node(g: _Graph, gm, node, catalog: _Catalog, placeholders: list) -> None:
    from torch import fx

    if node.op == "placeholder":
        node_id = g.add("input", INPUT, str(node.target), {})
        g.value_of[node] = (node_id, "out")
        placeholders.append((node, node_id))
    elif node.op == "call_module":
        module = gm.get_submodule(node.target)
        spec = catalog.layers.get(type(module))
        if spec is None:
            raise ImportFailure(f"{type(module).__module__}.{type(module).__name__}（self.{node.target}）")
        params = _layer_params(spec, module)
        node_id = g.add("layer", spec["type"], _module_name(str(node.target), g), params)
        if node.kwargs or len(node.args) > len(spec["inputs"]):
            raise ImportFailure(f"self.{node.target} 的额外参数（如 mask、初始状态）暂不支持")
        for port, arg in zip(spec["inputs"], node.args):
            g.link(arg, node_id, port["name"])
        g.produces(node, node_id, _outputs(spec, params))
    elif node.op == "call_function" and node.target is operator.getitem:
        source, index = node.args
        if source not in g.tuple_of or not isinstance(index, int):
            raise ImportFailure("张量切片或索引（如 x[:, 0]）")
        node_id, path, outputs = g.tuple_of[source]
        path = [*path, index]
        match = [o for o in outputs if o.get("path") == path]
        if match:
            g.value_of[node] = (node_id, match[0]["name"])
        elif any(o.get("path", [])[: len(path)] == path for o in outputs):
            g.tuple_of[node] = (node_id, path, outputs)
        else:
            raise ImportFailure(f"元组下标 {index} 超出范围")
    elif node.op in ("call_function", "call_method"):
        spec = catalog.methods.get(node.target) if node.op == "call_method" else catalog.functions.get(node.target)
        if spec is None:
            raise ImportFailure(_describe(node))
        params, edges = _op_params(spec, node)
        node_id = g.add("op", spec["type"], node.name, params)
        for port, producer in edges:
            g.link(producer, node_id, port)
        g.produces(node, node_id, _outputs(spec, params))
    elif node.op == "output":
        results = node.args[0]
        results = list(results) if isinstance(results, (tuple, list)) else [results]
        for i, value in enumerate(results):
            if not isinstance(value, fx.Node):
                raise ImportFailure(f"返回值中的非张量 {value!r}")
            out_id = g.add("output", OUTPUT, "output" if i == 0 else f"output_{i}", {})
            g.link(value, out_id, "in")
    else:
        raise ImportFailure(f"{node.op} {getattr(node.target, '__name__', node.target)}")


def _guess_input(node, gm, g: _Graph) -> InputSpec:
    """An input shape from the layer that consumes the input directly, when that tells enough."""
    from torch import nn

    first = next(iter(node.users), None)
    module = gm.get_submodule(first.target) if first is not None and first.op == "call_module" else None
    if isinstance(module, nn.Linear):
        dims: Dims = ["B", module.in_features]
        g.warnings.append(f"输入 {node.target} 的形状由 {first.target} 推断为 [{_shape_text(dims)}]")
        return InputSpec(dims)
    raise ImportFailure(f"请提供输入 {node.target} 的形状，例如 {node.target}=B,3,224,224")


def _probe_shapes(gm, cls, graph, placeholders, inputs: dict[str, InputSpec], g: _Graph) -> dict[str, dict[str, list[int]]]:
    """Runs the traced graph once and records every value's shape: on the meta
    device when the ops support it, otherwise on the CPU with a fresh model."""
    import torch
    from torch import fx
    from torch.fx.passes.shape_prop import ShapeProp

    def examples(device: str):
        tensors = []
        for node, _ in placeholders:
            spec = inputs[str(node.target)]
            size = [d if isinstance(d, int) else BATCH_PROBE for d in spec.dims]
            if spec.dtype == "int64":
                tensors.append(torch.zeros(size, dtype=torch.long, device=device))
            else:
                tensors.append(torch.randn(size, device=device) if device == "cpu" else torch.empty(size, device=device))
        return tensors

    try:
        ShapeProp(gm).propagate(*examples("meta"))
    except Exception:
        try:
            with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
                cpu = fx.GraphModule(cls(), graph)
                ShapeProp(cpu).propagate(*examples("cpu"))
        except Exception as e:
            raise ImportFailure(f"按声明的输入形状运行模型失败：{e}") from e

    observed: dict[str, dict[str, list[int]]] = {}
    for fx_node, (node_id, port) in g.value_of.items():
        meta = fx_node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            observed.setdefault(node_id, {})[port] = list(meta.shape)
    return observed


def import_many(requests: list[dict]) -> list[dict]:
    """Imports several sources in one process (used by the round-trip tests)."""
    results = []
    for request in requests:
        try:
            inputs = {k: parse_input(v) for k, v in (request.get("inputShapes") or {}).items()}
            results.append(import_source(request["source"], request.get("className"), inputs))
        except ImportFailure as e:
            results.append({"error": {"message": str(e), "details": e.details}})
    return results
