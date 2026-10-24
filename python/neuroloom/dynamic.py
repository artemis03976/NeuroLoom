"""Dynamic import (PRD 11.2 L2, 11.4): run the model, keep its hierarchy, merge with the static result.

The model is instantiated once (in the CLI process, which the server starts as a
child with a timeout). Every module instance then becomes a module definition,
chosen class by class:

- A class the static importer structured completely (L1) keeps its static
  definition, so its expressions, hyperparameters and variable names survive;
  each instance passes the arguments it was really constructed with.
- Any other module is traced with torch.fx with every child module as a leaf, so
  the hierarchy and the state_dict keys are kept: nn.Sequential children become
  inline containers, nn.ModuleList loops become repeats, third-party classes
  external modules. Instances that trace to the same graph share a definition.
- Where tracing fails, the static definition is used, its children resolved the
  same way. If even that is impossible the static result is returned unchanged,
  so a dynamic import is never worse than a static one.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import operator
import re
import sys
import types
from typing import Any

from . import SCHEMA_VERSION
from .i18n import t
from .importer import BATCH_PROBE, INPUT, OUTPUT, ImportFailure, InputSpec, _catalog, _describe, _layer_params, _op_params, _outputs, _state_node
from .probe import RUNS
from .static import import_static

USER_MODULE = "neuroloom_user_code"


class TraceFailed(Exception):
    """One module could not be traced; its static definition is used instead."""


class Fallback(Exception):
    """The dynamic result cannot be completed; the static result is returned."""


# ---------------------------------------------------------------- running the model


_recording: dict[int, tuple[tuple, dict]] | None = None


def _module_new(cls, *args, **kwargs):
    obj = object.__new__(cls)
    if _recording is not None:
        _recording[id(obj)] = (args, kwargs)
    return obj


@contextlib.contextmanager
def _record_constructor_args(records: dict[int, tuple[tuple, dict]]):
    """
    Remembers the arguments every nn.Module is constructed with, keyed by id().
    The hook stays on nn.Module once set: deleting a class's __new__ does not give
    back object.__new__ in CPython, and every model built later would fail with
    "object.__new__() takes exactly one argument".
    """
    global _recording
    from torch import nn

    if nn.Module.__dict__.get("__new__") is not _module_new:
        nn.Module.__new__ = _module_new
    _recording = records
    try:
        yield
    finally:
        _recording = None


@contextlib.contextmanager
def _weights_on_meta():
    """Parameters and buffers move to the meta device as modules register them; other tensors stay where they are."""
    from torch import nn

    register_parameter = nn.Module.register_parameter
    register_buffer = nn.Module.register_buffer

    def parameter(self, name, param):
        register_parameter(self, name, param)
        if param is not None:
            self._parameters[name] = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    def buffer(self, name, tensor, persistent=True):
        register_buffer(self, name, tensor, persistent)
        if tensor is not None:
            self._buffers[name] = tensor.to("meta")

    nn.Module.register_parameter = parameter
    nn.Module.register_buffer = buffer
    try:
        yield
    finally:
        nn.Module.register_parameter = register_parameter
        nn.Module.register_buffer = register_buffer


def _literal(value: Any) -> Any:
    """A constructor argument as IR JSON; TypeError when it is an object (a module, a class, a config)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_literal(v) for v in value]
    raise TypeError(value)


def _tensors(value: Any, path: tuple = ()) -> list[tuple[tuple, Any]]:
    """Tensors in a (nested) result with their index paths."""
    import torch

    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        return [item for i, v in enumerate(value) for item in _tensors(v, (*path, i))]
    return []


class _Observation:
    """What each module instance was called with, and the shapes it returned, in two runs (see probe.RUNS)."""

    def __init__(self) -> None:
        self.calls: dict[int, dict[str, Any]] = {}  # id -> forward parameter -> "tensor" or the constant passed
        self.outputs: list[dict[int, list[tuple[tuple, list[int], str]]]] = [{}, {}]
        self.values: list[dict[str, int]] = []


def _observe(model, specs: list[tuple[str, InputSpec]]) -> _Observation:
    import torch

    obs = _Observation()
    run = 0

    def before(module, args, kwargs):
        if id(module) in obs.calls:
            return
        with contextlib.suppress(TypeError, ValueError):
            bound = inspect.signature(module.forward).bind(*args, **kwargs)
            obs.calls[id(module)] = {k: "tensor" if isinstance(v, torch.Tensor) else v for k, v in bound.arguments.items()}

    def after(module, _args, output):
        obs.outputs[run].setdefault(
            id(module), [(path, list(v.shape), str(v.dtype).replace("torch.", "")) for path, v in _tensors(output)]
        )

    symbols: list[str] = []
    for _, spec in specs:
        for d in spec.dims:
            if isinstance(d, str) and d not in symbols:
                symbols.append(d)
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    handles = []
    for m in model.modules():
        handles.append(m.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(m.register_forward_hook(after))
    try:
        for run, sequence in enumerate(RUNS):
            value = {s: sequence[i] for i, s in enumerate(symbols)}
            obs.values.append(value)
            tensors = []
            for _, spec in specs:
                size = [d if isinstance(d, int) else value[d] for d in spec.dims]
                if spec.dtype == "int64":
                    tensors.append(torch.zeros(size, dtype=torch.long, device=device))
                else:
                    tensors.append(torch.empty(size, device=device) if device.type == "meta" else torch.randn(size, device=device))
            with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
                model(*tensors)
    finally:
        for h in handles:
            h.remove()
    return obs


def _dim_from_runs(a: int, b: int, first: dict[str, int], second: dict[str, int]) -> int | str | None:
    """As dimFromRuns in packages/core/src/shape.ts: a constant, or a product of symbols times a constant."""
    if a == b:
        return a
    symbols = list(first)
    for mask in range(1, 1 << len(symbols)):
        picked = [s for i, s in enumerate(symbols) if mask & (1 << i)]
        pa = 1
        pb = 1
        for s in picked:
            pa *= first[s]
            pb *= second[s]
        if a % pa == 0 and b % pb == 0 and a // pa == b // pb:
            coef = a // pa
            return "*".join(sorted(picked) + ([] if coef == 1 else [str(coef)]))
    return None


# ---------------------------------------------------------------- building definitions


def _leaf_tracer():
    from torch import fx

    class LeafTracer(fx.Tracer):
        """Every child module is a call, never traced into: each one gets a definition of its own."""

        def is_leaf_module(self, m, module_qualified_name):
            return True

    return LeafTracer()


class _Builder:
    def __init__(self, static_project: dict, static_report: list[dict], records: dict, obs: _Observation):
        self.static = static_project["modules"]
        self.levels = {r["className"]: r for r in static_report}
        self.records = records
        self.obs = obs
        self.catalog = _catalog()
        self.modules: dict[str, dict] = {}
        self.by_content: dict[str, str] = {}
        self.cache: dict[int, tuple[str, dict]] = {}
        self.report: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.origin: dict[str, str] = {}  # registered id -> the static definition it came from
        self.pure_memo: dict[str, bool] = {}
        self.import_of: dict[str, tuple[str, str]] = {}  # external definition -> (import line, name it binds)
        self.root = None  # the model being imported
        self.root_graph = None  # (fx graph, definition, fx node -> (node id, port)) when the entry was traced

    # .................................................... registry

    def register(self, mod: dict, level: str, notes: list[str] | None = None, origin: str | None = None) -> str:
        """
        Adds a definition, or returns the id of an identical one; ids and class
        names stay unique. `origin` is the static definition it was made from,
        whose hyperparameters it keeps.
        """
        content = {k: v for k, v in mod.items() if k not in ("id", "className", "layout")}
        key = json.dumps({"base": mod["className"], **content}, sort_keys=True)
        if self.by_content.get(key) in self.modules:  # not pruned since
            return self.by_content[key]
        base = mod["id"]
        module_id, i = base, 2
        while module_id in self.modules:
            module_id, i = f"{base}_{i}", i + 1
        class_name, i = mod["className"], 2
        taken = {m["className"] for m in self.modules.values()}
        # An external call keeps the name it is imported by; other classes get unique names.
        external = mod.get("opaque") is not None and "source" not in mod["opaque"]
        while class_name in taken and not mod.get("sequential") and not external:
            class_name, i = f"{mod['className']}{i}", i + 1
        self.modules[module_id] = {**mod, "id": module_id, "className": class_name, "layout": {}}
        self.by_content[key] = module_id
        if origin is not None:
            self.origin[module_id] = origin
        if not mod.get("sequential") and not (mod.get("opaque") is not None and "source" not in mod.get("opaque", {})):
            entry = {"className": class_name, "level": level, "codeBlocks": sum(1 for n in mod["nodes"].values() if n["kind"] == "code"), "notes": notes or []}
            self.report.setdefault(class_name, entry)
        return module_id

    # .................................................... instances

    def is_user(self, cls) -> bool:
        return cls.__module__ == USER_MODULE

    def constructor_params(self, m) -> dict | None:
        """Hyperparameter values an instance was constructed with, or None when one is not a literal."""
        args, kwargs = self.records.get(id(m), ((), {}))
        try:
            bound = inspect.signature(type(m)).bind(*args, **kwargs)
        except (TypeError, ValueError):
            return None
        params = {}
        for name, value in bound.arguments.items():
            kind = inspect.signature(type(m)).parameters[name].kind
            if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return None
            try:
                params[name] = {"literal": _literal(value)}
            except TypeError:
                return None
        return params

    def verbatim_params(self, m) -> dict | None:
        """Arguments as written for a class whose signature NeuroLoom does not model: "#0", "#1", keywords."""
        args, kwargs = self.records.get(id(m), ((), {}))
        try:
            params = {f"#{i}": {"literal": _literal(v)} for i, v in enumerate(args)}
            params.update({k: {"literal": _literal(v)} for k, v in kwargs.items()})
        except TypeError:
            return None
        return params

    def resolve(self, m) -> tuple[str, dict]:
        """The definition an instance uses and the hyperparameter values it passes."""
        from torch import nn

        if id(m) in self.cache:
            return self.cache[id(m)]
        cls = type(m)
        name = cls.__name__
        static = self.static.get(name) if self.is_user(cls) else None
        result: tuple[str, dict] | None = None
        notes: list[str] = []

        if static is not None and "opaque" not in static and self.levels.get(name, {}).get("level") == "L1":
            params = self.constructor_params(m)
            if params is not None:
                result = (self.register(self.retarget(static, m), "L1", origin=name), params)
            else:
                notes.append(t("构造参数不是常量，改为追踪", "constructor arguments are not constants, traced instead"))
        if result is None and isinstance(m, nn.Sequential) and type(m).forward is nn.Sequential.forward:
            result = (self.sequential(m, name), {})
        # A library module without submodules (DropPath) stays one call; one with submodules
        # (a whole torchvision model) is traced for its structure, and called only if that fails.
        composite = next(m.children(), None) is not None
        if result is None and not self.is_user(cls) and not composite:
            params = self.verbatim_params(m)
            if params is not None:
                result = (self.external(m), params)
        if result is None:
            try:
                mod = self.trace(m, name)
                result = (self.register(mod, "L2", notes + [t("用 torch.fx 追踪", "traced with torch.fx")]), {})
            except TraceFailed as e:
                reason = str(e)
                params = None if self.is_user(cls) or static is not None else self.verbatim_params(m)
                if params is not None:
                    self.warnings.append(
                        t(f"{cls.__module__}.{name} 无法追踪（{reason}），按外部模块导入", f"{cls.__module__}.{name} cannot be traced ({reason}) and is imported as an external module")
                    )
                    result = (self.external(m), params)
                elif static is not None and "opaque" not in static:
                    params = self.constructor_params(m)
                    if params is None:
                        raise Fallback(t(f"{name} 无法追踪（{reason}），构造参数也不是常量", f"{name} cannot be traced ({reason}) and its constructor arguments are not constants")) from e
                    level = self.levels.get(name, {}).get("level", "L1+L3")
                    result = (self.register(self.retarget(static, m), level, [t(f"无法追踪：{reason}", f"cannot be traced: {reason}")], origin=name), params)
                elif static is not None:
                    params = self.verbatim_params(m) if static["opaque"].get("verbatimArgs") else self.constructor_params(m)
                    if params is None:
                        raise Fallback(t(f"{name} 无法追踪（{reason}）", f"{name} cannot be traced ({reason})")) from e
                    result = (self.register(copy.deepcopy(static), "L3", [t(f"无法追踪：{reason}", f"cannot be traced: {reason}")], origin=name), params)
                else:
                    raise Fallback(t(f"{cls.__module__}.{name} 无法追踪（{reason}）", f"{cls.__module__}.{name} cannot be traced ({reason})")) from e
        self.cache[id(m)] = result
        return result

    def external(self, m) -> str:
        """A third-party class kept as a call: ports from its forward signature and observed outputs."""
        cls = type(m)
        # The name as the code writes it (timm.layers.DropPath, DropPath): from the static result when it has one.
        # Otherwise the class imported by name (from torchvision.ops.stochastic_depth import StochasticDepth):
        # a module path can be shadowed (torchvision.ops.stochastic_depth is also a function).
        static_name = next(
            (s["className"] for s in self.static.values() if s.get("opaque") == {} and s["className"].rsplit(".", 1)[-1] == cls.__name__), None
        )
        torch_nn = cls.__module__.startswith("torch.nn")
        imported = None if static_name or torch_nn else self.import_name(cls)
        written = static_name or (f"nn.{cls.__name__}" if torch_nn else imported[1] + cls.__qualname__[len(cls.__qualname__.split(".")[0]) :])
        mod = self.empty(f"ext_{cls.__name__}", written)
        mod["opaque"] = {}
        params = self.verbatim_params(m) or {}
        mod["hyperparams"] = [{"name": k} for k in params]
        for param, kind in self.obs.calls.get(id(m), {}).items():
            if kind == "tensor":
                self.add_input(mod, param)
        outputs = self.obs.outputs[0].get(id(m), [])
        for i in range(max(1, len(outputs))):
            self.add_output(mod, i)
        if not cls.__module__.startswith("torch."):
            self.warnings.append(t(f"{cls.__module__}.{cls.__name__} 来自其他库，按外部模块导入", f"{cls.__module__}.{cls.__name__} comes from another library and is imported as an external module"))
        module_id = self.register(mod, "L3")
        if imported is not None:
            self.import_of[module_id] = imported
        return module_id

    def import_name(self, cls) -> tuple[str, str]:
        """The import line for a library class and the name it binds; another class of the same name gets an alias."""
        root = cls.__qualname__.split(".")[0]
        taken = {name for _, name in self.import_of.values()} | set(self.static) | {m["className"] for m in self.modules.values()}
        line = f"from {cls.__module__} import {root}"
        existing = next((name for imp, name in self.import_of.values() if imp.startswith(line)), None)
        if existing:
            return next(imp for imp, name in self.import_of.values() if name == existing), existing
        name = root
        for i in range(2, 100):
            if name not in taken:
                break
            name = f"{root}_{i}"
        return (line if name == root else f"{line} as {name}"), name

    # .................................................... definitions

    def empty(self, module_id: str, class_name: str) -> dict:
        return {"id": module_id, "className": class_name, "hyperparams": [], "nodes": {}, "edges": [], "layout": {}, "interface": {"inputs": [], "outputs": []}}

    def add_input(self, mod: dict, name: str) -> str:
        node_id = f"n{len(mod['nodes']) + 1}"
        mod["nodes"][node_id] = {"id": node_id, "kind": "input", "type": INPUT, "name": name, "params": {"shape": {"literal": ""}, "dtype": {"literal": "float32"}}}
        mod["interface"]["inputs"].append(node_id)
        return node_id

    def add_output(self, mod: dict, i: int) -> str:
        node_id = f"n{len(mod['nodes']) + 1}"
        mod["nodes"][node_id] = {"id": node_id, "kind": "output", "type": OUTPUT, "name": "output" if i == 0 else f"output_{i}", "params": {}}
        mod["interface"]["outputs"].append(node_id)
        return node_id

    def child_node(self, child, name: str) -> dict:
        """Node content (kind, type, params) for a child module, without an id."""
        from torch import nn

        spec = self.catalog.layers.get(type(child))
        if spec is not None:
            try:
                return {"kind": "layer", "type": spec["type"], "params": {k: {"literal": v} for k, v in _layer_params(spec, child).items()}}
            except ImportFailure as e:
                raise TraceFailed(str(e)) from e
        if isinstance(child, nn.Sequential) and type(child).forward is nn.Sequential.forward:
            return {"kind": "module", "type": self.sequential(child, name), "params": {}}
        module_id, params = self.resolve(child)
        node = {"kind": "module", "type": module_id, "params": params}
        shapes = self.declared_shapes(child, module_id)
        if shapes:
            node["shapes"] = shapes
        return node

    def declared_shapes(self, child, module_id: str) -> dict:
        """Output shapes of opaque and external modules, as observed in the two runs."""
        mod = self.modules[module_id]
        if "opaque" not in mod:
            return {}
        first = self.obs.outputs[0].get(id(child), [])
        second = self.obs.outputs[1].get(id(child), [])
        shapes = {}
        for port, a, b in zip(mod["interface"]["outputs"], first, second):
            dims = [_dim_from_runs(x, y, self.obs.values[0], self.obs.values[1]) for x, y in zip(a[1], b[1])]
            if len(a[1]) == len(b[1]) and None not in dims:
                shapes[port] = ", ".join(str(d) for d in dims) + ("" if a[2] == "float32" else f":{a[2]}")
        return shapes

    def sequential(self, container, base: str) -> str:
        """An nn.Sequential as an inline container: its children in a chain, numbered (0, 1, …) or named (conv0, norm0)."""
        if next(container.children(), None) is None:
            raise TraceFailed(t("空的 nn.Sequential", "an empty nn.Sequential"))
        mod = self.empty(f"{base}_seq", f"{base}_seq")
        mod["sequential"] = True
        prev, port = self.add_input(mod, "x"), "out"
        for name, child in container.named_children():
            content = self.child_node(child, f"{base}_{name}")
            node_id = f"n{len(mod['nodes']) + 1}"
            mod["nodes"][node_id] = {"id": node_id, "name": name, **content}
            first_in, first_out = self.first_ports(content)
            mod["edges"].append({"id": f"e{len(mod['edges']) + 1}", "source": {"node": prev, "port": port}, "target": {"node": node_id, "port": first_in}})
            prev, port = node_id, first_out
        out = self.add_output(mod, 0)
        mod["edges"].append({"id": f"e{len(mod['edges']) + 1}", "source": {"node": prev, "port": port}, "target": {"node": out, "port": "in"}})
        return self.register(mod, "L2")

    def first_ports(self, content: dict) -> tuple[str, str]:
        """The single input and output port of a node in a chain."""
        if content["kind"] == "layer":
            spec = next(s for s in self.catalog.layers.values() if s["type"] == content["type"])
            outputs = _outputs(spec, {k: v["literal"] for k, v in content["params"].items()})
            if len(outputs) != 1:
                raise TraceFailed(t("链中的层有多个输出", "a layer in the chain has several outputs"))
            return spec["inputs"][0]["name"], outputs[0]["name"]
        mod = self.modules[content["type"]]
        if len(mod["interface"]["inputs"]) != 1 or len(mod["interface"]["outputs"]) != 1:
            raise TraceFailed(t("链中的模块不是单输入单输出", "a module in the chain does not have one input and one output"))
        return mod["interface"]["inputs"][0], mod["interface"]["outputs"][0]

    def pure(self, module_id: str) -> bool:
        """A static definition that needs nothing from the run: completely structured, all the way down."""
        if module_id not in self.pure_memo:
            mod = self.static.get(module_id)
            if mod is None:
                result = False
            elif mod.get("opaque") is not None:
                result = "source" not in mod["opaque"]  # a third-party call is kept as written
            elif not mod.get("sequential") and self.levels.get(mod["className"], {}).get("level") != "L1":
                result = False
            else:
                self.pure_memo[module_id] = False  # a module inside itself is not pure
                result = all(self.pure(n["type"]) for n in mod["nodes"].values() if n["kind"] == "module")
            self.pure_memo[module_id] = result
        return self.pure_memo[module_id]

    def static_subtree(self, module_id: str) -> str:
        """Registers a pure static definition and the ones it uses, as they are."""
        mod = copy.deepcopy(self.static[module_id])
        for node in mod["nodes"].values():
            if node["kind"] == "module":
                node["type"] = self.static_subtree(node["type"])
        static = self.levels.get(mod["className"], {})
        external = mod.get("opaque") is not None and "source" not in mod["opaque"]
        return self.register(mod, "L3" if external else static.get("level", "L1"), static.get("notes"), origin=module_id)

    def retarget(self, static: dict, m) -> dict:
        """A static definition whose module nodes point at the definitions this instance's children resolve to."""
        from torch import nn

        mod = copy.deepcopy(static)
        for node in mod["nodes"].values():
            if node["kind"] != "module":
                continue
            target = self.static.get(node["type"], {})
            child = getattr(m, node["name"], None)
            if not isinstance(child, nn.Module):
                raise Fallback(t(f"找不到子模块 {node['name']}", f"submodule {node['name']} was not found"))
            if self.pure(node["type"]):
                # Fully structured statically (or a third-party call as written): keep it, expressions and all.
                node["type"] = self.static_subtree(node["type"])
                continue
            if node.get("repeat"):
                contents = [self.child_node(c, node["name"]) for c in child]
                kinds = {(c["kind"], c["type"]) for c in contents}
                if len(kinds) != 1:
                    raise Fallback(t(f"重复块 {node['name']} 的各份不同", f"the copies of repeat {node['name']} differ"))
                rest = contents[-1]["params"]
                first = {k: v for k, v in contents[0]["params"].items() if rest.get(k) != v}
                node["type"] = contents[0]["type"]
                node["params"] = rest
                node["repeat"] = {"count": {"literal": len(contents)}, **({"first": first} if first else {})}
                continue
            content = self.child_node(child, node["name"])
            if self.origin.get(content["type"]) == node["type"]:
                # The child's definition keeps this class's hyperparameters, so the expressions still apply.
                node["type"] = content["type"]
                continue
            node["type"] = content["type"]
            node["params"] = content["params"]
            if content.get("shapes"):
                node["shapes"] = content["shapes"]
        return mod

    # .................................................... tracing

    def concrete_args(self, m) -> dict:
        """Forward arguments that are not tensors (None masks, flags), and omitted ones with defaults, specialized away."""
        calls = self.obs.calls.get(id(m))
        concrete = {}
        for name, p in inspect.signature(m.forward).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TraceFailed(t("forward 使用了 *args 或 **kwargs", "forward() takes *args or **kwargs"))
            if calls is None:
                # Never called in the run (a training-only head): required arguments are tensors.
                if p.default is not inspect.Parameter.empty:
                    concrete[name] = p.default
            elif name in calls and calls[name] != "tensor":
                concrete[name] = calls[name]
            elif name not in calls:
                if p.default is inspect.Parameter.empty:
                    raise TraceFailed(t(f"forward 的参数 {name} 没有被传入", f"forward() argument {name} was never passed"))
                concrete[name] = p.default
        return concrete

    def trace(self, m, class_name: str) -> dict:
        from torch import fx, nn

        concrete = self.concrete_args(m)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                graph = _leaf_tracer().trace(m, concrete_args=concrete or None)
        except Exception as e:  # fx raises many types; all mean "cannot trace"
            raise TraceFailed(f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}") from e

        mod = self.empty(class_name, class_name)
        values: dict[Any, tuple[str, str]] = {}  # fx node -> (node id, port)
        tuples: dict[Any, tuple[str, list[int], list[dict]]] = {}
        used_names: dict[str, int] = {}
        groups = self.module_list_groups(m, graph)
        skip: set = set()

        def add(content: dict, name: str) -> str:
            node_id = f"n{len(mod['nodes']) + 1}"
            mod["nodes"][node_id] = {"id": node_id, "name": name, **content}
            return node_id

        def link(producer, node_id: str, port: str) -> None:
            if producer not in values:
                raise TraceFailed(t("把整个元组结果作为输入", "a whole tuple result used as an input"))
            source = values[producer]
            mod["edges"].append({"id": f"e{len(mod['edges']) + 1}", "source": {"node": source[0], "port": source[1]}, "target": {"node": node_id, "port": port}})

        def produces(fx_node, node_id: str, outputs: list[dict]) -> None:
            if len(outputs) == 1 and not outputs[0].get("path"):
                values[fx_node] = (node_id, outputs[0]["name"])
            else:
                tuples[fx_node] = (node_id, [], outputs)

        def unique(name: str) -> str:
            taken = {n["name"] for n in mod["nodes"].values()}
            if name not in taken:
                return name
            i = used_names.get(name, 1)
            while f"{name}_{i}" in taken:
                i += 1
            used_names[name] = i + 1
            return f"{name}_{i}"

        for node in graph.nodes:
            if node in skip:
                continue
            if node.op == "placeholder":
                name = str(node.target).lstrip("*")
                if name in concrete:
                    continue
                values[node] = (self.add_input(mod, name), "out")
            elif node.op == "get_attr":
                # The module's own nn.Parameter or buffer (ConvNeXt's layer_scale): a Parameter / Buffer node.
                target = str(node.target)
                held = self.state_node(m, target)
                if held is None:
                    raise TraceFailed(t(f"forward 直接使用了张量 self.{target}", f"forward() uses the tensor self.{target} directly"))
                existing = next((k for k, n in mod["nodes"].items() if n["name"] == target), None)
                values[node] = (existing or add(held, target), "out")
            elif node.op == "call_module":
                target = str(node.target)
                if target in groups:
                    group = groups[target]
                    node_id, outputs = self.repeat_node(mod, group, add, unique)
                    for i, port in enumerate(group["ports"]):
                        arg = group["args"][i]
                        if isinstance(arg, fx.Node):
                            link(arg, node_id, port)
                    produces(group["last"], node_id, outputs)
                    skip.update(group["skip"])
                    continue
                if "." in target:
                    raise TraceFailed(t(f"调用了更深一层的子模块 self.{target}", f"calls the nested submodule self.{target}"))
                child = m.get_submodule(target)
                content = self.traced_child(child, f"{class_name}_{target}")
                # Without parameters or buffers, a second call is the same as a second copy (ReLU, StochasticDepth).
                stateless = (content["kind"] == "layer" and self.layer_spec(content["type"]).get("stateless")) or (
                    next(child.parameters(), None) is None and next(child.buffers(), None) is None
                )
                if any(n["name"] == target for n in mod["nodes"].values()) and not stateless:
                    raise TraceFailed(t(f"self.{target} 被调用了多次（共享权重）", f"self.{target} is called more than once (shared weights)"))
                node_id = add(content, unique(target))
                ports, outputs = self.call_ports(content, child)
                bound = self.bind_call(child, node)
                defaults = {k: p.default for k, p in inspect.signature(child.forward).parameters.items()}
                for param, value in bound.items():
                    if isinstance(value, fx.Node):
                        if param not in ports:
                            raise TraceFailed(t(f"self.{target} 的参数 {param}", f"argument {param} of self.{target}"))
                        link(value, node_id, ports[param])
                    elif content["kind"] == "layer" and value is not defaults.get(param) and value != defaults.get(param):
                        # A layer node passes only tensors; a constant like is_causal=True would be lost.
                        raise TraceFailed(t(f"self.{target} 的参数 {param}={value!r}", f"argument {param}={value!r} of self.{target}"))
                produces(node, node_id, outputs)
            elif node.op == "call_function" and node.target is operator.getitem:
                source, index = node.args
                if source not in tuples or not isinstance(index, int):
                    raise TraceFailed(t("张量切片或索引（如 x[:, 0]）", "tensor slicing or indexing (such as x[:, 0])"))
                node_id, path, outputs = tuples[source]
                path = [*path, index]
                match = [o for o in outputs if o.get("path") == path]
                if match:
                    values[node] = (node_id, match[0]["name"])
                elif any(o.get("path", [])[: len(path)] == path for o in outputs):
                    tuples[node] = (node_id, path, outputs)
                else:
                    raise TraceFailed(t(f"元组下标 {index} 超出范围", f"tuple index {index} is out of range"))
            elif node.op == "call_function" and getattr(node.target, "__name__", "") in ("_assert_is_none", "_assert"):
                continue  # fx's checks for specialized arguments
            elif node.op in ("call_function", "call_method"):
                spec = self.catalog.methods.get(node.target) if node.op == "call_method" else self.catalog.functions.get(node.target)
                if spec is None:
                    raise TraceFailed(_describe(node))
                try:
                    params, edges = _op_params(spec, node)
                except ImportFailure as e:
                    raise TraceFailed(str(e)) from e
                node_id = add({"kind": "op", "type": spec["type"], "params": {k: {"literal": v} for k, v in params.items()}}, unique(spec["defaultName"]))
                for port, producer in edges:
                    link(producer, node_id, port)
                produces(node, node_id, _outputs(spec, params))
            elif node.op == "output":
                results = node.args[0]
                results = list(results) if isinstance(results, (tuple, list)) else [results]
                for i, value in enumerate(results):
                    if not isinstance(value, fx.Node):
                        raise TraceFailed(t(f"返回值中的非张量 {value!r}", f"a non-tensor {value!r} in the return value"))
                    out = self.add_output(mod, i)
                    link(value, out, "in")
            else:
                raise TraceFailed(f"{node.op} {getattr(node.target, '__name__', node.target)}")

        # Submodules forward() did not call in this run (GoogLeNet's aux heads only run in training)
        # are still built, so the model keeps their weights.
        present = {n["name"] for n in mod["nodes"].values()} | {g["container"] for g in groups.values()}
        for name, child in m.named_children():
            if name not in present:
                add(self.traced_child(child, f"{class_name}_{name}"), name)

        if m is self.root:
            self.root_graph = (graph, mod, values)
        return mod

    @staticmethod
    def state_node(m, name: str) -> dict | None:
        """Node content for a Parameter or Buffer the module itself holds, or None (a child's weight, an fx constant)."""
        state = None if "." in name else _state_node(m, name)
        return state and {"kind": "layer", "type": state[0], "params": {k: {"literal": v} for k, v in state[1].items()}}

    def traced_child(self, child, name: str) -> dict:
        """A child while tracing its parent: if the child cannot be expressed, the parent falls back instead of the import."""
        try:
            return self.child_node(child, name)
        except Fallback as e:
            raise TraceFailed(str(e)) from e

    def layer_spec(self, type_: str) -> dict:
        return next(s for s in self.catalog.layers.values() if s["type"] == type_)

    def call_ports(self, content: dict, child) -> tuple[dict[str, str], list[dict]]:
        """Forward parameter -> input port, and the output ports, of a node."""
        if content["kind"] == "layer":
            spec = self.layer_spec(content["type"])
            names = list(inspect.signature(child.forward).parameters)
            ports = {name: p["name"] for name, p in zip(names, spec["inputs"])}
            return ports, _outputs(spec, {k: v["literal"] for k, v in content["params"].items()})
        mod = self.modules[content["type"]]
        if mod.get("sequential") and child is not None:
            # nn.Sequential.forward(input): its one argument is the container's input.
            ports = {next(iter(inspect.signature(child.forward).parameters)): mod["interface"]["inputs"][0]}
        else:
            ports = {mod["nodes"][i]["name"]: i for i in mod["interface"]["inputs"]}
        outs = mod["interface"]["outputs"]
        return ports, [{"name": o} for o in outs] if len(outs) == 1 else [{"name": o, "path": [i]} for i, o in enumerate(outs)]

    def bind_call(self, child, node) -> dict:
        try:
            return dict(inspect.signature(child.forward).bind(*node.args, **node.kwargs).arguments)
        except (TypeError, ValueError) as e:
            raise TraceFailed(t(f"无法匹配 self.{node.target} 的参数", f"cannot match the arguments of self.{node.target}")) from e

    def module_list_groups(self, m, graph) -> dict:
        """`for blk in self.blocks: x = blk(x, ...)`, unrolled by fx into blocks.0, blocks.1, …: one repeat each."""
        from torch import fx, nn

        calls: dict[str, list] = {}
        for node in graph.nodes:
            if node.op == "call_module" and str(node.target).count(".") == 1:
                container, index = str(node.target).split(".")
                if index.isdigit() and isinstance(getattr(m, container, None), nn.ModuleList):
                    calls.setdefault(container, []).append((int(index), node))
        groups = {}
        for container, items in calls.items():
            modules = list(getattr(m, container))
            if [i for i, _ in items] != list(range(len(modules))):
                raise TraceFailed(t(f"self.{container} 不是按顺序串联调用的", f"self.{container} is not called in sequence"))
            nodes = [n for _, n in items]
            for prev, cur in zip(nodes, nodes[1:]):
                if not cur.args or cur.args[0] is not prev or set(prev.users) != {cur} or cur.args[1:] != nodes[0].args[1:] or cur.kwargs != nodes[0].kwargs:
                    raise TraceFailed(t(f"self.{container} 不是按顺序串联调用的", f"self.{container} is not called in sequence"))
            groups[str(nodes[0].target)] = {
                "container": container,
                "modules": modules,
                "args": list(nodes[0].args) + list(nodes[0].kwargs.values()),
                "kwnames": list(nodes[0].kwargs),
                "last": nodes[-1],
                "skip": set(nodes[1:]),
            }
        return groups

    def repeat_node(self, mod: dict, group: dict, add, unique) -> tuple[str, list[dict]]:
        """A ModuleList chain as a repeat (all copies alike, the first may differ in parameters), or as a Sequential."""
        contents = [self.traced_child(c, group["container"]) for c in group["modules"]]
        name = unique(group["container"])
        same = len({(c["kind"], c["type"]) for c in contents}) == 1
        if same:
            rest = contents[-1]["params"]
            first = {k: v for k, v in contents[0]["params"].items() if rest.get(k) != v}
            node_id = add({**contents[0], "params": rest, "repeat": {"count": {"literal": len(contents)}, **({"first": first} if first else {})}}, name)
            ports, outputs = self.call_ports(contents[0], group["modules"][0])
        else:
            if len(group["args"]) != 1:
                raise TraceFailed(t(f"self.{group['container']} 的各份不同且有多个输入", f"the copies of self.{group['container']} differ and take several inputs"))
            module_id = self.sequential_of(contents, group["container"])
            node_id = add({"kind": "module", "type": module_id, "params": {}}, name)
            _, outputs = self.call_ports({"kind": "module", "type": module_id}, None)
            group["ports"] = [self.modules[module_id]["interface"]["inputs"][0]]  # its one argument
            return node_id, outputs
        params = list(inspect.signature(group["modules"][0].forward).parameters)
        positional = len(group["args"]) - len(group["kwnames"])
        port_list = [ports.get(p, "") for p in params[:positional]] + [ports.get(k, "") for k in group["kwnames"]]
        if "" in port_list:
            raise TraceFailed(t(f"无法匹配 self.{group['container']} 的参数", f"cannot match the arguments of self.{group['container']}"))
        group["ports"] = port_list
        return node_id, outputs

    def sequential_of(self, contents: list[dict], base: str) -> str:
        mod = self.empty(f"{base}_seq", f"{base}_seq")
        mod["sequential"] = True
        prev, port = self.add_input(mod, "x"), "out"
        for i, content in enumerate(contents):
            node_id = f"n{len(mod['nodes']) + 1}"
            mod["nodes"][node_id] = {"id": node_id, "name": str(i), **content}
            first_in, first_out = self.first_ports(content)
            mod["edges"].append({"id": f"e{len(mod['edges']) + 1}", "source": {"node": prev, "port": port}, "target": {"node": node_id, "port": first_in}})
            prev, port = node_id, first_out
        out = self.add_output(mod, 0)
        mod["edges"].append({"id": f"e{len(mod['edges']) + 1}", "source": {"node": prev, "port": port}, "target": {"node": out, "port": "in"}})
        return self.register(mod, "L2")


def _verbatim_text(builder: _Builder, preamble: str) -> str:
    """All Python the project keeps as written: it may name the user's classes."""
    parts = [preamble]
    for mod in builder.modules.values():
        source = (mod.get("opaque") or {}).get("source", "")
        parts += [mod.get("initCode", ""), mod.get("members", ""), re.sub(r"^class\s+\w+", "", source, flags=re.M)]
        parts += [n["code"]["source"] for n in mod["nodes"].values() if n["kind"] == "code"]
    return "\n".join(parts)


def _define_referenced_classes(builder: _Builder, static_project: dict) -> None:
    """
    Code kept as written calls the user's classes by their original names and
    signatures (`DoubleConv(in_ch, out_ch)` in an __init__ tail). Those classes
    are defined as the static importer read them; traced variants of the same
    class take other names.
    """
    user_classes = [k for k, m in static_project["modules"].items() if k == m["className"] and not m.get("sequential")]
    done: set[str] = set()
    while True:
        text = _verbatim_text(builder, static_project.get("preamble", ""))
        missing = [
            c
            for c in user_classes
            if c not in done
            and re.search(rf"\b{re.escape(c)}\b", text)
            and not any(builder.origin.get(k) == c and m["className"] == c for k, m in builder.modules.items())
        ]
        if not missing:
            return
        for name in missing:
            done.add(name)
            module_id = builder.static_subtree(name)
            static_mod = builder.modules[module_id]
            entry = builder.report.pop(static_mod["className"], None) if static_mod["className"] != name else None
            taken = {m["className"] for m in builder.modules.values()}
            for k, mod in builder.modules.items():
                if k != module_id and mod["className"] == name:
                    renamed = next(f"{name}{i}" for i in range(2, 10_000) if f"{name}{i}" not in taken)
                    taken.add(renamed)
                    mod["className"] = renamed
                    if name in builder.report:
                        builder.report[renamed] = {**builder.report.pop(name), "className": renamed}
            static_mod["className"] = name
            builder.origin[module_id] = name
            if entry is not None:
                builder.report[name] = {**entry, "className": name}


# ---------------------------------------------------------------- entry point


def _observed_root(builder: _Builder, model, specs: list[tuple[str, InputSpec]]) -> dict:
    """Shapes of the entry's own values with every symbol = BATCH_PROBE, for the round-trip tests."""
    if builder.root_graph is None:
        return {}
    import torch
    from torch import fx
    from torch.fx.passes.shape_prop import ShapeProp

    graph, mod, values = builder.root_graph
    try:
        gm = fx.GraphModule(model, graph)
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        tensors = []
        for _, spec in specs:
            size = [d if isinstance(d, int) else BATCH_PROBE for d in spec.dims]
            tensors.append(torch.zeros(size, dtype=torch.long, device=device) if spec.dtype == "int64" else torch.empty(size, device=device))
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            ShapeProp(gm).propagate(*tensors)
    except Exception:
        return {}
    observed: dict[str, dict[str, list[int]]] = {}
    for fx_node, (node_id, port) in values.items():
        meta = fx_node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            observed.setdefault(node_id, {})[port] = list(meta.shape)
    return observed


def import_dynamic(
    source: str,
    class_name: str | None = None,
    inputs: dict[str, InputSpec] | None = None,
    filename: str = "<model.py>",
) -> dict[str, Any]:
    import torch

    static = import_static(source, class_name, inputs, filename)
    static_project = static["project"]
    entry = static_project["entry"]
    entry_static = static_project["modules"][entry]

    # Input shapes: as given, otherwise as the static importer guessed them.
    specs: list[tuple[str, InputSpec]] = []
    for node_id in entry_static["interface"]["inputs"]:
        node = entry_static["nodes"][node_id]
        given = (inputs or {}).get(node["name"])
        if given is None:
            text = node["params"]["shape"]["literal"]
            dims = [int(d) if d.strip().isdigit() else d.strip() for d in text.split(",") if d.strip()]
            given = InputSpec(dims, node["params"]["dtype"]["literal"])
        specs.append((node["name"], given))

    records: dict[int, tuple[tuple, dict]] = {}
    # A real module, so dataclasses and pickling can find it in sys.modules.
    user_module = types.ModuleType(USER_MODULE)
    namespace = user_module.__dict__

    def instantiate(device: str):
        with _record_constructor_args(records), contextlib.redirect_stdout(sys.stderr), torch.device(device):
            return namespace[entry]()

    sys.modules[USER_MODULE] = user_module
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source, filename, "exec"), namespace)

    # Forward arguments with defaults (padding_mask=None) are left out unless the user gave their shape.
    forward = inspect.signature(namespace[entry].forward).parameters
    specs = [(name, spec) for name, spec in specs if name in (inputs or {}) or forward.get(name) is None or forward[name].default is inspect.Parameter.empty]
    try:
        model = instantiate("meta")
    except Exception:
        # Some constructors compute with tensors (RegNet's widths): those stay on the CPU,
        # while weights still go to the meta device; failing that, everything on the CPU.
        records.clear()
        try:
            with _weights_on_meta():
                model = instantiate("cpu")
        except Exception:
            records.clear()
            try:
                model = instantiate("cpu")
            except Exception as e:
                raise ImportFailure(t(f"无法用默认参数实例化 {entry}：{e}", f"cannot instantiate {entry} with default arguments: {e}")) from e
    model.eval()
    try:
        obs = _observe(model, specs)
    except Exception:
        # Some ops have no meta implementation: run on the CPU with real weights instead.
        records.clear()
        try:
            model = instantiate("cpu")
            model.eval()
            obs = _observe(model, specs)
        except Exception as e:
            raise ImportFailure(t(f"按声明的输入形状运行模型失败：{e}", f"running the model with the declared input shapes failed: {e}")) from e

    builder = _Builder(static_project, static["report"], records, obs)
    builder.root = model
    try:
        root_id, _ = builder.resolve(model)
    except Fallback as e:
        static["warnings"].insert(0, t(f"动态导入未能完成（{e}），以下是静态解析的结果", f"the dynamic import could not be completed ({e}); this is the static result"))
        return static

    # Definitions registered for a subtree that later fell back are not used: keep what the entry reaches.
    reachable, stack = set(), [root_id]
    while stack:
        module_id = stack.pop()
        if module_id in reachable:
            continue
        reachable.add(module_id)
        stack.extend(n["type"] for n in builder.modules[module_id]["nodes"].values() if n["kind"] == "module")
    builder.modules = {k: v for k, v in builder.modules.items() if k in reachable}
    _define_referenced_classes(builder, static_project)
    kept_classes = {m["className"] for m in builder.modules.values()}
    builder.report = {k: v for k, v in builder.report.items() if k in kept_classes}

    root = builder.modules[root_id]
    for name, spec in specs:
        node = next((n for n in root["nodes"].values() if n["kind"] == "input" and n["name"] == name), None)
        if node is not None:
            node["params"]["shape"] = {"literal": ", ".join(str(d) for d in spec.dims)}
            node["params"]["dtype"] = {"literal": spec.dtype}

    # The entry first in the report and last in the file, as in a static import.
    project: dict[str, Any] = {"schemaVersion": SCHEMA_VERSION, "name": entry, "entry": root_id, "modules": builder.modules}
    # Classes that external definitions are written as (from torchvision.ops.stochastic_depth import StochasticDepth).
    preamble = static_project.get("preamble", "")
    imports = sorted({builder.import_of[k][0] for k in builder.modules if k in builder.import_of})
    missing = [line for line in imports if not re.search(rf"^{re.escape(line)}$", preamble, re.M)]
    if missing:
        preamble = "\n".join([*missing, preamble] if preamble else missing)
    if preamble:
        project["preamble"] = preamble
    observed = _observed_root(builder, model, specs)
    report = list(builder.report.values())
    return {
        "project": project,
        "report": report,
        "warnings": [w for w in static["warnings"] if "来自其他库" not in w and "comes from another library" not in w] + builder.warnings,
        "observedShapes": observed,
        "batchProbe": BATCH_PROBE,
    }
