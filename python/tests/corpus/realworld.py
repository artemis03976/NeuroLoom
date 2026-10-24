"""The third-party import corpus (PRD 13): model files from common repositories.

Builds each model the way its repository does and checks code NeuroLoom generated from
it against the original: same state_dict keys and shapes, the original weights load with
strict=True, and the same input gives the same output. manifest.json says how to build
each model; the files are in third_party/<id>/ at their repository paths, unmodified.

    python tests/corpus/realworld.py originals [id ...]    build and run the originals
    python tests/corpus/realworld.py check                 compare generated code: reads
                                                           [{"id", "generated"}] on stdin

Runs offline: the network is closed, and pretrained checkpoints a constructor asks for
(torch.hub / torchvision downloads) come back empty, so models keep random weights.
Library files (timm, transformers, diffusers) run from the installed package, whose
version the manifest pins.
"""
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib
import json
import os
import socket
import sys
import time
import traceback
import types
import warnings
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
sys.dont_write_bytecode = True  # third_party/ stays as downloaded
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.hub  # noqa: E402
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE / "third_party"
MANIFEST = {e["id"]: e for e in json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))}
GENERATED = "neuroloom_generated"


class NoWeights(dict):
    """A pretrained checkpoint that was not downloaded: loading it keeps the random weights."""


def _no_weights(*args, **kwargs):
    return NoWeights()


def _closed(*args, **kwargs):
    raise OSError("the corpus test runs offline")


@contextlib.contextmanager
def offline():
    """No network; checkpoint downloads, wherever a module holds the function, return NoWeights."""
    download = torch.hub.load_state_dict_from_url
    patched = [(socket.socket, "connect", socket.socket.connect), (torch.hub, "download_url_to_file", torch.hub.download_url_to_file)]
    for module in list(sys.modules.values()):
        for name in ("load_state_dict_from_url", "load_url"):
            if getattr(module, name, None) is download:
                patched.append((module, name, download))
    load = nn.Module.load_state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        if isinstance(state_dict, NoWeights):
            return nn.modules.module._IncompatibleKeys([], [])
        return load(self, state_dict, *args, **kwargs)

    socket.socket.connect = _closed
    torch.hub.download_url_to_file = _closed
    for module, name, _ in patched[2:]:
        setattr(module, name, _no_weights)
    nn.Module.load_state_dict = load_state_dict
    try:
        yield
    finally:
        nn.Module.load_state_dict = load
        for owner, name, value in patched:
            setattr(owner, name, value)


def definitions(source: str) -> str:
    """A script's imports, functions and classes, without what it runs (downloads, training).

    An assignment is kept when it calls nothing and reads only names bound above it
    (`hidden_size = 500`, not `x = test_data[0]`).
    """
    tree = ast.parse(source)
    bound, star, keep = set(dir(builtins)), False, []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            keep.append(stmt)
            star = star or any(a.name == "*" for a in stmt.names)
            bound |= {(a.asname or a.name).split(".")[0] for a in stmt.names}
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(stmt)
            bound.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            reads = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            if not any(isinstance(n, ast.Call) for n in ast.walk(stmt.value)) and (star or reads <= bound):
                keep.append(stmt)
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                bound |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


@contextlib.contextmanager
def repository(entry: dict):
    """The entry's repository on sys.path and its stub modules; its modules are dropped afterwards."""
    folder = THIRD_PARTY / entry["id"]
    paths = [] if "library" in entry else [str(folder / entry["root"]), str(folder)]
    # Repositories reuse top-level names (models, utils): nothing of another one may linger.
    if "library" not in entry:
        tops = {Path(p).relative_to(entry["root"]).parts[0].removesuffix(".py") for p in entry["files"] if Path(p).is_relative_to(entry["root"])}
        tops |= {Path(p).parts[0].removesuffix(".py") for p in entry["files"]}
        for name in [n for n in sys.modules if n.split(".")[0] in tops]:
            del sys.modules[name]
        importlib.invalidate_caches()
    before, saved_path = set(sys.modules), list(sys.path)
    sys.path[:0] = paths
    for name, source in entry.get("stubs", {}).items():
        stub = types.ModuleType(name)
        exec(source, stub.__dict__)
        sys.modules[name] = stub
    try:
        with offline():
            yield
    finally:
        sys.path[:] = saved_path  # some files append to sys.path themselves (Swin's kernels)
        inside = str(folder)

        def ours(name: str) -> bool:
            module = sys.modules[name]
            try:
                # A namespace package recomputes __path__ from its parent: read it before deleting anything.
                where = [getattr(module, "__file__", None) or "", *list(getattr(module, "__path__", None) or [])]
            except Exception:  # noqa: BLE001
                where = [inside]
            return name.endswith(GENERATED) or name in entry.get("stubs", {}) or any(str(w).startswith(inside) for w in where)

        for name in [n for n in set(sys.modules) - before if ours(n)]:
            del sys.modules[name]


def module_of(entry: dict, generated: str | None = None) -> types.ModuleType:
    """The original module, imported as its repository imports it, or generated code beside it."""
    name, package = entry["module"], entry["package"]
    if generated is None and not entry.get("script"):
        module = importlib.import_module(name)
        if "library" in entry:
            # The installed library must be the pinned one: the import reads the copy in third_party/.
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if digest != entry["sha256"][entry["path"]]:
                library = entry["library"]
                raise RuntimeError(f"installed {library['name']} is not {library['version']} (pip install -r requirements-dev.txt)")
        return module
    if package:
        importlib.import_module(package)
    path = THIRD_PARTY / entry["id"] / entry["path"]
    source = generated if generated is not None else path.read_text(encoding="utf-8")
    if entry.get("script"):
        source = definitions(source)
    if generated is not None:
        name = f"{package}.{GENERATED}" if package else GENERATED
    module = types.ModuleType(name)
    module.__file__ = str(path.with_name(f"{GENERATED}.py") if generated is not None else path)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def build(entry: dict, module: types.ModuleType) -> nn.Module:
    if entry.get("setup"):
        exec(entry["setup"], module.__dict__)
    torch.manual_seed(0)
    model = eval(entry["construct"], module.__dict__)
    return model.eval()


def example(entry: dict, module: types.ModuleType, model: nn.Module):
    torch.manual_seed(1)
    with torch.no_grad():
        args = eval(entry["example"], {**module.__dict__, "torch": torch, "model": model})
    return args if isinstance(args, (tuple, dict)) else (args,)


def run(model: nn.Module, args):
    args = copy.deepcopy(args)
    torch.manual_seed(2)
    with torch.no_grad():
        return model(**args) if isinstance(args, dict) else model(*args)


def tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return []


def same(a, b) -> str | None:
    ta, tb = tensors(a), tensors(b)
    if len(ta) != len(tb):
        return f"{len(ta)} output tensors, generated code gives {len(tb)}"
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x.shape != y.shape:
            return f"output {i}: shape {tuple(x.shape)}, generated code gives {tuple(y.shape)}"
        if x.is_floating_point() or x.is_complex():
            if not torch.allclose(x, y, atol=1e-5, rtol=1e-4, equal_nan=True):
                return f"output {i}: differs by up to {float((x - y).abs().max()):.3g}"
        elif not torch.equal(x, y):
            return f"output {i}: values differ"
    return None


def short(e: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(e.__traceback__) if "realworld.py" not in f.filename][-2:]
    where = "; ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames)
    return f"{type(e).__name__}: {str(e)[:300]}" + (f" ({where})" if where else "")


def original(entry_id: str) -> dict:
    entry = MANIFEST[entry_id]
    start = time.perf_counter()
    try:
        with repository(entry):
            module = module_of(entry)
            model = build(entry, module)
            out = run(model, example(entry, module, model))
        shapes = [list(t.shape) for t in tensors(out)]
        params = sum(p.numel() for p in model.parameters())
        return {"id": entry_id, "ok": True, "outputs": shapes, "params": params, "seconds": round(time.perf_counter() - start, 2)}
    except BaseException as e:  # noqa: BLE001  a corpus file may raise anything, SystemExit included
        return {"id": entry_id, "ok": False, "error": short(e)}


def check(entry_id: str, generated: str) -> dict:
    """{"ok"} or the first stage that fails: original (a manifest problem), load, build, state_dict, run, outputs."""
    entry = MANIFEST[entry_id]
    stage = "original"
    try:
        with repository(entry):
            module = module_of(entry)
            a = build(entry, module)
            args = example(entry, module, a)
            ya = run(a, args)
            stage = "load"
            module_b = module_of(entry, generated)
            stage = "build"
            b = build(entry, module_b).eval()
            stage = "state_dict"
            ka = {k: tuple(v.shape) for k, v in a.state_dict().items()}
            kb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
            if ka != kb:
                only_a = sorted(set(ka) - set(kb))[:4]
                only_b = sorted(set(kb) - set(ka))[:4]
                shapes = sorted(k for k in set(ka) & set(kb) if ka[k] != kb[k])[:4]
                return {"id": entry_id, "ok": False, "stage": stage, "error": f"only in original {only_a}; only in generated {only_b}; shapes differ {shapes}"}
            b.load_state_dict(a.state_dict(), strict=True)
            stage = "run"
            yb = run(b, args)
            stage = "outputs"
            difference = same(ya, yb)
            if difference:
                return {"id": entry_id, "ok": False, "stage": stage, "error": difference}
        return {"id": entry_id, "ok": True}
    except BaseException as e:  # noqa: BLE001
        return {"id": entry_id, "ok": False, "stage": stage, "error": short(e)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    command = sys.argv[1]
    if command == "originals":
        ids = sys.argv[2:] or list(MANIFEST)
        for entry_id in ids:
            with contextlib.redirect_stdout(sys.stderr):
                result = original(entry_id)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    elif command == "check":
        requests = json.load(sys.stdin)
        results = []
        for r in requests:
            print(f"check {r['id']}", file=sys.stderr, flush=True)
            with contextlib.redirect_stdout(sys.stderr):
                results.append(check(r["id"], r["generated"]))
        print(json.dumps(results, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
