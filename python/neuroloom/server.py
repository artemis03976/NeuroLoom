"""JSON-RPC 2.0 over stdio, one JSON message per line (PRD section 15.3).

The server never executes user code itself: each import runs in a fresh child
process with a timeout, so a crash, an infinite loop or stray prints in a
model file cannot take the backend down or corrupt the protocol stream.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
IMPORT_TIMEOUT_S = 30


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "backend": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": None,
        "cuda": False,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info


def import_code(source: str, className: str | None = None, inputShapes: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "neuroloom", "import", "-"]
    if className:
        cmd += ["--class", className]
    for name, dims in (inputShapes or {}).items():
        cmd += ["--input", f"{name}={dims}"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            cmd,
            input=source,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=IMPORT_TIMEOUT_S,
            cwd=PACKAGE_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RpcError(-32002, f"导入超时（{IMPORT_TIMEOUT_S} 秒），已终止执行") from e

    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as e:
        raise RpcError(-32603, "导入进程没有返回结果", {"stderr": proc.stderr[-4000:]}) from e
    if "error" in result:
        raise RpcError(-32001, result["error"]["message"], {**result["error"].get("details", {}), "stderr": proc.stderr[-4000:]})
    return result


METHODS = {
    "ping": lambda: {"pong": True, "backend": __version__},
    "env_info": env_info,
    "import_code": import_code,
}


def handle(message: dict[str, Any]) -> dict[str, Any]:
    msg_id = message.get("id")
    try:
        method = METHODS.get(message.get("method", ""))
        if method is None:
            raise RpcError(-32601, f"unknown method {message.get('method')!r}")
        params = message.get("params") or {}
        return {"jsonrpc": "2.0", "id": msg_id, "result": method(**params)}
    except RpcError as e:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": e.code, "message": str(e), "data": e.data}}
    except TypeError as e:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": f"invalid params: {e}"}}


def serve() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {e}"}}
        else:
            response = handle(message)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
