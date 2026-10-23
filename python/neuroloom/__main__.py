"""Command line: `python -m neuroloom [--lang zh|en] import|import-batch|probe|serve|env`."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .i18n import set_locale


def _utf8_stdio() -> None:
    # Windows defaults to the ANSI code page; the protocol is UTF-8 everywhere.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _cmd_import(args: argparse.Namespace) -> int:
    from .importer import ImportFailure, import_code, parse_input

    source = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    try:
        inputs = {}
        for item in args.input:
            name, _, spec = item.partition("=")
            inputs[name.strip()] = parse_input(spec)
        result = import_code(source, args.class_name, inputs, args.mode, filename=args.file if args.file != "-" else "<model.py>")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except ImportFailure as e:
        print(json.dumps({"error": {"message": str(e), "details": e.details}}, ensure_ascii=False))
        return 2
    except Exception as e:  # user code can raise anything
        tb = traceback.format_exc(limit=6)
        print(json.dumps({"error": {"message": f"{type(e).__name__}: {e}", "details": {"traceback": tb}}}, ensure_ascii=False))
        return 2


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = argparse.ArgumentParser(prog="neuroloom")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="language of the messages shown to the user (default: zh); goes before the command")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import a PyTorch model file into NeuroLoom IR (prints JSON)")
    p_import.add_argument("file", help="path to a .py file, or - to read stdin")
    p_import.add_argument("--class", dest="class_name", help="model class to import (default: last nn.Module defined)")
    p_import.add_argument(
        "--input", action="append", default=[], metavar="NAME=DIMS[:DTYPE]", help="input shape, e.g. x=B,784 or ids=B,16:int64"
    )
    p_import.add_argument(
        "--mode", choices=["static", "trace"], default="static", help="static: read the source without running it (default); trace: run it under torch.fx"
    )
    sub.add_parser("import-batch", help="import a JSON list of {source, className, inputShapes, mode} from stdin (prints a JSON list)")

    sub.add_parser("probe", help="run instrumented generated code once per symbol assignment; reads {code, className, inputs} JSON from stdin")
    sub.add_parser("serve", help="JSON-RPC 2.0 over stdio, one message per line")
    sub.add_parser("env", help="print Python / torch environment info")

    args = parser.parse_args(argv)
    set_locale(args.lang)
    if args.command == "import":
        return _cmd_import(args)
    if args.command == "import-batch":
        from .importer import import_many

        print(json.dumps(import_many(json.loads(sys.stdin.read())), ensure_ascii=False))
        return 0
    if args.command == "probe":
        from .importer import ImportFailure
        from .probe import probe

        request = json.loads(sys.stdin.read())
        try:
            print(json.dumps(probe(request["code"], request["className"], request["inputs"]), ensure_ascii=False))
        except ImportFailure as e:
            print(json.dumps({"error": {"message": str(e), "details": e.details}}, ensure_ascii=False))
            return 2
        return 0
    if args.command == "serve":
        from .server import serve

        serve()
        return 0
    if args.command == "env":
        from .server import env_info

        print(json.dumps(env_info(), ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
