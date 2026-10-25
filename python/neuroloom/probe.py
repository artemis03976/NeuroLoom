"""Runs instrumented generated code to observe shapes inference cannot see.

The editor generates the project's code with `_nl_probe(...)` calls after every
code block and opaque-module call (codegen option `probe`). The model runs
twice, with different values for each symbolic input dimension, so the editor
can tell constant dimensions from symbolic ones (B, B*T) by comparing the runs.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from .i18n import t
from .importer import ImportFailure, example_tensor, parse_input

# Values for the first, second, ... symbolic dimension in each run.
RUNS = ((2, 3, 5, 7, 11, 13), (3, 5, 7, 11, 13, 17))


def probe(code: str, class_name: str, inputs: list[str]) -> dict[str, Any]:
    import torch

    specs = [parse_input(text) for text in inputs]
    symbols: list[str] = []
    for spec in specs:
        for d in spec.dims:
            if isinstance(d, str) and d not in symbols:
                symbols.append(d)
    if len(symbols) > len(RUNS[0]):
        raise ImportFailure(t(f"符号维度太多（最多 {len(RUNS[0])} 个）", f"too many symbolic dimensions (at most {len(RUNS[0])})"))

    runs, values = [], []
    for sequence in RUNS:
        value = {s: sequence[i] for i, s in enumerate(symbols)}
        namespace: dict[str, Any] = {"__name__": "neuroloom_probe"}
        # Model code must not write to stdout: in the CLI that is where the result goes.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                exec(compile(code, "<neuroloom>", "exec"), namespace)
                model = namespace[class_name]()
                model.eval()
                tensors = []
                for spec in specs:
                    size = [d if isinstance(d, int) else value[d] for d in spec.dims]
                    tensors.append(example_tensor(spec, size))
                with torch.no_grad():
                    model(*tensors)
            except ImportFailure:
                raise
            except Exception as e:  # the model may raise anything
                raise ImportFailure(t(f"运行模型失败：{type(e).__name__}: {e}", f"running the model failed: {type(e).__name__}: {e}")) from e
        runs.append(namespace.get("_NL_SHAPES", {}))
        values.append(value)
    return {"runs": runs, "values": values}
