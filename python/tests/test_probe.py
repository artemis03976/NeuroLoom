"""Running instrumented generated code to observe shapes inference cannot see."""

import pytest

from neuroloom.importer import ImportFailure
from neuroloom.probe import probe
from neuroloom.server import handle

CODE = """
import torch
from torch import nn


_NL_SHAPES = {}


def _nl_probe(module, node, port, value):
    if isinstance(value, torch.Tensor):
        _NL_SHAPES.setdefault(f"{module}/{node}/{port}", [list(value.shape), str(value.dtype).replace("torch.", "")])
    return value


class M(nn.Module):
    def forward(self, x):
        y = x.reshape(-1, 4)
        _nl_probe("M", "n2", "y", y)
        return y
"""


def test_runs_once_per_symbol_assignment():
    result = probe(CODE, "M", ["B,T,4"])
    assert result["values"] == [{"B": 2, "T": 3}, {"B": 3, "T": 5}]
    assert result["runs"][0]["M/n2/y"] == [[6, 4], "float32"]
    assert result["runs"][1]["M/n2/y"] == [[15, 4], "float32"]


def test_reports_model_errors():
    with pytest.raises(ImportFailure, match="运行模型失败"):
        probe(CODE.replace("reshape(-1, 4)", "reshape(-1, 5)"), "M", ["B,4"])


def test_rpc_runs_in_a_subprocess():
    ok = handle({"jsonrpc": "2.0", "id": 1, "method": "probe_shapes", "params": {"code": CODE, "className": "M", "inputs": ["B,8"]}})
    assert ok["result"]["runs"][0]["M/n2/y"] == [[4, 4], "float32"]
