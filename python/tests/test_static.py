"""Static import (L1 + L3): structure without running the code."""

import json
import subprocess
import sys
from pathlib import Path

from neuroloom.importer import InputSpec
from neuroloom.static import import_static

ROOT = Path(__file__).resolve().parent.parent


def modules(result):
    return result["project"]["modules"]


def by_name(module):
    return {n["name"]: n for n in module["nodes"].values()}


def edges_by_name(module):
    names = {n["id"]: n["name"] for n in module["nodes"].values()}
    return sorted((names[e["source"]["node"]], e["source"]["port"], names[e["target"]["node"]], e["target"]["port"]) for e in module["edges"])


RESNET = """
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Net(nn.Module):
    def __init__(self, width: int = 16, depth: int = 3, num_classes: int = 10):
        super().__init__()
        self.stage = nn.ModuleList([Block(3, width, stride=2)] + [Block(width, width) for _ in range(depth - 1)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, num_classes))

    def forward(self, x):
        for block in self.stage:
            x = block(x)
        return self.head(x)
"""


def test_keeps_classes_hyperparameters_repeats_and_sequential():
    result = import_static(RESNET, inputs={"x": InputSpec(["B", 3, 32, 32])})
    mods = modules(result)
    assert result["project"]["entry"] == "Net"
    assert [r["level"] for r in result["report"]] == ["L1", "L1"]

    block = mods["Block"]
    assert [h["name"] for h in block["hyperparams"]] == ["cin", "cout", "stride"]
    assert block["hyperparams"][2]["default"] == {"literal": 1}
    conv = by_name(block)["conv"]["params"]
    assert conv["in_channels"] == {"expr": "cin"} and conv["stride"] == {"expr": "stride"} and conv["bias"] == {"literal": False}

    net = mods["Net"]
    stage = by_name(net)["stage"]
    assert stage["kind"] == "module" and stage["type"] == "Block"
    assert stage["params"] == {"cin": {"expr": "width"}, "cout": {"expr": "width"}}
    assert stage["repeat"] == {"count": {"expr": "depth"}, "first": {"cin": {"literal": 3}, "stride": {"literal": 2}}}
    head = by_name(net)["head"]
    seq = mods[head["type"]]
    assert seq["sequential"] is True
    assert [by_name(seq)[str(i)]["type"] for i in range(3)] == ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Flatten", "torch.nn.Linear"]
    assert by_name(seq)["2"]["params"]["out_features"] == {"expr": "num_classes"}
    assert by_name(net)["x"]["params"]["shape"] == {"literal": "B, 3, 32, 32"}


CODE_BLOCKS = """
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(8 * 6 * 6, 10)

    def forward(self, x):
        x = self.conv(x)
        b = x.size(0)
        x = x.reshape(b, -1)
        y = self.fc(x)
        return y[:, :5]
"""


def test_falls_back_to_code_blocks_statement_by_statement():
    result = import_static(CODE_BLOCKS, inputs={"x": InputSpec(["B", 3, 8, 8])})
    net = modules(result)["Net"]
    nodes = by_name(net)
    codes = [n for n in net["nodes"].values() if n["kind"] == "code"]
    assert len(codes) == 2
    # Consecutive statements become one block; its outputs are what later code reads.
    assert codes[0]["code"] == {"source": "b = x.size(0)\nx = x.reshape(b, -1)", "inputs": ["x"], "outputs": ["x"]}
    # An unsupported return value is computed by a block of its own.
    assert codes[1]["code"] == {"source": "output = y[:, :5]", "inputs": ["y"], "outputs": ["output"]}
    edges = edges_by_name(net)
    assert ("conv", "out", codes[0]["name"], "x") in edges
    assert (codes[0]["name"], "x", "fc", "in") in edges
    assert nodes["fc"]["kind"] == "layer"
    assert result["report"][0]["level"] == "L1+L3" and result["report"][0]["codeBlocks"] == 2


def test_shared_layers_and_unknown_calls_stay_code():
    source = """
from torch import nn
import torch


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        h = self.fc(h)
        return torch.einsum("bi->b", h)
"""
    net = modules(import_static(source, inputs={"x": InputSpec(["B", 4])}))["Tied"]
    codes = [n["code"]["source"] for n in net["nodes"].values() if n["kind"] == "code"]
    assert codes == ["h = self.fc(h)", 'output = torch.einsum("bi->b", h)']


def test_init_code_constants_and_globals():
    source = """
import math
import torch
from torch import nn

HIDDEN = 32


class Net(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.fc = nn.Linear(dim, HIDDEN)
        self.register_buffer("mask", torch.ones(dim))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x * self.scale)
"""
    result = import_static(source, inputs={"x": InputSpec(["B", 8])})
    net = modules(result)["Net"]
    assert net["constants"] == [{"name": "scale", "expr": "dim ** -0.5"}]
    assert net["initCode"] == 'self.register_buffer("mask", torch.ones(dim))\nnn.init.zeros_(self.fc.bias)'
    assert by_name(net)["fc"]["params"]["out_features"] == {"expr": "HIDDEN"}
    mul = next(n for n in net["nodes"].values() if n["type"] == "torch.mul")
    assert mul["params"]["other"] == {"expr": "scale"}
    assert result["project"]["preamble"] == "import math\n\nHIDDEN = 32"


def test_unstructured_classes_and_third_party_layers_are_kept_as_python():
    source = """
from torch import nn
from timm.layers import DropPath


class Gate(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x, y):
        if x.sum() > 0:
            return x
        return y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate(4, 4)
        self.drop = DropPath(0.1, scale_by_keep=True)

    def forward(self, x):
        return self.drop(self.gate(x, x))
"""
    result = import_static(source, inputs={"x": InputSpec(["B", 4])})
    mods = modules(result)
    gate = mods["Gate"]
    assert gate["opaque"]["source"].startswith("class Gate(nn.Module):") and gate["opaque"]["verbatimArgs"] is True
    assert [n["name"] for n in gate["nodes"].values() if n["kind"] == "input"] == ["x", "y"]
    ext = mods["ext_DropPath"]
    assert ext["opaque"] == {} and ext["className"] == "DropPath"
    drop = by_name(mods["Net"])["drop"]
    assert drop["params"] == {"#0": {"literal": 0.1}, "scale_by_keep": {"literal": True}}
    assert by_name(mods["Net"])["gate"]["params"] == {"#0": {"literal": 4}, "#1": {"literal": 4}}
    assert result["report"][0]["level"] == "L3"
    assert result["project"]["preamble"] == "from timm.layers import DropPath"


def test_never_runs_the_code():
    source = 'raise SystemExit("should not run")\nimport not_a_module\nfrom torch import nn\n\nclass M(nn.Module):\n    def forward(self, x):\n        return x\n'
    result = import_static(source)
    assert result["project"]["entry"] == "M"
    assert any("形状" in w for w in result["warnings"])


def test_cli_modes():
    for mode in ("static", "trace"):
        proc = subprocess.run(
            [sys.executable, "-m", "neuroloom", "import", "-", "--mode", mode, "--input", "x=B,4"],
            input="from torch import nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n    def forward(self, x):\n        return self.fc(x)\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert json.loads(proc.stdout)["project"]["entry"] in ("M", "main")
