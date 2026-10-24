"""Dynamic import: run the model, keep the hierarchy, merge with what the static reader understood."""

import re

from neuroloom.dynamic import import_dynamic
from neuroloom.importer import InputSpec
from neuroloom.static import import_static

X = {"x": InputSpec(["B", 3, 16, 16])}


def levels(result):
    return {r["className"]: r["level"] for r in result["report"]}


def by_class(result):
    return {m["className"]: m for m in result["project"]["modules"].values()}


LOOPED = """
import torch
from torch import nn


class Stage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.Conv2d(cin, cout, 3, padding=1))
            if i < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Head(nn.Module):
    def __init__(self, cin: int, num_classes: int = 10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.flat(self.pool(x)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage = Stage([3, 8, 16])
        self.head = Head(16)

    def forward(self, x):
        return self.head(self.stage(x))
"""


def test_structures_from_the_run_what_the_reader_cannot_follow():
    static = import_static(LOOPED, inputs=X)
    assert levels(static)["Stage"] != "L1"

    result = import_dynamic(LOOPED, inputs=X)
    assert levels(result) == {"Stage": "L2", "Head": "L1", "Net": "L1"}
    stage = by_class(result)["Stage"]
    assert not any(n["kind"] == "code" for n in stage["nodes"].values())
    # The ModuleList of unlike layers, chained in a loop, becomes a Sequential of them.
    layers = next(n for n in stage["nodes"].values() if n["name"] == "layers")
    chain = result["project"]["modules"][layers["type"]]
    assert chain.get("sequential")
    assert [chain["nodes"][n]["type"] for n in sorted(chain["nodes"]) if chain["nodes"][n]["kind"] == "layer"] == ["torch.nn.Conv2d", "torch.nn.ReLU", "torch.nn.Conv2d"]


def test_keeps_what_the_reader_understood_expressions_and_all():
    result = import_dynamic(LOOPED, inputs=X)
    head = by_class(result)["Head"]
    assert [h["name"] for h in head["hyperparams"]] == ["cin", "num_classes"]
    fc = next(n for n in head["nodes"].values() if n["name"] == "fc")
    assert fc["params"]["in_features"] == {"expr": "cin"}
    entry = result["project"]["modules"][result["project"]["entry"]]
    x = next(n for n in entry["nodes"].values() if n["kind"] == "input")
    assert x["params"]["shape"] == {"literal": "B, 3, 16, 16"}


BRANCHING = """
import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        if x.sum() > 0:
            return self.a(x)
        return self.b(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(4, 8)
        self.gate = Gate()

    def forward(self, x):
        return self.gate(self.inp(x))
"""


def test_a_class_that_cannot_be_traced_stays_as_the_reader_left_it():
    inputs = {"x": InputSpec(["B", 4])}
    static = import_static(BRANCHING, inputs=inputs)
    result = import_dynamic(BRANCHING, inputs=inputs)
    assert levels(result)["Gate"] == levels(static)["Gate"]
    assert by_class(result)["Gate"].get("opaque") == by_class(static)["Gate"].get("opaque")
    assert levels(result)["Net"] in ("L1", "L2")


REFERENCED = """
import torch
from torch import nn


class Pair(nn.Module):
    def __init__(self, cin, cout, mid=None):
        super().__init__()
        if not mid:
            mid = cout
        self.a = nn.Conv2d(cin, mid, 3, padding=1)
        self.b = nn.Conv2d(mid, cout, 3, padding=1)

    def forward(self, x):
        return self.b(self.a(x))


class Up(nn.Module):
    def __init__(self, cin, cout, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
            self.conv = Pair(cin, cout, cin // 2)
        else:
            self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
            self.conv = Pair(cin // 2, cout)

    def forward(self, x):
        return self.conv(self.up(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Pair(3, 8)
        self.up = Up(8, 4)

    def forward(self, x):
        return self.up(self.first(x))
"""


def test_classes_named_in_kept_code_keep_their_signature():
    result = import_dynamic(REFERENCED, inputs=X)
    classes = by_class(result)
    kept = "\n".join(m.get("initCode", "") + (m.get("opaque") or {}).get("source", "") for m in classes.values())
    for name in set(re.findall(r"\b(Pair|Up)\(", kept)):
        static = by_class(import_static(REFERENCED, inputs=X))[name]
        assert [h["name"] for h in classes[name]["hyperparams"]] == [h["name"] for h in static["hyperparams"]]
    names = [m["className"] for m in result["project"]["modules"].values()]
    assert len(names) == len(set(names))
    assert {r["className"] for r in result["report"]} <= set(names)


def test_falls_back_to_the_static_result_when_the_entry_cannot_be_resolved():
    source = """
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        y = self.fc(x)
        if y.sum() > 0:  # data-dependent: neither reader nor tracer can follow it
            y = y * 2
        return y
"""
    inputs = {"x": InputSpec(["B", 4])}
    result = import_dynamic(source, inputs=inputs)
    static = import_static(source, inputs=inputs)
    assert result["project"]["modules"].keys() == static["project"]["modules"].keys()
    assert levels(result) == levels(static)


STATE = """
import torch
from torch import nn


class Scaled(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-6)
        self.register_buffer("offset", torch.zeros(dim))

    def forward(self, x):
        return self.layer_scale * self.fc(x) + self.offset
"""


def test_parameters_and_buffers_used_in_forward_become_nodes():
    inputs = {"x": InputSpec(["B", 8])}
    # The reader keeps an initial value it cannot express (ones * 1e-6) as code; the run turns it into a node.
    result = import_dynamic(STATE, inputs=inputs)
    scaled = by_class(result)["Scaled"]
    states = {n["name"]: (n["type"], n["params"]) for n in scaled["nodes"].values() if n["type"] in ("neuroloom.Parameter", "neuroloom.Buffer")}
    assert states == {
        "layer_scale": ("neuroloom.Parameter", {"shape": {"literal": [8]}, "init": {"literal": "zeros"}}),
        "offset": ("neuroloom.Buffer", {"shape": {"literal": [8]}, "init": {"literal": "zeros"}}),
    }
    assert not any(n["kind"] == "code" for n in scaled["nodes"].values())
