import json
import subprocess
import sys
from pathlib import Path

import pytest

from neuroloom.importer import ImportFailure, InputSpec, import_source
from neuroloom.server import handle

ROOT = Path(__file__).resolve().parent.parent

MLP = """
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
"""


def by_name(module):
    return {n["name"]: n for n in module["nodes"].values()}


def test_imports_a_functional_mlp():
    result = import_source(MLP)
    module = result["project"]["modules"]["main"]
    nodes = by_name(module)
    assert module["className"] == "MLP"
    assert nodes["fc1"]["params"]["in_features"] == {"literal": 784}
    assert nodes["fc2"]["params"]["bias"] == {"literal": False}
    assert nodes["relu"]["type"] == "torch.nn.functional.relu"
    # Input shape inferred from the first Linear, and the probe ran on the meta device.
    assert nodes["x"]["params"]["shape"] == {"literal": "B, 784"}
    assert result["observedShapes"][nodes["fc2"]["id"]] == {"out": [2, 10]}
    assert len(module["edges"]) == 4


def test_explicit_input_shape_and_class_choice():
    source = MLP + "\nclass Other(nn.Module):\n    def forward(self, x):\n        return x\n"
    result = import_source(source, class_name="MLP", inputs={"x": InputSpec(["N", 784])})
    nodes = by_name(result["project"]["modules"]["main"])
    assert nodes["x"]["params"]["shape"] == {"literal": "N, 784"}


def test_unsupported_layers_and_slicing_are_listed():
    source = (
        "from torch import nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.b = nn.Bilinear(3, 3, 8)\n"
        "    def forward(self, x):\n"
        "        return self.b(x, x)[:, 0]\n"
    )
    with pytest.raises(ImportFailure, match="Bilinear") as e:
        import_source(source, inputs={"x": InputSpec(["B", 3])})
    assert "切片" in str(e.value)


def test_runtime_dependent_arguments_are_explained():
    source = (
        "from torch import nn\n"
        "class M(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x.view(x.size(0), -1)\n"
    )
    with pytest.raises(ImportFailure, match="运行时"):
        import_source(source, inputs={"x": InputSpec(["B", 3, 4])})


def test_int64_inputs_and_tuple_outputs():
    source = (
        "from torch import nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.embed = nn.Embedding(50, 8)\n"
        "        self.lstm = nn.LSTM(8, 16, batch_first=True)\n"
        "    def forward(self, ids):\n"
        "        out, (h, c) = self.lstm(self.embed(ids))\n"
        "        return out, h\n"
    )
    result = import_source(source, inputs={"ids": InputSpec(["B", 12], "int64")})
    module = result["project"]["modules"]["main"]
    lstm = by_name(module)["lstm"]
    assert by_name(module)["ids"]["params"]["dtype"] == {"literal": "int64"}
    assert lstm["params"]["batch_first"] == {"literal": True}
    sources = sorted((e["source"]["port"], module["nodes"][e["target"]["node"]]["name"]) for e in module["edges"] if e["source"]["node"] == lstm["id"])
    assert sources == [("h_n", "output_1"), ("out", "output")]
    assert result["observedShapes"][lstm["id"]] == {"out": [2, 12, 16], "h_n": [1, 2, 16], "c_n": [1, 2, 16]}


def test_data_dependent_control_flow_is_reported():
    source = "from torch import nn\nclass Dyn(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 4)\n    def forward(self, x):\n        if x.sum() > 0:\n            return self.fc(x)\n        return x\n"
    with pytest.raises(ImportFailure, match="torch.fx"):
        import_source(source)


def test_cli_prints_json_and_keeps_user_prints_off_stdout():
    noisy = MLP.replace("super().__init__()", "super().__init__()\n        print('hello from user code')")
    proc = subprocess.run(
        [sys.executable, "-m", "neuroloom", "import", "-", "--mode", "trace"],
        input=noisy, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "hello from user code" in proc.stderr
    assert json.loads(proc.stdout)["project"]["name"] == "MLP"


def test_rpc_import_and_errors():
    ok = handle({"jsonrpc": "2.0", "id": 1, "method": "import_code", "params": {"source": MLP}})
    assert ok["result"]["project"]["entry"] == "MLP"
    traced = handle({"jsonrpc": "2.0", "id": 4, "method": "import_code", "params": {"source": MLP, "mode": "trace"}})
    assert traced["result"]["project"]["modules"]["main"]["className"] == "MLP"
    bad = handle({"jsonrpc": "2.0", "id": 2, "method": "import_code", "params": {"source": "def broken(:\n    pass\n"}})
    assert bad["error"]["code"] == -32001 and "语法错误" in bad["error"]["message"]
    unknown = handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
    assert unknown["error"]["code"] == -32601
