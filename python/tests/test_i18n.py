"""Messages follow the editor's language (zh by default, en on request)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from neuroloom.i18n import get_locale, set_locale
from neuroloom.importer import ImportFailure, import_many
from neuroloom.server import RpcError, _child, handle
from neuroloom.static import import_static

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile("[一-鿿]")

BROKEN = "def broken(:\n    pass\n"

CODE_BLOCK = """
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
"""


@pytest.fixture(autouse=True)
def chinese_afterwards():
    """Every other test expects the default language."""
    yield
    set_locale("zh")


def test_set_locale_switches_messages():
    set_locale("en")
    with pytest.raises(ImportFailure, match=r"^syntax error \(line 1\)"):
        import_static(BROKEN)
    set_locale("zh")
    with pytest.raises(ImportFailure, match="语法错误"):
        import_static(BROKEN)
    set_locale("fr")
    assert get_locale() == "zh"


def test_cli_lang_option():
    def run(*lang):
        proc = subprocess.run(
            [sys.executable, "-m", "neuroloom", *lang, "import", "-", "--mode", "static"],
            input=BROKEN, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        return json.loads(proc.stdout)["error"]["message"]

    english = run("--lang", "en")
    assert english.startswith("syntax error (line 1)") and not CJK.search(english)
    assert "语法错误" in run()


def test_rpc_report_notes_in_the_requested_locale():
    en = handle({"jsonrpc": "2.0", "id": 1, "method": "import_code", "params": {"source": CODE_BLOCK, "locale": "en"}})["result"]
    notes = en["report"][0]["notes"]
    assert en["report"][0]["codeBlocks"] == 1
    assert notes and all(not CJK.search(n) for n in notes + en["warnings"])
    assert "kept as a code block" in notes[0] and "run-time" in notes[0]

    zh = handle({"jsonrpc": "2.0", "id": 2, "method": "import_code", "params": {"source": CODE_BLOCK}})["result"]
    assert "保留为代码块" in zh["report"][0]["notes"][0]

    bad = handle({"jsonrpc": "2.0", "id": 3, "method": "import_code", "params": {"source": BROKEN, "locale": "en"}})
    assert bad["error"]["code"] == -32001 and bad["error"]["message"].startswith("syntax error")
    # Errors raised by the server itself use the request's language too.
    with pytest.raises(RpcError, match="no result"):
        _child(["no-such-command"], "", "en")
    with pytest.raises(RpcError, match="子进程"):
        _child(["no-such-command"], "", "zh")


def test_batch_requests_carry_their_own_locale():
    results = import_many([{"source": BROKEN, "locale": "en"}, {"source": BROKEN}])
    assert results[0]["error"]["message"].startswith("syntax error")
    assert "语法错误" in results[1]["error"]["message"]
    assert get_locale() == "zh"


def test_default_is_chinese():
    assert get_locale() == "zh"
    with pytest.raises(ImportFailure, match="语法错误"):
        import_static(BROKEN)
    result = import_static(CODE_BLOCK)
    assert "保留为代码块" in result["report"][0]["notes"][0]
    assert any("形状" in w for w in result["warnings"])
