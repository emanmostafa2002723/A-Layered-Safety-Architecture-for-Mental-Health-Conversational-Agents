"""Execute the shipped UI logic without making claims about browser layout."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_account_inline_script_contracts():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is optional for inline JavaScript contracts')
    script=Path(__file__).with_name('account_ui_contracts.cjs')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"passed":true' in result.stdout
