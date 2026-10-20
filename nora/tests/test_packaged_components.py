"""Ensure trimmed components still import without model weights or inference."""
from pathlib import Path
import subprocess
import sys


def test_current_boundary_source_imports_without_weights():
    root = Path(__file__).resolve().parents[1]
    code = 'import sys; sys.path.insert(0,"components/boundary"); from boundary.conservative import ConservativeGuard; from boundary.recalibrated import RecalibratedBackend'
    result = subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=25)
    assert result.returncode == 0, result.stderr


def test_current_qwen_output_rule_component_initializes_without_model_calls():
    root = Path(__file__).resolve().parents[1]
    code = ('import sys; from pathlib import Path; sys.path.insert(0,"src"); '
        'from mental_health_pipeline.local_output import LocalJudge,JudgeSettings; '
        'LocalJudge(Path("components/output_guardrails"),JudgeSettings())')
    result = subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=25)
    assert result.returncode == 0, result.stderr
