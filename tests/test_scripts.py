"""Basic smoke tests for evaluate.py and demo.py."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TREC_2021_QRELS = ROOT / "data" / "trec" / "2021" / "qrels.txt"


def _run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env if env is not None else os.environ.copy(),
    )


def _run_with_no_key(script: str, extra_args: list[str]) -> subprocess.CompletedProcess:
    """
    Run scripts/{script}.py with load_dotenv mocked out so no .env files are read,
    and OPENAI_API_KEY stripped from the environment.

    Both scripts import load_dotenv lazily inside their functions, so patching
    dotenv.load_dotenv before calling main() prevents any .env loading.
    """
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / 'scripts')!r})
        from unittest.mock import patch
        import {script}
        sys.argv = [{script + '.py'!r}] + {extra_args!r}
        with patch('dotenv.load_dotenv', return_value=False):
            {script}.main()
    """).strip()
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    return subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, env=env)


class TestMissingApiKey:
    def test_evaluate_prints_clear_error(self):
        result = _run_with_no_key("evaluate", ["--quick", "--year", "2021"])
        combined = result.stdout + result.stderr
        assert result.returncode != 0, f"Expected exit 1:\n{combined}"
        assert "OPENAI_API_KEY" in combined

    def test_demo_prints_clear_error(self):
        result = _run_with_no_key("demo", ["58yo male with NSCLC"])
        combined = result.stdout + result.stderr
        assert result.returncode != 0, f"Expected exit 1:\n{combined}"
        assert "OPENAI_API_KEY" in combined

    def test_demo_no_args_shows_usage(self):
        result = _run([sys.executable, str(ROOT / "scripts" / "demo.py")])
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Usage" in combined


@pytest.mark.skipif(
    not TREC_2021_QRELS.exists(),
    reason="TREC 2021 data not present",
)
class TestIntegration:
    def test_evaluate_quick_year_2021(self):
        """evaluate.py --quick --year 2021 runs end-to-end without crashing."""
        result = _run(
            [
                sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                "--quick", "--year", "2021",
            ]
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"evaluate.py failed:\n{combined[-2000:]}"
        assert "TREC 2021" in combined

    def test_demo_sample_patient_returns_results(self):
        """demo.py returns trial results for a sample patient."""
        result = _run(
            [
                sys.executable, str(ROOT / "scripts" / "demo.py"),
                "58yo male with NSCLC EGFR mutation, prior platinum chemotherapy",
            ]
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"demo.py failed:\n{combined[-2000:]}"
        assert "NCT" in combined
        assert "Patient:" in combined
