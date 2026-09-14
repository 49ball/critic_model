import os
from pathlib import Path
import subprocess
import sys
from critic_poc.schema import read_scenes


def test_cli_generates_usable_data_and_rejects_overwrite(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    command = [
        sys.executable,
        "-m",
        "critic_poc",
        "synthetic",
        "--out",
        str(tmp_path / "toy"),
        "--scenes",
        "20",
        "--steps",
        "8",
        "--candidates",
        "3",
    ]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert len(read_scenes(tmp_path / "toy" / "test.jsonl")) == 2
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "exists" in second.stderr
