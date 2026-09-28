"""Smoke tests for the weight-size gate."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WG_PATH = os.path.join(ROOT, "tools", "weight_gate.py")


def _load():
    spec = importlib.util.spec_from_file_location("weight_gate", WG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extensions_cover_expected_formats():
    wg = _load()
    for ext in (".pt", ".onnx", ".safetensors", ".ckpt", ".engine", ".npz"):
        assert ext in wg.WEIGHT_EXTS


def test_pass_on_empty_dir(tmp_path):
    wg = _load()
    assert wg.main(["--dir", str(tmp_path)]) == 0


def test_fail_over_limit(tmp_path):
    wg = _load()
    (tmp_path / "big.onnx").write_bytes(b"0" * 1024)
    assert wg.main(["--dir", str(tmp_path), "--limit-gb", "0.0000001"]) == 1


def _init_repo(path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    # deterministic, isolated identity so `git add` / ls-files work everywhere
    subprocess.run(["git", "config", "user.email", "gate@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "gate"], cwd=path, check=True)


def _track(path, rel):
    subprocess.run(["git", "add", "--", rel], cwd=path, check=True)


def test_git_mode_counts_only_tracked_weights(tmp_path):
    """Tracked deliverable ONNX counts; untracked/ignored dev checkpoints do not."""
    wg = _load()
    _init_repo(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "artifacts" / "best.onnx").write_bytes(b"0" * 1024)
    (tmp_path / "runs" / "dev.pt").write_bytes(b"0" * (50 * 1024 * 1024))
    _track(tmp_path, "artifacts/best.onnx")

    got = {os.path.basename(p) for p, _ in wg.iter_git_weights(str(tmp_path))}
    assert got == {"best.onnx"}
    assert wg.main(["--dir", str(tmp_path), "--git"]) == 0


def test_git_mode_fails_over_limit_with_tracked_file(tmp_path):
    wg = _load()
    _init_repo(tmp_path)
    (tmp_path / "big.onnx").write_bytes(b"0" * 1024)
    _track(tmp_path, "big.onnx")
    assert wg.main(["--dir", str(tmp_path), "--git", "--limit-gb", "0.0000001"]) == 1


def test_git_mode_errors_outside_repo(tmp_path):
    wg = _load()
    if shutil.which("git") is None:
        pytest.skip("git not available")
    # git present but tmp_path is not a work tree -> loud failure (exit 2)
    assert wg.main(["--dir", str(tmp_path), "--git"]) == 2


def test_git_and_include_dir_are_mutually_exclusive(tmp_path):
    wg = _load()
    with pytest.raises(SystemExit):
        wg.main(["--dir", str(tmp_path), "--git", "--include-dir", "artifacts"])
