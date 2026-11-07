"""The corpus sandbox wrapper must keep third-party test code isolated."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "training" / "corpus" / "sandbox" / "sandbox_python.sh"
)
REQUIREMENTS = SCRIPT.with_name("requirements-sandbox.txt")


def _dry_run(cwd: Path, env_extra: dict[str, str], *args: str) -> list[str]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": "/home/should-never-be-mounted",
        "FL_SANDBOX_DRY_RUN": "1",
        "HF_TOKEN": "secret-token-value",
        **env_extra,
    }
    done = subprocess.run(
        [str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.splitlines()


def test_command_is_isolated(tmp_path: Path) -> None:
    work = tmp_path / "work"
    plugin = tmp_path / "plugin"
    trace = tmp_path / "trace"
    for directory in (work, plugin, trace):
        directory.mkdir()
    out = _dry_run(
        work,
        {
            "PYTHONPATH": f"{work.resolve()}/src:{plugin.resolve()}",
            "FL_TRACE_OUT": f"{trace.resolve()}/trace.json",
            "FL_TRACE_ROOT": str(work.resolve()),
        },
        "-m",
        "pytest",
        "tests/test_a.py::test_x[a b]",
    )
    text = "\n".join(out)
    assert out[0] == "run"
    for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL", "--pids-limit"):
        assert flag in out
    assert out[out.index("--network") + 1] == "none"
    assert "no-new-privileges" in text
    assert "should-never-be-mounted" not in text
    assert "secret-token-value" not in text and "HF_TOKEN" not in text
    mounts = [out[i + 1] for i, item in enumerate(out) if item == "-v"]
    resolved = work.resolve()
    assert f"{resolved}:{resolved}" in mounts
    assert f"{plugin.resolve()}:{plugin.resolve()}:ro" in mounts
    assert f"{trace.resolve()}:{trace.resolve()}" in mounts
    assert len(mounts) == 3
    assert "tests/test_a.py::test_x[a b]" in out
    assert "/opt/venv/bin/python" in out


def test_unsafe_path_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "has space"
    bad.mkdir()
    done = subprocess.run(
        [str(SCRIPT), "-V"],
        cwd=bad,
        env={"PATH": os.environ["PATH"], "FL_SANDBOX_DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 64


def test_requirements_are_hash_pinned_wheels() -> None:
    lines = [
        line
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines
    for line in lines:
        assert "==" in line and "--hash=sha256:" in line, line


def test_dockerfile_pins_base_image_by_digest() -> None:
    text = SCRIPT.with_name("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python@sha256:" in text
    assert "--require-hashes" in text


def test_trace_plugin_emulates_install_metadata(tmp_path: Path) -> None:
    package = tmp_path / "src" / "demo_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from ._version import __version__\n\n\ndef answer():\n    return 42\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_demo.py").write_text(
        "import importlib.metadata\nimport demo_pkg\n\n\n"
        "def test_version():\n"
        "    assert demo_pkg.__version__\n"
        "    assert importlib.metadata.version('demo-pkg')\n",
        encoding="utf-8",
    )
    plugin_dir = Path(__file__).resolve().parents[2] / "training" / "corpus"
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": f"{tmp_path / 'src'}{os.pathsep}{plugin_dir}",
        "FL_TRACE_ROOT": str(tmp_path),
        "FL_TRACE_CALLS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [sys.executable, "-m", "pytest", "-p", "trace_plugin", "-p", "no:cacheprovider"]
    command += ["-q", "-o", "addopts=", str(tests)]
    done = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout
    assert not (package / "_version.py").exists()
