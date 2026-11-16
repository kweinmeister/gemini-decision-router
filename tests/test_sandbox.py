"""Unit tests for air-gapped container sandbox command generation and environment scrubbing."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from evals import sandbox
from evals.sandbox import (
    _HARNESS_DONE_PREFIX,
    SandboxBackend,
    SandboxResult,
    _completion_marker,
    _completion_marker_source,
    _scrubbed_host_env,
    build_sandbox_command,
    detect_available_backend,
    extract_python_code,
    run_python_in_sandbox,
)


def test_build_sandbox_command_apple_container() -> None:
    cmd = build_sandbox_command("container", binary_path="/usr/local/bin/container")
    assert cmd[:4] == ["/usr/local/bin/container", "run", "--rm", "-i"]
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--no-dns" in cmd
    assert "--read-only" in cmd
    # Apple container CLI requires bare path for --tmpfs
    assert cmd[cmd.index("--tmpfs") + 1] == "/tmp"
    assert cmd[-3:] == ["python3", "-I", "-"]


def test_build_sandbox_command_colima_docker() -> None:
    cmd = build_sandbox_command("docker", binary_path="/opt/homebrew/bin/docker")
    assert cmd[:4] == ["/opt/homebrew/bin/docker", "run", "--rm", "-i"]
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd
    assert "--pids-limit" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert cmd[cmd.index("--user") + 1] == "65534:65534"
    assert cmd[cmd.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,size=64m"
    assert cmd[-3:] == ["python3", "-I", "-"]


def test_detect_available_backend_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_available_backend.cache_clear()
    monkeypatch.setattr(sandbox.os.path, "exists", lambda p: True)
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=0, stdout="", stderr=""
        ),
    )
    b1 = detect_available_backend("container")
    b2 = detect_available_backend("container")
    assert b1 == b2 == "container"
    assert detect_available_backend.cache_info().hits >= 1
    detect_available_backend.cache_clear()


def test_scrubbed_host_env_excludes_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-key")
    env = _scrubbed_host_env()
    assert "GEMINI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "PATH" in env


def test_extract_python_code_strips_markdown_fences() -> None:
    fenced = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert extract_python_code(fenced) == "def add(a, b):\n    return a + b"


# --- Harness completion marker (mocked subprocess; nothing is executed) ---

_FIXED_NONCE = "0123456789abcdef" * 2
_MARKER = _completion_marker(_FIXED_NONCE)
_TIMEOUT_STDERR = "Execution timed out inside container sandbox."


class _FakeRun:
    """Stand-in for `subprocess.run`: records each call and replays a canned outcome."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outcome: tuple[int, str, str] | BaseException = (0, "", "")

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"cmd": cmd, **kwargs})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        returncode, stdout, stderr = self.outcome
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> _FakeRun:
    fake = _FakeRun()
    monkeypatch.setattr(sandbox, "detect_available_backend", lambda backend="auto": "docker")
    monkeypatch.setattr(sandbox.subprocess, "run", fake)
    return fake


@pytest.fixture
def fixed_nonce(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda nbytes=None: _FIXED_NONCE)
    return _FIXED_NONCE


def _run(**kwargs: Any) -> SandboxResult:
    return run_python_in_sandbox("```python\nx = 1\n```", "assert x == 1", **kwargs)


def test_sandbox_result_harness_completed_defaults_to_false() -> None:
    res = SandboxResult(passed=False, exit_code=0, stdout="", stderr="")
    assert res.harness_completed is False


def test_exit_zero_with_marker_passes_and_marker_is_stripped(
    fake_run: _FakeRun, fixed_nonce: str
) -> None:
    fake_run.outcome = (0, f"GDR_S2_HIDDEN_SUITE_OK 64\n\n{_MARKER}\n", "")
    res = _run()
    assert res.passed is True
    assert res.harness_completed is True
    assert res.exit_code == 0
    assert res.timed_out is False
    assert res.backend_used == "docker"
    assert res.stdout == "GDR_S2_HIDDEN_SUITE_OK 64\n"


def test_exit_zero_without_marker_is_not_a_pass(fake_run: _FakeRun, fixed_nonce: str) -> None:
    # e.g. candidate code ran `sys.exit(0)` before the appended harness.
    fake_run.outcome = (0, "candidate output\n", "")
    res = _run()
    assert res.passed is False
    assert res.harness_completed is False
    assert res.exit_code == 0
    assert res.stdout == "candidate output\n"


def test_nonzero_exit_with_marker_is_not_a_pass(fake_run: _FakeRun, fixed_nonce: str) -> None:
    # e.g. an atexit hook failed after the harness finished: the exit status still counts.
    fake_run.outcome = (1, f"\n{_MARKER}\n", "Traceback ...\nRuntimeError: atexit boom\n")
    res = _run()
    assert res.passed is False
    assert res.harness_completed is True
    assert res.exit_code == 1
    assert res.stdout == ""
    assert res.stderr.endswith("RuntimeError: atexit boom")


@pytest.mark.parametrize(
    ("stdout", "kept"),
    [
        # Marker near the head, followed by >4096 chars of later output (e.g. atexit noise):
        # the marker falls outside the kept tail but must still count.
        (f"\n{_MARKER}\n" + "x" * 5000, "x" * 4096),
        # Marker at the end of a long stdout: it is stripped before truncation, so the kept
        # tail is pure program output.
        ("y" * 6000 + f"\n{_MARKER}\n", "y" * 4096),
    ],
    ids=["marker-before-tail", "marker-at-end"],
)
def test_marker_is_checked_on_full_stdout_before_truncation(
    fake_run: _FakeRun, fixed_nonce: str, stdout: str, kept: str
) -> None:
    fake_run.outcome = (0, stdout, "")
    res = _run()
    assert res.passed is True
    assert res.harness_completed is True
    assert res.stdout == kept
    assert _HARNESS_DONE_PREFIX not in res.stdout


def test_marker_from_another_run_does_not_count(fake_run: _FakeRun, fixed_nonce: str) -> None:
    stale = _completion_marker("f" * 32)
    fake_run.outcome = (0, f"\n{stale}\n", "")
    res = _run()
    assert res.passed is False
    assert res.harness_completed is False
    # Only this run's marker is stripped; a foreign one is left visible.
    assert stale in res.stdout


def test_script_layout_and_subprocess_kwargs(fake_run: _FakeRun, fixed_nonce: str) -> None:
    fake_run.outcome = (0, f"\n{_MARKER}\n", "")
    response = "Sure:\n```python\ndef f():\n    return 1\n```\nDone."
    res = run_python_in_sandbox(response, "assert f() == 1", timeout_sec=7.5)
    assert res.passed is True

    assert len(fake_run.calls) == 1
    call = fake_run.calls[0]
    script: str = call["input"]
    # Candidate code first, then the harness, then the completion epilogue as the very end.
    assert script.startswith("def f():\n    return 1\n")
    assert (
        script.index("def f():")
        < script.index("# --- UNIT TEST HARNESS ---")
        < script.index("assert f() == 1")
        < script.index("# --- HARNESS COMPLETION MARKER ---")
    )
    assert script.endswith(_completion_marker_source(_FIXED_NONCE))
    # The full marker never appears verbatim in the source; both nonce halves do.
    assert _MARKER not in script
    assert _FIXED_NONCE[:16] in script and _FIXED_NONCE[16:] in script
    compile(script, "<sandbox>", "exec")

    assert call["cmd"][-3:] == ["python3", "-I", "-"]
    assert call["env"] == _scrubbed_host_env()
    assert call["capture_output"] is True
    assert call["encoding"] == "utf-8"
    assert call["errors"] == "replace"
    assert call["timeout"] == 7.5


def test_each_run_uses_a_fresh_nonce(fake_run: _FakeRun) -> None:
    fake_run.outcome = (0, "", "")
    first, second = _run(), _run()
    assert not first.passed and not second.passed
    scripts = [call["input"] for call in fake_run.calls]
    assert scripts[0] != scripts[1]


def test_completion_epilogue_writes_marker_after_buffered_output(
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Executes only the trusted epilogue (no candidate code) to check what it writes to fd 1.
    nonce = "a" * 16 + "b" * 16
    source = "print('candidate output', end='')\n" + _completion_marker_source(nonce)
    exec(compile(source, "<epilogue>", "exec"), {})
    assert capfd.readouterr().out == f"candidate output\n{_completion_marker(nonce)}\n"


@pytest.mark.parametrize(
    ("partial", "completed", "stdout"),
    [
        (b"partial\n", False, "partial\n"),
        (f"partial\n\n{_MARKER}\n".encode(), True, "partial\n"),
        ("partial text\n", False, "partial text\n"),
        (None, False, ""),
    ],
    ids=["bytes-no-marker", "bytes-with-marker", "str-no-marker", "no-output"],
)
def test_timeout_never_passes(
    fake_run: _FakeRun,
    fixed_nonce: str,
    partial: bytes | str | None,
    completed: bool,
    stdout: str,
) -> None:
    fake_run.outcome = subprocess.TimeoutExpired(cmd=["docker"], timeout=1.0, output=partial)
    res = _run(timeout_sec=1.0)
    assert res.passed is False
    assert res.timed_out is True
    assert res.exit_code == -1
    assert res.harness_completed is completed
    assert res.stdout == stdout
    assert res.stderr == _TIMEOUT_STDERR
    assert res.backend_used == "docker"


# --- Real sandbox execution (skipped when no container backend is running) ---


@pytest.mark.slow
@pytest.mark.parametrize(
    "candidate",
    ["import sys\nsys.exit(0)", "raise SystemExit", "import os\nos._exit(0)", "quit()"],
    ids=["sys.exit", "SystemExit", "os._exit", "quit"],
)
def test_real_sandbox_early_exit_is_not_a_pass(
    real_backend: SandboxBackend, candidate: str
) -> None:
    harness = "raise AssertionError('harness ran')"
    res = run_python_in_sandbox(candidate, harness, timeout_sec=60.0, backend=real_backend)
    assert res.exit_code == 0
    assert res.harness_completed is False
    assert res.passed is False
    assert "harness ran" not in res.stderr


@pytest.mark.slow
def test_real_sandbox_completed_harness_passes_and_hides_marker(
    real_backend: SandboxBackend,
) -> None:
    candidate = "```python\ndef f():\n    return 1\nprint('candidate says hi', end='')\n```"
    harness = "assert f() == 1\nprint('HARNESS_OK')"
    res = run_python_in_sandbox(candidate, harness, timeout_sec=60.0, backend=real_backend)
    assert res.passed is True
    assert res.harness_completed is True
    assert res.stdout == "candidate says hiHARNESS_OK\n"


@pytest.mark.slow
def test_real_sandbox_undecodable_stdout_does_not_crash(real_backend: SandboxBackend) -> None:
    candidate = "import os\nos.write(1, b'\\xff\\xfe raw bytes\\n')"
    res = run_python_in_sandbox(candidate, "pass", timeout_sec=60.0, backend=real_backend)
    assert res.passed is True
    assert "\ufffd" in res.stdout


@pytest.mark.slow
def test_real_sandbox_timeout_enforced(real_backend: SandboxBackend) -> None:
    candidate = "while True:\n    pass"
    harness = "assert True"
    res = run_python_in_sandbox(candidate, harness, timeout_sec=1.5, backend=real_backend)
    assert res.timed_out is True
    assert res.passed is False
    assert res.harness_completed is False
    assert res.exit_code == -1
