"""Air-gapped container sandbox (`container` or `docker`) for grading untrusted LLM code."""

from __future__ import annotations

import contextlib
import functools
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SandboxBackend = Literal["container", "docker"]


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of running candidate code + unit tests inside the isolated container."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    backend_used: str = ""
    # True only if the per-run completion marker (written after the harness) reached stdout,
    # i.e. control fell off the end of the harness instead of exiting early.
    harness_completed: bool = False


def build_sandbox_command(
    backend: SandboxBackend,
    *,
    image: str = "python:3.14-slim",
    binary_path: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    """Construct the exact CLI argument vector for `container` (Apple) or `docker` (Colima).

    Verified against installed CLI syntax on macOS:
    - Apple `container` (`/usr/local/bin/container`): `--tmpfs` accepts only a filesystem path
      (no `:rw,noexec` mount options) and does not have `--pids-limit`; supports `--no-dns`,
      `--network none`, `--read-only`, `-m 256M`, `-c 1`, `--name`.
    - Colima `docker` (`/opt/homebrew/bin/docker`): supports `--network none`, `--read-only`,
      `--tmpfs /tmp:rw,noexec,nosuid,size=64m`, `--pids-limit 64`, `--memory 256m`, `--cpus 1`, `--name`.
    """
    default_bin, backend_flags = _SANDBOX_BACKEND_CONFIGS.get(backend, (None, None))
    if default_bin is None or backend_flags is None:
        raise ValueError(f"Unsupported sandbox backend: {backend!r}")
    bin_cmd = binary_path or shutil.which(backend) or default_bin
    name_args = ["--name", container_name] if container_name else []
    return [
        bin_cmd,
        "run",
        "--rm",
        *name_args,
        "-i",
        "--network",
        "none",
        "--read-only",
        *backend_flags,
        image,
        "python3",
        "-I",
        "-",
    ]


_SANDBOX_BACKEND_CONFIGS: dict[str, tuple[str, list[str]]] = {
    "container": (
        "/usr/local/bin/container",
        ["--no-dns", "--tmpfs", "/tmp", "-m", "256M", "-c", "1"],
    ),
    "docker": (
        "/opt/homebrew/bin/docker",
        [
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
        ],
    ),
}


def _scrubbed_host_env() -> dict[str, str]:
    """Return a minimal host environment for invoking the container CLI without leaking API keys."""
    safe_env: dict[str, str] = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
    }
    if "HOME" in os.environ:
        safe_env["HOME"] = os.environ["HOME"]
    if "DOCKER_HOST" in os.environ:
        safe_env["DOCKER_HOST"] = os.environ["DOCKER_HOST"]
    return safe_env


def _resolve_backend_binary(backend: SandboxBackend) -> str:
    """Resolve the container runtime CLI path, falling back to its standard install location."""
    fallback, _ = _SANDBOX_BACKEND_CONFIGS[backend]
    return shutil.which(backend) or fallback


@functools.lru_cache(maxsize=4)
def detect_available_backend(
    preferred: SandboxBackend | Literal["auto"] = "auto",
) -> SandboxBackend:
    """Detect which container runtime (`container` or `docker`) is active and ready."""
    order: list[SandboxBackend] = ["container", "docker"] if preferred == "auto" else [preferred]
    env = _scrubbed_host_env()
    for candidate in order:
        bin_path = _resolve_backend_binary(candidate)
        if not Path(bin_path).exists():
            continue
        status_args = ["system", "status"] if candidate == "container" else ["info"]
        try:
            proc = subprocess.run(
                [bin_path, *status_args],
                check=False,
                env=env,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if proc.returncode == 0:
                return candidate
        except Exception:
            continue
    raise RuntimeError(
        "Neither Apple `container` nor `docker` (Colima) daemon is currently running. "
        "Start one via `container system start` or `colima start` before grading code slices."
    )


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


_CONTAINER_PROGRESS_RE = re.compile(r"^\s*\[\d+/\d+\]\s.*$")


def strip_container_progress(stderr: str) -> str:
    """Drop Apple `container` CLI startup progress lines (e.g. `[3/6] Fetching kernel [0s]`).

    The runtime writes these to stderr before the Python process starts, so keeping
    the head of stderr hides the actual traceback.
    """
    kept: list[str] = []
    for raw_line in stderr.replace("\r", "\n").splitlines():
        if not raw_line.strip() or _CONTAINER_PROGRESS_RE.match(raw_line):
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def extract_python_code(model_output: str) -> str:
    """Extract Python source code from a markdown code block or raw response."""
    matches = _CODE_FENCE_RE.findall(model_output)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return model_output.strip()


_HARNESS_DONE_PREFIX = "__GDR_HARNESS_DONE_"


def _completion_marker(nonce: str) -> str:
    """The stdout line that proves the harness ran to completion in the run keyed by `nonce`."""
    return f"{_HARNESS_DONE_PREFIX}{nonce}__"


def _completion_marker_source(nonce: str) -> str:
    """Python appended after the harness that writes the run's completion marker to fd 1.

    It only executes when control falls off the end of the harness, so candidate code that exits
    early (`sys.exit(0)`, `raise SystemExit`, `os._exit(0)`, `quit()`, `unittest.main()`) never
    emits it. Fresh aliased imports keep candidate rebinding of `os` / `sys` / `print` out of the
    way, buffered `print` output is flushed first so the marker comes last, and the nonce is split
    so the full marker never appears verbatim in the script.
    """
    half = len(nonce) // 2
    return (
        "import os as _gdr_os_mark\n"
        "import sys as _gdr_sys_mark\n"
        "try:\n"
        "    _gdr_sys_mark.stdout.flush()\n"
        "    _gdr_sys_mark.__stdout__.flush()\n"
        "except Exception:\n"
        "    pass\n"
        f'_gdr_os_mark.write(1, ("\\n{_HARNESS_DONE_PREFIX}" + "{nonce[:half]}" + '
        f'"{nonce[half:]}" + "__\\n").encode())\n'
    )


def _strip_completion_marker(stdout: str, marker: str) -> str:
    """Remove the completion marker (and the newlines written around it) from `stdout`."""
    return re.sub(r"\n?" + re.escape(marker) + r"\n?", "", stdout)


def run_python_in_sandbox(
    candidate_response: str,
    test_harness: str,
    *,
    timeout_sec: float = 5.0,
    backend: SandboxBackend | Literal["auto"] = "auto",
    image: str = "python:3.14-slim",
) -> SandboxResult:
    """Execute `candidate_response` + `test_harness` inside an air-gapped container (`--network none`).

    A run passes only if the process exits 0 AND the per-run completion marker appended after the
    harness shows up in the full stdout (checked before truncation). Exit status 0 alone is not
    enough: a candidate that exits early skips the tests. Harnesses must therefore signal success
    by running to completion and failure by raising, never by calling `sys.exit(0)`. The marker is
    stripped from the returned stdout.
    """
    resolved_backend = detect_available_backend(backend)
    container_name = f"gdr_sb_{secrets.token_hex(8)}"
    cmd = build_sandbox_command(resolved_backend, image=image, container_name=container_name)
    code_body = extract_python_code(candidate_response)
    nonce = secrets.token_hex(16)
    marker = _completion_marker(nonce)
    combined_script = (
        f"{code_body}\n\n# --- UNIT TEST HARNESS ---\n{test_harness}\n"
        f"\n# --- HARNESS COMPLETION MARKER ---\n{_completion_marker_source(nonce)}"
    )

    try:
        proc = subprocess.run(
            cmd,
            input=combined_script,
            check=False,
            env=_scrubbed_host_env(),
            capture_output=True,
            text=True,
            # Candidate output is untrusted: undecodable bytes must not crash the grader.
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        full_stdout = proc.stdout or ""
        harness_completed = marker in full_stdout
        return SandboxResult(
            passed=(proc.returncode == 0 and harness_completed),
            exit_code=proc.returncode,
            stdout=_strip_completion_marker(full_stdout, marker)[-4096:],
            stderr=strip_container_progress(proc.stderr or "")[-4096:],
            timed_out=False,
            backend_used=resolved_backend,
            harness_completed=harness_completed,
        )
    except subprocess.TimeoutExpired as exc:
        # Force-kill and remove the microVM container if it timed out so Apple Virtualization
        # does not leave orphaned VM processes pegging CPU in the background.
        bin_cleanup = _resolve_backend_binary(resolved_backend)
        with contextlib.suppress(Exception):
            subprocess.run(
                [bin_cleanup, "rm", "-f", container_name],
                check=False,
                capture_output=True,
                env=_scrubbed_host_env(),
                timeout=5.0,
            )

        partial_stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return SandboxResult(
            passed=False,
            exit_code=-1,
            stdout=_strip_completion_marker(partial_stdout, marker)[:2048],
            stderr="Execution timed out inside container sandbox.",
            timed_out=True,
            backend_used=resolved_backend,
            harness_completed=marker in partial_stdout,
        )
