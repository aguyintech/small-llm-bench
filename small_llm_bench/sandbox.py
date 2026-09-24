"""Sandboxed execution of model-generated Python (always-on, fail-closed).

Only the ``code`` module runs LLM output. This module takes a script string plus
resource limits, runs it isolated, and returns a :class:`SandboxResult`. It auto-
detects the best available backend; when no *real* sandbox (network + filesystem
confinement) is available it fails closed unless ``allow_unsandboxed`` is set.

Backend chain, best first:
    1. docker        — daemon, cross-platform (macOS + Linux)
    2. podman        — drop-in for docker, rootless
    3. bwrap         — bubblewrap, Linux, no daemon
    4. sandbox-exec  — macOS built-in (deprecated but ships everywhere)
    5. rlimit        — resource caps only, NO fs/net isolation (not a real sandbox)
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

# Backends that confine both network and filesystem.
_REAL_BACKENDS = ("docker", "podman", "bwrap", "sandbox-exec")
_BACKEND_ORDER = ("docker", "podman", "bwrap", "sandbox-exec", "rlimit")
_DEFAULT_IMAGE = "python:3.12-slim"

# Sequence counter for unique container names (avoids Math.random-style nondeterminism).
_run_counter = 0


@dataclass
class SandboxResult:
    """Outcome of one sandboxed script run."""

    stdout: str
    stderr: str
    returncode: int          # -1 reserved for "did not run"
    # "error" is the CANDIDATE's failure (it ran and exited non-zero);
    # "unavailable" and "skipped_no_sandbox" are the HARNESS's — the script
    # never ran, so neither may be graded against the model.
    status: str              # "ok"|"timeout"|"oom"|"error"|"unavailable"
                             # |"skipped_no_sandbox"
    backend: str             # "docker"|"podman"|"bwrap"|"sandbox-exec"|"rlimit"|"none"


def is_real_sandbox(backend: str) -> bool:
    """True for backends that confine network and filesystem."""
    return backend in _REAL_BACKENDS


def _probe(cmd: list[str]) -> bool:
    """Run a cheap probe command; True iff it exits 0 within a short window."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _available(backend: str) -> bool:
    """True if a specific backend is usable on this host right now."""
    if backend == "docker":
        return bool(shutil.which("docker")) and _probe(["docker", "info"])
    if backend == "podman":
        return bool(shutil.which("podman")) and _probe(["podman", "info"])
    if backend == "bwrap":
        return (platform.system() == "Linux" and bool(shutil.which("bwrap"))
                and _probe(["bwrap", "--version"]))
    if backend == "sandbox-exec":
        return platform.system() == "Darwin" and bool(shutil.which("sandbox-exec"))
    if backend == "rlimit":
        return resource is not None and hasattr(os, "fork")
    return False


@functools.lru_cache(maxsize=None)
def detect_backend(prefer: str = "auto") -> str:
    """Pick the best available backend. Cached once per process.

    ``prefer`` other than ``auto`` forces that backend if it is available, else
    falls back to ``none``. ``auto`` walks the chain best-first.
    """
    if prefer != "auto":
        return prefer if _available(prefer) else "none"
    for backend in _BACKEND_ORDER:
        if _available(backend):
            return backend
    return "none"


def run_sandboxed(script: str, *, timeout: float, memory_mb: int,
                  backend: str, allow_unsandboxed: bool,
                  image: str = _DEFAULT_IMAGE) -> SandboxResult:
    """Run ``script`` isolated under ``backend``; return a :class:`SandboxResult`.

    Fail-closed: if the resolved backend is not a real sandbox and the caller did
    not opt in with ``allow_unsandboxed``, the script is NOT run.
    """
    resolved = detect_backend(backend)
    if resolved == "none" or (not is_real_sandbox(resolved) and not allow_unsandboxed):
        return SandboxResult(stdout="", stderr="", returncode=-1,
                             status="skipped_no_sandbox", backend=resolved)

    if resolved in ("docker", "podman"):
        return _run_container(resolved, script, timeout=timeout,
                              memory_mb=memory_mb, image=image)
    if resolved == "bwrap":
        return _run_bwrap(script, timeout=timeout, memory_mb=memory_mb)
    if resolved == "sandbox-exec":
        return _run_sandbox_exec(script, timeout=timeout, memory_mb=memory_mb)
    if resolved == "rlimit":
        return _run_rlimit(script, timeout=timeout, memory_mb=memory_mb)
    return SandboxResult(stdout="", stderr="", returncode=-1,
                         status="skipped_no_sandbox", backend=resolved)


def _classify(returncode: int, status_on_zero: str = "ok") -> str:
    """Map a process return code to a sandbox status string."""
    if returncode == 0:
        return status_on_zero
    if returncode == 137:        # 128 + SIGKILL → cgroup OOM kill
        return "oom"
    return "error"


def _finish(proc: subprocess.CompletedProcess, backend: str) -> SandboxResult:
    """Build a SandboxResult from a finished process."""
    return SandboxResult(
        stdout=proc.stdout or "", stderr=proc.stderr or "",
        returncode=proc.returncode, status=_classify(proc.returncode),
        backend=backend,
    )


def _run_container(backend: str, script: str, *, timeout: float,
                   memory_mb: int, image: str) -> SandboxResult:
    """Run the script inside a locked-down docker/podman container."""
    if not _ensure_image(backend, image):
        # Not "error": the candidate never ran. A daemon that is up but cannot
        # produce the image (pruned store, a credential helper the OS blocked)
        # otherwise reads downstream as the model writing fifteen broken
        # functions in a row.
        return SandboxResult(stdout="", stderr=f"image unavailable: {image}",
                             returncode=-1, status="unavailable", backend=backend)
    global _run_counter
    _run_counter += 1
    name = f"slb-sandbox-{os.getpid()}-{_run_counter}"
    cmd = [
        backend, "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,noexec",
        "--workdir", "/tmp",
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--pids-limit", "64",
        "--cpus", "1",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        image, "python", "-",
    ]
    try:
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run([backend, "kill", name], capture_output=True, text=True)
        return SandboxResult(stdout="", stderr="", returncode=-1,
                             status="timeout", backend=backend)
    return _finish(proc, backend)


@functools.lru_cache(maxsize=None)
def _ensure_image(backend: str, image: str) -> bool:
    """Ensure the runtime image exists locally, pulling once per session."""
    if _probe([backend, "image", "inspect", image]):
        return True
    print(f"sandbox: pulling {image} (one-time)...", file=sys.stderr)
    try:
        proc = subprocess.run([backend, "pull", image], capture_output=True,
                              text=True, timeout=300.0)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _run_bwrap(script: str, *, timeout: float, memory_mb: int) -> SandboxResult:
    """Run under bubblewrap: unshare namespaces, read-only system, tmpfs /tmp."""
    py = sys.executable
    cmd = [
        "bwrap", "--unshare-all", "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--chdir", "/tmp",
        "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
    ]
    for path in ("/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives"):
        if os.path.exists(path):
            cmd += ["--ro-bind", path, path]
    cmd += [py, "-"]
    try:
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=_rlimit_preexec(memory_mb))
    except subprocess.TimeoutExpired:
        return SandboxResult(stdout="", stderr="", returncode=-1,
                             status="timeout", backend="bwrap")
    return _finish(proc, "bwrap")


_SANDBOX_EXEC_PROFILE = """(version 1)
(deny default)
(allow process-fork process-exec)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(deny file-write* (subpath "/"))
(allow file-write* (subpath "{scratch}"))
(allow file-write-data (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
(deny network*)
"""


def _run_sandbox_exec(script: str, *, timeout: float,
                      memory_mb: int) -> SandboxResult:
    """Run under macOS sandbox-exec with a deny-by-default profile.

    Note: ``sandbox-exec`` is deprecated by Apple but ships on every Mac and
    needs no daemon; kept as the zero-dependency macOS backend.
    """
    py = sys.executable
    with tempfile.TemporaryDirectory(prefix="slb-sbx-") as scratch:
        profile = os.path.join(scratch, "profile.sb")
        # /private/tmp resolves the symlinked scratch on macOS; allow both.
        with open(profile, "w") as handle:
            handle.write(_SANDBOX_EXEC_PROFILE.format(scratch=scratch))
        cmd = ["sandbox-exec", "-f", profile, py, "-"]
        try:
            proc = subprocess.run(
                cmd, input=script, capture_output=True, text=True,
                timeout=timeout, cwd=scratch,
                preexec_fn=_rlimit_preexec(memory_mb),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(stdout="", stderr="", returncode=-1,
                                 status="timeout", backend="sandbox-exec")
    return _finish(proc, "sandbox-exec")


def _run_rlimit(script: str, *, timeout: float, memory_mb: int) -> SandboxResult:
    """Resource-capped fallback: NO network/fs isolation. Opt-in only."""
    with tempfile.TemporaryDirectory(prefix="slb-rl-") as scratch:
        env = {"PATH": "/usr/bin:/bin", "HOME": scratch, "TMPDIR": scratch}
        try:
            proc = subprocess.run(
                [sys.executable, "-"], input=script, capture_output=True,
                text=True, timeout=timeout, cwd=scratch, env=env,
                preexec_fn=_rlimit_preexec(memory_mb, block_writes=True),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(stdout="", stderr="", returncode=-1,
                                 status="timeout", backend="rlimit")
    return _finish(proc, "rlimit")


def _rlimit_preexec(memory_mb: int, block_writes: bool = False):
    """Build a POSIX ``preexec_fn`` applying CPU/memory/proc/file rlimits.

    Returns None on platforms without ``resource`` so callers stay portable.
    Note: macOS (Darwin) does not enforce ``RLIMIT_AS``, so the memory cap is a
    no-op there — another reason ``rlimit`` is not a real sandbox.
    """
    if resource is None:
        return None

    def _apply() -> None:
        # CPU-seconds backstop; the wall-clock timeout is the primary guard.
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        mem = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass
        if block_writes:
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            except (ValueError, OSError):
                pass

    return _apply
