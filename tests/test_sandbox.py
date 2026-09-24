"""Tests for the code-execution sandbox: detection, mapping, and isolation."""

from __future__ import annotations

import platform

import pytest

from small_llm_bench import sandbox
from small_llm_bench.sandbox import (detect_backend, is_real_sandbox,
                                     run_sandboxed)

POSIX = hasattr(__import__("os"), "fork")


def _force_available(monkeypatch, *available: str) -> None:
    """Make only the named backends report as available, and clear the cache."""
    allowed = set(available)
    monkeypatch.setattr(sandbox, "_available", lambda backend: backend in allowed)
    detect_backend.cache_clear()


class TestDetectBackend:
    def teardown_method(self):
        detect_backend.cache_clear()

    def test_docker_preferred(self, monkeypatch):
        _force_available(monkeypatch, "docker", "podman", "rlimit")
        assert detect_backend("auto") == "docker"

    def test_podman_when_no_docker(self, monkeypatch):
        _force_available(monkeypatch, "podman", "bwrap", "rlimit")
        assert detect_backend("auto") == "podman"

    def test_bwrap_before_sandbox_exec(self, monkeypatch):
        _force_available(monkeypatch, "bwrap", "sandbox-exec", "rlimit")
        assert detect_backend("auto") == "bwrap"

    def test_rlimit_last_resort(self, monkeypatch):
        _force_available(monkeypatch, "rlimit")
        assert detect_backend("auto") == "rlimit"

    def test_none_when_nothing(self, monkeypatch):
        _force_available(monkeypatch)
        assert detect_backend("auto") == "none"

    def test_explicit_prefer_available(self, monkeypatch):
        _force_available(monkeypatch, "docker", "rlimit")
        assert detect_backend("docker") == "docker"

    def test_explicit_prefer_unavailable(self, monkeypatch):
        _force_available(monkeypatch, "rlimit")
        assert detect_backend("bwrap") == "none"


class TestIsRealSandbox:
    @pytest.mark.parametrize("backend", ["docker", "podman", "bwrap", "sandbox-exec"])
    def test_real(self, backend):
        assert is_real_sandbox(backend)

    @pytest.mark.parametrize("backend", ["rlimit", "none", ""])
    def test_not_real(self, backend):
        assert not is_real_sandbox(backend)


class TestFailClosed:
    def teardown_method(self):
        detect_backend.cache_clear()

    def test_no_backend_skips(self, monkeypatch):
        _force_available(monkeypatch)
        result = run_sandboxed("print('hi')", timeout=5, memory_mb=128,
                               backend="auto", allow_unsandboxed=True)
        assert result.status == "skipped_no_sandbox"
        assert result.backend == "none"

    def test_rlimit_without_optin_skips(self, monkeypatch):
        _force_available(monkeypatch, "rlimit")
        result = run_sandboxed("print('hi')", timeout=5, memory_mb=128,
                               backend="auto", allow_unsandboxed=False)
        assert result.status == "skipped_no_sandbox"
        assert result.backend == "rlimit"


@pytest.mark.skipif(not POSIX, reason="rlimit backend is POSIX-only")
class TestRlimitFallback:
    def _run(self, script, **kw):
        params = {"timeout": 5.0, "memory_mb": 256, "backend": "rlimit",
                  "allow_unsandboxed": True}
        params.update(kw)
        return run_sandboxed(script, **params)

    def test_normal_script_ok(self):
        result = self._run("print('hello')")
        assert result.status == "ok"
        assert result.stdout.strip() == "hello"

    def test_timeout_is_bounded(self):
        result = self._run("while True:\n    pass\n", timeout=1.0)
        assert result.status == "timeout"

    @pytest.mark.skipif(platform.system() == "Darwin",
                        reason="macOS does not enforce RLIMIT_AS")
    def test_memory_cap_blocks_huge_alloc(self):
        script = "x = bytearray(600 * 1024 * 1024)\nprint(len(x))\n"
        result = self._run(script, memory_mb=128)
        assert result.status != "ok"

    def test_fsize_cap_blocks_file_write(self):
        script = ("with open('out.txt', 'w') as f:\n"
                  "    f.write('x' * 100000)\n"
                  "print('wrote')\n")
        result = self._run(script)
        assert result.status != "ok"
        assert "wrote" not in result.stdout


def _docker_ready() -> bool:
    """True only if docker runs AND the image is already cached locally.

    Requiring the image avoids triggering a multi-minute pull inside the suite.
    """
    detect_backend.cache_clear()
    ready = detect_backend("docker") == "docker"
    detect_backend.cache_clear()
    if not ready:
        return False
    return sandbox._probe(["docker", "image", "inspect", sandbox._DEFAULT_IMAGE])


@pytest.mark.skipif(not _docker_ready(), reason="docker not available")
class TestDockerIsolation:
    def _run(self, script):
        return run_sandboxed(script, timeout=30.0, memory_mb=256,
                             backend="docker", allow_unsandboxed=False)

    def test_normal_runs(self):
        result = self._run("print(2 + 2)")
        assert result.status == "ok"
        assert result.stdout.strip() == "4"

    def test_filesystem_write_blocked(self):
        script = ("import sys\n"
                  "try:\n"
                  "    open('/etc/hosts', 'w').write('x')\n"
                  "    print('WROTE')\n"
                  "except OSError as e:\n"
                  "    print('BLOCKED', file=sys.stderr)\n")
        result = self._run(script)
        assert "WROTE" not in result.stdout

    def test_network_blocked(self):
        script = ("import urllib.request, sys\n"
                  "try:\n"
                  "    urllib.request.urlopen('http://example.com', timeout=5)\n"
                  "    print('CONNECTED')\n"
                  "except Exception:\n"
                  "    print('NO_NET', file=sys.stderr)\n")
        result = self._run(script)
        assert "CONNECTED" not in result.stdout
