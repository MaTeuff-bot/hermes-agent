"""Durable supervision for env-backed background processes.

Integration-style: a real subprocess backend (LocalEnvironment, which spawns a
real bash per execute call like a Docker-exec transport) wrapped to emulate
Docker-exec semantics (per-run container id, fault injection). Short supervision
constants so the suite stays fast. Covers the base failure: a transient
transport outage during a live job must not fabricate an exit, must launch the
command exactly once, and must deliver the real receipt.
"""
import os
import signal
import time
import uuid

import pytest

import tools.process_registry_supervision as sup
from tools.process_registry import ProcessRegistry
from tools.environments.local import LocalEnvironment


def _short_supervision(monkeypatch):
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.05)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 4)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_CAP_S", 0.1)


class SandboxLikeEnv:
    """LocalEnvironment with Docker-exec-like identity + fault injection.

    Delegates every command to a real subprocess (real bash, real background
    job). ``_container_id`` is the continuity proof. ``fail_probe_calls`` raises
    a transient transport error on probe commands only — the launch command
    (containing ``nohup``) always passes through exactly once, so tests prove no
    relaunch of the original job. ``nonzero_probe_calls`` returns a nonzero
    transport instead of raising.
    """

    def __init__(self):
        self.inner = LocalEnvironment()
        self._container_id = "ctr-" + uuid.uuid4().hex[:12]
        self.launch_count = 0
        self.probe_count = 0
        self.fail_probe_calls = 0
        self.nonzero_probe_calls = 0
        self.always_fail_probes = False
        self.commands = []

    def get_temp_dir(self):
        return self.inner.get_temp_dir()

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if "nohup" in command:
            self.launch_count += 1
            return self.inner.execute(command, **kwargs)
        self.probe_count += 1
        if self.always_fail_probes:
            raise RuntimeError("injected persistent transport outage")
        if self.fail_probe_calls > 0:
            self.fail_probe_calls -= 1
            raise RuntimeError("injected transient transport outage")
        if self.nonzero_probe_calls > 0:
            self.nonzero_probe_calls -= 1
            return {"output": "injected nonzero transport", "returncode": 1}
        return self.inner.execute(command, **kwargs)


def _wait_for(predicate, timeout=25.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _cleanup(registry, session):
    try:
        pid = getattr(session, "sandbox_pid", None) or getattr(session, "pid", None)
        if pid:
            with open(f"/proc/{pid}/cmdline", "rb"):
                pass
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def test_transient_probe_outage_then_real_success_single_launch(monkeypatch):
    """Base-failure repro: transport exceptions mid-run are uncertainty, not exit.

    Three injected probe outages while ``sleep 2`` runs; the job must still
    finish with its real exit 0 and the command must have launched exactly once
    (no relaunch of the original job on ambiguous supervision).
    """
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    env.fail_probe_calls = 3
    session = registry.spawn_via_env(env, "sleep 2; exit 0", timeout=15)
    try:
        assert not session.exited
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done, "live job never delivered its real receipt after transient outage"
        assert session.exit_code == 0, f"phantom exit: {session.exit_code!r}"
        assert session.completion_reason == "exited"
        assert env.launch_count == 1, f"original job relaunched {env.launch_count}x"
    finally:
        _cleanup(registry, session)


def test_nonzero_transport_is_uncertain_not_exit(monkeypatch):
    """Nonzero probe transports retry; the later real receipt still wins."""
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    env.nonzero_probe_calls = 2
    session = registry.spawn_via_env(env, "echo ok; exit 0", timeout=15)
    try:
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done
        assert session.exit_code == 0
        assert env.launch_count == 1
    finally:
        _cleanup(registry, session)


def test_nonzero_job_exit_surfaced_honestly(monkeypatch):
    """A real nonzero wrapper receipt is a real exit (not unresolved, not 0)."""
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "exit 3", timeout=15)
    try:
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done
        assert session.exit_code == 3
        assert session.completion_reason == "exited"
        assert getattr(session, "supervision_state", "ok") != "unresolved"
    finally:
        _cleanup(registry, session)


@pytest.mark.parametrize("output,token,expected", [
    ("", "tok", None),
    ("garbage\n", "tok", None),
    ("tok\n", "tok", None),
    ("other\n0\n", "tok", None),
    ("tok\nnotanint\n", "tok", None),
    ("tok\n0\n", "tok", 0),
    ("tok\n3\n", "tok", 3),
    ("__NO_RECEIPT__\n", "tok", None),
])
def test_malformed_exit_never_fabricates_zero(output, token, expected):
    assert sup.parse_verified_exit(output, token) is expected


def test_budget_timeout_owned_by_wrapper_without_polling(monkeypatch):
    """The job budget fires inside the wrapper even if nobody polls.

    ``sleep 30`` with ``budget_seconds=1`` must record the wrapper's 124 as
    ``budget_timeout`` (non-success) — never overwritten by a later child exit.
    """
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "sleep 30", timeout=15, budget_seconds=1)
    try:
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done, "wrapper-owned budget never fired"
        assert session.exit_code == 124
        assert session.completion_reason == "budget_timeout"
    finally:
        _cleanup(registry, session)


def test_unrecoverable_transport_escalates_unresolved_no_false_exit(monkeypatch):
    """Persistent outage: durable ``unresolved``, retained evidence, no faked exit."""
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "sleep 30; exit 0", timeout=15)
    assert not session.exited
    env.always_fail_probes = True
    try:
        done = _wait_for(
            lambda: getattr(session, "supervision_state", "") == "unresolved", timeout=25.0)
        assert done, "unrecoverable supervision never escalated"
        assert not session.exited, "unresolved run must not be finished with an invented exit"
        assert session.exit_code is None
        assert session.completion_reason == "unresolved"
        assert session.last_supervision_error, "evidence for operator triage required"
        assert env.launch_count == 1, "original job must never be relaunched"
    finally:
        _cleanup(registry, session)


def test_pid_number_alone_never_proves_identity():
    """PIDs recycle: without a token match the verdict is never ``match``."""
    live = SandboxLikeEnv()
    live._container_id = "ctr-aaa"
    assert sup.sandbox_identity_verdict("ctr-aaa", 1234, live, 1234, "", None) == "unknown"
    assert sup.sandbox_identity_verdict("", 1234, live, 1234, "", None) == "unknown"
    other = SandboxLikeEnv()
    other._container_id = "ctr-bbb"
    assert sup.sandbox_identity_verdict("ctr-aaa", 1234, other, 1234, "", None) == "mismatch"


def test_token_mismatch_is_replacement_not_continuity():
    """Same PID number, different run token: a reused PID, never the same run."""
    live = SandboxLikeEnv()
    live._container_id = "ctr-same"
    assert sup.sandbox_identity_verdict(
        "ctr-same", 4242, live, 4242, "token-run-a", "token-run-b") == "mismatch"
    assert sup.sandbox_identity_verdict(
        "ctr-same", 4242, live, 4242, "token-run-a", "token-run-a") == "match"


def test_container_replacement_refuses_reattach(monkeypatch):
    """A replaced container (new id, same paths) must not resume the old run."""
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "sleep 30; exit 0", timeout=15)
    env.always_fail_probes = True
    try:
        assert _wait_for(
            lambda: getattr(session, "supervision_state", "") == "unresolved", timeout=25.0)
        replacement = SandboxLikeEnv()
        assert replacement._container_id != env._container_id
        assert registry.reattach_env_session(session.id, replacement) is False
        assert getattr(session, "supervision_state", "") == "unresolved"
        assert not session.exited
    finally:
        _cleanup(registry, session)


def test_checkpoint_recovery_retains_unresolved_and_reattach_resumes(monkeypatch, tmp_path):
    """Restart recovery: sandbox entries are retained as durable ``unresolved``
    (never dropped, never host-signalled) and resume in the same container."""
    import hermes_constants
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "sleep 20; exit 7", timeout=15)
    assert not session.exited
    session_id = session.id
    try:
        registry._write_checkpoint()
        restarted = ProcessRegistry()
        recovered = restarted.recover_from_checkpoint()
        assert recovered >= 1, "sandbox run must be retained across restart, not dropped"
        got = restarted.get(session_id)
        assert got is not None
        assert getattr(got, "supervision_state", "") == "unresolved"
        assert not got.exited
        assert got.pid is None, "no host PID to signal for a sandbox run"
        assert got.completion_reason == "unresolved"
        assert got.env_exit_path and got.env_token_path
        assert restarted.reattach_env_session(session_id, env) is True
    finally:
        _cleanup(registry, session)
