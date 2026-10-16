"""Idle-reaper container-scope regression + D2/D4/PID guards.

Real reaper + real Docker cleanup with a subprocess-backed fake exec (no
Docker engine): the fake subclasses DockerEnvironment only for its
_container_id identity/cleanup, overrides execute to a local backend. Forced
fake-idle clock, no 300s sleep. Asserts the leased handle is not cleared, one
launch, true exit, and owner vs container scope separation.
"""
import os
import signal
import time
import uuid

import pytest

import tools.process_registry_supervision as sup
from tools.environments.docker import DockerEnvironment
from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry
from tests.waiting import wait_until


def _short_supervision(monkeypatch):
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.05)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 4)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_CAP_S", 0.1)


class PersistentDockerFake(DockerEnvironment):
    """Subprocess-backed Docker-identity fake (no engine).

    Subclasses DockerEnvironment only for _container_id identity and the real
    cleanup() persist path. execute is overridden to a local backend and
    asserts the handle, so idle-reaper invalidation surfaces as
    'Container not started'. _docker_exe/cwd are intentionally absent so the
    pinned non-recreating path uses this execute directly (real engines always
    carry Base execution state and still pin).
    """

    def __init__(self):
        self.inner = LocalEnvironment()
        self._container_id = "disposable-container-identity"
        self._persist_across_processes = True
        self._persistent = True
        self._session_scoped = False
        self.launch_count = 0

    def get_temp_dir(self):
        return self.inner.get_temp_dir()

    def execute(self, command, **kwargs):
        assert self._container_id, "Container not started"
        if "nohup" in command:
            self.launch_count += 1
        return self.inner.execute(command, **kwargs)


def _cleanup_session(registry, session):
    try:
        try:
            registry.kill_process(session.id)
        except Exception:
            pass
        pid = getattr(session, "sandbox_pid", None) or getattr(session, "pid", None)
        if pid is not None:
            try:
                with open(f"/proc/{pid}/cmdline", "rb"):
                    pass
            except OSError:
                return
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def test_idle_reaper_keeps_leased_container(monkeypatch):
    """Leased default container survives a forced-idle reaper with true exit."""
    _short_supervision(monkeypatch)
    import tools.process_registry as pr_module
    import tools.terminal_tool as tt
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs

    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    env = PersistentDockerFake()
    session = registry.spawn_via_env(
        env, "sleep 2; exit 7", task_id="default",
        owner_task_id="raw-session-owner", timeout=10, persist_on_release=True)
    assert not session.exited
    monkeypatch.setitem(tt._active_environments, "default", env)
    monkeypatch.setitem(tt._last_activity, "default", time.time() - 301)
    assert registry.has_active_processes("default") is False
    assert registry.has_active_processes("raw-session-owner") is True
    assert registry.has_active_for_container("default", env) is True
    assert registry.has_active_for_container("default") is True
    _cleanup_inactive_envs(300)
    assert env._container_id == "disposable-container-identity"
    assert "default" in tt._active_environments
    try:
        wait_until(lambda: session.exited, timeout=15.0, what="leased job true exit")
        assert session.exit_code == 7
        assert session.completion_reason == "exited"
        assert env.launch_count == 1
    finally:
        _cleanup_session(registry, session)


def test_idle_reaper_cleans_truly_idle(monkeypatch):
    """No leased jobs: a stale env is still reaped."""
    import tools.terminal_tool as tt
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    env = PersistentDockerFake()
    monkeypatch.setitem(tt._active_environments, "default", env)
    monkeypatch.setitem(tt._last_activity, "default", time.time() - 301)
    assert registry.has_active_for_container("default", env) is False
    _cleanup_inactive_envs(300)
    assert "default" not in tt._active_environments


def test_owner_scope_preserved_for_delegated_lifetime():
    """has_active_processes stays owner-scoped; container query is separate."""
    from tools.process_registry import ProcessSession
    registry = ProcessRegistry()
    env = PersistentDockerFake()
    session = ProcessSession(
        id="proc_owner_scope", command="sleep 2", task_id="default",
        owner_task_id="child-owner", session_key="k", env_ref=env)
    registry._running[session.id] = session
    try:
        assert registry.has_active_processes("default") is False
        assert registry.has_active_processes("child-owner") is True
        assert registry.has_active_for_container("default", env) is True
        assert registry.has_active_for_container("default") is True
        assert registry.has_active_for_container("other-key", None) is False
        other_env = PersistentDockerFake()
        assert registry.has_active_for_container("other-key", other_env) is False
        assert registry.has_active_for_container("other-key", env) is True
        assert session.task_id == "default"
        assert session.owner_task_id == "child-owner"
    finally:
        registry._running.pop(session.id, None)


def test_outbox_disk_failure_escalates_with_evidence(monkeypatch, tmp_path):
    """Permanent outbox failure finishes honestly with retained evidence.

    Bounded synchronous retry inside the production finisher, then an explicit
    ``unresolved`` marker: the real exit is preserved, the query-only result is
    retained best-effort, one completion event is queued (never a silent stop),
    and poll stays honest. No endless poll or watcher CPU.
    """
    import hermes_constants
    import tools.process_registry_delivery as delivery

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    registry = ProcessRegistry()
    env = PersistentDockerFake()
    session = registry.spawn_via_env(env, "exit 0", task_id="t", timeout=10)
    wait_until(lambda: session.exited, timeout=15.0, what="job exit")
    assert session.exit_code == 0
    fresh = ProcessRegistry()
    from tools.process_registry import ProcessSession
    pending = ProcessSession(
        id="proc_disk_fail", command="echo hi", task_id="t",
        owner_task_id="t", started_at=time.time(),
        output_buffer="hi\n", exited=False,
        watcher_interval=5, notify_on_complete=True,
        session_key="k", parent_session_id="p")
    fresh._running[pending.id] = pending
    calls = {"n": 0}

    def always_fail(_s):
        calls["n"] += 1
        return False

    monkeypatch.setattr(delivery, "write_pending_delivery", always_fail)
    bound = delivery.outbox_failure_bound()
    fresh._finish_exited(pending, 0)
    assert calls["n"] == bound
    assert pending.id not in fresh._running
    assert pending.id in fresh._finished
    assert pending.exited is True
    assert pending.exit_code == 0
    assert pending.completion_reason == "exited"
    assert getattr(pending, "supervision_state", "ok") == "unresolved"
    assert fresh.completion_queue.qsize() == 1
    polled = fresh.poll(pending.id)
    assert polled["status"] == "exited"
    assert polled.get("supervision_state") == "unresolved"


def test_expired_outbox_retained_and_escalated(monkeypatch, tmp_path):
    """Expired undelivered outbox is retained with escalation, never deleted."""
    import hermes_constants
    import tools.process_registry_delivery as delivery

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    from tools.process_registry import ProcessSession
    session = ProcessSession(
        id="proc_expired_retain", command="echo hi", task_id="t",
        owner_task_id="t", started_at=time.time(), exit_code=0,
        completion_reason="exited", output_buffer="hi\n", exited=True,
        watcher_interval=5, notify_on_complete=True,
        session_key="k", parent_session_id="p")
    assert delivery.write_pending_delivery(session) is True
    path = delivery._delivery_dir() / f"{session.id}.json"
    assert path.exists()
    old = time.time() - (8 * 24 * 60 * 60)
    os.utime(path, (old, old))
    records = delivery.list_pending_deliveries()
    assert any(r.get("session_id") == session.id and r.get("expired") is True for r in records)
    assert path.exists()
    filtered = delivery.list_pending_deliveries(include_expired=False)
    assert all(r.get("session_id") != session.id for r in filtered)
    assert path.exists()


def test_signal_verdict_ticks_binding():
    """Start-ticks bind the live PID: reuse is mismatch, unreadable is unknown."""
    live = PersistentDockerFake()
    assert sup.sandbox_signal_verdict(
        "disposable-container-identity", 4242, live, 4242,
        "tok-a", "tok-a", "111", "111", "", None) == "match"
    assert sup.sandbox_signal_verdict(
        "disposable-container-identity", 4242, live, 4242,
        "tok-a", "tok-a", "111", "222", "", None) == "mismatch"
    assert sup.sandbox_signal_verdict(
        "disposable-container-identity", 4242, live, 4242,
        "tok-a", "tok-a", "111", None, "", None) == "unknown"
    assert sup.sandbox_signal_verdict(
        "disposable-container-identity", 4242, live, 4242,
        "tok-a", "tok-a", "", "222", "", None) == "unknown"


def test_reattach_refuses_mismatch_without_env_update(monkeypatch, tmp_path):
    """Failed reattach never swaps env_ref; success swaps once verified."""
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    env = PersistentDockerFake()
    session = registry.spawn_via_env(env, "sleep 2; exit 0", timeout=10)
    session_id = session.id
    old_ref = session.env_ref
    try:
        registry._write_checkpoint()
        restarted = ProcessRegistry()
        restarted.recover_from_checkpoint()
        got = restarted.get(session_id)
        assert got is not None
        replacement = PersistentDockerFake()
        replacement._container_id = "replacement-container"
        assert restarted.reattach_env_session(session_id, replacement) is False
        assert restarted.get(session_id).env_ref is not replacement
        assert restarted.get(session_id).env_ref is None or restarted.get(session_id).env_ref is not replacement
        assert old_ref is not None
    finally:
        _cleanup_session(registry, session)


def test_pinned_bypass_taken_for_recreating_fake(monkeypatch):
    """Recreating backends bypass execute via Base (no adopt of fresh files)."""
    from tools.environments.base import BaseEnvironment
    seen = {}

    class RecreatingFake:
        _container_id = "ctr-pinned"
        _recreate_container = lambda self: True

        def execute(self, command, **kwargs):
            seen["via"] = "fake"
            return {"output": "fake", "returncode": 0}

    def _base_execute(self, command, timeout=None, **kwargs):
        seen["via"] = "base"
        return {"output": "pinned", "returncode": 0}

    monkeypatch.setattr(BaseEnvironment, "execute", _base_execute)
    fake = RecreatingFake()
    fake._recreate_container = lambda: True
    import tools.process_registry_supervision as supmod
    out = supmod._pinned_env_exec(fake, "cat /tmp/x", 5)
    assert out["output"] == "pinned"
    assert seen.get("via") == "base"
