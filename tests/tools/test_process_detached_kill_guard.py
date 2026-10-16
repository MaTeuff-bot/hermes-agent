"""R4 recovered host kill guard: unverifiable identities are refused, never signalled.

Drives the real ``_signal_kill_detached`` with an intercepted ``_terminate_host_pid``
(zero real signals by construction): an unreadable start-time probe or a missing
baseline refuses and keeps the session unresolved, a reused PID closes without a
signal, and a proven identity passes its baseline through to termination.
"""
import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def isolated_registry(monkeypatch):
    import tools.process_registry as pr_module
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _detached(sid="proc_r4_det", pid=999999, host_start_time=11111):
    return ProcessSession(
        id=sid, command="sleep 999", task_id="t", owner_task_id="o",
        session_key="s", started_at=0.0, pid=pid, pid_scope="host",
        detached=True, host_start_time=host_start_time)


def test_unreadable_probe_refuses_with_zero_signals(isolated_registry, monkeypatch):
    """Alive PID, unreadable start time: refuse, unresolved, no signal sent."""
    import tools.process_registry as pr_module

    session = _detached()
    isolated_registry._running[session.id] = session
    calls = []
    monkeypatch.setattr(ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: None))
    monkeypatch.setattr(
        ProcessRegistry, "_terminate_host_pid",
        classmethod(lambda _cls, pid, expected_start=None: calls.append((pid, expected_start))))

    result = isolated_registry._signal_kill_detached(session, session.id, consume_output=False)

    assert calls == []
    assert result is not None and result.get("status") == "error"
    assert "unverifiable" in result.get("error", "")
    assert session.supervision_state == "unresolved"
    assert not session.exited


def test_missing_baseline_refuses_with_zero_signals(isolated_registry, monkeypatch):
    """Legacy checkpoint without a baseline: refuse even when the PID is alive."""
    session = _detached(sid="proc_r4_legacy", host_start_time=None)
    isolated_registry._running[session.id] = session
    calls = []
    monkeypatch.setattr(ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: 22222))
    monkeypatch.setattr(
        ProcessRegistry, "_terminate_host_pid",
        classmethod(lambda _cls, pid, expected_start=None: calls.append((pid, expected_start))))

    result = isolated_registry._signal_kill_detached(session, "proc_r4_legacy", consume_output=False)

    assert calls == []
    assert result is not None and result.get("status") == "error"
    assert session.supervision_state == "unresolved"


def test_reused_pid_closes_without_signal(isolated_registry, monkeypatch):
    """Positively mismatched start time: already-exited close, never a signal."""
    session = _detached(sid="proc_r4_reused")
    isolated_registry._running[session.id] = session
    calls = []
    monkeypatch.setattr(ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: 99999))
    monkeypatch.setattr(
        ProcessRegistry, "_terminate_host_pid",
        classmethod(lambda _cls, pid, expected_start=None: calls.append((pid, expected_start))))

    result = isolated_registry._signal_kill_detached(session, "proc_r4_reused", consume_output=False)

    assert calls == []
    assert result is not None and result.get("status") == "already_exited"


def test_proven_identity_passes_baseline_through(isolated_registry, monkeypatch):
    """Matching start time: exactly one termination call carrying the baseline."""
    session = _detached(sid="proc_r4 proven".replace(" ", "_"))
    isolated_registry._running[session.id] = session
    calls = []
    monkeypatch.setattr(ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: 11111))
    monkeypatch.setattr(
        ProcessRegistry, "_terminate_host_pid",
        classmethod(lambda _cls, pid, expected_start=None: calls.append((pid, expected_start))))

    result = isolated_registry._signal_kill_detached(session, session.id, consume_output=False)

    assert result is None
    assert calls == [(999999, 11111)]
