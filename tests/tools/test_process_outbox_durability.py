"""L1 outbox-durability: bounded synchronous retry inside the finisher.

The finisher keeps original ``mark_exited`` semantics (a raced kill keeps its
own code/reason) and retries the durable outbox write in place BEFORE the
query-only result save and final publication. One ``_finish_exited`` call
handles a transient disk failure autonomously: fail once then recover still
yields one completion event and one receipt, for the local reader and the
sandbox poller alike. Exhausted writes escalate to an explicit retained
``unresolved`` marker with in-memory notification evidence, never a silent
stop. Disk loss cannot promise disk durability; the checkpoint is best-effort
and the user-facing state stays honest.
"""
import time

import pytest

import tools.process_registry_delivery as delivery
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module
    import hermes_constants as hc
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(hc, "get_hermes_home", lambda: tmp_path)
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _staged(sid="proc_l1_stage", **fields):
    base = dict(
        id=sid, command="sleep 0.1; exit 7", task_id="t", owner_task_id="t",
        session_key="k", parent_session_id="p", started_at=time.time(),
        output_buffer="tail\n", watcher_interval=5, notify_on_complete=True,
    )
    base.update(fields)
    return ProcessSession(**base)


def _outbox_path(tmp_path, sid):
    return tmp_path / "logs" / "process-delivery" / f"{sid}.json"


def _result_path(tmp_path, sid):
    return tmp_path / "logs" / "process-results" / f"{sid}.json"


def test_transient_outbox_failure_recovers_autonomously_local(tmp_path, monkeypatch, isolated_home):
    """Local reader: one production ``_finish_exited`` survives fail-once."""
    import tools.process_registry_delivery as delivery_module
    real_write = delivery_module.write_pending_delivery
    calls = {"n": 0}

    def flaky(session):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real_write(session)

    monkeypatch.setattr(delivery_module, "write_pending_delivery", flaky)
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self, *a, **k: None)
    registry = ProcessRegistry()
    session = _staged("proc_l1_flaky_local")
    registry._running[session.id] = session

    registry._finish_exited(session, 7)

    assert calls["n"] == 2
    assert session.id not in registry._running
    assert session.id in registry._finished
    assert session.exited is True
    assert session.exit_code == 7
    assert session.completion_reason == "exited"
    assert _outbox_path(tmp_path, session.id).exists()
    assert _result_path(tmp_path, session.id).exists()
    assert registry.completion_queue.qsize() == 1
    assert registry.poll(session.id)["status"] == "exited"
    assert registry.poll(session.id)["exit_code"] == 7
    assert registry.wait(session.id, timeout=1)["status"] == "exited"

    registry._finish_exited(session, 7)
    assert registry.completion_queue.qsize() == 1


def test_transient_outbox_failure_recovers_autonomously_sandbox(tmp_path, monkeypatch, isolated_home):
    """Sandbox poller: one production ``_finish_env_exit`` survives fail-once."""
    import tools.process_registry_delivery as delivery_module
    real_write = delivery_module.write_pending_delivery
    calls = {"n": 0}

    def flaky(session):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real_write(session)

    monkeypatch.setattr(delivery_module, "write_pending_delivery", flaky)
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self, *a, **k: None)
    registry = ProcessRegistry()
    session = _staged("proc_l1_flaky_sandbox")
    registry._running[session.id] = session

    registry._finish_env_exit(session, 7)

    assert calls["n"] == 2
    assert session.id not in registry._running
    assert session.id in registry._finished
    assert session.exited is True
    assert session.exit_code == 7
    assert _outbox_path(tmp_path, session.id).exists()
    assert registry.completion_queue.qsize() == 1
    assert registry.poll(session.id)["status"] == "exited"


def test_permanent_outbox_failure_escalates_with_evidence(tmp_path, monkeypatch, isolated_home):
    """Exhausted writes finish honestly with retained unresolved evidence."""
    import tools.process_registry_delivery as delivery_module
    calls = {"n": 0}

    def always_fail(_session):
        calls["n"] += 1
        return False

    monkeypatch.setattr(delivery_module, "write_pending_delivery", always_fail)
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self, *a, **k: None)
    registry = ProcessRegistry()
    session = _staged("proc_l1_permanent")
    registry._running[session.id] = session

    bound = delivery_module.outbox_failure_bound()
    registry._finish_exited(session, 3)

    assert calls["n"] == bound
    assert session.id not in registry._running
    assert session.id in registry._finished
    assert session.exited is True
    assert session.exit_code == 3
    assert session.completion_reason == "exited"
    assert getattr(session, "supervision_state", "ok") == "unresolved"
    assert getattr(session, "_outbox_failures", 0) == bound
    assert "Pending delivery" in getattr(session, "last_supervision_error", "")
    assert not _outbox_path(tmp_path, session.id).exists()
    assert _result_path(tmp_path, session.id).exists()
    assert registry.completion_queue.qsize() == 1
    polled = registry.poll(session.id)
    assert polled["status"] == "exited"
    assert polled["exit_code"] == 3
    assert polled.get("supervision_state") == "unresolved"
    assert registry.wait(session.id, timeout=1)["status"] == "exited"


def test_crash_recovery_replays_durable_outbox_once(tmp_path, monkeypatch, isolated_home):
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self, *a, **k: None)
    registry = ProcessRegistry()
    session = _staged("proc_l1_replay")
    registry._running[session.id] = session
    registry._finish_exited(session, 9)
    assert _outbox_path(tmp_path, session.id).exists()
    assert registry.completion_queue.qsize() == 1

    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() >= 1
    assert restarted.get(session.id) is not None
    watchers = [w for w in restarted.pending_watchers if w.get("session_id") == session.id]
    assert len(watchers) == 1
    before = len(restarted.pending_watchers)
    restarted.recover_from_checkpoint()
    after = len([w for w in restarted.pending_watchers if w.get("session_id") == session.id])
    assert after == 1
    assert len(restarted.pending_watchers) == before


def test_finish_preserves_kill_receipt(monkeypatch, isolated_home):
    """A raced kill keeps its own code/reason through the durable finish."""
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self, *a, **k: None)
    registry = ProcessRegistry()
    session = _staged("proc_l1_killwins")
    session.exited = True
    session.exit_code = -15
    session.completion_reason = "killed"
    session.termination_source = "process.kill"
    registry._running[session.id] = session
    registry._finish_exited(session, 0)
    assert (session.exit_code, session.completion_reason, session.termination_source) == (
        -15, "killed", "process.kill")
