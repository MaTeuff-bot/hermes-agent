"""R5 prune/delivery decoupling: pending or unsurfaced-unresolved finishes survive pruning.

Drives the production ``_prune_if_needed``: an outage-aged finished session with an
unacked outbox (or unsurfaced unresolved evidence) stays resolvable via ``get`` so
the in-process watcher retries without a restart or user ping. Ordinary expired
sessions still prune, and the 7-day expired-outbox retention contract is preserved.
"""
import time

import pytest

import tools.process_registry_delivery as delivery
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module
    import hermes_constants as hc
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(hc, "get_hermes_home", lambda: tmp_path)
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _finished(sid, age_s, **fields):
    base = dict(
        id=sid, command="echo done", task_id="t", owner_task_id="t",
        session_key="k", parent_session_id="p", started_at=time.time() - age_s,
        output_buffer="done\n", watcher_interval=5, notify_on_complete=True,
        watcher_platform="telegram", watcher_chat_id="123", exit_code=0)
    base.update(fields)
    session = ProcessSession(**base)
    session.exited = True
    return session


def test_outage_prune_then_send_recovers_in_process(tmp_path, monkeypatch, isolated_home):
    """Outage-aged undelivered finish survives prune and stays watcher-resolvable."""
    aged = _finished("proc_r5_pending", age_s=4000)
    isolated_home._running[aged.id] = aged
    isolated_home._finish_exited(aged, 0)
    outbox = tmp_path / "logs" / "process-delivery" / "proc_r5_pending.json"
    assert outbox.exists()

    old = _finished("proc_r5_ordinary", age_s=4000, watcher_interval=0,
                    notify_on_complete=False)
    isolated_home._finished[old.id] = old

    with isolated_home._lock:
        isolated_home._prune_if_needed()

    assert isolated_home.get("proc_r5_pending") is not None
    assert outbox.exists()
    assert isolated_home.get("proc_r5_ordinary") is None


def test_unsurfaced_unresolved_finish_survives_prune(tmp_path, monkeypatch, isolated_home):
    """A failed-outbox unresolved finish is not pruned before its escalation lands."""
    monkeypatch.setattr(delivery, "write_pending_delivery", lambda _s: False)
    session = _finished("proc_r5_unresolved", age_s=4000)
    isolated_home._running[session.id] = session
    isolated_home._finish_exited(session, 0)
    assert session.supervision_state == "unresolved"

    with isolated_home._lock:
        isolated_home._prune_if_needed()

    assert isolated_home.get("proc_r5_unresolved") is not None


def test_expired_outbox_retained_not_deleted(tmp_path, monkeypatch, isolated_home):
    """The 7-day retention policy still holds: expired records are kept + flagged."""
    aged = _finished("proc_r5_expired", age_s=10)
    isolated_home._running[aged.id] = aged
    isolated_home._finish_exited(aged, 0)
    outbox = tmp_path / "logs" / "process-delivery" / "proc_r5_expired.json"
    assert outbox.exists()

    monkeypatch.setattr(delivery, "DELIVERY_RETENTION_SECONDS", -1)
    records = delivery.list_pending_deliveries()
    assert [r for r in records if r.get("session_id") == "proc_r5_expired"
            and r.get("expired") is True]
    assert outbox.exists()
