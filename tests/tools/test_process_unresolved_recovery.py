"""R2 durable unresolved-finish route: outbox failure with a writable checkpoint/result.

Drives the production finisher with a failing outbox: the session must finish with
its real exit preserved and unresolved evidence, the checkpoint must carry a
replayable ``unresolved_finish`` record, and a cold restart must rebuild the
finished session and re-arm an explicit owner escalation (never a re-run, never a
query-only orphan). Also covers the stale-entry double-adoption guard and the
all-storage-failure honesty case.
"""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def _staged(sid="proc_r2_stage", **fields):
    base = dict(
        id=sid, command="echo hi; exit 3", task_id="t", owner_task_id="t",
        session_key="k", parent_session_id="p", started_at=time.time(),
        output_buffer="hello tail\n", watcher_interval=5, notify_on_complete=True,
        watcher_platform="telegram", watcher_chat_id="123",
        watcher_user_id="u1", watcher_user_name="bob",
        watcher_thread_id="", watcher_message_id="m1",
    )
    base.update(fields)
    return ProcessSession(**base)


def _finish_with_dead_outbox(monkeypatch, isolated_home, sid, exit_code=3):
    monkeypatch.setattr(delivery, "write_pending_delivery", lambda _s: False)
    session = _staged(sid)
    isolated_home._running[session.id] = session
    isolated_home._finish_exited(session, exit_code)
    return session


def test_finish_with_dead_outbox_keeps_exit_and_checkpoint_evidence(
        tmp_path, monkeypatch, isolated_home):
    """Permanent outbox failure finishes honestly: exit kept, checkpoint carries replay."""
    import json

    session = _finish_with_dead_outbox(monkeypatch, isolated_home, "proc_r2_evidence")
    assert session.exited and session.exit_code == 3
    assert session.supervision_state == "unresolved"
    assert session._outbox_failures == delivery.outbox_failure_bound()
    assert "proc_r2_evidence" in isolated_home._finished
    assert not (tmp_path / "logs" / "process-delivery" / "proc_r2_evidence.json").exists()
    assert (tmp_path / "logs" / "process-results" / "proc_r2_evidence.json").exists()

    entries = json.loads((tmp_path / "processes.json").read_text(encoding="utf-8"))
    matches = [e for e in entries if e.get("session_id") == "proc_r2_evidence"]
    assert len(matches) == 1
    assert matches[0].get("unresolved_finish") is True
    assert matches[0].get("exit_code") == 3

    note = isolated_home.poll("proc_r2_evidence").get("supervision_note", "")
    assert "Exit 3 was recorded" in note
    assert "no exit has been recorded" not in note


def test_cold_restart_rebuilds_finished_and_rearms_escalation(tmp_path, monkeypatch, isolated_home):
    """A new registry recovers the finished-unresolved session plus an escalation watcher."""
    _finish_with_dead_outbox(monkeypatch, isolated_home, "proc_r2_replay")

    import tools.process_registry as pr_module
    fresh = pr_module.ProcessRegistry()
    assert fresh.recover_from_checkpoint() >= 1
    session = fresh.get("proc_r2_replay")
    assert session is not None and session.exited and session.exit_code == 3
    assert session.supervision_state == "unresolved"
    watchers = [w for w in fresh.pending_watchers if w.get("session_id") == "proc_r2_replay"]
    assert len(watchers) == 1
    assert watchers[0].get("unresolved_escalation_replay") is True


@pytest.mark.asyncio
async def test_rearmed_watcher_sends_explicit_escalation_only(tmp_path, monkeypatch, isolated_home):
    """The replay watcher sends one explicit escalation, never a normal completion."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    import tools.process_registry as pr_module

    _finish_with_dead_outbox(monkeypatch, isolated_home, "proc_r2_watch")
    fresh = pr_module.ProcessRegistry()
    assert fresh.recover_from_checkpoint() >= 1
    monkeypatch.setattr(pr_module, "process_registry", fresh)

    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: concise\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    adapter.send.return_value = SimpleNamespace(success=True)
    runner.adapters[Platform.TELEGRAM] = adapter

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    watcher = dict(
        next(w for w in fresh.pending_watchers if w.get("session_id") == "proc_r2_watch"))
    watcher["check_interval"] = 0
    await runner._run_process_watcher(watcher)

    assert adapter.send.await_count == 1
    text = adapter.send.await_args.args[1]
    assert "supervision LOST" in text
    assert "Exit 3 was recorded" in text


def test_stale_checkpoint_entry_after_outbox_replay_adopts_nothing(
        tmp_path, monkeypatch, isolated_home):
    """Crash between outbox write and checkpoint rewrite: no RUNNING twin, one watcher."""
    import json
    import tools.process_registry as pr_module

    session = _staged("proc_r2_stale")
    isolated_home._running[session.id] = session
    assert delivery.write_pending_delivery(session) is True
    ckpt = tmp_path / "processes.json"
    ckpt.write_text(json.dumps([{
        "session_id": "proc_r2_stale", "command": "echo hi", "pid": 999999,
        "pid_scope": "host", "host_start_time": None, "task_id": "t",
        "owner_task_id": "t", "session_key": "k", "watcher_interval": 5,
        "parent_session_id": "p", "notify_on_complete": True,
    }]), encoding="utf-8")

    fresh = pr_module.ProcessRegistry()
    with monkeypatch.context() as patcher:
        patcher.setattr(pr_module.ProcessRegistry, "_is_host_pid_alive", lambda _pid: False)
        assert fresh.recover_from_checkpoint() >= 1
    assert "proc_r2_stale" in fresh._finished
    assert "proc_r2_stale" not in fresh._running
    assert len([w for w in fresh.pending_watchers
                if w.get("session_id") == "proc_r2_stale"]) == 1


def test_all_storage_failure_stays_honest_and_best_effort(tmp_path, monkeypatch, isolated_home):
    """Outbox + result + checkpoint all failing: in-memory unresolved finish, loudly kept."""
    import utils

    def _dead_disk(*_a, **_kw):
        raise OSError("disk gone")
    monkeypatch.setattr(utils, "atomic_json_write", _dead_disk)

    session = _staged("proc_r2_diskless")
    isolated_home._running[session.id] = session
    isolated_home._finish_exited(session, 3)

    assert session.exited and session.exit_code == 3
    assert session.supervision_state == "unresolved"
    assert "proc_r2_diskless" in isolated_home._finished
    assert isolated_home.completion_queue.qsize() == 1
    assert not (tmp_path / "processes.json").exists()


def test_acked_delivery_does_not_rearm_on_restart(tmp_path, monkeypatch, isolated_home):
    """An acked outbox plus a query-only result must not re-arm a watcher on restart."""
    import tools.process_registry as pr_module

    session = _staged("proc_r2_acked")
    isolated_home._running[session.id] = session
    isolated_home._finish_exited(session, 0)
    outbox = tmp_path / "logs" / "process-delivery" / "proc_r2_acked.json"
    assert outbox.exists()
    delivery.ack_delivery(session.id)
    assert not outbox.exists()
    assert (tmp_path / "logs" / "process-results" / "proc_r2_acked.json").exists()

    fresh = pr_module.ProcessRegistry()
    fresh.recover_from_checkpoint()
    assert fresh.get("proc_r2_acked") is None
    assert not [w for w in fresh.pending_watchers if w.get("session_id") == "proc_r2_acked"]

