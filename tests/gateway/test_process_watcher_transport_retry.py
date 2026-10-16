"""R1 transport-exception retry + R6 terminal-suppression settlement for the gateway watcher.

Drives the real ``_send_watcher_message`` / ``_watcher_deliver_text_notice`` /
``_run_process_watcher`` against stub adapters: a first-send ``OSError`` must retry
with backoff (never ``UnboundLocalError``), exhaustion must not ack, and off-mode /
ownership-suppressed exits must settle the outbox without sending (no re-arm on
cold restart, no handoff regression for surfaced jobs).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


def _build_runner(monkeypatch, tmp_path, mode="concise"):
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {mode}\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner


def _watcher_dict(session_id="proc_test"):
    return {
        "session_id": session_id, "check_interval": 0, "platform": "telegram",
        "chat_id": "123", "session_key": "agent:main:telegram:dm:123",
    }


def _exited_session(session_id="proc_test"):
    return SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="echo done",
        owner_task_id="t", task_id="t", started_at=None, supervision_state="ok",
        completion_reason="exited", termination_source="", last_supervision_error="",
        _unresolved_surfaced=False,
    )


class _FakeRegistry:
    def __init__(self, sessions, consumed=False):
        self._sessions = list(sessions)
        self._consumed = consumed

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return self._consumed


@pytest.mark.asyncio
async def test_send_retries_transport_exception_then_succeeds(monkeypatch, tmp_path):
    """R1: a first-send OSError retries with backoff and succeeds (no UnboundLocalError)."""
    import tools.process_registry as pr_module

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.side_effect = [OSError("transient outage"), SimpleNamespace(success=True)]

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    monkeypatch.setattr("gateway.run_notifications._WATCHER_SEND_RETRY_DELAY_S", 0.0)

    watcher = _watcher_dict()
    assert await runner._send_watcher_message("telegram", "123", "", "hello", watcher) is True
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_send_exhaustion_does_not_ack_and_later_tick_recovers(monkeypatch, tmp_path):
    """R1: exhausted exceptions return False without ack; the next tick still delivers."""
    import tools.process_registry as pr_module
    import tools.process_registry_delivery as delivery

    monkeypatch.setattr("gateway.run_notifications._WATCHER_SEND_RETRY_DELAY_S", 0.0)

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    session = _exited_session()
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry([session, session]))
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.side_effect = OSError("down")
    acked = []
    monkeypatch.setattr(delivery, "ack_delivery", acked.append)
    watcher = _watcher_dict()

    assert await runner._watcher_deliver_text_notice(
        watcher, session, watcher["session_id"], "telegram", "123", "", "concise") == "continue"
    assert acked == []

    adapter.send.side_effect = None
    adapter.send.return_value = SimpleNamespace(success=True)
    assert await runner._watcher_deliver_text_notice(
        watcher, session, watcher["session_id"], "telegram", "123", "", "concise") == "break"
    assert acked == [watcher["session_id"]]


@pytest.mark.asyncio
async def test_watcher_survives_first_send_exception(monkeypatch, tmp_path):
    """R1: the real watcher loop survives a first-send exception (no dead watcher)."""
    import tools.process_registry as pr_module

    monkeypatch.setattr("gateway.run_notifications._WATCHER_SEND_RETRY_DELAY_S", 0.0)

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry([_exited_session(), _exited_session()]))
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.side_effect = [OSError("boom"), SimpleNamespace(success=True)]

    await runner._run_process_watcher(_watcher_dict())
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_off_mode_settles_outbox_without_send_or_rearm(monkeypatch, tmp_path):
    """R6: off-mode exits ack the outbox without sending; cold restart re-arms nothing."""
    import tools.process_registry as pr_module
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools.process_registry_delivery import _delivery_dir, write_pending_delivery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants as hc
    monkeypatch.setattr(hc, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    import time
    session = ProcessSession(
        id="proc_off_settle", command="echo done", task_id="t", owner_task_id="t",
        session_key="k", parent_session_id="p", started_at=time.time(),
        output_buffer="done\n", watcher_interval=5, notify_on_complete=False,
        watcher_platform="telegram", watcher_chat_id="123", exit_code=0)
    session.exited = True
    registry._finished[session.id] = session
    assert write_pending_delivery(session) is True
    outbox = _delivery_dir() / "proc_off_settle.json"
    assert outbox.exists()

    runner = _build_runner(monkeypatch, tmp_path, "off")
    await runner._run_process_watcher(_watcher_dict("proc_off_settle"))

    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.assert_not_awaited()
    assert not outbox.exists()

    fresh = ProcessRegistry()
    assert fresh.recover_from_checkpoint() == 0
    assert [w for w in fresh.pending_watchers if w.get("session_id") == "proc_off_settle"] == []


@pytest.mark.asyncio
async def test_suppressed_child_settles_surfaced_child_delivers(monkeypatch, tmp_path):
    """R6: ownership-suppressed exits settle without send; surfaced ones still deliver."""
    import tools.process_registry as pr_module
    import tools.process_registry_notifications as notifications
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools.process_registry_delivery import _delivery_dir, write_pending_delivery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants as hc
    monkeypatch.setattr(hc, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    import time

    def _stage(sid):
        registry = ProcessRegistry()
        session = ProcessSession(
            id=sid, command="echo done", task_id="child", owner_task_id="parent",
            session_key="k", parent_session_id="p", started_at=time.time(),
            output_buffer="done\n", watcher_interval=5, notify_on_complete=False,
            watcher_platform="telegram", watcher_chat_id="123", exit_code=0)
        session.exited = True
        registry._finished[session.id] = session
        assert write_pending_delivery(session) is True
        return registry

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.return_value = SimpleNamespace(success=True)

    suppressed = _stage("proc_child_suppressed")
    monkeypatch.setattr(pr_module, "process_registry", suppressed)
    monkeypatch.setattr(notifications, "should_surface_notification", lambda _evt: False)
    await runner._run_process_watcher(_watcher_dict("proc_child_suppressed"))
    adapter.send.assert_not_awaited()
    assert not (_delivery_dir() / "proc_child_suppressed.json").exists()

    surfaced = _stage("proc_child_surfaced")
    monkeypatch.setattr(pr_module, "process_registry", surfaced)
    monkeypatch.setattr(notifications, "should_surface_notification", lambda _evt: True)
    await runner._run_process_watcher(_watcher_dict("proc_child_surfaced"))
    adapter.send.assert_awaited()
    assert not (_delivery_dir() / "proc_child_surfaced.json").exists()
