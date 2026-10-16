"""Cross-restart durable admission: accepted-before-ack, single admission replay.

Agent-notify outbox ack follows durable admission (the admitted turn carries the
result); text-notice ack follows send success. A crash between admission and ack
replays the outbox on restart but the durable producer identity dedupes
re-admission for the same producer event. No live Slack, no Docker engine, no model
inference, no outbound send claimed.

TIERS (honest labels): the first two tests are seam-level -- a stubbed
``completion_admission`` and an AsyncMock ``handle_message`` transport
acceptance, which proves outbox/ack sequencing, NOT workflow resumption.
``test_process_completion_native_admission_exactly_once_across_restart`` is native
admission processing proof (real SessionDB-backed authority, real
``admit_automation`` identity, real ``execute_admission`` into a stubbed turn
handler with ``_schedule`` disabled and an explicit manual ``execute_admission``
call): it is NOT autonomous-scheduler / no-user-ping root-job proof, and its
exactly-once scope is one admission replay for the same producer event, not
exactly-once external actions.
"""
import asyncio
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tools.process_registry_supervision as sup
from tools.process_registry import ProcessRegistry, ProcessSession
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.platforms.base import SendResult
from tests.waiting import wait_until


def _short_supervision(monkeypatch):
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.05)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 4)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_CAP_S", 0.1)


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


class _SessionDB:
    def __init__(self, row):
        self._row = row

    async def get_session(self, session_id):
        return self._row

    async def get_compression_tip(self, session_id):
        return None


def _runner(adapter, *, session_db):
    async def _admit(event):
        event._gateway_accepted = True
    adapter.handle_message.side_effect = _admit
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._session_db = session_db
    runner._background_tasks = set()
    runner._completion_notification_batches = {}
    runner._completion_notification_batch_tasks = {}
    runner._completion_notification_batch_flush_tasks = set()
    runner._completion_notification_batch_window = 0.05
    runner._completion_notification_batches_stopping = False
    return runner


def _watcher_for(session_id, session_key="agent:main:telegram:dm:123", parent="sess-live"):
    return {
        "session_id": session_id,
        "check_interval": 0.1,
        "session_key": session_key,
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
        "parent_session_id": parent,
    }


def _run_watcher_sync(monkeypatch, runner, watcher):
    async def _instant_sleep(*_a, **_kw):
        return None
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    return asyncio.run(runner._run_process_watcher(watcher))


def test_crash_between_admission_and_ack_replays_once(monkeypatch, isolated_home):
    """Admit -> crash before ack -> recover: durable identity admits once."""
    import tools.process_registry as pr_module
    from tools.process_registry_delivery import write_pending_delivery, list_pending_deliveries

    registry = isolated_home
    session = ProcessSession(
        id="proc_crash_admit", command="echo crash", task_id="t",
        owner_task_id="t", started_at=1234.5, output_buffer="crash\n",
        exited=True, exit_code=0, notify_on_complete=True,
        session_key="agent:main:telegram:dm:123", parent_session_id="sess-live",
    )
    registry._finished[session.id] = session
    assert write_pending_delivery(session) is True
    assert len(list_pending_deliveries()) == 1
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SendResult(success=True)),
        handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
    _run_watcher_sync(monkeypatch, runner, _watcher_for("proc_crash_admit"))
    assert adapter.handle_message.await_count == 1
    assert list_pending_deliveries() == []
    # Simulate crash between admission and ack: outbox still present on restart.
    assert write_pending_delivery(session) is True
    fresh_runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
    # Durable admission already exists: stub the authority to report it.
    import gateway.session_automation as automation
    monkeypatch.setattr(automation, "completion_admission", lambda _r, _e: {"durable": True})
    fresh_runner.session_authority = SimpleNamespace(db=SimpleNamespace())
    before = adapter.handle_message.await_count
    _run_watcher_sync(monkeypatch, fresh_runner, _watcher_for("proc_crash_admit"))
    assert adapter.handle_message.await_count == before
    assert list_pending_deliveries() == []


def test_send_failure_after_admission_retries_then_escalates(monkeypatch, isolated_home):
    """Failure after admission never loses the outbox: retry succeeds."""
    import tools.process_registry_delivery as delivery

    registry = isolated_home
    session = ProcessSession(
        id="proc_retry_after_admit", command="echo retry", task_id="t",
        owner_task_id="t", started_at=time.time(), output_buffer="retry\n",
        exited=True, exit_code=0, notify_on_complete=False,
        session_key="agent:main:telegram:dm:123",
        watcher_platform="telegram", watcher_chat_id="123",
        watcher_interval=5, parent_session_id="sess-live")
    registry._finished[session.id] = session
    assert delivery.write_pending_delivery(session) is True
    calls = {"n": 0}

    async def _flaky(chat_id, text, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SendResult(success=False, error="injected refuse")
        return SendResult(success=True)

    adapter = SimpleNamespace(send=_flaky, handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    ok_first = asyncio.run(runner._send_watcher_message(
        "telegram", "123", "", "hello",
        {"session_key": "agent:main:telegram:dm:123"}))
    assert ok_first is True
    assert calls["n"] >= 2
    assert len(delivery.list_pending_deliveries()) == 1
    delivery.ack_delivery(session.id)
    assert delivery.list_pending_deliveries() == []


@pytest.mark.asyncio
async def test_process_completion_native_admission_exactly_once_across_restart(
    tmp_path, monkeypatch, isolated_home,
):
    """TIER: native admission processing (honest label, not autonomous dispatch).

    Real SessionDB-backed session authority, real ``admit_automation`` durable producer
    identity, the real ``_deliver_completion_notification`` scoped path, and the real
    ``execute_admission`` dispatcher into a STUBBED turn handler (``_schedule`` patched
    to None, ``execute_admission`` called manually after completion): no model, no
    provider API calls, no outbound adapter send, no autonomous scheduler/watcher
    dispatch without user input. NOT proven here: model inference, outbound delivery,
    automatic owning-workflow triggering.

    Failure exercised: crash after durable admission, before outbox ack
    (accepted-before-outbox-ack). Recovery replays the outbox, finds the durable
    admission, and admits nothing twice for the same producer event: exactly 1
    automation admission row and exactly 1 stub-turn execution in this replay.
    Exactly-once external actions are not guaranteed. /new/compression/profile
    identity is untouched: the admission lands on the live root the pre-flight proved.
    """
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from gateway.session_automation import completion_admission
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.gateway.test_automation_compression_media import _admit_human
    from tools.process_registry import ProcessSession
    from tools.process_registry_delivery import (
        ack_delivery,
        list_pending_deliveries,
        write_pending_delivery,
    )

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")

    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")

    receipt = await _admit_human(
        runner, authority, MessageEvent(text="human", source=source, message_id="human"))
    root = receipt.ref.session_id
    key = runner.session_store._generate_session_key(source)

    session = ProcessSession(
        id="proc_native_admit", command="echo native", task_id="t-native",
        owner_task_id="t-native", started_at=1234.5, output_buffer="native\n",
        exited=True, exit_code=0, completion_reason="exited",
        notify_on_complete=True, session_key=key, parent_session_id=root,
    )
    evt = {
        "type": "completion",
        "session_id": session.id,
        "session_key": key,
        "task_id": "t-native",
        "owner_task_id": "t-native",
        "command": "echo native",
        "output": "native\n",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "started_at": 1234.5,
        "parent_session_id": root,
        "platform": "discord",
        "chat_type": "dm",
        "chat_id": "42",
        "user_id": "42",
    }
    synth_text = "[Background process proc_native_admit finished with exit code 0]\nnative\n"

    processed = []

    async def stub_turn(event):
        processed.append(event.text)
        return "stub-reply"

    monkeypatch.setattr(runner, "_handle_message", stub_turn)

    # Crash-before-ack setup: the durable outbox record exists before delivery.
    assert write_pending_delivery(session) is True
    assert len(list_pending_deliveries()) == 1

    assert await runner._deliver_completion_notification(synth_text, evt) is True

    rows = list_session_admissions(authority.db, session_id=root, pending_only=False)
    assert len(rows) == 2, rows  # the human turn plus exactly one automation admission
    auto_row = rows[-1]
    assert "automation" in auto_row["payload"].get("native_text_v1", {}), auto_row["payload"]
    assert completion_admission(runner, evt)["admission_id"] == auto_row["admission_id"]

    # The admitted turn reaches the owning workflow's turn handler (stubbed model).
    await execute_admission(authority, SessionRef(authority.profile_id, root), auto_row)
    assert len(processed) == 1
    assert "proc_native_admit" in processed[0]
    assert auto_row["admission_id"] in authority.pending_results

    # Crash after admission, before outbox ack: the outbox still holds the record.
    assert len(list_pending_deliveries()) == 1

    # Restart: a fresh runner reopens the same home/state; nothing in-memory survives.
    runner2 = GatewayRunner(GatewayConfig())
    adapter2 = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner2.adapters = {Platform.DISCORD: adapter2}
    authority2 = await initialize_session_authority(runner2, profile_id="default", instance_id="owner2")
    runner2._wire_adapter_handlers(adapter2)
    resurrected = []

    async def tripwire_turn(event):
        resurrected.append(event.text)
        raise AssertionError("redelivery must not re-execute an admitted completion")

    monkeypatch.setattr(runner2, "_handle_message", tripwire_turn)

    assert completion_admission(runner2, evt)["admission_id"] == auto_row["admission_id"]
    assert await runner2._deliver_completion_notification(synth_text, evt) is True
    assert resurrected == []
    assert len(processed) == 1, "owning turn executed twice across restart"

    rows2 = list_session_admissions(authority2.db, session_id=root, pending_only=False)
    assert len(rows2) == 2, rows2  # still exactly one automation admission, no duplicate

    ack_delivery(session.id)
    assert list_pending_deliveries() == []
