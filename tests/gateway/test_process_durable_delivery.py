"""Durable supervision gateway delivery: real subprocess + real handler.

Integration, not production: a real local subprocess backend emulating
Docker-exec semantics (per-run container id, fault injection) plus the ACTUAL
gateway completion handler (``_run_process_watcher`` /
``_deliver_completion_notification`` / ``_send_watcher_message``) with stubbed
adapter/model transports. No live Slack, no Docker engine.

Covers the supervision contract end to end:
- transient transport outage mid-job launches once, delivers the true receipt
  into the owning session turn (admission, not a mocked handler);
- nonzero and wrapper-budget timeout route the same way (non-success, honest);
- duplicate/replayed completions admit at most once;
- ``SendResult(success=False)`` retries durably until ack, never silently acked;
- pending-delivery replay after restart (absent checkpoint) rebuilds the
  finished session with correct route keys and delivers once.
"""
import asyncio
import os
import signal
import time
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tools.process_registry_supervision as sup
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.environments.local import LocalEnvironment
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.platforms.base import SendResult


def _short_supervision(monkeypatch):
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.05)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 4)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_CAP_S", 0.1)


class SandboxLikeEnv:
    """Real subprocess backend with Docker-exec-like identity + fault injection."""

    def __init__(self):
        self.inner = LocalEnvironment()
        self._container_id = "ctr-" + uuid.uuid4().hex[:12]
        self.launch_count = 0
        self.fail_probe_calls = 0
        self.commands = []

    def get_temp_dir(self):
        return self.inner.get_temp_dir()

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if "nohup" in command:
            self.launch_count += 1
            return self.inner.execute(command, **kwargs)
        if self.fail_probe_calls > 0:
            self.fail_probe_calls -= 1
            raise RuntimeError("injected transient transport outage")
        return self.inner.execute(command, **kwargs)


def _wait_for(predicate, timeout=20.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _cleanup(session):
    try:
        pid = getattr(session, "sandbox_pid", None) or getattr(session, "pid", None)
        if pid:
            with open(f"/proc/{pid}/cmdline", "rb"):
                pass
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


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


def test_transient_outage_then_gateway_admits_success_once(monkeypatch, isolated_home):
    """Live job survives 3 probe outages, launches once, routes to owning turn once."""
    _short_supervision(monkeypatch)
    registry = isolated_home
    env = SandboxLikeEnv()
    env.fail_probe_calls = 3
    session = registry.spawn_via_env(env, "echo durable-hello; exit 0", timeout=15)
    session.session_key = "agent:main:telegram:dm:123"
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    session.watcher_interval = 5
    session.notify_on_complete = True
    session.parent_session_id = "sess-live"
    try:
        assert _wait_for(lambda: session.exited, timeout=20.0)
        assert session.exit_code == 0
        assert env.launch_count == 1
        adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
        runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
        _run_watcher_sync(monkeypatch, runner, _watcher_for(session.id))
        assert adapter.handle_message.await_count == 1
        # Pending outbox acked on admission: no replay after restart.
        from tools.process_registry_delivery import list_pending_deliveries
        assert list_pending_deliveries() == []
    finally:
        _cleanup(session)


def test_nonzero_exit_routes_honestly_once(monkeypatch, isolated_home):
    """Real nonzero receipt (exit 3) routes once, never as success or unresolved."""
    _short_supervision(monkeypatch)
    registry = isolated_home
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "exit 3", timeout=15)
    session.session_key = "agent:main:telegram:dm:123"
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    session.watcher_interval = 5
    session.notify_on_complete = True
    session.parent_session_id = "sess-live"
    try:
        assert _wait_for(lambda: session.exited, timeout=20.0)
        assert session.exit_code == 3
        assert session.completion_reason == "exited"
        assert env.launch_count == 1
        adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
        runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
        _run_watcher_sync(monkeypatch, runner, _watcher_for(session.id))
        assert adapter.handle_message.await_count == 1
    finally:
        _cleanup(session)


def test_budget_timeout_routes_as_non_success(monkeypatch, isolated_home):
    """Wrapper-owned budget fires without polling: 124/budget_timeout, routed once."""
    _short_supervision(monkeypatch)
    registry = isolated_home
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(env, "sleep 5", timeout=15, budget_seconds=1)
    session.session_key = "agent:main:telegram:dm:123"
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    session.watcher_interval = 5
    session.notify_on_complete = True
    session.parent_session_id = "sess-live"
    try:
        assert _wait_for(lambda: session.exited, timeout=20.0)
        assert session.exit_code == 124
        assert session.completion_reason == "budget_timeout"
        adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
        runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
        _run_watcher_sync(monkeypatch, runner, _watcher_for(session.id))
        assert adapter.handle_message.await_count == 1
    finally:
        _cleanup(session)


def test_duplicate_replayed_completion_admits_once(monkeypatch, isolated_home):
    """Same producer identity twice: at most one admitted turn, second deduped."""
    registry = isolated_home
    session = ProcessSession(
        id="proc_dup", command="echo dup", task_id="t", started_at=1234.5,
        output_buffer="dup\n", exited=True, exit_code=0, notify_on_complete=True,
        session_key="agent:main:telegram:dm:123", parent_session_id="sess-live",
    )
    registry._finished[session.id] = session
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
    _run_watcher_sync(monkeypatch, runner, _watcher_for("proc_dup"))
    assert adapter.handle_message.await_count == 1
    # Replay the same completion (retry/restart): identity dedups, no second turn.
    _run_watcher_sync(monkeypatch, runner, _watcher_for("proc_dup"))
    assert adapter.handle_message.await_count == 1


def test_send_success_false_retries_until_ack(monkeypatch, isolated_home):
    """``SendResult(success=False)`` is failure: no ack, retry succeeds, then acked."""
    import tools.process_registry as pr_module
    registry = isolated_home
    session = ProcessSession(
        id="proc_send_retry", command="echo hi", task_id="t", started_at=time.time(),
        output_buffer="hi\n", exited=True, exit_code=0, notify_on_complete=False,
        session_key="agent:main:telegram:dm:123",
        watcher_platform="telegram", watcher_chat_id="123", watcher_interval=5,
        parent_session_id="sess-live",
    )
    registry._finished[session.id] = session
    from tools.process_registry_delivery import write_pending_delivery, list_pending_deliveries, ack_delivery
    assert write_pending_delivery(session) is True
    assert len(list_pending_deliveries()) == 1
    calls = {"n": 0}

    async def _flaky_send(chat_id, text, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SendResult(success=False, error="injected refuse")
        return SendResult(success=True)

    adapter = SimpleNamespace(send=_flaky_send, handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    ok_first = asyncio.run(runner._send_watcher_message("telegram", "123", "", "hello", {"session_key": "agent:main:telegram:dm:123"}))
    assert ok_first is True
    assert calls["n"] >= 2, "success=False must retry, not ack silently"
    ack_delivery(session.id)
    assert list_pending_deliveries() == []


def test_pending_replay_absent_checkpoint_rebuilds_and_delivers_once(monkeypatch, isolated_home, tmp_path):
    """Restart with no checkpoint file still replays pending delivery exactly once."""
    import tools.process_registry as pr_module
    registry = isolated_home
    session = ProcessSession(
        id="proc_replay", command="echo replay", task_id="t", started_at=1234.5,
        output_buffer="replay-output\n", exited=True, exit_code=0, notify_on_complete=True,
        session_key="agent:main:telegram:dm:123",
        watcher_platform="telegram", watcher_chat_id="123", watcher_user_id="u1",
        watcher_interval=5, parent_session_id="sess-live",
    )
    registry._finished[session.id] = session
    from tools.process_registry_delivery import write_pending_delivery, list_pending_deliveries
    assert write_pending_delivery(session) is True
    # Simulate gateway/registry crash: fresh registry, no checkpoint file.
    fresh = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", fresh)
    assert not (tmp_path / "processes.json").exists()
    assert fresh.recover_from_checkpoint() >= 1
    got = fresh.get("proc_replay")
    assert got is not None and got.exited and got.exit_code == 0
    assert "replay-output" in (got.output_buffer or "")
    assert len(fresh.pending_watchers) == 1
    watcher = fresh.pending_watchers[0]
    assert watcher.get("platform") == "telegram"
    assert watcher.get("chat_id") == "123"
    assert "watcher_platform" not in watcher
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
    # Point the handler at the fresh registry for this delivery.
    monkeypatch.setattr(pr_module, "process_registry", fresh)
    # Drain the replayed watcher through the real handler.
    watchers = fresh.pending_watchers
    fresh.pending_watchers = []
    import asyncio as _asyncio
    async def _instant_sleep(*_a, **_kw):
        return None
    monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
    _asyncio.run(runner._run_process_watcher(watchers[0]))
    assert adapter.handle_message.await_count == 1
    assert list_pending_deliveries() == []


def test_closed_session_completion_never_wakes_new_session(monkeypatch, isolated_home):
    """/new boundary: stamped completion from a user-closed session is dropped."""
    registry = isolated_home
    session = ProcessSession(
        id="proc_closed", command="echo old", task_id="t", started_at=1234.5,
        output_buffer="old\n", exited=True, exit_code=0, notify_on_complete=True,
        session_key="agent:main:telegram:dm:123", parent_session_id="sess-closed",
    )
    registry._finished[session.id] = session
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)), handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": 1786288000.0, "end_reason": "session_reset"}))
    _run_watcher_sync(monkeypatch, runner, _watcher_for("proc_closed", parent="sess-closed"))
    adapter.handle_message.assert_not_awaited()
