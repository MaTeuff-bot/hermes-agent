"""Autonomous resume: real watcher + real authority scheduler wake owning turn.

TIER (honest): automatic NATIVE dispatch to a stub turn callback, not full
inference or outbound send. Real ``SessionAuthority._schedule``/``_drain``,
real ``_run_process_watcher``, real fault-injecting subprocess
(``PersistentDockerFake`` subprocess backend, no Docker engine), real
SessionDB-backed authority, real ``admit_automation``. Downstream
``runner._handle_message`` is stubbed solely to observe dispatch; the
authority schedule/drain and watcher are never mocked. This proves
no-human-ping owning-workflow entry beyond the manual ``execute_admission``
proof in ``test_process_restart_dedup``. No live Slack, no Docker engine, no
model inference, no outbound send claimed.
"""
import asyncio
import time

import pytest

import tools.process_registry_supervision as sup
from tests.waiting import async_wait_until
from tools.environments.docker import DockerEnvironment
from tools.environments.local import LocalEnvironment


def _short_supervision(monkeypatch):
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.05)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 4)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_BACKOFF_CAP_S", 0.1)


class PersistentDockerFake(DockerEnvironment):
    """Subprocess-backed Docker-identity fake (no engine).

    Mirrors the idle-reaper suite: subclasses DockerEnvironment only for the
    container identity/cleanup path, overrides execute with a local backend and
    asserts the handle, so idle-reaper invalidation surfaces as a failure.
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
    import os
    import signal

    try:
        try:
            registry.kill_process(session.id)
        except Exception:
            pass
        pid = getattr(session, "sandbox_pid", None) or getattr(session, "pid", None)
        if pid is None:
            return
        try:
            with open(f"/proc/{pid}/cmdline", "rb"):
                pass
        except OSError:
            return
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _isolated_registry(tmp_path, monkeypatch):
    # Process-registry isolation only. HERMES_HOME stays on the hermetic conftest
    # home so the runner DB, the transport home and native_callback resolve one
    # home; re-pointing hc.get_hermes_home here leaks a per-param lambda into
    # modules that bound it at import, and the next param then admits under a
    # stale home (authority_for_home -> profile_mismatch).
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_process_autonomous_resume_without_ping(tmp_path, monkeypatch, exit_code):
    """Real bounded job completes, forced-idle reaper keeps lease, owning turn wakes alone."""
    _short_supervision(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    registry = _isolated_registry(tmp_path, monkeypatch)

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import completion_admission
    from gateway.session_ingress_context import register_transport_home
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.gateway.test_automation_compression_media import _admit_human

    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner-auto")
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    wrong_source = adapter.build_source(chat_id="99", chat_type="dm", user_id="42")

    processed = []
    arrived = asyncio.Event()

    async def stub_turn(event):
        processed.append((str(event.text), str(getattr(event.source, "chat_id", ""))))
        arrived.set()
        return "stub-reply"

    monkeypatch.setattr(runner, "_handle_message", stub_turn)

    receipt = await _admit_human(
        runner, authority, MessageEvent(text="human", source=source, message_id=f"human-{exit_code}"))
    root = receipt.ref.session_id
    key = runner.session_store._generate_session_key(source)
    wrong_receipt = await _admit_human(
        runner, authority, MessageEvent(text="wrong-human", source=wrong_source, message_id=f"wrong-{exit_code}"))
    wrong_root = wrong_receipt.ref.session_id

    await async_wait_until(lambda: len(processed) >= 2, timeout=15.0, what="seeded human turns dispatched")
    processed.clear()
    arrived.clear()

    import tools.terminal_tool as tt
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs

    env = PersistentDockerFake()
    command = f"sleep 0.5; echo auto-resume-{exit_code}; exit {exit_code}"
    session = registry.spawn_via_env(
        env, command, task_id="default", owner_task_id="raw-session-owner",
        session_key=key, timeout=10, persist_on_release=True)
    proc_id = session.id
    session.session_key = key
    session.parent_session_id = root
    session.notify_on_complete = True
    session.watcher_interval = 1

    monkeypatch.setitem(tt._active_environments, "default", env)
    monkeypatch.setitem(tt._last_activity, "default", time.time() - 301)
    assert registry.has_active_processes("default") is False
    assert registry.has_active_processes("raw-session-owner") is True
    assert registry.has_active_for_container("default", env) is True
    _cleanup_inactive_envs(300)
    assert env._container_id == "disposable-container-identity"
    assert "default" in tt._active_environments

    watcher = {
        "session_id": session.id,
        "check_interval": 0.2,
        "session_key": key,
        "platform": "discord",
        "chat_type": "dm",
        "chat_id": "42",
        "user_id": "42",
        "notify_on_complete": True,
        "parent_session_id": root,
    }
    watcher_task = asyncio.create_task(runner._run_process_watcher(watcher))
    try:
        await asyncio.wait_for(arrived.wait(), timeout=20.0)
        await async_wait_until(lambda: session.exited, timeout=15.0, what="leased job true exit")
        assert session.exit_code == exit_code
        assert session.completion_reason == "exited"
        assert env.launch_count == 1
        assert len(processed) == 1
        completion_text, completion_chat = processed[0]
        assert proc_id in completion_text
        assert completion_chat == "42"
        assert f"auto-resume-{exit_code}" in completion_text

        rows = list_session_admissions(authority.db, session_id=root, pending_only=False)
        assert len(rows) == 2, rows
        auto_row = rows[-1]
        assert "automation" in auto_row["payload"].get("native_text_v1", {}), auto_row["payload"]
        assert auto_row["admission_id"] in authority.pending_results
        wrong_rows = list_session_admissions(authority.db, session_id=wrong_root, pending_only=False)
        assert len(wrong_rows) == 1, wrong_rows

        from tools.process_registry_notifications import format_process_notification

        fresh_evt = runner._build_process_completion_event(watcher, session, session.id)
        fresh_text = format_process_notification(fresh_evt)
        assert fresh_text
        assert completion_admission(runner, fresh_evt)["admission_id"] == auto_row["admission_id"]
        assert await runner._deliver_completion_notification(fresh_text, fresh_evt) is True
        await asyncio.sleep(0)
        assert len(processed) == 1
        rows_after = list_session_admissions(authority.db, session_id=root, pending_only=False)
        assert len(rows_after) == 2, rows_after
        wrong_after = list_session_admissions(authority.db, session_id=wrong_root, pending_only=False)
        assert len(wrong_after) == 1, wrong_after
    finally:
        watcher_task.cancel()
        try:
            await watcher_task
        except (asyncio.CancelledError, Exception):
            pass
        _cleanup_session(registry, session)
        try:
            from tools.process_registry_delivery import ack_delivery

            ack_delivery(session.id)
        except Exception:
            pass
        pending = []
        for live in list(getattr(authority, "sessions", {}).values()):
            task = getattr(live, "task", None)
            if task is not None and not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        authority.db.close()
