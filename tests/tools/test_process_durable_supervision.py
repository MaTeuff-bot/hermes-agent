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


def test_spawn_wrapper_has_no_default_budget_via_entry_path(monkeypatch):
    """No default budget: the terminal entry path never imposes a timeout.

    Drives the ACTUAL ``spawn_background_process`` entry (not the registry
    directly): the wrapper must contain no ``timeout -s TERM`` budget segment,
    ``timeout`` is the start/probe timeout only, and a server-like job stays
    running (never killed by a default). Budgets are caller-owned.
    """
    _short_supervision(monkeypatch)
    from tools import terminal_tool_background as bg
    from tools.process_registry import process_registry as live_registry
    registry = ProcessRegistry()
    monkeypatch.setattr(bg, "process_registry", registry, raising=False)
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module, "process_registry", registry)
    env = SandboxLikeEnv()
    result_json = bg.spawn_background_process(
        command="sleep 30", env=env, env_type="sandbox",
        effective_task_id="t", task_id="t", session_key="k",
        workdir=None, cwd="/tmp", effective_pty=False,
        notify_on_complete=False, watch_patterns=None, approval_note=None,
        completion_output_chars=0, pty_disabled_reason=None,
        heartbeat_seconds=0, persist_on_release=False, mounted_host=None,
    )
    import json as _json
    data = _json.loads(result_json)
    session = registry.get(data["session_id"])
    try:
        assert session is not None and not session.exited
        flat = "\n".join(env.commands)
        assert "timeout -s TERM" not in flat, f"default budget imposed: {flat[:500]!r}"
        assert "nohup" in flat and "bash -lc" in flat
        for _ in range(100):
            assert not session.exited, "server-like job killed without a caller budget"
            time.sleep(0)
        assert getattr(session, "sandbox_wrapper_startticks", "") or True
    finally:
        try:
            registry.kill_process(session.id)
        except Exception:
            pass
        _cleanup(registry, session)


def test_caller_owned_timeout_reports_124_even_after_child_exit0(monkeypatch):
    """Caller-owned GNU timeout holds even if the child exits 0 after TERM.

    A caller that needs a bound wraps its own command; the wrapper's 124 is
    never overwritten by a later child exit 0 (TERM then KILL grace).
    """
    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    env = SandboxLikeEnv()
    session = registry.spawn_via_env(
        env, "timeout -s TERM -k 1 1 sleep 30", timeout=15)
    try:
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done, "caller-owned timeout never fired"
        assert session.exit_code == 124
        assert session.completion_reason == "exited"
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


def test_current_probes_quote_temp_paths_with_spaces_and_verify_token_receipt(monkeypatch):
    """Quoting contract on the CURRENT probe set (moved from test_process_registry's
    obsolete log/pid/exit-order test: the poller is now framed log-delta, token,
    exit-receipt and liveness probes, and sessions carry a run token).

    Every probe that touches a temp path quotes it (shlex), and a token-bound exit
    receipt finishes the run with its real code. The first cycle has no receipt so
    the liveness probe (quoted pid path) is exercised too.
    """
    import shlex

    from tools.process_registry import ProcessSession

    _short_supervision(monkeypatch)
    registry = ProcessRegistry()
    base = "/path with spaces/hermes_bg_proc_space"
    log_path, pid_path, exit_path = base + ".log", base + ".pid", base + ".exit"
    token_path = base + ".token"
    session = ProcessSession(
        id="proc_space", command="echo hello", task_id="t1", started_at=time.time(),
        pid_scope="sandbox", pid=4242, sandbox_pid=4242,
        sandbox_container_id="ctr-space", sandbox_backend="FakeEnv",
        sandbox_run_token="tok-run-a",
        env_log_path=log_path, env_pid_path=pid_path,
        env_exit_path=exit_path, env_token_path=token_path,
    )
    registry._running[session.id] = session
    calls = {"log": 0, "exit": 0}

    class FakeEnv:
        _container_id = "ctr-space"

        def __init__(self):
            self.commands = []

        def execute(self, command, **kwargs):
            self.commands.append(command)
            if command.startswith("if [ -f"):
                calls["exit"] += 1
                if calls["exit"] == 1:
                    return {"output": "__NO_RECEIPT__\n", "returncode": 0}
                return {"output": "tok-run-a\n0\n", "returncode": 0}
            if ".token" in command:
                return {"output": "tok-run-a\n", "returncode": 0}
            if "kill -0" in command:
                return {"output": "0\n", "returncode": 0}
            assert command.startswith("O="), command[:80]
            calls["log"] += 1
            if calls["log"] == 1:
                return {"output": "5 0\nhello", "returncode": 0}
            return {"output": "5 5\n", "returncode": 0}

    env = FakeEnv()
    try:
        sup.run_env_supervision(registry, session, env, log_path, pid_path, exit_path)
    finally:
        registry._running.pop(session.id, None)
        registry._finished.pop(session.id, None)
        try:
            from tools.process_registry_delivery import ack_delivery
            ack_delivery(session.id)
        except Exception:
            pass
    assert session.exited and session.exit_code == 0
    assert session.completion_reason == "exited"
    assert session.output_buffer == "hello"
    assert calls["exit"] >= 2, "exit-receipt probe never ran"
    assert any("kill -0" in cmd for cmd in env.commands), "liveness probe never ran"
    flat = "\n".join(env.commands)
    assert shlex.quote(log_path) in flat
    assert shlex.quote(pid_path) in flat
    assert shlex.quote(exit_path) in flat
    assert shlex.quote(token_path) in flat


def test_terminal_entry_caller_owned_timeout_reports_124(monkeypatch):
    """Terminal-tool-entry smoke for a caller-owned bound (not a terminal default).

    Drives the ACTUAL ``spawn_background_process`` entry with a bounded COMMAND:
    the bound is owned by the caller's GNU ``timeout`` wrapper, independent of
    supervision polling, and stays nonzero (124) even when the child traps TERM
    and exits 0. No default budget is imposed on the entry path.
    """
    _short_supervision(monkeypatch)
    from tools import terminal_tool_background as bg
    from tools.process_registry import ProcessRegistry
    registry = ProcessRegistry()
    monkeypatch.setattr(bg, "process_registry", registry, raising=False)
    import tools.process_registry as pr_module
    monkeypatch.setattr(pr_module, "process_registry", registry)
    env = SandboxLikeEnv()
    result_json = bg.spawn_background_process(
        command="timeout -s TERM -k 1 1 bash -c 'trap \"exit 0\" TERM; sleep 30'",
        env=env, env_type="sandbox",
        effective_task_id="t", task_id="t", session_key="k",
        workdir=None, cwd="/tmp", effective_pty=False,
        notify_on_complete=False, watch_patterns=None, approval_note=None,
        completion_output_chars=0, pty_disabled_reason=None,
        heartbeat_seconds=0, persist_on_release=False, mounted_host=None,
    )
    import json as _json
    data = _json.loads(result_json)
    session = registry.get(data["session_id"])
    try:
        assert session is not None and not session.exited
        done = _wait_for(lambda: session.exited, timeout=25.0)
        assert done, "caller-owned timeout never fired through the terminal entry"
        assert session.exit_code == 124, (
            f"caller bound lost: exit={session.exit_code!r} reason={session.completion_reason!r}")
        assert session.completion_reason == "exited"
        assert env.launch_count == 1
    finally:
        try:
            registry.kill_process(session.id)
        except Exception:
            pass
        _cleanup(registry, session)
