"""R7 stale-poller generation guard: commits under lock, never from an old epoch.

Drives the production poller commit points with a bumped generation (simulating a
``reattach_env_session`` that verified a new backend and started a new poller): a
stale reconcile, degraded mark, or exit finish must commit nothing, while the
current generation still works. A barrier test pins the in-flight interleave.
"""
import threading

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
import tools.process_registry_supervision as sup


@pytest.fixture()
def isolated_registry(monkeypatch):
    import tools.process_registry as pr_module
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _sandbox_session(sid="proc_r7_gen"):
    return ProcessSession(
        id=sid, command="sleep 999", task_id="default", owner_task_id="owner1",
        session_key="s1", started_at=0.0, pid=42421, pid_scope="sandbox",
        sandbox_container_id="cidAAA", sandbox_run_token="tokAAA",
        sandbox_wrapper_startticks="12345", sandbox_boot_id="bootAAA",
        env_pid_path="/tmp/x.pid", env_token_path="/tmp/x.token",
        env_exit_path="/tmp/x.exit", env_log_path="/tmp/x.log",
        env_ref=None, supervision_state="ok")


def _reattached(session):
    """Simulate what ``reattach_env_session`` commits on success: new epoch, clean state."""
    with session._lock:
        session._supervision_generation += 1
        session.supervision_state = "ok"
        session.supervision_failures = 0
        session.last_supervision_error = ""
    return session._supervision_generation


def test_stale_reconcile_commits_nothing_after_reattach(isolated_registry):
    """An old-epoch ``_reconcile_or_escalate`` escalation lands nowhere after a bump."""
    session = _sandbox_session()
    isolated_registry._running[session.id] = session
    stale = session._supervision_generation
    current = _reattached(session)

    assert current == stale + 1
    sup._reconcile_or_escalate(
        isolated_registry, session, object(), __import__("shlex").quote,
        "/tmp/x.pid", "/tmp/x.exit", "boom", stale)

    assert session.supervision_state == "ok"
    assert session.last_supervision_error == ""
    assert session._supervision_generation == current
    assert not session.exited


def test_stale_degraded_and_finish_commit_nothing(isolated_registry):
    """Stale degraded marks and exit finishes are dropped; current ones apply."""
    session = _sandbox_session()
    isolated_registry._running[session.id] = session
    stale = session._supervision_generation
    _reattached(session)

    isolated_registry._note_supervision_degraded(session, "stale noise", 3, stale)
    assert session.supervision_state == "ok"
    isolated_registry._finish_env_exit(session, 0, stale)
    assert not session.exited

    isolated_registry._note_supervision_degraded(session, "live noise", 1,
                                                 session._supervision_generation)
    assert session.supervision_state == "degraded"


def test_barrier_old_blocked_reconcile_vs_new_reattach(isolated_registry, monkeypatch):
    """Old poller blocked in transport while reattach lands: no stale commit, no exit."""
    import tools.process_registry as pr_module

    session = _sandbox_session("proc_r7_barrier")
    isolated_registry._running[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", isolated_registry)
    stale = session._supervision_generation

    entered = threading.Event()
    release = threading.Event()

    def _blocked_receipt(env, quoted_exit_path, expected_token=""):
        entered.set()
        assert release.wait(timeout=15)
        raise sup._SupervisionTransportError("old poller unblocked into a new epoch")

    monkeypatch.setattr(sup, "_read_exit_receipt", _blocked_receipt)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 1)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.0)

    worker = threading.Thread(
        target=sup._reconcile_or_escalate,
        args=(isolated_registry, session, object(), __import__("shlex").quote,
              "/tmp/x.pid", "/tmp/x.exit", "uncertain", stale),
        daemon=True)
    worker.start()
    assert entered.wait(timeout=15)

    current = _reattached(session)
    release.set()
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert session._supervision_generation == current
    assert session.supervision_state == "ok"
    assert session.last_supervision_error == ""
    assert not session.exited


def test_barrier_old_blocked_reconcile_vs_real_reattach(isolated_registry, monkeypatch):
    """Old reconcile blocked while the production reattach path lands: stale commits nothing."""
    import shlex as _shlex
    import tools.process_registry as pr_module

    session = _sandbox_session("proc_r7_real_reattach")
    session.env_log_path = "/tmp/x.log"
    session.env_pid_path = "/tmp/x.pid"
    session.env_exit_path = "/tmp/x.exit"
    session.env_token_path = "/tmp/x.token"
    isolated_registry._running[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", isolated_registry)
    stale = session._supervision_generation

    class _MatchEnv:
        _container_id = "cidAAA"
        _docker_exe = None

        def execute(self, command, timeout=5, **kwargs):
            if "boot_id" in command:
                return {"output": "bootAAA\n", "returncode": 0}
            if "/proc/" in command and "/stat" in command:
                fields = ["S"] + ["1"] * 18 + ["12345", "20", "21"]
                return {"output": f"42421 (bash) {' '.join(fields)}\n", "returncode": 0}
            if ".token" in command:
                return {"output": "tokAAA\n", "returncode": 0}
            if command.startswith("cat ") and ".pid" in command:
                return {"output": "42421\n", "returncode": 0}
            if command.startswith("if [ -f"):
                return {"output": "__NO_RECEIPT__\n", "returncode": 0}
            return {"output": "", "returncode": 0}

    old_env = _MatchEnv()
    new_env = _MatchEnv()
    entered = __import__("threading").Event()
    release = __import__("threading").Event()

    def _blocked_receipt(env, quoted_exit_path, expected_token=""):
        entered.set()
        assert release.wait(timeout=15)
        raise sup._SupervisionTransportError("old poller unblocked into a new epoch")

    monkeypatch.setattr(sup, "_read_exit_receipt", _blocked_receipt)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_MAX_FAILURES", 1)
    monkeypatch.setattr(sup, "ENV_SUPERVISION_POLL_S", 0.0)
    monkeypatch.setattr(isolated_registry, "_track_started", lambda *a, **k: None)
    monkeypatch.setattr(isolated_registry, "_write_checkpoint", lambda *_a, **_k: None)

    worker = __import__("threading").Thread(
        target=sup._reconcile_or_escalate,
        args=(isolated_registry, session, old_env, _shlex.quote,
              "/tmp/x.pid", "/tmp/x.exit", "uncertain", stale),
        daemon=True)
    worker.start()
    assert entered.wait(timeout=15)

    assert isolated_registry.reattach_env_session(session.id, new_env) is True
    current = session._supervision_generation
    assert current == stale + 1
    release.set()
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert session._supervision_generation == current
    assert session.supervision_state == "ok"
    assert session.last_supervision_error == ""
    assert not session.exited
    assert session.env_ref is new_env


def test_stale_finish_after_real_reattach_commits_nothing(isolated_registry):
    """Check-then-mark atomicity: a stale finish cannot land between check and mark."""
    session = _sandbox_session("proc_r7_stale_finish")
    isolated_registry._running[session.id] = session
    stale = session._supervision_generation
    _reattached(session)
    isolated_registry._finish_env_exit(session, 0, stale)
    assert not session.exited
    assert session._supervision_generation == stale + 1

