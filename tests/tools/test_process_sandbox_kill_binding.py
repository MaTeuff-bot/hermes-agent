"""R3 sandbox kill binding: pidfd helper inside the container, pinned docker target.

Drives the real ``_signal_kill_sandbox`` with disposable fake envs (no real PIDs,
no real signals: the low-level exec is always intercepted). The signal is an
identity-safe pidfd helper executed INSIDE the target container (open pidfd,
verify AFTER open, signal via the fd in the SAME helper); no fd crosses the
transport. Missing inside-python/pidfd exits 10 (unsupported, fail closed with
no shell-kill fallback). The docker exec argv carries the immutable expected
container ID, never the mutable env field.
"""
import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
import tools.process_registry_supervision as sup


@pytest.fixture()
def isolated_registry(monkeypatch):
    import tools.process_registry as pr_module
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _sandbox_session(sid="proc_r3_kill"):
    return ProcessSession(
        id=sid, command="sleep 999", task_id="default", owner_task_id="owner1",
        session_key="s1", started_at=0.0, pid=42421, pid_scope="sandbox",
        sandbox_container_id="cidAAA", sandbox_run_token="tokAAA",
        sandbox_wrapper_startticks="12345", sandbox_boot_id="bootAAA",
        env_pid_path="/tmp/x.pid", env_token_path="/tmp/x.token",
        env_ref=None,
    )


class _FakeEnv:
    """Disposable canary backend: identity reads stable until a flip is armed."""

    def __init__(self):
        self._container_id = "cidAAA"
        self.exec_calls = []

    def execute(self, command, timeout=5, **kwargs):
        self.exec_calls.append(command)
        return {"output": "", "returncode": 0}


def _match_reads(monkeypatch):
    monkeypatch.setattr(sup, "_read_pid_file", lambda _env, _p: 42421)
    monkeypatch.setattr(sup, "_read_token_file", lambda _env, _p: "tokAAA")
    monkeypatch.setattr(sup, "_read_wrapper_startticks", lambda _env, _p: "12345")
    monkeypatch.setattr(sup, "_read_boot_id", lambda _env: "bootAAA")


def test_supported_path_sends_one_pinned_pidfd_kill(isolated_registry, monkeypatch):
    """Matching identity: exactly one pidfd helper exec, pinned ticks/token, no shell kill."""
    session = _sandbox_session()
    fake = _FakeEnv()
    session.env_ref = fake
    _match_reads(monkeypatch)
    monkeypatch.setattr(sup, "_exec_probe_pinned", lambda _e, _c, _t, label="": "1\n")

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is None
    assert len(fake.exec_calls) == 1
    command = fake.exec_calls[0]
    assert "pidfd_send_signal" in command
    assert "pidfd_open" in command
    assert "12345" in command
    assert "tokAAA" in command
    assert "42421" in command
    assert 'kill "$PID"' not in command
    assert not command.startswith("kill ")
    assert "kill -" not in command


def test_container_replacement_visible_at_authorization_refuses(isolated_registry, monkeypatch):
    """A replaced backend fails authorization: zero kill execs, explicit refusal."""
    session = _sandbox_session()
    fake = _FakeEnv()
    fake._container_id = "cidBBB"
    session.env_ref = fake
    _match_reads(monkeypatch)
    monkeypatch.setattr(sup, "_exec_probe_pinned", lambda *a, **k: "1\n")

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is not None and result.get("status") == "error"
    assert "identity changed or is" in result.get("error", "")
    assert fake.exec_calls == []


def test_container_replacement_during_kill_exec_refuses(isolated_registry, monkeypatch):
    """The env handle flips mid-exec: the pinned kill refuses instead of trusting it."""
    session = _sandbox_session()

    class _FlippingEnv(_FakeEnv):
        def execute(self, command, timeout=5, **kwargs):
            self.exec_calls.append(command)
            self._container_id = "cidBBB"
            return {"output": "", "returncode": 0}

    fake = _FlippingEnv()
    session.env_ref = fake
    _match_reads(monkeypatch)

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is not None and result.get("status") == "error"
    assert "container replaced during kill" in result.get("error", "")


def test_in_exec_identity_failure_refuses_without_post_verify(isolated_registry, monkeypatch):
    """Verdict matched, but the pidfd helper reports refusal: no post-verify probe runs."""
    session = _sandbox_session()
    fake = _FakeEnv()
    session.env_ref = fake
    _match_reads(monkeypatch)

    def _refusing_kill(env, expected_container, command, timeout=5, **kwargs):
        assert expected_container == "cidAAA"
        assert "pidfd_send_signal" in command
        assert 'kill "$PID"' not in command
        return {"output": "", "returncode": 4}

    monkeypatch.setattr(sup, "pinned_kill_execute", _refusing_kill)
    probed = []
    monkeypatch.setattr(sup, "_exec_probe_pinned",
                        lambda *a, **k: probed.append(a) or "1\n")

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is not None and result.get("status") == "error"
    assert "nothing was signalled" in result.get("error", "")
    assert probed == []


def test_unsupported_helper_refuses_without_fallback(isolated_registry, monkeypatch):
    """Inside-python/pidfd missing (exit 10): fail closed, no shell kill, no post-verify."""
    session = _sandbox_session()
    fake = _FakeEnv()
    session.env_ref = fake
    _match_reads(monkeypatch)

    def _unsupported_kill(env, expected_container, command, timeout=5, **kwargs):
        assert "pidfd_send_signal" in command
        assert 'kill "$PID"' not in command
        return {"output": "UNSUPPORTED\n", "returncode": 10}

    monkeypatch.setattr(sup, "pinned_kill_execute", _unsupported_kill)
    probed = []
    monkeypatch.setattr(sup, "_exec_probe_pinned",
                        lambda *a, **k: probed.append(a) or "1\n")
    isolated_registry._running[session.id] = session

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is not None and result.get("status") == "error"
    assert "identity-safe helper unavailable" in result.get("error", "")
    assert probed == []
    assert session.id in isolated_registry._running
    assert not session.exited


def test_missing_stored_ticks_refuses_before_any_exec(isolated_registry, monkeypatch):
    """No stored start-ticks continuity proof: refuse with zero execs, never signal."""
    session = _sandbox_session()
    session.sandbox_wrapper_startticks = ""
    fake = _FakeEnv()
    session.env_ref = fake
    _match_reads(monkeypatch)
    monkeypatch.setattr(sup, "_exec_probe_pinned", lambda *a, **k: "1\n")

    result = isolated_registry._signal_kill_sandbox(session, session.id)

    assert result is not None and result.get("status") == "error"
    assert "incomplete" in result.get("error", "")
    assert fake.exec_calls == []


def test_helper_command_binds_ticks_and_never_shell_kills():
    """The helper carries pid/ticks/token, requires ticks, and has zero shell kill."""
    command = sup.sandbox_pidfd_kill_command(
        pid_path="/tmp/x.pid", token_path="/tmp/x.token",
        live_pid=42421, expected_token="tokAAA", expected_startticks="12345")
    assert "pidfd_send_signal" in command
    assert "pidfd_open" in command
    assert "42421" in command
    assert "12345" in command
    assert "tokAAA" in command
    assert 'kill "$PID"' not in command
    assert not command.strip().startswith("kill ")
    assert "kill -" not in command

    import shlex
    nasty = 'tok with "quotes" and spaces'
    quoted = sup.sandbox_pidfd_kill_command(
        pid_path="/tmp/x.pid", token_path="/tmp/x.token",
        live_pid=42421, expected_token=nasty, expected_startticks="12345")
    assert shlex.quote(nasty) in quoted
    assert shlex.quote("12345") in quoted


def test_docker_exec_argv_uses_expected_container(monkeypatch):
    """Pinned docker argv carries the snapshot ID even when the env field mutated."""
    captured = {}

    class _DockerFake:
        _docker_exe = "docker"
        _container_id = "cidBBB"

    def _fake_cap(argv, timeout=5, **kwargs):
        captured["argv"] = list(argv)
        assert argv[0] == "docker"
        assert argv[1] == "exec"
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr("tools.environments.remote_common.run_capture", _fake_cap)
    fake = _DockerFake()
    result = sup._docker_exec_pinned(fake, "cidAAA", "echo hi", timeout=5)

    assert result["returncode"] == 0
    assert captured["argv"][2] == "cidAAA"
    assert captured["argv"][2] != fake._container_id


def test_local_pidfd_supported_path_via_local_env():
    """LocalEnvironment can run the pidfd helper (supported, not UNSUPPORTED)."""
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    command = sup.sandbox_pidfd_kill_command(
        pid_path="-", token_path="-",
        live_pid=999999, expected_token="tok-local-probe", expected_startticks="99999")
    assert "pidfd_send_signal" in command
    result = env.execute(command, timeout=10)
    assert int(result.get("returncode", 0) or 0) != 10 or "UNSUPPORTED" in result.get("output", "")
    if int(result.get("returncode", 0) or 0) == 10:
        pytest.skip("inside-python pidfd unavailable in this image")
    assert int(result.get("returncode", 0) or 0) in (3, 4, 5, 6, 7)
