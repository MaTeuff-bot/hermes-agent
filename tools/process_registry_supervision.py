"""Durable supervision for env-backed (sandbox) background processes.

Owns the poll/reconcile policy that keeps ``ProcessRegistry._env_poller_loop`` thin:
a transport exception, timeout or malformed probe means *uncertain supervision*,
never an exit. The loop retries bounded with backoff, reconciles a verified exit
receipt first and identity-checked liveness second, and escalates to a durable
``unresolved`` state instead of inventing an exit code or relaunching the job.

Never relaunch: once the poller loses the handle the original command must not run
again (a retry could double side effects). Never signal: in-sandbox PIDs are never
passed to a host ``kill``; a mismatched identity refuses the signal.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("tools.process_registry")

# Poll/retry policy. Module-level so tests can tighten them with monkeypatch.
ENV_SUPERVISION_POLL_S = 2.0
ENV_SUPERVISION_MAX_FAILURES = 5
ENV_SUPERVISION_BACKOFF_BASE_S = 2.0
ENV_SUPERVISION_BACKOFF_CAP_S = 30.0

# Job budgets are caller-owned, never terminal-owned. ``spawn_via_env(timeout=...)``
# is the transport/start timeout only (launch + probes); it never bounds the job.
# A caller that needs a bound wraps its own command with GNU ``timeout``
# (TERM then KILL grace, exit 124 even if the child exits 0 after TERM).
# No default budget is imposed: background servers must not be killed.


class _SupervisionTransportError(Exception):
    """One poll cycle could not reach the backend; supervision is uncertain."""


def supervision_backoff_delay(failures: int) -> float:
    """Backoff after ``failures`` consecutive failed cycles (1-based)."""
    delay = ENV_SUPERVISION_BACKOFF_BASE_S * (2.0 ** max(0, failures - 1))
    return min(delay, ENV_SUPERVISION_BACKOFF_CAP_S)


def parse_verified_exit(output: str, expected_token: str = "") -> Optional[int]:
    """The exit-file content as a verified exit code, or None.

    Token-bound format written atomically at spawn/exit is ``"<token>\\n<code>\\n"``.
    When ``expected_token`` is known the first non-empty line must equal it,
    otherwise the receipt belongs to another run (or a torn write) and is refused.
    Only a clean transport whose last non-empty line parses as an integer counts.
    Empty, malformed, missing or token-mismatched content is *no receipt*, never
    exit 0: acting on a half-read value would fabricate a completion.
    """
    if "__NO_RECEIPT__" in (output or ""):
        return None
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return None
    if expected_token:
        if len(lines) < 2:
            return None
        if lines[0] != expected_token:
            return None
        try:
            return int(lines[-1])
        except ValueError:
            return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def capture_sandbox_identity(env: Any) -> tuple[str, str]:
    """Best-effort (container_id, backend) snapshot of a live env handle.

    The Docker backend recreates a removed container and retries the probe, so the
    exact container ID is the continuity proof: a changed ID is a replacement, not
    the same run. Unknown backends report what they can; empty means unidentified.
    """
    container_id = ""
    for attr in ("_container_id", "container_id"):
        value = getattr(env, attr, None)
        if value:
            container_id = str(value)
            break
    backend = type(env).__name__ if env is not None else ""
    return container_id, backend


def sandbox_identity_verdict(
    stored_container_id: str,
    stored_pid: Optional[int],
    env: Any,
    pid_file_value: Optional[int] = None,
    stored_token: str = "",
    live_token: Optional[str] = None,
) -> str:
    """``match`` / ``mismatch`` / ``unknown`` for one sandbox session.

    The per-run token written at spawn is the continuity proof: only an exact
    token match proves the live backend is the same run. A changed Docker
    container ID or a changed token means replacement, never continuity. PID
    number equality alone is NEVER a reuse guard (PIDs recycle inside and
    outside the sandbox), so without a token match the verdict is at best
    ``unknown``. ``unknown`` must not be signalled and must not be finished
    as an exit.
    """
    current_container, _ = capture_sandbox_identity(env)
    if stored_container_id and current_container:
        if current_container != stored_container_id:
            return "mismatch"
    elif bool(stored_container_id) != bool(current_container):
        return "unknown"
    if stored_token:
        if live_token is None:
            return "unknown"
        if live_token != stored_token:
            return "mismatch"
        if pid_file_value is not None and stored_pid is not None:
            if pid_file_value != stored_pid:
                return "mismatch"
        return "match"
    return "unknown"


def sandbox_wrapper_inner(command: str, quote) -> str:
    """The inner command for the ``spawn_via_env`` nohup wrapper, always unbounded.

    No job budget is imposed here: background servers must not be killed by a
    default. A caller that needs a bound owns it (GNU ``timeout -s TERM -k 5 N``),
    which reports 124 even if the child exits 0 after TERM.
    """
    return f"bash -lc {quote(command)}"


def _parse_proc_stat_startticks(stat_text: str) -> Optional[str]:
    """Kernel start ticks (field 22) from ``/proc/<pid>/stat``, or None.

    The comm field may contain spaces/parens, so split after the last ``)``.
    A malformed read is None (fail closed), never a guessed identity."""
    try:
        text = (stat_text or "").strip()
        if not text:
            return None
        after = text.rsplit(")", 1)[-1].strip() if ")" in text else text
        parts = after.split()
        if ")" in text:
            if len(parts) < 20:
                return None
            return parts[19]
        if len(parts) < 22:
            return None
        return parts[21]
    except (ValueError, IndexError):
        return None


def _read_wrapper_startticks(env: Any, quoted_pid_path: str) -> Optional[str]:
    """Live start ticks for the wrapper PID, or None when gone/unreadable-clean.

    Transport failures raise ``_SupervisionTransportError`` (uncertain, retry);
    a clean transport with no parseable ticks is None (fail closed: no signal,
    no finish as exit, escalate to ``unknown``).
    """
    try:
        result = _pinned_env_exec(
            env, f"cat /proc/$(cat {quoted_pid_path} 2>/dev/null)/stat 2>/dev/null", 5)
    except _ContainerReplacedError:
        raise
    except _SupervisionTransportError:
        raise
    except Exception as exc:
        raise _SupervisionTransportError(f"startticks read: {exc}") from exc
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(
            f"startticks read transport failed: {result.get('output', '')[:120]}")
    return _parse_proc_stat_startticks(result.get("output", ""))


def _read_boot_id(env: Any) -> Optional[str]:
    """Host/container boot ID via the pinned path, or None when unreadable-clean."""
    try:
        result = _pinned_env_exec(
            env, "cat /proc/sys/kernel/random/boot_id 2>/dev/null", 5)
    except _ContainerReplacedError:
        raise
    except _SupervisionTransportError:
        raise
    except Exception as exc:
        raise _SupervisionTransportError(f"boot-id read: {exc}") from exc
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(
            f"boot-id read transport failed: {result.get('output', '')[:120]}")
    lines = [ln.strip() for ln in result.get("output", "").splitlines() if ln.strip()]
    return lines[0] if lines else None


def sandbox_signal_verdict(
    stored_container_id: str,
    stored_pid: Optional[int],
    env: Any,
    pid_file_value: Optional[int] = None,
    stored_token: str = "",
    live_token: Optional[str] = None,
    stored_startticks: str = "",
    live_startticks: Optional[str] = None,
    stored_boot_id: str = "",
    live_boot_id: Optional[str] = None,
) -> str:
    """``match`` / ``mismatch`` / ``unknown`` for signalling a sandbox run.

    Extends :func:`sandbox_identity_verdict` with live ``/proc`` binding in the
    same pinned execution namespace: the token proves the run, the container ID
    proves no replacement, and the wrapper start ticks (+ boot ID when known)
    prove the live PID is still the spawned wrapper shell, not a reused number.
    A stale token/PID file with a reused PID (same container, same number, old
    token) is ``mismatch`` when the live start ticks differ, ``unknown`` when
    the live ticks cannot be read. Legacy runs without stored ticks are
    ``unknown`` (fail closed: explicit escalate, never signal).
    """
    base = sandbox_identity_verdict(
        stored_container_id, stored_pid, env, pid_file_value, stored_token, live_token)
    if base != "match":
        return base
    if stored_startticks:
        if live_startticks is None:
            return "unknown"
        if live_startticks != stored_startticks:
            return "mismatch"
    else:
        return "unknown"
    if stored_boot_id:
        if live_boot_id is None:
            return "unknown"
        if live_boot_id != stored_boot_id:
            return "mismatch"
    if pid_file_value is not None and stored_pid is not None:
        if pid_file_value != stored_pid:
            return "mismatch"
    return "match"


def _read_pid_file(env: Any, quoted_pid_path: str) -> Optional[int]:
    """The wrapper PID file as an int, or None when missing/malformed/unreadable.

    Transport failures raise ``_SupervisionTransportError`` (uncertain, retry);
    a clean transport with no parseable PID is a fact (None), not an error.
    """
    try:
        result = _pinned_env_exec(env, f"cat {quoted_pid_path} 2>/dev/null", 5)
    except _ContainerReplacedError:
        raise
    except _SupervisionTransportError:
        raise
    except Exception as exc:
        raise _SupervisionTransportError(f"pid read: {exc}") from exc
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(
            f"pid read transport failed: {result.get('output', '')[:120]}")
    lines = [line.strip() for line in result.get("output", "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def _exec_probe(env: Any, command: str, timeout: int, label: str) -> str:
    """One sandbox probe; transport failure (exception, nonzero returncode or
    timeout hint) raises ``_SupervisionTransportError`` instead of returning
    output that could be mistaken for a fact."""
    try:
        result = env.execute(command, timeout=timeout)
    except Exception as exc:
        raise _SupervisionTransportError(f"{label}: {exc}") from exc
    output = result.get("output", "")
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(f"{label} transport failed: {output[:120]}")
    lowered = str(output).lower()
    if "timed out" in lowered and "command timed out" in lowered:
        raise _SupervisionTransportError(f"{label} timed out")
    return output


class _ContainerReplacedError(_SupervisionTransportError):
    """The live backend is a replacement run, not the supervised one."""


def _is_recreating_docker_env(env: Any) -> bool:
    """True for backends whose ``execute`` recreates a gone container and retries.

    ``isinstance`` when the class imports, else duck-typing on the recreate
    marker so test fakes and renamed subclasses still pin correctly. A
    subprocess-backed test fake that subclasses ``DockerEnvironment`` only for
    its ``_container_id`` identity/cleanup (no ``_docker_exe``/``cwd`` Base
    init, ``execute`` overridden to a local backend) is NOT recreating: its
    ``execute`` already asserts the handle and never recreates, so pinned
    probes must use it directly. Real Docker backends always carry the Base
    execution state, so this never misreports a real engine.
    """
    try:
        from tools.environments.docker import DockerEnvironment
        if isinstance(env, DockerEnvironment):
            if type(env).execute is not DockerEnvironment.execute:
                return False
            if not (hasattr(env, "_docker_exe") and hasattr(env, "cwd")):
                return False
            return True
    except ImportError as exc:
        logger.debug("Docker backend unavailable for recreate check: %s", exc, exc_info=True)
    return type(getattr(env, "__class__", None)).__name__ == "DockerEnvironment" or (
        hasattr(env, "_recreate_container") and hasattr(env, "_container_id"))


def _pinned_env_exec(env: Any, command: str, timeout: int) -> dict:
    """Execute a pure-read probe without recreating the backend.

    Docker's ``DockerEnvironment.execute`` recreates a removed container and
    retries once, which would let a receipt read adopt a fresh container's
    files. Bypass the recreating override via ``BaseEnvironment.execute`` and
    refuse when the container ID changed around the call.
    """
    before, _ = capture_sandbox_identity(env)
    try:
        if _is_recreating_docker_env(env):
            from tools.environments.base import BaseEnvironment
            result = BaseEnvironment.execute(env, command, timeout=timeout)
        else:
            result = env.execute(command, timeout=timeout)
    except Exception as exc:
        raise _SupervisionTransportError(f"probe: {exc}") from exc
    after, _ = capture_sandbox_identity(env)
    if before and after and before != after:
        raise _ContainerReplacedError(
            f"container replaced during probe ({before[:12]} -> {after[:12]})")
    return result


def pinned_launch_execute(env: Any, command: str, timeout: int, **kwargs) -> dict:
    """One-shot wrapper launch without backend auto-recreate-and-retry.

    A recreating ``execute`` would re-run the nohup wrapper after an ambiguous
    launch, double-executing the original job in a fresh container. Bypass it
    so at most one launch runs; a gone container fails fast as
    ``failed_start`` instead of risking duplicate side effects.
    """
    if _is_recreating_docker_env(env):
        from tools.environments.base import BaseEnvironment
        return BaseEnvironment.execute(env, command, timeout=timeout, **kwargs)
    return env.execute(command, timeout=timeout, **kwargs)


SANDBOX_PIDFD_UNSUPPORTED_EXIT = 10


_SANDBOX_PIDFD_HELPER_PY = (
    "import os, signal, sys\n"
    "def _ticks(pid):\n"
    "    try:\n"
    "        with open(\"/proc/%d/stat\" % pid, encoding=\"utf-8-sig\") as f:\n"
    "            t = f.read().strip()\n"
    "        if not t:\n"
    "            return None\n"
    "        a = t.rsplit(\")\", 1)[-1].strip() if \")\" in t else t\n"
    "        p = a.split()\n"
    "        if \")\" in t:\n"
    "            return p[19] if len(p) >= 20 else None\n"
    "        return p[21] if len(p) >= 22 else None\n"
    "    except Exception:\n"
    "        return None\n"
    "def main():\n"
    "    if len(sys.argv) != 6:\n"
    "        sys.exit(3)\n"
    "    _, pid_s, exp_ticks, exp_token, pid_path, token_path = sys.argv\n"
    "    if not exp_token:\n"
    "        sys.exit(4)\n"
    "    try:\n"
    "        pid = int(pid_s)\n"
    "    except Exception:\n"
    "        sys.exit(3)\n"
    "    pfo = getattr(os, 'pidfd_open', None)\n"
    "    pss = getattr(signal, 'pidfd_send_signal', None)\n"
    "    if pfo is None or pss is None:\n"
    "        sys.stdout.write(\"UNSUPPORTED\\n\")\n"
    "        sys.exit(10)\n"
    "    try:\n"
    "        fd = pfo(pid, 0)\n"
    "    except OSError as e:\n"
    "        import errno as _e\n"
    "        if e.errno in (_e.ENOSYS, _e.EINVAL):\n"
    "            sys.stdout.write(\"UNSUPPORTED\\n\")\n"
    "            sys.exit(10)\n"
    "        sys.exit(3)\n"
    "    except Exception:\n"
    "        sys.exit(3)\n"
    "    try:\n"
    "        if pid_path != \"-\":\n"
    "            try:\n"
    "                with open(pid_path, encoding=\"utf-8-sig\") as f:\n"
    "                    parts = f.read().strip().split()\n"
    "                if not parts:\n"
    "                    sys.exit(3)\n"
    "                if int(parts[-1]) != pid:\n"
    "                    sys.exit(3)\n"
    "            except SystemExit:\n"
    "                raise\n"
    "            except Exception:\n"
    "                sys.exit(3)\n"
    "        if token_path != \"-\":\n"
    "            try:\n"
    "                with open(token_path, encoding=\"utf-8-sig\") as f:\n"
    "                    lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]\n"
    "                live_tok = lines[0] if lines else ''\n"
    "            except SystemExit:\n"
    "                raise\n"
    "            except Exception:\n"
    "                sys.exit(4)\n"
    "            if not exp_token or live_tok != exp_token:\n"
    "                sys.exit(4)\n"
    "        elif exp_token:\n"
    "            sys.exit(4)\n"
    "        live = _ticks(pid)\n"
    "        if live is None:\n"
    "            sys.exit(5)\n"
    "        if not exp_ticks or live != exp_ticks:\n"
    "            sys.exit(6)\n"
    "        try:\n"
    "            pss(fd, signal.SIGTERM)\n"
    "        except OSError as e:\n"
    "            import errno as _e2\n"
    "            if e.errno in (_e2.ENOSYS, _e2.EINVAL):\n"
    "                sys.stdout.write(\"UNSUPPORTED\\n\")\n"
    "                sys.exit(10)\n"
    "            sys.exit(7)\n"
    "        except Exception:\n"
    "            sys.exit(7)\n"
    "        sys.exit(0)\n"
    "    finally:\n"
    "        try:\n"
    "            os.close(fd)\n"
    "        except Exception:\n"
    "            pass\n"
    "main()\n"
)


def sandbox_pidfd_kill_command(*, pid_path: str = "", token_path: str = "",
                               live_pid: Optional[int],
                               expected_token: str = "",
                               expected_startticks: str = "") -> str:
    """Identity-safe kill helper executed INSIDE the target container.

    Opens a pidfd for ``live_pid``, verifies pid-file/token/start-ticks AFTER
    the open, then signals via ``pidfd_send_signal`` in the SAME helper — no
    fd crosses the transport, so a PID number reused between authorization and
    signal cannot redirect the signal to a stranger. The helper is optional
    inside-Python (no install): missing ``python3`` or missing
    ``pidfd_open``/``pidfd_send_signal`` exits 10 (unsupported, fail closed).
    Exit 0 means the pidfd signal was sent; 3/4/5/6/7 mean nothing was
    signalled (pid/token/ticks mismatch or signal failure). The command never
    contains a plain shell ``kill``. ``expected_startticks`` is required: an
    empty value refuses (exit 5) instead of signalling on pidfile/token alone.
    """
    import shlex
    target = int(live_pid)
    ticks = str(expected_startticks or "")
    token = str(expected_token or "")
    pid_arg = str(pid_path) if pid_path else "-"
    token_arg = str(token_path) if token_path else "-"
    if token and not token_path:
        token_arg = "-"
    inner = (
        "command -v python3 >/dev/null 2>&1 || exit 10; "
        "exec python3 -c " + shlex.quote(_SANDBOX_PIDFD_HELPER_PY) + " "
        + shlex.quote(str(target)) + " "
        + shlex.quote(ticks) + " "
        + shlex.quote(token) + " "
        + shlex.quote(pid_arg) + " "
        + shlex.quote(token_arg)
    )
    return inner


def _is_docker_like_env(env: Any) -> bool:
    """True when execs target a Docker container via ``_docker_exe``/``_container_id``."""
    exe = getattr(env, "_docker_exe", None)
    cid = getattr(env, "_container_id", None)
    return bool(exe and isinstance(cid, str) and cid)


def _docker_exec_pinned(env: Any, expected_container_id: str, shell_command: str,
                        timeout: int, **kwargs) -> dict:
    """Run ``shell_command`` in ``expected_container_id`` without reading mutable state.

    Builds the ``docker exec`` argv from the immutable snapshot, never from
    ``env._container_id``: a concurrent mutation of the shared env object cannot
    redirect the signal into a stranger's namespace. No auto-recreate-and-retry:
    a gone container fails fast instead of signalling a fresh replacement.
    Uses the backend's ``_docker_exe`` seam; values never touch the shell
    unquoted (``bash -c`` tail matches ``DockerEnvironment._run_bash``).
    """
    import subprocess as _sp
    docker_exe = getattr(env, "_docker_exe", None) or "docker"
    argv = [str(docker_exe), "exec", str(expected_container_id),
            "bash", "-c", str(shell_command)]
    try:
        from tools.environments.remote_common import run_capture as _cap
        completed = _cap(argv, timeout=float(timeout or 5))
        out = (getattr(completed, "stdout", "") or "")
        err = (getattr(completed, "stderr", "") or "")
        combined = out + (err if err and err not in out else "")
        code = getattr(completed, "returncode", 1)
        return {"output": combined, "returncode": int(code) if code is not None else 1}
    except Exception as exc:
        raise _SupervisionTransportError(f"pinned docker exec: {exc}") from exc


def pinned_kill_execute(env: Any, expected_container_id: str, command: str,
                        timeout: int, **kwargs) -> dict:
    """Container-bound one-shot exec for sandbox signalling with pinned target.

    The ``docker exec`` argv carries the immutable ``expected_container_id``
    snapshot (never the mutable ``env._container_id``), so authorization cannot
    be redirected after the check. A pre-call mismatch refuses with zero execs;
    a post-call mismatch raises (verification is then uncertain, even though the
    signal itself went to the pinned target). Non-Docker backends (Local,
    fakes) execute directly with the same pre/post guards. Raises
    ``_SupervisionTransportError`` on mismatch/transport failure.
    """
    current, _ = capture_sandbox_identity(env)
    if expected_container_id and current and current != expected_container_id:
        raise _SupervisionTransportError(
            f"refusing signal: container replaced before kill "
            f"({expected_container_id[:12]} -> {current[:12]})")
    if _is_docker_like_env(env) and expected_container_id:
        result = _docker_exec_pinned(
            env, expected_container_id, command, timeout=timeout, **kwargs)
    elif _is_recreating_docker_env(env):
        from tools.environments.base import BaseEnvironment
        result = BaseEnvironment.execute(env, command, timeout=timeout, **kwargs)
    else:
        result = env.execute(command, timeout=timeout, **kwargs)
    after, _ = capture_sandbox_identity(env)
    if expected_container_id and after and after != expected_container_id:
        raise _ContainerReplacedError(
            f"container replaced during kill ({expected_container_id[:12]} -> {after[:12]})")
    return result


def _exec_probe_pinned(env: Any, command: str, timeout: int, label: str) -> str:
    """Like ``_exec_probe`` but through the non-recreating pinned path."""
    try:
        result = _pinned_env_exec(env, command, timeout)
    except _ContainerReplacedError:
        raise
    except _SupervisionTransportError:
        raise
    output = result.get("output", "")
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(f"{label} transport failed: {output[:120]}")
    lowered = str(output).lower()
    if "timed out" in lowered and "command timed out" in lowered:
        raise _SupervisionTransportError(f"{label} timed out")
    return output


def _read_token_file(env: Any, quoted_token_path: str) -> Optional[str]:
    """The spawn run-token file content, or None when absent/unreadable-clean.

    Transport failures raise ``_SupervisionTransportError`` (uncertain, retry);
    a clean transport with no token is a fact (None) that forces ``unknown``.
    """
    try:
        result = _pinned_env_exec(env, f"cat {quoted_token_path} 2>/dev/null", 5)
    except _ContainerReplacedError:
        raise
    except _SupervisionTransportError:
        raise
    except Exception as exc:
        raise _SupervisionTransportError(f"token read: {exc}") from exc
    if int(result.get("returncode", 0) or 0) != 0:
        raise _SupervisionTransportError(
            f"token read transport failed: {result.get('output', '')[:120]}")
    lines = [line.strip() for line in result.get("output", "").splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def _read_exit_receipt(env: Any, quoted_exit_path: str, expected_token: str = "") -> Optional[int]:
    """Framed exit-receipt read distinguishing absent receipt from failure.

    ``cat`` on a nonexistent file exits nonzero during every healthy running
    poll; treating that as a transport failure would burn the retry budget while
    the job runs. The framed ``if [ -f ... ]`` command succeeds with a
    ``__NO_RECEIPT__`` marker when nothing was written yet, raises transport
    error only on real failures, and token-binds the receipt before trusting it.
    """
    try:
        output = _exec_probe_pinned(
            env,
            f"if [ -f {quoted_exit_path} ]; then cat {quoted_exit_path}; "
            f"else printf '__NO_RECEIPT__\n'; fi",
            5, "exit read")
    except _ContainerReplacedError:
        raise
    return parse_verified_exit(output, expected_token)


def run_env_supervision(
    registry: Any, session: Any, env: Any, log_path: str, pid_path: str, exit_path: str
) -> None:
    """Poll one sandbox background process until its verified exit or a durable
    ``unresolved`` escalation. Runs on the session's reader thread; returns when
    the session finishes, escalates, or a newer poller generation takes over
    (``reattach_env_session`` after a restart)."""
    import shlex

    q = shlex.quote
    generation = session._supervision_generation
    prev_output_bytes = 0
    failures = 0

    def _current() -> bool:
        return (
            not session.exited
            and getattr(session, "supervision_state", "ok") != "unresolved"
            and session._supervision_generation == generation
        )

    while _current():
        time.sleep(supervision_backoff_delay(failures) if failures else ENV_SUPERVISION_POLL_S)
        if not _current():
            return
        try:
            prev_output_bytes = _poll_cycle(
                registry, session, env, q, log_path, pid_path, exit_path, prev_output_bytes,
                generation)
        except _SupervisionTransportError as exc:
            failures += 1
            registry._note_supervision_degraded(session, str(exc), failures, generation)
            logger.warning(
                "Background process %s supervision uncertain (%d/%d): %s",
                session.id, failures, ENV_SUPERVISION_MAX_FAILURES, exc)
            if failures >= ENV_SUPERVISION_MAX_FAILURES:
                _reconcile_or_escalate(
                    registry, session, env, q, pid_path, exit_path, str(exc), generation)
                return
            continue
        failures = 0
        if getattr(session, "supervision_state", "ok") != "ok":
            with session._lock:
                session.supervision_state = "ok"
                session.last_supervision_error = ""


def _poll_ingest_log(registry: Any, session: Any, env: Any, q: Any,
                     log_path: str, prev_output_bytes: int) -> int:
    """One log-delta read; returns the new offset. Raises on uncertain transport."""
    raw = _exec_probe_pinned(
        env, registry._log_delta_command(q(log_path), prev_output_bytes),
        timeout=10, label="log read")
    header, _, delta = raw.partition("\n")
    try:
        size_str, offset_str = header.split()
        new_size = int(size_str)
        used_offset = int(offset_str)
    except ValueError:
        raise _SupervisionTransportError(f"unusable log header: {header[:120]!r}")
    if used_offset < prev_output_bytes:
        with session._lock:
            if session.exited:
                return prev_output_bytes
            session.output_buffer = ""
    prev_output_bytes = new_size
    if delta:
        with session._lock:
            if session.exited:
                return prev_output_bytes
            session.output_buffer += delta
            if len(session.output_buffer) > session.max_output_chars:
                session.output_buffer = session.output_buffer[-session.max_output_chars:]
        registry._check_watch_patterns(session, delta)
        registry._emit_output(session, delta)
    return prev_output_bytes


def _poll_drain_final_log(registry: Any, session: Any, env: Any, q: Any,
                          log_path: str, prev_output_bytes: int) -> int:
    """Best-effort final log drain after a verified receipt; never fails the finish."""
    try:
        raw = _exec_probe_pinned(
            env, registry._log_delta_command(q(log_path), prev_output_bytes),
            timeout=10, label="final log drain")
    except _SupervisionTransportError as exc:
        logger.debug("Final log drain uncertain, finishing from receipt: %s", exc, exc_info=True)
        return prev_output_bytes
    header, _, delta = raw.partition("\n")
    try:
        size_str, _offset_str = header.split()
        prev_output_bytes = int(size_str)
    except ValueError:
        logger.debug("Final log drain header unusable: %r", header[:120], exc_info=True)
    if delta:
        with session._lock:
            if not session.exited:
                session.output_buffer += delta
                if len(session.output_buffer) > session.max_output_chars:
                    session.output_buffer = session.output_buffer[-session.max_output_chars:]
    return prev_output_bytes


def _poll_handle_receipt(registry: Any, session: Any, env: Any, q: Any,
                         log_path: str, exit_path: str, prev_output_bytes: int,
                         stored_token: str, live_token: Optional[str],
                         generation: Any = None) -> Optional[int]:
    """Verified-receipt phase; returns new offset when handled, None to continue."""
    try:
        receipt = _read_exit_receipt(env, q(exit_path), stored_token)
    except _ContainerReplacedError as exc:
        registry._mark_supervision_unresolved(session, str(exc), generation)
        return prev_output_bytes
    if receipt is None:
        return None
    if sandbox_identity_verdict(
            session.sandbox_container_id, session.sandbox_pid, env,
            None, stored_token, live_token) != "match":
        registry._mark_supervision_unresolved(
            session, "exit receipt present but run identity unverified (refusing to trust it)",
            generation)
        return prev_output_bytes
    prev_output_bytes = _poll_drain_final_log(
        registry, session, env, q, log_path, prev_output_bytes)
    registry._finish_env_exit(session, receipt, generation)
    return prev_output_bytes


def _poll_check_liveness(registry: Any, session: Any, env: Any, q: Any,
                         pid_path: str, stored_token: str,
                         live_token: Optional[str],
                         generation: Any = None) -> Optional[int]:
    """Liveness phase; returns offset when resolved, None never (raises or escalates)."""
    verdict = sandbox_identity_verdict(
        session.sandbox_container_id, session.sandbox_pid, env,
        None, stored_token, live_token)
    if verdict == "mismatch":
        registry._mark_supervision_unresolved(
            session, "sandbox run identity changed (replacement, not continuity)",
            generation)
        return 0
    if verdict == "unknown":
        raise _SupervisionTransportError(
            "sandbox run identity unverifiable (no token/container continuity proof)")
    alive_raw = _exec_probe_pinned(
        env, f"kill -0 \"$(cat {q(pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?",
        timeout=5, label="liveness probe")
    if alive_raw.strip().splitlines()[-1].strip() == "0":
        return 0
    pid_value = _read_pid_file(env, q(pid_path))
    if sandbox_identity_verdict(
            session.sandbox_container_id, session.sandbox_pid, env, pid_value,
            stored_token, live_token) != "match":
        registry._mark_supervision_unresolved(
            session, "worker unreachable and wrapper identity changed (no verified exit receipt)",
            generation)
        return 0
    raise _SupervisionTransportError("worker not alive and no exit receipt yet")


def _poll_cycle(
    registry: Any, session: Any, env: Any, q: Any,
    log_path: str, pid_path: str, exit_path: str, prev_output_bytes: int,
    generation: Any = None,
) -> int:
    """One full poll cycle; returns the new log offset. Raises on uncertain probe."""
    prev_output_bytes = _poll_ingest_log(registry, session, env, q, log_path, prev_output_bytes)
    stored_token = getattr(session, "sandbox_run_token", "") or ""
    token_path = getattr(session, "env_token_path", "") or ""
    try:
        live_token: Optional[str] = None
        if stored_token and token_path:
            live_token = _read_token_file(env, q(token_path))
    except _ContainerReplacedError as exc:
        registry._mark_supervision_unresolved(session, str(exc), generation)
        return prev_output_bytes
    except _SupervisionTransportError:
        raise
    handled = _poll_handle_receipt(
        registry, session, env, q, log_path, exit_path, prev_output_bytes,
        stored_token, live_token, generation)
    if handled is not None:
        return handled
    alive = _poll_check_liveness(registry, session, env, q, pid_path, stored_token, live_token,
                                 generation)
    if alive is not None:
        return prev_output_bytes
    return prev_output_bytes


def _reconcile_or_escalate(
    registry: Any, session: Any, env: Any, q: Any,
    pid_path: str, exit_path: str, last_error: str,
    generation: Any = None,
) -> None:
    """Final reconcile after the retry bound: one verified-receipt read wins, else
    a durable ``unresolved`` escalation. Never invents an exit, never relaunches."""
    stored_token = getattr(session, "sandbox_run_token", "") or ""
    token_path = getattr(session, "env_token_path", "") or ""
    live_token: Optional[str] = None
    try:
        if stored_token and token_path:
            live_token = _read_token_file(env, q(token_path))
    except _SupervisionTransportError:
        live_token = None
    try:
        receipt = _read_exit_receipt(env, q(exit_path), stored_token)
    except _SupervisionTransportError as exc:
        registry._mark_supervision_unresolved(
            session, f"supervision lost after {ENV_SUPERVISION_MAX_FAILURES} attempts: {exc}",
            generation)
        return
    if receipt is not None:
        if sandbox_identity_verdict(
                session.sandbox_container_id, session.sandbox_pid, env, None,
                stored_token, live_token) != "match":
            registry._mark_supervision_unresolved(
                session, "supervision lost: exit receipt present but identity unverified",
                generation)
            return
        registry._finish_env_exit(session, receipt, generation)
        return
    pid_value: Optional[int] = None
    try:
        pid_value = _read_pid_file(env, q(pid_path))
    except _SupervisionTransportError:
        pid_value = None
    if sandbox_identity_verdict(
            session.sandbox_container_id, session.sandbox_pid, env, pid_value,
            stored_token, live_token) != "match":
        registry._mark_supervision_unresolved(
            session, "supervision lost and wrapper identity changed (no verified exit receipt)",
            generation)
        return
    registry._mark_supervision_unresolved(
        session, f"supervision lost after {ENV_SUPERVISION_MAX_FAILURES} attempts: {last_error}",
        generation)
