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

# Exit code GNU ``timeout`` reports when the job budget fires.
BUDGET_TIMEOUT_EXIT = 124


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


def budget_wrapper_inner(command: str, budget_seconds: Any, quote) -> str:
    """The inner command for the ``spawn_via_env`` nohup wrapper.

    ``budget_seconds`` is an explicit job budget owned by the wrapper, independent
    of client polling: GNU ``timeout`` sends TERM at the budget and the wrapper's
    exit file keeps 124, so a later exit 0 of the child can never overwrite the
    timeout verdict. ``None`` leaves the wrapper unbounded, as before.
    """
    if budget_seconds is None:
        return f"bash -lc {quote(command)}"
    budget = int(budget_seconds)
    if budget <= 0:
        raise ValueError(f"budget_seconds must be positive (got {budget_seconds!r})")
    return f"timeout -s TERM -k 5 {budget} bash -lc {quote(command)}"


def budget_completion_reason(exit_code: int, budget_seconds: int) -> tuple[str, str]:
    """(reason, source) for a verified wrapper exit: a 124 under a budget is the
    budget firing, still a non-success even though the child may have exited 0
    after the signal — the recorded code is the wrapper's, never the child's."""
    if budget_seconds and exit_code == BUDGET_TIMEOUT_EXIT:
        return "budget_timeout", "budget"
    return "exited", ""


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
    marker so test fakes and renamed subclasses still pin correctly.
    """
    try:
        from tools.environments.docker import DockerEnvironment
        if isinstance(env, DockerEnvironment):
            return True
    except Exception:
        pass
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
                registry, session, env, q, log_path, pid_path, exit_path, prev_output_bytes)
        except _SupervisionTransportError as exc:
            failures += 1
            registry._note_supervision_degraded(session, str(exc), failures)
            logger.warning(
                "Background process %s supervision uncertain (%d/%d): %s",
                session.id, failures, ENV_SUPERVISION_MAX_FAILURES, exc)
            if failures >= ENV_SUPERVISION_MAX_FAILURES:
                _reconcile_or_escalate(registry, session, env, q, pid_path, exit_path, str(exc))
                return
            continue
        failures = 0
        if getattr(session, "supervision_state", "ok") != "ok":
            with session._lock:
                session.supervision_state = "ok"
                session.last_supervision_error = ""


def _poll_cycle(
    registry: Any, session: Any, env: Any, q: Any,
    log_path: str, pid_path: str, exit_path: str, prev_output_bytes: int,
) -> int:
    """One full poll cycle; returns the new log offset. Raises
    ``_SupervisionTransportError`` on any uncertain probe."""
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

    stored_token = getattr(session, "sandbox_run_token", "") or ""
    token_path = getattr(session, "env_token_path", "") or ""
    try:
        live_token: Optional[str] = None
        if stored_token and token_path:
            live_token = _read_token_file(env, q(token_path))
    except _ContainerReplacedError as exc:
        registry._mark_supervision_unresolved(session, str(exc))
        return prev_output_bytes
    except _SupervisionTransportError:
        raise
    # Verified exit receipt FIRST: token-bound, framed (absent != failed), and
    # identity-checked before trusting. kill -0 races the wrapper's final write.
    try:
        receipt = _read_exit_receipt(env, q(exit_path), stored_token)
    except _ContainerReplacedError as exc:
        registry._mark_supervision_unresolved(session, str(exc))
        return prev_output_bytes
    if receipt is not None:
        if sandbox_identity_verdict(
                session.sandbox_container_id, session.sandbox_pid, env,
                None, stored_token, live_token) != "match":
            registry._mark_supervision_unresolved(
                session, "exit receipt present but run identity unverified (refusing to trust it)")
            return prev_output_bytes
        try:
            raw = _exec_probe_pinned(
                env, registry._log_delta_command(q(log_path), prev_output_bytes),
                timeout=10, label="final log drain")
            header, _, delta = raw.partition("\n")
            try:
                size_str, _offset_str = header.split()
                prev_output_bytes = int(size_str)
            except ValueError:
                pass
            if delta:
                with session._lock:
                    if not session.exited:
                        session.output_buffer += delta
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
        except _SupervisionTransportError:
            pass
        registry._finish_env_exit(session, receipt)
        return prev_output_bytes

    # Identity-safe liveness SECOND: a replaced container is never continuity;
    # an unknown identity is uncertainty, never an exit and never a signal.
    verdict = sandbox_identity_verdict(
        session.sandbox_container_id, session.sandbox_pid, env,
        None, stored_token, live_token)
    if verdict == "mismatch":
        registry._mark_supervision_unresolved(
            session, "sandbox run identity changed (replacement, not continuity)")
        return prev_output_bytes
    if verdict == "unknown":
        raise _SupervisionTransportError(
            "sandbox run identity unverifiable (no token/container continuity proof)")
    alive_raw = _exec_probe_pinned(
        env, f"kill -0 \"$(cat {q(pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?",
        timeout=5, label="liveness probe")
    if alive_raw.strip().splitlines()[-1].strip() == "0":
        return prev_output_bytes

    # Dead without a receipt: cross-check the wrapper PID file before concluding
    # anything — a changed PID file is a replacement; an unreadable backend is
    # uncertainty, not an exit.
    pid_value = _read_pid_file(env, q(pid_path))
    if sandbox_identity_verdict(
            session.sandbox_container_id, session.sandbox_pid, env, pid_value,
            stored_token, live_token) != "match":
        registry._mark_supervision_unresolved(
            session, "worker unreachable and wrapper identity changed (no verified exit receipt)")
        return prev_output_bytes
    raise _SupervisionTransportError("worker not alive and no exit receipt yet")


def _reconcile_or_escalate(
    registry: Any, session: Any, env: Any, q: Any,
    pid_path: str, exit_path: str, last_error: str,
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
            session, f"supervision lost after {ENV_SUPERVISION_MAX_FAILURES} attempts: {exc}")
        return
    if receipt is not None:
        if sandbox_identity_verdict(
                session.sandbox_container_id, session.sandbox_pid, env, None,
                stored_token, live_token) != "match":
            registry._mark_supervision_unresolved(
                session, "supervision lost: exit receipt present but identity unverified")
            return
        registry._finish_env_exit(session, receipt)
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
            session, "supervision lost and wrapper identity changed (no verified exit receipt)")
        return
    registry._mark_supervision_unresolved(
        session, f"supervision lost after {ENV_SUPERVISION_MAX_FAILURES} attempts: {last_error}")
