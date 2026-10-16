"""Sandbox (env-backed) background-process lifecycle for ProcessRegistry.

Sandbox spawn, spawn-identity capture, verified-exit finish, supervision
degraded/unresolved escalation, and identity-verified reattach. The facade
(``tools/process_registry.py``) stays thin; policy lives in
``tools/process_registry_supervision.py`` (probes, verdicts, backoff).
"""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


logger = logging.getLogger("tools.process_registry")


class ProcessSandboxMixin:
    def has_active_for_container(self, container_key: str, env: object = None) -> bool:
        """Whether any running session is leased to container ``container_key``.

        Container scope, not owner scope: matches ``session.task_id`` (the
        collapsed/profile-qualified container key the idle reaper caches under)
        or the live ``env_ref`` identity when the same env object is shared
        across alias keys. Owner scope stays in ``has_active_processes`` for
        delegated lifetime; the idle reaper must use this, never the owner
        query, or a ``default`` container running an owner-scoped job reads as
        idle and its handle is invalidated mid-job."""
        return self._any_running(  # type: ignore[attr-defined]
            lambda sess: sess.task_id == container_key or (env is not None and sess.env_ref is env))
    """The subclass supplies the registry core (``_new_session``, ``get``,
    ``_env_temp_dir``, ``_track_started``, ``_prune_if_needed``,
    ``_finish_exited``, ``_write_checkpoint``). See ProcessRegistry."""

    def spawn_via_env(
        self, env: Any, command: str, cwd: str | None = None, task_id: str = "", session_key: str = "",
        timeout: int = 10, owner_task_id: str = "", persist_on_release: bool = False) -> ProcessSession:
        """Spawn a background process inside a non-local backend's sandbox.
        The command is wrapped to capture its in-sandbox PID and redirect output to a
        log file that later execute() calls poll. No live pipe or stdin, but it runs in
        the correct sandbox context. ``persist_on_release`` keeps the process out of
        agent-lifecycle kill sweeps (#41225). ``timeout`` is the transport/start
        timeout only, never a job budget: no default budget is imposed so servers
        are never killed; a caller that needs a bound owns it (GNU ``timeout``).
        Wrapper identity (PID + start ticks + boot ID) is captured at spawn
        atomically in the same pinned namespace and verified before any signal."""
        from tools.process_registry_supervision import sandbox_wrapper_inner, capture_sandbox_identity, pinned_launch_execute
        import secrets as _secrets
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd, env_ref=env, pid_scope="sandbox",
                                    persist_on_release=persist_on_release)
        container_id, backend = capture_sandbox_identity(env)
        session.sandbox_container_id = container_id
        session.sandbox_backend = backend
        run_token = _secrets.token_hex(8)
        session.sandbox_run_token = run_token
        temp_dir = self._env_temp_dir(env)
        log_path, pid_path, exit_path = (f"{temp_dir}/hermes_bg_{session.id}.{ext}" for ext in ("log", "pid", "exit"))
        token_path = f"{temp_dir}/hermes_bg_{session.id}.token"
        start_path = f"{temp_dir}/hermes_bg_{session.id}.start"
        boot_path = f"{temp_dir}/hermes_bg_{session.id}.boot"
        session.env_log_path, session.env_pid_path, session.env_exit_path = log_path, pid_path, exit_path
        session.env_token_path = token_path
        session.env_start_path = start_path
        session.env_boot_path = boot_path
        q = shlex.quote
        bg_command = (
            f"mkdir -p {q(temp_dir)} && "
            f"printf '%s\\n' {q(run_token)} > {q(token_path)} && "
            f"( nohup {sandbox_wrapper_inner(command, q)} > {q(log_path)} 2>&1; "
            f"rc=$?; printf '%s\\n%s\\n' {q(run_token)} \"$rc\" > {q(exit_path)}.tmp && "
            f"mv {q(exit_path)}.tmp {q(exit_path)} ) & "
            f"WRAP_PID=$!; echo $WRAP_PID > {q(pid_path)}; "
            f"cat /proc/$WRAP_PID/stat > {q(start_path)} 2>/dev/null || true; "
            f"cat /proc/sys/kernel/random/boot_id > {q(boot_path)} 2>/dev/null || true; "
            f"cat {q(pid_path)}")
        try:
            result = pinned_launch_execute(env, bg_command, timeout=timeout, rewrite_compound_background=False)
            output = result.get("output", "").strip()
            session.pid = next((int(ln) for ln in map(str.strip, output.splitlines()) if ln.isdigit()), None)
            session.sandbox_pid = session.pid
            if session.pid is not None:
                self._capture_sandbox_spawn_identity(session)
            # No PID from the wrapper (syntax error, broken redirect): a failed launch,
            # not a fake running session.
            if session.pid is None:
                session.mark_exited(int(result.get("returncode", -1)) or -1, "failed_start", "failed_start")
                session.output_buffer = output
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Sandbox spawn transport failed for %s: %s", session.id, exc, exc_info=True)
            session.mark_exited(-1, "failed_start", "failed_start")
            session.output_buffer = f"Failed to start: {exc}"
        except Exception as exc:
            logger.warning("Sandbox spawn failed for %s: %s", session.id, exc, exc_info=True)
            session.mark_exited(-1, "failed_start", "failed_start")
            session.output_buffer = f"Failed to start: {exc}"
        if session.exited:
            with self._lock:
                self._prune_if_needed()
        else:
            self._track_started(
                session, self._env_poller_loop, f"proc-poller-{session.id}", (env, log_path, pid_path, exit_path))
        return session

    # ----- Reader / Poller Threads -----

    def _capture_sandbox_spawn_identity(self, session) -> None:
        """Capture wrapper start ticks + boot ID in the pinned namespace.

        Best effort at spawn: failures leave empty fields, which later verify as
        ``unknown`` (fail closed, never signal). Never raises."""
        try:
            from tools.process_registry_supervision import (
                _parse_proc_stat_startticks, _pinned_env_exec)
            import shlex as _shlex
            if session.env_start_path:
                try:
                    res = _pinned_env_exec(
                        session.env_ref, f"cat {_shlex.quote(session.env_start_path)} 2>/dev/null", 5)
                    if int(res.get("returncode", 0) or 0) == 0:
                        parsed = _parse_proc_stat_startticks(res.get("output", ""))
                        if parsed:
                            session.sandbox_wrapper_startticks = parsed
                except Exception as exc:
                    logger.debug("Spawn start-ticks capture uncertain for %s: %s", session.id, exc, exc_info=True)
            if session.env_boot_path:
                try:
                    res = _pinned_env_exec(
                        session.env_ref, f"cat {_shlex.quote(session.env_boot_path)} 2>/dev/null", 5)
                    if int(res.get("returncode", 0) or 0) == 0:
                        lines = [ln.strip() for ln in res.get("output", "").splitlines() if ln.strip()]
                        if lines:
                            session.sandbox_boot_id = lines[0]
                except Exception as exc:
                    logger.debug("Spawn boot-ID capture uncertain for %s: %s", session.id, exc, exc_info=True)
        except Exception as exc:
            logger.debug("Spawn identity capture failed for %s: %s", getattr(session, "id", "?"), exc, exc_info=True)

    def _finish_env_exit(self, session: ProcessSession, exit_code: int) -> None:
        """Finish on a verified wrapper exit receipt (a raced kill keeps its own)."""
        self._finish_exited(session, exit_code, reason="exited", source="")

    def _note_supervision_degraded(self, session: ProcessSession, error: str, failures: int) -> None:
        """Record one uncertain cycle; the session keeps running and retrying."""
        with session._lock:
            if session.exited:
                return
            session.supervision_state = "degraded"
            session.supervision_failures = failures
            session.last_supervision_error = error[:300]

    def _mark_supervision_unresolved(self, session: ProcessSession, error: str) -> None:
        """Durable escalation: the retry bound is exhausted (or identity changed) with
        no verified exit. The session stays tracked with its evidence — checkpointed,
        visible to poll(), surfaced once by the gateway watcher — and is never finished
        with an invented exit code. ``reattach_env_session`` can still resume it."""
        with session._lock:
            if session.exited:
                return
            session.supervision_state = "unresolved"
            session.supervision_failures += 1
            session.last_supervision_error = error[:300]
            session.completion_reason = "unresolved"
            session.termination_source = "backend_unreachable"
        logger.warning("Background process %s supervision unresolved: %s", session.id, error)
        self._write_checkpoint()

    def reattach_env_session(self, session_id: str, env: Any) -> bool:
        """Resume supervision of an ``unresolved`` sandbox session through a live env
        handle after a restart. Identity (container, token, wrapper start ticks +
        boot ID in the same pinned namespace) must match: a replacement or an
        unverifiable legacy run is refused and stays ``unresolved``. ``env_ref``
        is updated only after the candidate verifies; the old handle is never
        reused after success. Reads never recreate or relaunch work."""
        from tools.process_registry_supervision import (
            _read_pid_file, _read_token_file, _read_wrapper_startticks,
            _read_boot_id, sandbox_signal_verdict)
        from tools.process_registry_supervision import _SupervisionTransportError
        import shlex
        session = self.get(session_id)
        if session is None or session.exited or session.pid_scope != "sandbox":
            return False
        try:
            pid_value = _read_pid_file(env, shlex.quote(session.env_pid_path))
        except _SupervisionTransportError as exc:
            logger.debug("Reattach %s pid read uncertain: %s", session_id, exc, exc_info=True)
            return False
        except (OSError, ValueError) as exc:
            logger.debug("Reattach %s pid read failed: %s", session_id, exc, exc_info=True)
            return False
        except Exception as exc:
            logger.debug("Reattach %s pid read failed: %s", session_id, exc, exc_info=True)
            return False
        live_token = None
        if getattr(session, "sandbox_run_token", "") and getattr(session, "env_token_path", ""):
            try:
                live_token = _read_token_file(env, shlex.quote(session.env_token_path))
            except _SupervisionTransportError as exc:
                logger.debug("Reattach %s token read uncertain: %s", session_id, exc, exc_info=True)
                return False
            except (OSError, ValueError) as exc:
                logger.debug("Reattach %s token read failed: %s", session_id, exc, exc_info=True)
                return False
            except Exception as exc:
                logger.debug("Reattach %s token read failed: %s", session_id, exc, exc_info=True)
                return False
        try:
            live_ticks = _read_wrapper_startticks(env, shlex.quote(session.env_pid_path))
        except _SupervisionTransportError as exc:
            logger.debug("Reattach %s start-ticks uncertain: %s", session_id, exc, exc_info=True)
            return False
        except Exception as exc:
            logger.debug("Reattach %s start-ticks failed: %s", session_id, exc, exc_info=True)
            return False
        try:
            live_boot: object = _read_boot_id(env)
        except _SupervisionTransportError as exc:
            logger.debug("Reattach %s boot-ID uncertain: %s", session_id, exc, exc_info=True)
            return False
        except Exception as exc:
            logger.debug("Reattach %s boot-ID failed: %s", session_id, exc, exc_info=True)
            return False
        if sandbox_signal_verdict(
                session.sandbox_container_id, session.sandbox_pid, env, pid_value,
                getattr(session, "sandbox_run_token", ""), live_token,
                getattr(session, "sandbox_wrapper_startticks", "") or "",
                live_ticks,
                getattr(session, "sandbox_boot_id", "") or "",
                live_boot if isinstance(live_boot, str) else None) != "match":
            logger.warning(
                "Refusing to reattach %s: live backend identity does not match the checkpointed run",
                session_id)
            return False
        with session._lock:
            session.env_ref = env
            session.supervision_state = "ok"
            session.supervision_failures = 0
            session.last_supervision_error = ""
            session.completion_reason = "exited"
            session.termination_source = ""
            session._supervision_generation += 1
            session._unresolved_surfaced = False
        self._track_started(
            session, self._env_poller_loop, f"proc-poller-{session.id}",
            (env, session.env_log_path, session.env_pid_path, session.env_exit_path))
        logger.info("Resumed supervision of %s in its original sandbox", session_id)
        return True
