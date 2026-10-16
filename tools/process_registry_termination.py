"""Process-tree termination and post-kill survivor verification for ProcessRegistry."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from contextlib import suppress
from typing import TYPE_CHECKING, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger("tools.process_registry")


class ProcessTerminationMixin:
    """The subclass supplies ``_host_pid_is_ours``, ``_is_host_pid_alive``,
    ``_detached_host_fate`` and ``_daemon_term_grace_seconds``. See ProcessRegistry."""

    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie (already dead, just unreaped)."""
        try:
            import psutil
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.
        ``expected_start`` (kernel start time at spawn) is re-validated first: a mismatch
        or dead PID means the number was recycled onto a stranger and we refuse to touch
        it — a leaked orphan beats tree-killing someone's browser. POSIX: snapshot descendants,
        SIGTERM the parent alone so it can perform an orderly shutdown, then clean up snapshot
        descendants that survive its grace window. Survivors are SIGKILLed after a second
        ``terminal.daemon_term_grace_seconds`` window. Windows:
        ``taskkill /T /F`` (psutil's stale PPID links miss orphans there); ``os.kill``
        is the fallback."""
        from tools.process_registry import _IS_WINDOWS

        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid)
            return

        def _sigterm_quietly():
            with suppress(OSError, ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
        if _IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=10, creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL, check=False)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                _sigterm_quietly()
            return
        import psutil
        gone = (psutil.NoSuchProcess, psutil.AccessDenied, OSError)
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            _sigterm_quietly()
            return
        # Snapshot before signalling: once the parent exits, psutil can no longer
        # reliably find children that it failed to reap.
        try:
            descendants = parent.children(recursive=True)
        except gone:
            descendants = []

        # Let self-managing parents (notably Chromium/Electron) shut down their
        # tree before touching children. Killing their zygotes first can turn a
        # graceful browser shutdown into a crash dump.
        with suppress(gone):
            parent.terminate()

        grace = cls._daemon_term_grace_seconds()

        def _wait_for_exit(targets) -> None:
            if grace <= 0:
                return
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and any(cls._proc_alive(p) for p in targets):
                time.sleep(0.05)

        # Preserve descendants during the parent's configured shutdown window.
        _wait_for_exit([parent])

        # The snapshot is an anti-orphan guarantee: only descendants still alive
        # after the parent had its chance are asked to terminate themselves.
        remaining = descendants if grace <= 0 else [
            proc for proc in descendants if cls._proc_alive(proc)
        ]
        for proc in remaining:
            with suppress(gone):
                proc.terminate()

        # Preserve the existing SIGKILL escalation semantics for every owned
        # process that remains after its SIGTERM grace window. The parent is
        # included in case it ignored the first signal.
        targets = [parent, *remaining]
        # Escalate to SIGKILL for anything that ignored SIGTERM within the grace window.
        # ``psutil.wait_procs``' gone/alive partition is deliberately NOT trusted: it
        # reaps via ``Process.wait()`` and mis-partitions across zombie transitions in a
        # parent/child tree, leaving survivors un-killed. Re-probing every target is
        # deterministic.
        if grace <= 0:
            return
        _wait_for_exit(targets)
        # A parent that ignored SIGTERM (the interactive ``bash -lic`` wrapper does) keeps
        # running its script through both grace windows and can spawn children the first
        # snapshot never saw. Re-snapshot while it is still alive: once it is SIGKILLed
        # they reparent to init and nothing can find them again.
        with suppress(gone):
            if cls._proc_alive(parent):
                known = {proc.pid for proc in targets}
                targets.extend(p for p in parent.children(recursive=True) if p.pid not in known)
        for proc in targets:
            with suppress(gone):
                if cls._proc_alive(proc):
                    proc.kill()  # SIGKILL on POSIX
                    logger.info("Escalated to SIGKILL for pid %d (ignored SIGTERM within %.1fs grace)", proc.pid, grace)

    @staticmethod
    def _live_descendants(pid: int) -> list[int]:
        """PIDs of living non-zombie descendants of host PID ``pid`` (best-effort)."""
        try:
            import psutil
            children = psutil.Process(pid).children(recursive=True)
        except Exception:
            logger.debug("Could not list descendants of pid %s", pid, exc_info=True)
            return []
        return [c.pid for c in children if ProcessTerminationMixin._proc_alive(c)]

    # SIGKILL / taskkill are asynchronous: the kernel needs a scheduling tick to
    # tear the process down and the parent must reap it before poll()/isalive()
    # stop saying "alive". Verifying survivors in that window flagged every
    # escalated kill as incomplete.
    _KILL_SETTLE_SECONDS = 1.0

    def _post_kill_survivors(self, session: ProcessSession) -> list[int]:
        """Host PIDs still alive once the kill signals have had time to land (#115490).

        Fail-closed: anything unverifiable counts as a survivor, so a kill
        that leaves a live tree can never write a killed receipt. Sandbox
        (env) sessions have no host-visible tree and are unverifiable by
        design — they return no survivors, preserving existing behavior."""
        deadline = time.monotonic() + self._KILL_SETTLE_SECONDS
        while True:
            survivors = self._probe_survivors(session)
            if not survivors or time.monotonic() >= deadline:
                return survivors
            time.sleep(0.05)

    def _probe_survivors(self, session: ProcessSession) -> list[int]:
        survivors: list[int] = []
        proc = getattr(session, "process", None)
        if proc is not None:
            try:
                root_alive = proc.poll() is None
            except Exception:
                root_alive = True
            if root_alive:
                survivors.append(getattr(proc, "pid", None) or session.pid)
        pty = getattr(session, "_pty", None)
        if pty is not None:
            try:
                pty_alive = bool(pty.isalive())
            except Exception:
                pty_alive = self._is_host_pid_alive(session.pid)
            if pty_alive:
                survivors.append(session.pid)
        if session.pid_scope == "host" and session.pid:
            if self._detached_host_fate(session.pid, session.host_start_time) == "running":
                if session.pid not in survivors:
                    survivors.append(session.pid)
                survivors.extend(
                    pid for pid in self._live_descendants(session.pid)
                    if pid not in survivors)
            # A dead/recycled root has no PID-scope descendants left to find:
            # reparented orphans are outside PID scope (systemd scope stop,
            # issued before this check, covers the cgroup case).
        return [pid for pid in survivors if pid]

    def _signal_kill(self, session: ProcessSession, session_id: str, consume_output: bool) -> Optional[dict]:
        """Deliver the kill via PTY, local Popen tree, sandbox exec or recovered host
        PID. Returns a final result dict when the kill cannot proceed (recycled/dead
        recovered PID, or no runtime handle), else None."""
        if session._pty:
            try:
                session._pty.terminate(force=True)
            except Exception as exc:
                logger.debug("PTY terminate failed, falling back to SIGTERM: %s", exc, exc_info=True)
                if session.pid:
                    os.kill(session.pid, signal.SIGTERM)
        elif session.process:
            # Tree kill: on Windows Popen.terminate() only kills the shell wrapper and
            # leaves Git Bash descendants behind.
            self._terminate_host_pid(session.process.pid, session.host_start_time)
        elif session.env_ref and session.pid:
            return self._signal_kill_sandbox(session, session_id)
        elif session.detached and session.pid_scope == "host" and session.pid:
            return self._signal_kill_detached(session, session_id, consume_output)
        else:
            return {
                # Reject non-positive timeouts — the schema declares minimum=1, but not every caller
                # enforces schemas before dispatch. timeout=0 is falsy, so without this guard it silently
                # fell through (`0 or max_timeout`) to the DEFAULT wait instead of erroring. Salvaged from
                # PR #60004 (credit @isheng-eqi).
                "status": "error",
                "error": "Recovered process cannot be killed after restart because "
                         "its original runtime handle is no longer available",
            }
        return None

    def _signal_kill_sandbox(self, session: ProcessSession, session_id: str) -> Optional[dict]:
        """Signal through the sandbox exec transport after re-verifying run identity.

        Returns an error result when the live run is not provably ours; None when
        the signal was sent and post-signal verification should decide the outcome."""
        from tools.process_registry_supervision import (
            _read_pid_file, _read_token_file, _read_wrapper_startticks,
            _read_boot_id, sandbox_signal_verdict,
            _SupervisionTransportError)
        from tools.process_registry_supervision import pinned_launch_execute as _pinned_kill
        import shlex as _shlex
        try:
            _live_pid = _read_pid_file(
                session.env_ref, _shlex.quote(session.env_pid_path)) if session.env_pid_path else None
        except _SupervisionTransportError as exc:
            logger.debug("Kill %s pid read uncertain: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable (uncertain transport)."}
        except Exception as exc:
            logger.debug("Kill %s pid read failed: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable."}
        try:
            _live_token = _read_token_file(
                session.env_ref, _shlex.quote(session.env_token_path)
            ) if getattr(session, "sandbox_run_token", "") and getattr(session, "env_token_path", "") else None
        except _SupervisionTransportError as exc:
            logger.debug("Kill %s token read uncertain: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable (uncertain transport)."}
        except Exception as exc:
            logger.debug("Kill %s token read failed: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable."}
        try:
            _live_ticks = _read_wrapper_startticks(
                session.env_ref, _shlex.quote(session.env_pid_path)) if session.env_pid_path else None
        except _SupervisionTransportError as exc:
            logger.debug("Kill %s start-ticks uncertain: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable (uncertain transport)."}
        except Exception as exc:
            logger.debug("Kill %s start-ticks failed: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable."}
        try:
            _live_boot = _read_boot_id(session.env_ref)
        except _SupervisionTransportError as exc:
            logger.debug("Kill %s boot-ID uncertain: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable (uncertain transport)."}
        except Exception as exc:
            logger.debug("Kill %s boot-ID failed: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": "Refusing to signal: wrapper identity unverifiable."}
        if sandbox_signal_verdict(
                session.sandbox_container_id, session.sandbox_pid,
                session.env_ref, _live_pid,
                getattr(session, "sandbox_run_token", ""), _live_token,
                getattr(session, "sandbox_wrapper_startticks", "") or "", _live_ticks,
                getattr(session, "sandbox_boot_id", "") or "",
                _live_boot if isinstance(_live_boot, str) else None) != "match":
            return {
                "status": "error",
                "error": "Refusing to signal: the sandbox run identity changed or is "
                         "unverifiable (container replaced, token mismatch, or wrapper "
                         "PID reused by an unrelated process). "
                         "The recorded PID may belong to an unrelated process.",
            }
        try:
            _pinned_kill(session.env_ref, f"kill {int(session.pid)} 2>/dev/null", timeout=5)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Sandbox kill transport failed for %s: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": f"Kill signal not confirmed for {session_id}: {exc}"}
        except Exception as exc:
            logger.warning("Sandbox kill failed for %s: %s", session_id, exc, exc_info=True)
            return {"status": "error",
                    "error": f"Kill signal not confirmed for {session_id}: {exc}"}
        return self._verify_sandbox_kill(session, session_id)

    def _verify_sandbox_kill(self, session: ProcessSession, session_id: str) -> Optional[dict]:
        """Post-signal probe: a still-identical wrapper means the signal did not land.

        Fail-closed: an unverifiable wrapper is never claimed killed. Returns an
        error result on a proven-live or proven-reused wrapper, else None."""
        from tools.process_registry_supervision import (
            _read_wrapper_startticks, _SupervisionTransportError,
            _exec_probe_pinned as _probe)
        import shlex as _shlex
        try:
            from tools.process_registry_supervision import _exec_probe_pinned as _probe
            _alive_after = _probe(
                session.env_ref,
                f"kill -0 \"$(cat {_shlex.quote(session.env_pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?",
                5, label="kill verify")
            if _alive_after.strip().splitlines()[-1].strip() == "0":
                try:
                    _ticks_after = _read_wrapper_startticks(
                        session.env_ref, _shlex.quote(session.env_pid_path))
                except Exception as exc:
                    logger.debug("Kill verify start-ticks uncertain for %s: %s",
                                 session_id, exc, exc_info=True)
                    _ticks_after = None
                stored_ticks = getattr(session, "sandbox_wrapper_startticks", "") or ""
                if _ticks_after is None:
                    return {"status": "error",
                            "error": f"Kill signal sent to {session_id} but wrapper liveness unverifiable; not claimed killed."}
                if _ticks_after == stored_ticks:
                    return {"status": "error",
                            "error": f"Kill signal sent to {session_id} but wrapper still alive; not claimed killed."}
                return {"status": "error",
                        "error": f"Kill signal sent to {session_id} but wrapper identity changed (PID reused); not claimed killed."}
        except _SupervisionTransportError as exc:
            logger.debug("Kill verify uncertain for %s: %s", session_id, exc, exc_info=True)
        except Exception as exc:
            logger.debug("Kill verify failed for %s: %s", session_id, exc, exc_info=True)

    def _signal_kill_detached(self, session: ProcessSession, session_id: str, consume_output: bool) -> Optional[dict]:
        """Kill a recovered host-PID session whose live identity was just proven."""
        from tools.process_registry import _completion_output
        # Same fate as poll/list: a gone or reused PID means our process is
        # gone — never tree-kill the stranger — but a live PID with an
        # unreadable start time is still ours and must really be killed.
        # If this recovered session also carries an owned systemd scope, stop that scope before
        # returning: a daemonized descendant may still be alive there even though the wrapper PID exited
        # or was recycled across the gateway restart (#70716, teknium1 review).
        if self._detached_host_fate(session.pid, session.host_start_time) != "running":
            scope_stopped = self._stop_scope(session)
            with session._lock:
                output = _completion_output(session)
            if consume_output:
                self._completion_consumed.add(session_id)
            # No waitable handle, so this is not a collected exit. Close the
            # entry without queueing a completion, and do not signal a PID
            # whose start time does not match.
            self._close_reused_detached(session)
            result = {"status": "already_exited", "exit_code": session.exit_code, **output}
            if not scope_stopped:
                result["scope_stop_failed"] = session.systemd_unit
            return result
        # Identity was just proven above. Re-passing the start time would make
        # an unreadable probe refuse the kill and leave the re-adopted child running.
        self._terminate_host_pid(session.pid)
