"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[list[dict[str, Any]]] = None):
        """Write running process metadata to the checkpoint file atomically."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS

        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    if s.exited:
                        continue
                    # Backfill the start time so recovery can detect PID recycling
                    # even for sessions spawned before this field existed.
                    if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                        s.host_start_time = self._safe_host_start_time(s.pid)
                    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                    # Redact inline credentials before persisting (~/.hermes/processes.json).
                    # Recovery uses command only for display (adoption re-validates the
                    # PID, never re-runs it), so masking is lossless.
                    # See #77484.
                    entry["command"] = redact_sensitive_text(s.command, code_file=True)
                    entry["owner_task_id"] = s.owner_task_id or s.task_id
                    entries.append(entry)
                if extra_entries:
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
                known = {*self._running, *self._finished}
            from hermes_cli.active_sessions import _FileLock
            from utils import atomic_json_write
            path = _checkpoint_path()
            # Every interpreter of the profile (owner, each managed worker) rewrites this file from
            # its own table; a worker's output-log handoff is its only path to the next one, so a
            # writer keeps the handoffs it does not track (read-merge-write under one file lock).
            with _FileLock(path.with_name(path.name + ".lock")):
                entries.extend(self._foreign_handoffs(path, known))
                atomic_json_write(path, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def _foreign_handoffs(self, path, known) -> list[dict[str, Any]]:
        """On-disk worker handoffs this registry never tracked whose log still decides an outcome:
        the process is alive (same start time) or its recorded exit is waiting to be read."""
        from pathlib import Path
        from tools.process_registry_output_log import _exit_path, adoptable_log
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return []
        return [entry for entry in on_disk if isinstance(entry, dict) and entry.get("session_id") not in known
                and (log := adoptable_log(entry.get("output_log"))) and Path(log).exists()
                and (_exit_path(log).exists() or self._detached_host_fate(
                    entry.get("pid"), entry.get("host_start_time")) == "running")]

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions."""
        from tools.process_registry import _checkpoint_path

        checkpoint_path = _checkpoint_path()
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8-sig")) if checkpoint_path.exists() else []
        except (OSError, ValueError) as exc:
            logger.debug("Checkpoint read failed, starting empty: %s", exc, exc_info=True)
            entries = []
        except Exception as exc:
            logger.debug("Checkpoint read failed, starting empty: %s", exc, exc_info=True)
            entries = []
        recovered = self._replay_pending_deliveries()
        unresolved_scope_entries: list[dict[str, Any]] = []
        for entry in entries:
            recovered += self._adopt_checkpoint_entry(entry, unresolved_scope_entries)
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        return recovered


    def _replay_pending_deliveries(self) -> int:
        """Rebuild finished sessions from the durable outbox and re-arm watchers."""
        from tools.process_registry_delivery import list_pending_deliveries
        recovered = 0
        try:
            for record in list_pending_deliveries():
                session_id = record.get("session_id", "")
                if not session_id:
                    continue
                with self._lock:
                    if session_id in self._running or session_id in self._finished:
                        continue
                    known = {w.get("session_id") for w in self.pending_watchers}
                if session_id in known:
                    continue
                # Rebuild the finished terminal session so the re-armed watcher
                # resolves it (``get`` would otherwise return None and the
                # watcher would break without delivering). ``process-results/``
                # stays query-only; this outbox carries its own output tail.
                try:
                    self._rebuild_pending_delivery_session(record)
                    recovered += 1
                except (OSError, ValueError, KeyError) as exc:
                    logger.debug("Skipping unreadable pending delivery %s: %s", session_id, exc, exc_info=True)
                    continue
                except Exception as exc:
                    logger.debug("Skipping unreadable pending delivery %s: %s", session_id, exc, exc_info=True)
                    continue
                self.pending_watchers.append({
                    "session_id": session_id,
                    "check_interval": record.get("watcher_interval", 5) or 5,
                    "session_key": record.get("session_key", ""),
                    "platform": record.get("watcher_platform", ""),
                    "chat_id": record.get("watcher_chat_id", ""),
                    "user_id": record.get("watcher_user_id", ""),
                    "user_name": record.get("watcher_user_name", ""),
                    "thread_id": record.get("watcher_thread_id", ""),
                    "message_id": record.get("watcher_message_id", ""),
                    "notify_on_complete": record.get("notify_on_complete", False),
                    "parent_session_id": record.get("parent_session_id", ""),
                    "pending_delivery_replay": True,
                })
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Pending-delivery replay failed, continuing with checkpoint entries: %s", exc, exc_info=True)
        except Exception as exc:
            logger.warning("Pending-delivery replay failed, continuing with checkpoint entries: %s", exc, exc_info=True)
        return recovered

    def _adopt_checkpoint_entry(self, entry: dict[str, Any], unresolved_scope_entries: list[dict[str, Any]]) -> int:
        """Adopt one checkpoint entry; returns 1 when it counts as recovered."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS,
            _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log
        pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
        if not pid and pid_scope == "host":
            return 0
        # The registry is process-global, so every profile's checkpoint carries every live
        # process; a multiplexer recovering several homes must adopt each session once.
        with self._lock:
            already_tracked = entry.get("session_id") in self._running
        if already_tracked:
            return 0
        if pid_scope != "host":
            # In-sandbox PIDs mean nothing once the env handle is gone, but the run
            # is not over: retain the entry as durable ``unresolved`` evidence (never
            # a faked exit, never dropped) so ``reattach_env_session`` can resume the
            # same container when its backend returns, and the gateway surfaces it.
            entry = {**entry, "supervision_state": "unresolved",
                     "completion_reason": entry.get("completion_reason") or "unresolved",
                     "termination_source": entry.get("termination_source") or "backend_unreachable",
                     "last_supervision_error": entry.get("last_supervision_error") or (
                         "registry restarted while the sandbox run was unsupervised")}
            unresolved_scope_entries.append(entry)
            self._retain_unresolved_sandbox_session(entry)
            logger.warning(
                "Retaining unsupervised sandbox process: %s (session=%s, container=%s)",
                entry.get("command", "unknown")[:60], entry.get("session_id", "?"),
                entry.get("sandbox_container_id", "?"))
            return 1
        # Alive and still ours: re-attach. A start-time probe that cannot
        # be read is not proof the PID was reused — dropping it would leave
        # a live child unsupervised, and marking it exited would invent a
        # completion. A positive mismatch means the number was recycled:
        # do not adopt it and do not signal it.
        fate = self._detached_host_fate(pid, entry.get("host_start_time"))
        # A worker spawn's output log and recorded exit outlive both the spawner and the PID,
        # so its reader decides the outcome (a recycled PID is never signalled: fate != running).
        log = adoptable_log(entry.get("output_log"))
        if log:
            fate = "running"
        if fate == "reused":
            logger.info(
                "Not recovering session %s: pid %d is alive but its "
                "start time no longer matches — PID was recycled onto "
                "an unrelated process; refusing to adopt or signal it.",
                entry.get("session_id", "?"), pid)
            return 0
        if fate != "running":
            systemd_unit = entry.get("systemd_unit", "")
            if systemd_unit and not _stop_systemd_unit(systemd_unit):
                logger.warning(
                    "Could not reap persisted scope %s for dead wrapper pid %s; "
                    "retaining checkpoint entry for the next startup",
                    systemd_unit, pid)
                unresolved_scope_entries.append(entry)
            return 0
        fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
        fields.update(
            command=entry.get("command", "unknown"),
            owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
            started_at=entry.get("started_at", time.time()))
        # detached: no Popen handle, but can report status + kill; a worker spawn's output log
        # (and recorded exit) outlives its spawner, so tail it from the start.
        fields["output_log"] = log
        session = ProcessSession(id=entry["session_id"], detached=True, **fields)
        if session.output_log:
            self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
        else:
            with self._lock:
                self._running[session.id] = session
        logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
        # Re-enqueue watcher so gateway can resume notifications
        if session.watcher_interval > 0:
            self.pending_watchers.append({
                "session_id": session.id,
                "check_interval": session.watcher_interval,
                "session_key": session.session_key,
                **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                "notify_on_complete": session.notify_on_complete,
                "parent_session_id": session.parent_session_id,
            })
        return 1
    def _rebuild_pending_delivery_session(self, record: dict[str, Any]) -> None:
        """Rebuild a finished terminal session from a pending-delivery record.

        The watcher drain resolves sessions via ``get``; without a rebuilt
        entry it would see ``None`` and break without delivering. The record
        carries its own redacted output tail, so this never reads the
        query-only ``process-results/`` snapshots as notifications.
        """
        from tools.process_registry import ProcessSession
        session_id = record.get("session_id", "")
        if not session_id:
            raise ValueError("missing session_id")
        with self._lock:
            if session_id in self._running or session_id in self._finished:
                return
            session = ProcessSession(
                id=session_id,
                command=record.get("command", "unknown"),
                task_id=record.get("task_id", ""),
                owner_task_id=record.get("owner_task_id", "") or record.get("task_id", ""),
                session_key=record.get("session_key", ""),
                parent_session_id=record.get("parent_session_id", ""),
                started_at=record.get("started_at", time.time()),
                exit_code=record.get("exit_code"),
                completion_reason=record.get("completion_reason", "exited"),
                termination_source=record.get("termination_source", ""),
                output_buffer=record.get("output", "") or "",
                exited=True,
                watcher_platform=record.get("watcher_platform", ""),
                watcher_chat_id=record.get("watcher_chat_id", ""),
                watcher_user_id=record.get("watcher_user_id", ""),
                watcher_user_name=record.get("watcher_user_name", ""),
                watcher_thread_id=record.get("watcher_thread_id", ""),
                watcher_message_id=record.get("watcher_message_id", ""),
                watcher_interval=int(record.get("watcher_interval", 0) or 0),
                notify_on_complete=bool(record.get("notify_on_complete", False)),
                pid=None, pid_scope="host", detached=True,
            )
            session._completion_event.set()
            self._finished[session.id] = session

    def _retain_unresolved_sandbox_session(self, entry: dict[str, Any]) -> None:
        """Rebuild an in-memory marker for a checkpointed sandbox run whose backend is
        gone. No PID and no runtime handle: poll()/watchers report it as running with
        ``supervision_state=unresolved``, kill refuses to signal it, and only
        ``reattach_env_session`` (identity-verified) resumes supervision."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _CHECKPOINT_DEFAULTS,
            _WATCHER_ROUTE_KEYS,
        )
        session_id = entry.get("session_id", "")
        if not session_id:
            return
        with self._lock:
            if session_id in self._running or session_id in self._finished:
                return
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()),
                completion_reason="unresolved", termination_source="backend_unreachable",
                supervision_state="unresolved",
                last_supervision_error=entry.get("last_supervision_error", ""),
                pid=None, env_ref=None, detached=True)
            session = ProcessSession(id=session_id, **fields)
            self._running[session.id] = session
            if session.watcher_interval > 0:
                self.pending_watchers.append({
                    "session_id": session.id,
                    "check_interval": session.watcher_interval,
                    "session_key": session.session_key,
                    **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                    "notify_on_complete": session.notify_on_complete,
                    "parent_session_id": session.parent_session_id,
                })
