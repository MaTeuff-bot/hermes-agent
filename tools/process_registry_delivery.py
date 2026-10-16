"""Durable pending terminal-delivery outbox for background completions.

``process-results/`` receipts are read-only snapshots, never replayed as
notifications. This outbox is the complementary durable record: one small file
per finished session that still owns a watcher route. It is written exactly
once by the registry finisher (``_move_to_finished`` claim), replayed after a
registry/gateway restart via ``recover_from_checkpoint`` into
``pending_watchers`` (the existing startup drain re-arms them), and deleted
only after actual gateway admission + send success (``ack_delivery``). A failed
send never acks, so delivery retries instead of going silent. Dedup reuses the
gateway's existing completion identity/admission on the send path.
"""
import json
import logging
import time

logger = logging.getLogger("tools.process_registry")

DELIVERY_RETENTION_SECONDS = 7 * 24 * 60 * 60


def _delivery_dir():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "logs" / "process-delivery"


def write_pending_delivery(session) -> bool:
    """Persist one pending terminal delivery. True when written/kept."""
    if not (getattr(session, "watcher_interval", 0) > 0 or getattr(session, "notify_on_complete", False)):
        return False
    try:
        from hermes_constants import assert_named_profile_home_live
        from utils import atomic_json_write
        from agent.redact import redact_sensitive_text, redact_terminal_output
        directory = _delivery_dir()
        assert_named_profile_home_live(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            raw_output = getattr(session, "output_buffer", "") or ""
            output_tail = raw_output[-4000:]
            output_tail = redact_terminal_output(
                output_tail, getattr(session, "command", ""), force=True)
        except Exception:
            output_tail = ""
        record = {
            "session_id": session.id,
            "session_key": getattr(session, "session_key", ""),
            "task_id": getattr(session, "task_id", ""),
            "owner_task_id": getattr(session, "owner_task_id", ""),
            "parent_session_id": getattr(session, "parent_session_id", ""),
            "command": redact_sensitive_text(getattr(session, "command", ""), code_file=True),
            "output": output_tail,
            "exit_code": getattr(session, "exit_code", None),
            "completion_reason": getattr(session, "completion_reason", ""),
            "termination_source": getattr(session, "termination_source", ""),
            "started_at": getattr(session, "started_at", time.time()),
            "watcher_interval": getattr(session, "watcher_interval", 0),
            "notify_on_complete": bool(getattr(session, "notify_on_complete", False)),
            "created_at": time.time(),
        }
        for key in ("platform", "chat_id", "user_id", "user_name", "thread_id", "message_id"):
            try:
                record[f"watcher_{key}"] = getattr(session, f"watcher_{key}", "") or ""
            except Exception:
                record[f"watcher_{key}"] = ""
        atomic_json_write(directory / f"{session.id}.json", record, mode=0o600)
        return True
    except OSError:
        logger.warning("Could not persist pending delivery for %s", getattr(session, "id", "?"), exc_info=True)
        return False


def ack_delivery(session_id: str) -> None:
    """Delete the pending record after admission + send success. Failed sends never call this."""
    try:
        path = _delivery_dir() / f"{session_id}.json"
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not ack pending delivery %s", session_id, exc_info=True)


def list_pending_deliveries() -> list:
    """Pending records, oldest first; prunes expired. Read-only, never deletes unexpired."""
    try:
        directory = _delivery_dir()
    except Exception:
        return []
    if not directory.exists():
        return []
    cutoff = time.time() - DELIVERY_RETENTION_SECONDS
    staged = []
    for path in directory.glob("proc_*.json"):
        try:
            mtime = path.stat().st_mtime
            if mtime < cutoff:
                path.unlink(missing_ok=True)
                continue
            staged.append((mtime, path))
        except OSError:
            continue
    records = []
    for _, path in sorted(staged):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            continue
    return records
