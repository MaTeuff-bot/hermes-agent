"""Durable pending terminal-delivery outbox for background completions.

``process-results/`` receipts are read-only snapshots, never replayed as
notifications. This outbox is the complementary durable record: one small file
per finished session that still owns a watcher route. It is written BEFORE the
query-only result save by the registry finisher, replayed after a restart via
``recover_from_checkpoint`` into ``pending_watchers``, and deleted only after
gateway admission + send success (``ack_delivery``). A failed send never acks.

Retention SLA: 7 days of delivery attempts. Expired undelivered records are
NEVER silently deleted: they are retained with ``expired: true`` and escalated
once (loud warning, rate-limited per session), staying queryable for operator
triage. Host loss / route unavailability yields ``unknown``, never a fake exit.
No cron: expiry is evaluated lazily on list/recover.
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
        except (ValueError, RuntimeError, OSError) as exc:
            logger.debug("Delivery output redact failed for %s: %s", getattr(session, "id", "?"), exc, exc_info=True)
            output_tail = ""
        except Exception as exc:
            logger.debug("Delivery output redact failed for %s: %s", getattr(session, "id", "?"), exc, exc_info=True)
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
            except (AttributeError, ValueError) as exc:
                logger.debug("Delivery route copy failed for %s key %s: %s",
                             getattr(session, "id", "?"), key, exc, exc_info=True)
                record[f"watcher_{key}"] = ""
            except Exception as exc:
                logger.debug("Delivery route copy failed for %s key %s: %s",
                             getattr(session, "id", "?"), key, exc, exc_info=True)
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


_EXPIRED_ESCALATED: set[str] = set()
_EXPIRED_LOG_AT: dict[str, float] = {}
_EXPIRED_LOG_INTERVAL_S = 3600.0


def _escalate_expired_once(session_id: str, path_name: str) -> None:
    """Loud, rate-limited escalation for an expired undelivered outbox record."""
    now = time.time()
    last = _EXPIRED_LOG_AT.get(session_id, 0.0)
    if session_id in _EXPIRED_ESCALATED and (now - last) < _EXPIRED_LOG_INTERVAL_S:
        return
    _EXPIRED_ESCALATED.add(session_id)
    _EXPIRED_LOG_AT[session_id] = now
    logger.warning(
        "Pending delivery %s expired after %d days undelivered; retained as expired, "
        "not deleted (file %s). Route unavailable or host lost; operator triage required.",
        session_id, DELIVERY_RETENTION_SECONDS // 86400, path_name)


def list_pending_deliveries(*, include_expired: bool = True) -> list:
    """Pending records, oldest first. Expired undelivered are retained + escalated.

    Never silently deletes: an expired record is kept with ``expired: true`` and
    logged loudly once per hour per session (bounded). Callers that only want
    deliverable work pass ``include_expired=False``.
    """
    try:
        directory = _delivery_dir()
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("Delivery dir resolve failed: %s", exc, exc_info=True)
        return []
    except Exception as exc:
        logger.debug("Delivery dir resolve failed: %s", exc, exc_info=True)
        return []
    if not directory.exists():
        return []
    cutoff = time.time() - DELIVERY_RETENTION_SECONDS
    staged = []
    for path in directory.glob("proc_*.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            logger.debug("Delivery stat failed for %s: %s", path.name, exc, exc_info=True)
            continue
        except Exception as exc:
            logger.debug("Delivery stat failed for %s: %s", path.name, exc, exc_info=True)
            continue
        staged.append((mtime, path))
    records = []
    for _, path in sorted(staged):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            logger.debug("Delivery read failed for %s: %s", path.name, exc, exc_info=True)
            continue
        except Exception as exc:
            logger.debug("Delivery read failed for %s: %s", path.name, exc, exc_info=True)
            continue
        try:
            expired = path.stat().st_mtime < cutoff
        except (OSError, ValueError):
            expired = False
        if expired:
            record["expired"] = True
            _escalate_expired_once(str(record.get("session_id", path.stem)), path.name)
            if not include_expired:
                continue
        records.append(record)
    return records


def reconcile_finished_without_outbox(session) -> bool:
    """Re-create an outbox record for a finished routed session lacking one.

    Crash between outbox and result save must not lose the notification: if the
    query-only result exists but no pending file does, rebuild it. Returns True
    when a record is now present.
    """
    try:
        directory = _delivery_dir()
        if (directory / f"{session.id}.json").exists():
            return True
        return bool(write_pending_delivery(session))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Outbox reconcile failed for %s: %s", getattr(session, "id", "?"), exc, exc_info=True)
        return False
    except Exception as exc:
        logger.warning("Outbox reconcile failed for %s: %s", getattr(session, "id", "?"), exc, exc_info=True)
        return False
