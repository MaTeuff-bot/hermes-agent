---
sidebar_position: 16
title: "Background Process Supervision"
description: "Durable supervision for sandbox background processes: identity, uncertainty, admission vs delivery, crash recovery, and rollback"
---

# Background Process Supervision

How `tools/process_registry*.py` keeps env-backed (Docker/SSH) background processes
honest across transport outages, restarts, and container replacement. The host (local/PTY)
path is unchanged: kernel PIDs with start-time reuse guards.

## The reproduced root: owner query vs container question

The idle reaper asked a container-scoped question ("is this container idle?") with an
owner-scoped query (`has_active_processes`). A `default` container running an owner-scoped
job therefore read as idle and its handle was invalidated mid-job (repro:
`idle-reaper-base.json` harness in the supervision evidence bundle).

The fix separates the semantics and keeps them separated:

- `has_active_for_container(container_key, env)` is container scope: it matches
  `session.task_id` (the collapsed container key the reaper caches under) or a shared
  live `env_ref`. Owner scope stays in `has_active_processes` for delegated lifetime.
- New spawns capture wrapper identity atomically in the pinned namespace: in-sandbox PID,
  `/proc` start ticks, and boot ID, plus a per-run token and the container ID.
- Reviews must keep this split: any new "is it idle / may I reap" call site uses the
  container query, never the owner query.

## Reads never recreate; uncertainty is never an exit

Docker's `execute` recreates a removed container and retries once. A supervision probe
that rode that path could adopt a fresh container's files as the old run's. So:

- Every probe goes through the pinned path (`BaseEnvironment.execute`), bypassing the
  recreating override, and refuses when the container ID changed around the call.
- The one-shot launch does the same: at most one launch ever runs; a gone container
  fails fast as `failed_start` instead of double-executing the job in a fresh container.
- A transport exception, timeout, or malformed probe means *uncertain supervision*, never
  an exit. The poller retries bounded with backoff, reconciles a token-bound verified exit
  receipt first and identity-checked liveness second, then escalates to durable
  `unresolved` — never an invented exit code, never a relaunch, never a host `kill` of an
  in-sandbox PID.

## Wrapper budget vs transport timeout

`spawn_via_env(timeout=...)` is the transport/start timeout only (launch + probes). There
is no default job budget: background servers must never be killed by one. A caller that
needs a bound owns it with GNU `timeout -s TERM -k N` (TERM-then-KILL grace). The wrapper
exit is non-success when the bound fires: 124 when the TERM budget fires (the pinned
caller-owned case, even if the child exits 0 after TERM) and 137 when the KILL grace
fires. Tests pin both halves: the entry path imposes no `timeout -s TERM` segment, and a
caller-owned bound still reports non-success end to end.

## Admission vs processing vs outbound delivery

Three distinct boundaries; confusing any two double-delivers or drops completions.
Admitted, processed, and outbound-sent are separate receipts; exactly-once external
actions are not guaranteed.

1. **Admission** — `admit_automation` commits the producer wake to the session
   authority FIFO with a durable producer identity. A producer ACK means committed
   input, not successful inference or delivery.
2. **Processing** — the owning workflow's turn runs via the dispatcher
   (`execute_admission` → the turn handler). The durable producer identity prevents a
   second admission for the same producer event in the tested crash-before-ack replay;
   retried crashes may re-enter the dispatcher, and exactly-once execution of external
   effects is not promised.
3. **Outbound delivery** — the adapter send. Agent-notify completions ack the durable
   outbox (`logs/process-delivery/` via `ack_delivery`) after durable admission: the
   admitted turn carries the result, and the concise chat receipt sent afterwards is
   best-effort (a failed receipt only warns). Text-notice completions ack after send
   success. A failed send never acks in its own mode.

A crash between admission and ack replays the outbox on restart, and the durable
producer identity dedupes re-admission for the same producer event in that replay. The
seam-level restart tests (mocked admission/transport) prove only outbox/ack sequencing.
The native test in `tests/gateway/test_process_restart_dedup.py` is native admission
processing proof (real SessionDB-backed authority, real `admit_automation` identity,
real `execute_admission` into a stubbed turn handler with `_schedule` disabled and an
explicit `execute_admission` call, no model inference, no outbound send): it does not
prove autonomous scheduler dispatch without user ping, and its exactly-once claim is
scoped to that admission replay, not to external actions.

## Durable outbox and state retention

- The outbox write precedes the query-only result save and any finalized-state
  exposure: a crash between them leaves a recoverable pending event, never a silent loss.
- Retention is 7 days of delivery attempts. Expired undelivered records are never
  silently deleted: they are retained with `expired: true` and escalated once (loud,
  rate-limited), staying queryable for operator triage.
- Checkpoints retain sandbox runs as `unresolved` (never dropped, never host-signalled),
  with error evidence and a re-armed watch route. `reattach_env_session` resumes one
  only after identity verification (container, token, wrapper start ticks + boot ID in
  the same pinned namespace); a replacement or an unverifiable legacy run is refused.
- Legacy receipts without stored ticks/token fail closed to `unknown`: explicit
  escalation, never a signal. PID-number equality alone never proves identity (PIDs
  recycle inside and outside the sandbox).

## Failure matrix

| Case | Verdict | Action |
|---|---|---|
| Same container, token + ticks match | `match` | Poll / finish on verified receipt |
| Registry/gateway crash, same container returns | retained `unresolved` | `reattach_env_session` resumes |
| Container replaced (ID/token/ticks differ) | `mismatch` | Refuse: never adopt, never signal |
| Host lost / route unavailable / unreadable identity | `unknown` | Escalate `unresolved`, retain evidence |
| Exit receipt present but identity unverified | refuse receipt | `unresolved`, never trust it |

## Operator acceptance (read-only host inventory first)

Never restart a live gateway for verification without explicit approval. Inventory is
read-only; acceptance runs isolated (a scratch `HERMES_HOME`, never the live one).
No mutable live-host operation is part of acceptance.

```bash
# Read-only inventory only (no restarts, no mutations, no live-host writes).
hermes --version
hermes gateway status
# Installed-source revision, only when the operator substitutes their real checkout:
git -C <installed-source-checkout> rev-parse HEAD
```

Host acceptance gap: emulated backends in tests (`SandboxLikeEnv`-style: a real local
subprocess with a synthetic container ID and injected faults) prove no-relaunch,
no-fake-exit, and token/container verdicts — but they are **not** a real Docker engine,
and no real-engine run is claimed here. A real Docker acceptance harness through the
actual `DockerEnvironment` / registry / gateway API entry paths (bounded, disposable,
with precise isolation scope and canary cleanup) is still pending, as is the operator
approval to run it. The gated procedure, when approved, is: the operator provisions a
scratch `HERMES_HOME`/`HERMES_RUNTIME_DIR`, exercises the real entry paths against a
throwaway container, verifies single launch (no relaunch) and mismatch-on-replacement, removes
every canary, and records the daemon revision — no live gateway restart, no live-home
contact, no emulated backend presented as engine proof.

Host shutdown vs crash: a host shutdown that intentionally stops `persist_on_release`
jobs is an intentional stop, not a crash; crash replay (outbox/admission dedupe above)
and intentional-stop semantics are different paths, and shutdown persistence alone does
not promise notification delivery.

## Rollback

If a supervision candidate misbehaves: stop adopting it (revert the adopting change at
the seam the candidate entered through) and **preserve** `logs/process-delivery/`,
`logs/process-results/`, and `processes.json` — they are the recovery evidence for the
next restart. The supervision files are coupled (`process_registry.py` facade,
`..._supervision.py` policy, `..._sandbox.py` lifecycle, `..._checkpoint.py` recovery,
`..._delivery.py` outbox, `..._termination.py` kills): never revert one independently
of the rest; roll the set back together or not at all.
