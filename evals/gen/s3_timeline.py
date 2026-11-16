"""Subtype `longctx_ops_timeline_state_tracking` for `s3_frontier_reasoning`."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from operator import attrgetter
from typing import Any

MAX_ATTEMPTS = 60
MIN_TARGET_EVENTS = 10
MIN_TARGET_FIXES = 2

REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-south-1",
    "ap-southeast-1",
]
NEAR_REGION = {
    "us-east-1": "us-east-2",
    "us-east-2": "us-east-1",
    "us-west-2": "us-east-2",
    "eu-west-1": "eu-central-1",
    "eu-central-1": "eu-west-1",
    "ap-south-1": "ap-southeast-1",
    "ap-southeast-1": "ap-south-1",
}
TARGET_REGIONS = ["us-east-2", "eu-west-1", "ap-south-1", "us-east-1", "eu-central-1", "us-west-2"]

# (base service, near-duplicates). The base is a prefix of the first two near-duplicates.
FAMILIES = [
    ("billing-api", ("billing-apis", "billing-api-v2", "billing-api-canary")),
    ("auth-gw", ("auth-gw-int", "auth-gw-v2", "auth-gateway")),
    ("ledger-sync", ("ledger-syncd", "ledger-sync-v2", "ledger-sync-dr")),
    ("search-svc", ("search-svcs", "search-svc-v2", "search-svc-canary")),
    ("notify-api", ("notify-apis", "notify-api-v2", "notify-api-edge")),
    ("payments-core", ("payments-cores", "payments-core-v2", "payment-core")),
    ("orders-api", ("orders-apis", "orders-api-v2", "order-api")),
    ("profile-svc", ("profile-svcs", "profile-svc-v2", "profile-svc-int")),
]
FILLERS = [
    "inventory-svc",
    "cart-api",
    "media-cdn",
    "geo-lookup",
    "pricing-engine",
    "fraud-scorer",
    "session-store",
    "email-relay",
    "catalog-api",
    "reco-engine",
    "shipping-api",
    "tax-calc",
    "image-resize",
    "feature-flags",
]
TEAMS = [
    "payments-eng",
    "payments-platform",
    "payments-ops",
    "identity",
    "identity-edge",
    "search-infra",
    "search-quality",
    "growth-eng",
    "sre-tier1",
    "sre-tier2",
    "data-plane",
    "control-plane",
    "checkout-eng",
    "ledger-ops",
    "storefront",
    "messaging",
    "risk-eng",
    "core-infra",
]
HUMANS = [
    "alice",
    "bharat",
    "chen.w",
    "dmitri",
    "elena",
    "farah",
    "goran",
    "hiro",
    "ines",
    "jamal",
    "kofi",
    "lena",
    "mateo",
    "nadia",
    "oskar",
    "priya",
    "quinn",
    "rafael",
    "sven",
    "tomoko",
    "uma",
    "vikram",
    "wen",
    "ximena",
    "yusuf",
    "zoe",
]
BOTS = ["deploy-bot", "release-bot"]
COMPANIES = [
    "Tallgrass Commerce",
    "Bluefin Freight",
    "Copperleaf Health",
    "Northwind Ledger",
    "Juniper Mobility",
]

STATE_KINDS = ("DEPLOY", "ROLLBACK", "TRANSFER")
FIX_KINDS = ("CORRECTION", "VOID")
NOISE_WEIGHTS = {
    "DEPLOY": 20.0,
    "ROLLBACK": 6.0,
    "TRANSFER": 1.5,
    "CORRECTION": 2.2,
    "VOID": 1.3,
    "BUILD": 6.0,
    "DRYRUN": 4.0,
    "SCALE": 10.0,
    "CONFIG": 9.0,
    "ALERT": 12.0,
    "ACK": 5.0,
    "ONCALL": 4.0,
    "NOTE": 9.0,
}

KEY_SETS = [
    ("running_version", "deployed_by", "owner_team", "executed_rollbacks"),
    ("running_version", "rollback_target", "executed_rollbacks", "owner_team"),
    ("running_version", "deployed_by", "stack_depth", "rejected_rollbacks"),
    ("running_version", "owner_team", "executed_rollbacks", "rejected_rollbacks"),
    ("running_version", "rollback_target", "deployed_by", "executed_rollbacks", "owner_team"),
]
KEY_TEXT = {
    "running_version": (
        "string",
        "the running version of `{s}` in `{r}`, written exactly as in the log",
    ),
    "deployed_by": (
        "string",
        "the actor of the DEPLOY entry that pushed that running version onto the stack",
    ),
    "owner_team": ("string", "the team that owns `{s}`"),
    "executed_rollbacks": (
        "integer",
        "how many ROLLBACK entries of `{s}` in `{r}` were executed (not rejected)",
    ),
    "rejected_rollbacks": ("integer", "how many ROLLBACK entries of `{s}` in `{r}` were rejected"),
    "stack_depth": ("integer", "how many versions are on the release stack of `{s}` in `{r}`"),
    "rollback_target": (
        "string",
        "the version directly beneath the running version on that stack (what one more executed "
        "rollback would restore)",
    ),
}
KEY_PLACEHOLDER = {
    "running_version": "vX.Y.Z",
    "deployed_by": "<actor>",
    "owner_team": "<team>",
    "executed_rollbacks": 0,
    "rejected_rollbacks": 0,
    "stack_depth": 0,
    "rollback_target": "vX.Y.Z",
}

# Sloppy readings that must each change the answer (see `_evaluate`).
READINGS: dict[str, dict[str, Any]] = {
    "ignore_all_fixes": {"fixes": "none"},
    "ignore_fixes_after_cutoff": {"fixes": "pre_cutoff"},
    "line_order_not_time_order": {"order": "line"},
    "skip_late_replays_after_cutoff_line": {"skip_late": True},
    "conflate_near_duplicate_names": {"conflate": True},
    "count_rejected_rollbacks": {"count_rejected": True},
    "rollback_reverts_to_previous": {"revert": True},
    "ignore_cutoff": {"ignore_cutoff": True},
    "first_correction_wins": {"first_correction": True},
}

ROLLBACK_REASONS = [
    "p99 latency regression",
    "error budget burn",
    "failed health checks",
    "memory leak suspected",
    "elevated 5xx rate",
    "customer-reported checkout bug",
    "crash loop on startup",
]
CORRECTION_REASONS = [
    "deploy tool recorded the wrong region",
    "region label copied from the previous change",
    "pipeline template defaulted to the wrong region",
    "operator typo in region",
]
VOID_REASONS = [
    "pipeline aborted before any traffic shifted",
    "duplicate entry written by a retry",
    "entry written by a dry run misconfigured as live",
    "change was cancelled at approval",
]
TRANSFER_VOID_REASONS = [
    "reorg postponed; transfer never took effect",
    "ticket filed against the wrong service",
]
ALERT_TEXT = [
    ("p99_ms", (450, 2400), "p99 latency above SLO"),
    ("error_pct", (2, 35), "elevated 5xx rate"),
    ("restarts", (3, 60), "pod restarts"),
    ("queue_depth", (5000, 90000), "queue depth growing"),
    ("disk_pct", (85, 99), "disk usage high"),
    ("cert_days", (2, 14), "certificate expiring"),
]
ACK_TEXT = [
    "acknowledged, investigating",
    "known issue, tracking in incident channel",
    "noise, tuning threshold",
]
CONFIG_VALUES = {
    "max_conns": ["128", "256", "512", "1024"],
    "timeout_ms": ["250", "500", "1500", "3000"],
    "log_level": ["debug", "info", "warn"],
    "cache_ttl_s": ["30", "60", "300", "900"],
    "pool_size": ["8", "16", "32", "64"],
    "retry_budget": ["0.1", "0.2", "0.3"],
}
NOTE_TEXT = [
    "weekly capacity review moved to Thursday",
    "game day scheduled for {r} next week",
    "status page updated for {s} in {r}",
    "{s} {v} release notes published",
    "cost report: {s} spend in {r} up 4% week over week",
    "postmortem draft for INC-{n} shared",
    "dependency audit for {s} completed",
    "load test of {s} {v} planned in staging",
]


@dataclass
class _Event:
    idx: int
    slot: int
    ts: int
    actor: str
    kind: str
    service: str = ""
    region: str = ""
    version: str = ""
    team: str = ""
    ref: int = -1
    new_region: str = ""
    info: str = ""
    anchor: int = -1  # replayed entry: printed right after the entry in this slot
    eid: int = 0
    line: int = 0


@dataclass
class _Op:
    """One planned DEPLOY/ROLLBACK line of the target service whose raw or corrected region is the target."""

    kind: str
    raw: str
    fixes: list[tuple[str, str]] = field(
        default_factory=list
    )  # ("CORRECTION", region) | ("VOID", "")
    replay: str = (
        ""  # "" | "swap" (printed after the next deploy) | "late" (printed after the cutoff line)
    )
    tag: str = (
        ""  # "" | "exec" | "reject": the required outcome of a ROLLBACK (checked in `_replay_plan`)
    )


@dataclass
class _Log:
    events: list[_Event]
    cutoff: int
    cutoff_line: int
    service: str
    region: str
    partners: tuple[str, ...]
    initial_owner: str
    has_multi: bool
    target_fix_count: int
    target_event_count: int
    owners_at_start: dict[str, str]
    family: tuple[str, ...]


def _true_region(op: _Op) -> str:
    regions = [r for kind, r in op.fixes if kind == "CORRECTION"]
    return regions[-1] if regions else op.raw


def _replay_plan(ops: list[_Op], region: str) -> tuple[list[int], int, int, bool]:
    """(stack of op indices, executed, rejected, tags honoured) under the true semantics."""
    stack: list[int] = []
    executed = rejected = 0
    tags_ok = True
    for i, op in enumerate(ops):
        if any(kind == "VOID" for kind, _ in op.fixes) or _true_region(op) != region:
            continue
        if op.kind == "DEPLOY":
            stack.append(i)
        elif len(stack) < 2:
            rejected += 1
            tags_ok &= op.tag != "exec"
        else:
            stack.pop()
            executed += 1
            tags_ok &= op.tag != "reject"
    return stack, executed, rejected, tags_ok


def _plan_answer(ops: list[_Op], region: str, keys: tuple[str, ...]) -> tuple[int, ...]:
    """The answer keys at plan level (versions/actors identified by op index; ownership not modelled)."""
    stack, executed, rejected, _ = _replay_plan(ops, region)
    top = stack[-1] if stack else -1
    full = {
        "running_version": top,
        "deployed_by": top,
        "owner_team": 0,
        "executed_rollbacks": executed,
        "rejected_rollbacks": rejected,
        "stack_depth": len(stack),
        "rollback_target": stack[-2] if len(stack) >= 2 else -1,
    }
    return tuple(full[k] for k in keys)


def _plan_target(
    rng: random.Random, region: str, hard_index: int
) -> tuple[list[_Op], list[_Op], bool] | None:
    """Target-pair DEPLOY/ROLLBACK lines before (in time order) and after the cutoff."""
    others = [r for r in REGIONS if r != region]

    def elsewhere() -> str:
        return NEAR_REGION[region] if rng.random() < 0.55 else rng.choice(others)

    def dep(
        raw: str = region,
        fixes: list[tuple[str, str]] | None = None,
        replay: str = "",
        tag: str = "",
    ) -> _Op:
        return _Op("DEPLOY", raw, list(fixes or []), replay, tag)

    def rb(
        raw: str = region,
        fixes: list[tuple[str, str]] | None = None,
        replay: str = "",
        tag: str = "",
    ) -> _Op:
        return _Op("ROLLBACK", raw, list(fixes or []), replay, tag)

    # Opening: a rejected rollback; three of the four variants hinge on a fix. Depth is 1 afterwards.
    seed_variant = hard_index % 4
    if seed_variant == 0:  # plain rejection: one version stacked
        seed = [dep(), rb(tag="reject")]
    elif seed_variant == 1:  # the voided deploy makes the rollback a rejection
        seed = [dep(fixes=[("VOID", "")]), dep(), rb(tag="reject")]
    elif seed_variant == 2:  # the deploy moved out makes the rollback a rejection
        seed = [dep(), dep(fixes=[("CORRECTION", elsewhere())]), rb(tag="reject")]
    else:  # a rollback moved in while one version is stacked
        seed = [dep(), rb(raw=elsewhere(), fixes=[("CORRECTION", region)], tag="reject")]
    if rng.random() < 0.35:  # a second rejection in a row
        seed.append(rb(tag="reject"))
    # Floor: one rollback executes, the next is rejected (a "revert" reader executes both). When present,
    # the doubly corrected deploy decides how many versions are stacked here.
    has_multi = hard_index % 3 != 2
    pair = [rb(tag="exec"), rb(tag="reject")]
    if not has_multi:
        floor = [dep(), *pair]
    elif rng.random() < 0.5:  # moved out, then back in: it counts
        floor = [dep(fixes=[("CORRECTION", elsewhere()), ("CORRECTION", region)]), *pair]
    else:  # moved in, then out again: it does not count
        floor = [
            dep(),
            dep(raw=elsewhere(), fixes=[("CORRECTION", region), ("CORRECTION", elsewhere())]),
            *pair,
        ]
    blocks = [
        [dep() for _ in range(rng.randint(1, 2))],
        [dep(raw=elsewhere(), fixes=[("CORRECTION", region)]), dep()],
        [rb(tag="exec"), rb(fixes=[("VOID", "")]), rb(tag="exec")],
        [dep(fixes=[("CORRECTION", elsewhere())]), dep()],
    ]
    if (hard_index // 2) % 2 == 1:
        blocks.append([rb(fixes=[("CORRECTION", elsewhere())])])
    if rng.random() < 0.5:  # vary the counts
        blocks.append([dep(), rb(tag="exec")])
    if rng.random() < 0.4:
        blocks.append([rb(tag="exec")])
    if (
        hard_index % 2 == 0
    ):  # A (printed after B) runs: the late rollback, printed after the cutoff line, pops B
        tail = [dep(replay="swap"), dep(), rb(replay="late", tag="exec")]
    else:  # B runs: the late deploy E is popped by a rollback that is printed before E
        tail = [dep(replay="swap"), dep(), dep(replay="late"), rb(tag="exec")]
    for _ in range(40):
        rng.shuffle(blocks)
        ops = seed + floor + [op for block in blocks for op in block] + tail
        stack, executed, rejected, tags_ok = _replay_plan(ops, region)
        if len(stack) >= 3 and executed >= 3 and rejected >= 2 and tags_ok:
            break
    else:
        return None
    post = rng.choice(
        [
            [dep(), rb(), dep()],
            [rb(), dep()],
            [dep(), dep(), rb()],
            [dep(), rb(fixes=[("VOID", "")]), dep()],
        ]
    )
    return ops, post, has_multi


def _pick(rng: random.Random, used: dict[int, Any], lo: int, hi: int) -> int | None:
    if hi <= lo:
        return None
    for _ in range(12):
        slot = rng.randrange(lo, hi)
        if slot not in used:
            return slot
    free = [s for s in range(lo, hi) if s not in used]
    return rng.choice(free) if free else None


def _train(rng: random.Random, major: int, n: int = 90) -> list[str]:
    minor, patch, out = rng.randint(0, 30), rng.randint(0, 3), []
    for _ in range(n):
        out.append(f"v{major}.{minor}.{patch}")
        if rng.random() < 0.6:
            patch += 1
        else:
            minor, patch = minor + 1, 0
    return out


def _twin(rng: random.Random, eid: int) -> int:
    s = str(eid)
    spots = [i for i in range(3) if s[i] != s[i + 1] and not (i == 0 and s[1] == "0")]
    if not spots:
        return -1
    i = rng.choice(spots)
    return int(s[:i] + s[i + 1] + s[i] + s[i + 2 :])


def _evaluate(log: _Log, reading: dict[str, Any]) -> dict[str, Any]:
    """State of the target pair at the cutoff under the true semantics or one sloppy reading."""
    key = attrgetter("line") if reading.get("order") == "line" else attrgetter("ts")
    fixes = [e for e in log.events if e.kind in FIX_KINDS]
    if reading.get("fixes") == "none":
        fixes = []
    elif reading.get("fixes") == "pre_cutoff":
        fixes = [e for e in fixes if e.ts <= log.cutoff]
    voided = {e.ref for e in fixes if e.kind == "VOID"}
    region_of: dict[int, str] = {}
    corrections = sorted((e for e in fixes if e.kind == "CORRECTION"), key=key)
    for e in reversed(corrections) if reading.get("first_correction") else corrections:
        region_of[e.ref] = e.new_region
    main = [e for e in log.events if e.kind in STATE_KINDS and e.idx not in voided]
    if not reading.get("ignore_cutoff"):
        main = [e for e in main if e.ts <= log.cutoff]
    if reading.get("skip_late"):
        main = [e for e in main if e.line < log.cutoff_line]
    main.sort(key=key)

    def same_service(name: str) -> bool:
        return name == log.service or (bool(reading.get("conflate")) and name in log.partners)

    stack: list[_Event] = []
    executed = rejected = 0
    owner = log.initial_owner
    for e in main:
        if not same_service(e.service):
            continue
        if e.kind == "TRANSFER":
            owner = e.team
        elif region_of.get(e.idx, e.region) != log.region:
            continue
        elif e.kind == "DEPLOY":
            stack.append(e)
        elif len(stack) < 2:
            if reading.get("count_rejected"):
                executed += 1
            else:
                rejected += 1
        else:
            executed += 1
            if reading.get("revert"):  # "go back to the previously running version" = re-push it
                stack.append(stack[-2])
            else:
                stack.pop()
    return {
        "running_version": stack[-1].version if stack else None,
        "deployed_by": stack[-1].actor if stack else None,
        "owner_team": owner,
        "executed_rollbacks": executed,
        "rejected_rollbacks": rejected,
        "stack_depth": len(stack),
        "rollback_target": stack[-2].version if len(stack) >= 2 else None,
    }


def _build(rng: random.Random, hard_index: int) -> _Log | None:
    base, near_dups = FAMILIES[hard_index % len(FAMILIES)]
    if (hard_index // len(FAMILIES)) % 2 == 0:
        service, family_others = base, list(near_dups)
    else:  # target a longer member; the base becomes its prefix twin
        service, family_others = near_dups[1], [base, near_dups[0], near_dups[2]]
    region = TARGET_REGIONS[hard_index % len(TARGET_REGIONS)]
    other_base, other_dups = FAMILIES[(hard_index + 3) % len(FAMILIES)]
    fillers = rng.sample(FILLERS, 8)
    services = [service, *family_others, other_base, other_dups[0], *fillers]
    partners = tuple(
        s for s in services if s != service and (s.startswith(service) or service.startswith(s))
    )
    majors = rng.sample(
        range(2, 30), len(services)
    )  # distinct majors: versions never collide across services
    trains = {s: _train(rng, m) for s, m in zip(services, majors, strict=True)}
    owner_of = {s: rng.choice(TEAMS) for s in services}
    owners_at_start = dict(owner_of)
    initial_owner = owner_of[service]
    actors = HUMANS + BOTS

    n = rng.randint(585, 625)
    times: list[int] = []
    t = 0
    for _ in range(n):
        t += rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 41])
        times.append(t)
    cut = int(n * [0.52, 0.58, 0.64, 0.70, 0.55, 0.61, 0.67][hard_index % 7])
    cutoff = times[cut - 1] + rng.randint(1, times[cut] - times[cut - 1] - 1)

    plan = _plan_target(rng, region, hard_index)
    if plan is None:
        return None
    pre, post, has_multi = plan
    events: list[_Event] = []
    used: dict[int, _Event] = {}
    reserved: dict[int, str] = {}

    def new(slot: int, kind: str, actor: str, **kw: Any) -> _Event:
        ev = _Event(idx=len(events), slot=slot, ts=times[slot], actor=actor, kind=kind, **kw)
        events.append(ev)
        used[slot] = ev
        return ev

    # --- Target pair: slots, versions, actors -------------------------------------------------
    n_tail = 4 if hard_index % 2 else 3
    tail_span = max(14, cut // 7)
    lo = max(3, cut // 25)
    rest_slots = sorted(rng.sample(range(lo, cut - tail_span), len(pre) - n_tail))
    tail_slots = sorted(rng.sample(range(cut - tail_span, cut - 2), n_tail))
    post_slots = sorted(rng.sample(range(cut + 2, n - 3), len(post)))
    vi = rng.randint(3, 8)
    last_actor = ""
    op_events: list[tuple[_Op, _Event]] = []
    for op, slot in zip(pre + post, rest_slots + tail_slots + post_slots, strict=True):
        actor = rng.choice([a for a in actors if a != last_actor])
        if op.kind == "DEPLOY":
            last_actor = actor
            ev = new(
                slot,
                "DEPLOY",
                actor,
                service=service,
                region=op.raw,
                version=trains[service][vi],
                info=f"change=CHG-{rng.randint(10000, 99999)} sha={rng.getrandbits(28):07x}",
            )
            vi += rng.choice([1, 1, 2])
        else:
            ev = new(
                slot,
                "ROLLBACK",
                actor,
                service=service,
                region=op.raw,
                info=f'change=CHG-{rng.randint(10000, 99999)} reason="{rng.choice(ROLLBACK_REASONS)}"',
            )
        op_events.append((op, ev))

    swap_ev = next(ev for op, ev in op_events if op.replay == "swap")
    late_ev = next(ev for op, ev in op_events if op.replay == "late")
    swap_ev.anchor = rng.randrange(tail_slots[1] + 1, cut)
    late_ev.anchor = rng.randrange(cut, min(n - 1, cut + 30))

    # --- Target fixes: at least one load-bearing fix is logged after the cutoff --------------------
    def fix_event(slot: int, ref: _Event, kind: str, new_region: str = "") -> _Event:
        if kind == "VOID":
            why = TRANSFER_VOID_REASONS if ref.kind == "TRANSFER" else VOID_REASONS
            return new(
                slot,
                "VOID",
                rng.choice(HUMANS),
                ref=ref.idx,
                info=f'ticket=OPS-{rng.randint(1000, 9999)} reason="{rng.choice(why)}"',
            )
        return new(
            slot,
            "CORRECTION",
            rng.choice(HUMANS),
            ref=ref.idx,
            new_region=new_region,
            info=f'ticket=OPS-{rng.randint(1000, 9999)} reason="{rng.choice(CORRECTION_REASONS)}"',
        )

    keys = KEY_SETS[hard_index % len(KEY_SETS)]
    answer = _plan_answer(pre, region, keys)
    fixed = [k for k, op in enumerate(pre) if op.fixes]
    critical = [  # fixes whose last entry, if ignored, changes the answer
        k
        for k in fixed
        if _plan_answer(
            [replace(op, fixes=op.fixes[:-1]) if i == k else op for i, op in enumerate(pre)],
            region,
            keys,
        )
        != answer
    ]
    if not critical:
        return None
    forced_late = rng.choice(critical)
    target_fix_count = 0
    for k in fixed:
        op, ev = op_events[k]
        late_fix = k == forced_late or rng.random() < 0.25
        if len(op.fixes) == 1:
            slot = None if late_fix else _pick(rng, used, ev.slot + 1, cut)
            slot = slot if slot is not None else _pick(rng, used, cut, n)
            if slot is None:
                return None
            fix_event(slot, ev, op.fixes[0][0], op.fixes[0][1])
        else:  # two corrections; the later one wins
            c1 = _pick(rng, used, ev.slot + 1, cut - 1)
            if c1 is None:
                return None
            fix1 = fix_event(c1, ev, "CORRECTION", op.fixes[0][1])
            c2 = _pick(rng, used, cut, n) if late_fix else _pick(rng, used, c1 + 1, cut)
            c2 = c2 if c2 is not None else _pick(rng, used, cut, n)
            if c2 is None:
                return None
            fix_event(c2, ev, "CORRECTION", op.fixes[1][1])
            if rng.random() < 0.6:  # the earlier correction is printed after the later one
                fix1.anchor = min(n - 1, c2 + rng.randint(1, 25))
        target_fix_count += len(op.fixes)
    for op, ev in op_events[len(pre) :]:
        for kind, new_region in op.fixes:
            slot = _pick(rng, used, ev.slot + 1, n)
            if slot is None:
                return None
            fix_event(slot, ev, kind, new_region)

    # --- Ownership of the target service and its prefix twin ---------------------------------------
    teams = rng.sample([tm for tm in TEAMS if tm != initial_owner], 5)
    s_b = _pick(rng, used, cut // 10, cut // 2)
    if s_b is None:
        return None
    tr_b = new(
        s_b,
        "TRANSFER",
        rng.choice(HUMANS),
        service=service,
        team=teams[0],
        info=f"ticket=ORG-{rng.randint(100, 999)}",
    )
    last_transfer = s_b
    if hard_index % 2 == 1:  # a second effective transfer; the first is printed after it
        s_d = _pick(rng, used, s_b + 3, min(s_b + 40, cut - 30))
        if s_d is None:
            return None
        new(
            s_d,
            "TRANSFER",
            rng.choice(HUMANS),
            service=service,
            team=teams[3],
            info=f"ticket=ORG-{rng.randint(100, 999)}",
        )
        tr_b.anchor = min(cut - 1, s_d + rng.randint(1, 20))
        last_transfer = s_d
    s_c = _pick(rng, used, last_transfer + 1, cut - 5)
    if s_c is None:
        return None
    tr_c = new(
        s_c,
        "TRANSFER",
        rng.choice(HUMANS),
        service=service,
        team=teams[1],
        info=f"ticket=ORG-{rng.randint(100, 999)}",
    )
    s_void = _pick(rng, used, cut, n) if rng.random() < 0.5 else _pick(rng, used, s_c + 1, cut)
    if s_void is None:
        return None
    fix_event(s_void, tr_c, "VOID")
    s_e = _pick(rng, used, s_c + 1, cut)
    s_f = _pick(rng, used, cut, n)
    if s_e is None or s_f is None:
        return None
    new(
        s_e,
        "TRANSFER",
        rng.choice(HUMANS),
        service=rng.choice(partners),
        team=teams[2],
        info=f"ticket=ORG-{rng.randint(100, 999)}",
    )
    new(
        s_f,
        "TRANSFER",
        rng.choice(HUMANS),
        service=service,
        team=teams[4],
        info=f"ticket=ORG-{rng.randint(100, 999)}",
    )

    # --- Decoys: a prefix-twin deploy in the target region just before the cutoff, etc. ------------
    s_nd = _pick(rng, used, tail_slots[-1] + 1, cut)
    if s_nd is None:
        return None
    reserved[s_nd] = "PLANTED_TWIN_DEPLOY"
    s_dry = _pick(rng, used | dict.fromkeys(reserved), tail_slots[0], cut)
    if s_dry is not None and rng.random() < 0.7:
        reserved[s_dry] = "DRYRUN_TARGET"
    for cat in ("DECOY_IN", "DECOY_SVC", "DECOY_VOID", "DECOY_IN"):
        slot = _pick(rng, used | dict.fromkeys(reserved), int(n * 0.3), n - 5)
        if slot is not None:
            reserved[slot] = cat

    # --- Background, in time order --------------------------------------------------------------
    pair_list: list[tuple[str, str]] = []
    pair_w: list[float] = []
    for s in services:
        for r in REGIONS:
            if (s, r) == (service, region):
                continue
            if s == service:
                w = 3.0
            elif s in family_others:
                w = 3.0 if r == region else 0.6
            elif s in (other_base, other_dups[0]):
                w = 0.5
            else:
                w = 0.35
            pair_list.append((s, r))
            pair_w.append(w)
    stacks: dict[tuple[str, str], list[str]] = {p: [] for p in pair_list}
    vcount: dict[tuple[str, str], int] = {p: rng.randint(0, 4) for p in pair_list}
    bg_refs: list[_Event] = []
    bg_transfers: list[_Event] = []
    alerts: list[_Event] = []
    deploys: list[_Event] = []
    referenced: set[int] = set()
    cats = list(NOISE_WEIGHTS)
    weights = list(NOISE_WEIGHTS.values())

    def bg_deploy(slot: int, s: str, r: str) -> _Event:
        p = (s, r)
        vcount[p] = min(len(trains[s]) - 1, vcount[p] + rng.choice([1, 1, 2]))
        stacks[p].append(trains[s][vcount[p]])
        ev = new(
            slot,
            "DEPLOY",
            rng.choice(actors),
            service=s,
            region=r,
            version=trains[s][vcount[p]],
            info=f"change=CHG-{rng.randint(10000, 99999)} sha={rng.getrandbits(28):07x}",
        )
        bg_refs.append(ev)
        return ev

    def pick_ref(pool: list[_Event], ok: Any) -> _Event | None:
        cands = [e for e in pool if e.idx not in referenced and e.anchor < 0 and ok(e)]
        return rng.choice(cands) if cands else None

    for slot in range(n):
        if slot in used:
            ev = used[slot]
            if ev.kind in ("DEPLOY", "ROLLBACK"):
                deploys.append(ev)
            continue
        cat = reserved.get(slot) or rng.choices(cats, weights)[0]
        r_any = rng.choice(REGIONS)
        s_any = rng.choices(
            services, [3.0 if s == service or s in family_others else 1.0 for s in services]
        )[0]
        if cat == "PLANTED_TWIN_DEPLOY":
            bg_deploy(slot, rng.choice(partners), region)
        elif cat == "DRYRUN_TARGET":
            new(
                slot,
                "DRYRUN",
                rng.choice(actors),
                info=f"{service} {region} {trains[service][min(vi + 1, len(trains[service]) - 1)]}  result=no-diff",
            )
        elif cat == "DEPLOY" or (cat == "ROLLBACK" and rng.random() < 0.1):
            s, r = rng.choices(pair_list, pair_w)[0]
            if cat == "DEPLOY":
                deploys.append(bg_deploy(slot, s, r))
            else:  # a rollback that is (probably) rejected
                ev = new(
                    slot,
                    "ROLLBACK",
                    rng.choice(actors),
                    service=s,
                    region=r,
                    info=f'change=CHG-{rng.randint(10000, 99999)} reason="{rng.choice(ROLLBACK_REASONS)}"',
                )
                if len(stacks[(s, r)]) >= 2:
                    stacks[(s, r)].pop()
                bg_refs.append(ev)
        elif cat == "ROLLBACK":
            deep = [(p, w) for p, w in zip(pair_list, pair_w, strict=True) if len(stacks[p]) >= 2]
            if not deep:
                new(
                    slot,
                    "SCALE",
                    "autoscaler",
                    info=f"{s_any} {r_any} replicas={rng.randint(2, 8)}->{rng.randint(9, 24)}",
                )
                continue
            (s, r), _w = rng.choices(deep, [w for _, w in deep])[0]
            stacks[(s, r)].pop()
            ev = new(
                slot,
                "ROLLBACK",
                rng.choice(actors),
                service=s,
                region=r,
                info=f'change=CHG-{rng.randint(10000, 99999)} reason="{rng.choice(ROLLBACK_REASONS)}"',
            )
            bg_refs.append(ev)
            deploys.append(ev)
        elif cat == "TRANSFER":
            s = rng.choice([x for x in services if x != service])
            team = rng.choice([tm for tm in TEAMS if tm != owner_of[s]])
            owner_of[s] = team
            bg_transfers.append(
                new(
                    slot,
                    "TRANSFER",
                    rng.choice(HUMANS),
                    service=s,
                    team=team,
                    info=f"ticket=ORG-{rng.randint(100, 999)}",
                )
            )
        elif cat in ("CORRECTION", "VOID") or cat.startswith("DECOY"):
            if cat == "DECOY_IN":
                ref = pick_ref(bg_refs, lambda e: e.service in family_others and e.region != region)
            elif cat == "DECOY_SVC":
                ref = pick_ref(bg_refs, lambda e: e.service == service)
            elif cat == "DECOY_VOID":
                ref = pick_ref(bg_refs, lambda e: e.service in family_others and e.region == region)
            elif cat == "VOID":
                ref = pick_ref(bg_refs + bg_transfers, lambda e: True)
            else:
                ref = pick_ref(bg_refs, lambda e: True)
            if ref is None:
                new(slot, "CONFIG", rng.choice(HUMANS), info=f"{s_any} {r_any} log_level=info")
                continue
            referenced.add(ref.idx)
            if cat in ("VOID", "DECOY_VOID"):
                fix_event(slot, ref, "VOID")
            else:
                banned = {ref.region, region} if ref.service == service else {ref.region}
                target = (
                    region
                    if cat == "DECOY_IN"
                    else rng.choice([x for x in REGIONS if x not in banned])
                )
                fix_event(slot, ref, "CORRECTION", target)
        elif cat == "BUILD":
            v = trains[s_any][rng.randrange(len(trains[s_any]))]
            new(
                slot,
                "BUILD",
                "ci-runner",
                info=f"{s_any} {v}  pipeline=build-{rng.randint(10000, 99999)} status={rng.choice(['passed'] * 4 + ['failed'])}",
            )
        elif cat == "DRYRUN":
            v = trains[s_any][rng.randrange(len(trains[s_any]))]
            new(
                slot,
                "DRYRUN",
                rng.choice(actors),
                info=f"{s_any} {r_any} {v}  result={rng.choice(['no-diff', '3 resources changed', 'plan ok'])}",
            )
        elif cat == "SCALE":
            a = rng.randint(2, 12)
            new(
                slot,
                "SCALE",
                rng.choice(["autoscaler", *HUMANS[:6]]),
                info=f'{s_any} {r_any} replicas={a}->{a + rng.choice([-2, -1, 2, 3, 4])} reason="traffic forecast"',
            )
        elif cat == "CONFIG":
            ckey = rng.choice(list(CONFIG_VALUES))
            new(
                slot,
                "CONFIG",
                rng.choice(HUMANS),
                info=f"{s_any} {r_any} {ckey}={rng.choice(CONFIG_VALUES[ckey])}",
            )
        elif cat == "ALERT":
            metric, (lo_v, hi_v), text = rng.choice(ALERT_TEXT)
            alerts.append(
                new(
                    slot,
                    "ALERT",
                    "sre-pager",
                    info=f'{s_any} {r_any} {metric}={rng.randint(lo_v, hi_v)} sev={rng.randint(1, 4)} "{text}"',
                )
            )
        elif cat == "ACK" and alerts:
            new(
                slot,
                "ACK",
                rng.choice(HUMANS),
                ref=rng.choice(alerts[-12:]).idx,
                info=f'#E-{{ref}} "{rng.choice(ACK_TEXT)}"',
            )
        elif cat == "ONCALL":
            h1, h2 = rng.sample(HUMANS, 2)
            new(slot, "ONCALL", "pagerduty", info=f"{rng.choice(TEAMS)} {h1}->{h2}")
        elif deploys and rng.random() < 0.3:
            new(
                slot,
                "NOTE",
                rng.choice(HUMANS),
                ref=rng.choice(deploys[-40:]).idx,
                info='"re #E-{ref}: post-change smoke tests green"',
            )
        else:
            text = rng.choice(NOTE_TEXT).format(
                s=s_any, r=r_any, v=trains[s_any][rng.randrange(40)], n=rng.randint(100, 999)
            )
            new(slot, "NOTE", rng.choice(HUMANS), info=f'"{text}"')

    # Background late arrivals (never entries that fixes reference, never fixes' targets of the plan).
    planned_idx = {ev.idx for _, ev in op_events}
    pool = [
        e
        for e in events
        if e.anchor < 0
        and e.idx not in referenced
        and e.idx not in planned_idx
        and e.kind not in FIX_KINDS
        and e.kind != "TRANSFER"
        and e.slot < n - 10
    ]
    for e in rng.sample(pool, min(len(pool), rng.randint(14, 22))):
        e.anchor = min(n - 1, e.slot + rng.randint(4, 45))

    # --- Layout and IDs -----------------------------------------------------------------------
    by_line = sorted(
        events, key=lambda e: (times[e.anchor], 1, e.ts) if e.anchor >= 0 else (e.ts, 0, e.ts)
    )
    for line, (e, eid) in enumerate(
        zip(by_line, rng.sample(range(1000, 10000), len(by_line)), strict=True)
    ):
        e.line, e.eid = line, eid
    by_eid = {e.eid: e for e in events}
    decoy_pairs = []
    for op, ev in op_events[: len(pre)]:
        if op.fixes:
            cands = [
                e
                for e in events
                if e.kind == ev.kind
                and e.ts <= cutoff
                and e.idx not in planned_idx
                and (e.service in partners or (e.service == service and e.region != region))
            ]
            if cands:
                decoy_pairs.append((ev, rng.choice(cands)))
    for ref, decoy in rng.sample(decoy_pairs, min(2, len(decoy_pairs))):
        twin = _twin(rng, ref.eid)
        holder = by_eid.get(twin)
        if twin < 1000 or holder is ref or holder is decoy:
            continue
        if holder is not None:
            holder.eid = decoy.eid
            by_eid[holder.eid] = holder
        else:
            del by_eid[decoy.eid]
        decoy.eid = twin
        by_eid[twin] = decoy
    cutoff_line = min(e.line for e in events if e.ts > cutoff)
    target_events = sum(1 for _, ev in op_events[: len(pre)] if ev.ts <= cutoff)
    return _Log(
        events,
        cutoff,
        cutoff_line,
        service,
        region,
        partners,
        initial_owner,
        has_multi,
        target_fix_count,
        target_events,
        owners_at_start,
        (service, *family_others),
    )


def _stamp(start: datetime, minutes: int) -> str:
    return (start + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%MZ")


def _render_line(e: _Event, start: datetime, eid_of: dict[int, int]) -> str:
    if e.kind == "DEPLOY":
        args = f"{e.service} {e.region} {e.version}  {e.info}"
    elif e.kind == "ROLLBACK":
        args = f"{e.service} {e.region}  {e.info}"
    elif e.kind == "TRANSFER":
        args = f"{e.service} {e.team}  {e.info}"
    elif e.kind == "CORRECTION":
        args = f"#E-{eid_of[e.ref]} region={e.new_region}  {e.info}"
    elif e.kind == "VOID":
        args = f"#E-{eid_of[e.ref]}  {e.info}"
    else:
        args = e.info.replace("{ref}", str(eid_of[e.ref])) if e.ref >= 0 else e.info
    tag = "  [replayed from buffer]" if e.anchor >= 0 else ""
    return f"#E-{e.eid} {_stamp(start, e.ts)}  {e.actor:<11} {e.kind:<10} {args}{tag}"


def _prompt(
    company: str, start: datetime, log: _Log, lines: list[str], keys: tuple[str, ...]
) -> str:
    s, r = log.service, log.region
    family = ", ".join(f"`{x}`" for x in log.family)
    owner_rows = "\n".join(
        f"| {svc} | {team} |" for svc, team in sorted(log.owners_at_start.items())
    )
    key_lines = "\n".join(
        f"- `{k}` ({KEY_TEXT[k][0]}): {KEY_TEXT[k][1].format(s=s, r=r)}." for k in keys
    )
    example = "{" + ", ".join(f'"{k}": {json.dumps(KEY_PLACEHOLDER[k])}' for k in keys) + "}"
    log_text = "\n".join(lines)
    return f"""You are auditing the production change log of {company} (a fictional company). The log semantics below are complete and authoritative; apply them exactly.

## Log semantics

Each line is `#E-<id> <timestamp> <actor> <ENTRY TYPE> <arguments>`, optionally followed by informational `key=value` fields, quoted text and a `[replayed from buffer]` tag. Informational fields never change anything.

1. Event IDs (`#E-` plus four digits) are unique labels and are NOT in timestamp order. Timestamps are UTC with minute resolution, and no two entries share a timestamp.
2. Every entry takes effect at its timestamp, not at its position in the log. The log is mostly chronological, but late-arriving entries (tagged `[replayed from buffer]`) are printed further down than their timestamps would place them.
3. Every (service, region) pair has its own release stack, empty at the start of the log. `DEPLOY <service> <region> <version>` pushes `<version>` onto that pair's stack. The pair's running version is the version on top of its stack.
4. `ROLLBACK <service> <region>` pops the top version off that pair's stack, so the version beneath it becomes the running version again. If the stack holds fewer than 2 versions at that moment, the rollback is REJECTED: it changes nothing and is not an executed rollback.
5. Service and region names are exact identifiers: for example {family} are all different services, and `{r}` and `{NEAR_REGION[r]}` are different regions.
6. `TRANSFER <service> <team>` makes `<team>` the owner of `<service>` (in every region) from that timestamp on. Owners at the start of the log are listed below.
7. `CORRECTION #E-<id> region=<region>`: the referenced DEPLOY or ROLLBACK actually happened in `<region>` (same service, timestamp, actor and version); treat the original entry as if it had named `<region>`. If an entry has several corrections, only the correction with the latest timestamp counts.
8. `VOID #E-<id>`: the referenced DEPLOY, ROLLBACK or TRANSFER never happened; ignore it entirely.
9. CORRECTION and VOID entries are retroactive fixes to the historical record: they apply no matter when they were logged, including after the cutoff in the question. They only reference DEPLOY, ROLLBACK or TRANSFER entries (never each other), every referenced ID appears in the log, and no entry is both corrected and voided.
10. All other entry types (BUILD, DRYRUN, SCALE, CONFIG, ALERT, ACK, ONCALL, NOTE) never change release stacks or ownership, whatever their text says.

## Owners at the start of the log

| service | owner team |
|---|---|
{owner_rows}

## Change log ({len(lines)} entries)

```text
{log_text}
```

## Question

Cutoff: **{_stamp(start, log.cutoff)}**. Consider service `{s}` in region `{r}` at the cutoff. Entries with a timestamp after the cutoff have not happened yet, so ignore them, except that every CORRECTION and VOID in the log applies no matter when it was logged (rule 9). Counts include only entries with a timestamp at or before the cutoff, evaluated after applying all corrections and voids.

Report the state at the cutoff as a JSON object with exactly these lower-case keys:
{key_lines}

Reply with exactly one fenced code block, tagged `json`, containing only the object, for example:
```json
{example}
```
"""


def wrong_reading_answers(log: _Log, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Answers under each applicable sloppy reading in `READINGS` (restricted to `keys`)."""
    names = [k for k in READINGS if k != "first_correction_wins" or log.has_multi]
    return {name: {k: _evaluate(log, READINGS[name])[k] for k in keys} for name in names}


def _check_log(log: _Log) -> None:
    """Assert the structural promises the prompt makes (rules 1, 2 and 9)."""
    events = log.events
    by_idx = {e.idx: e for e in events}
    if len({e.eid for e in events}) != len(events) or len({e.ts for e in events}) != len(events):
        raise AssertionError("event IDs and timestamps must be unique")
    if any(e.ts == log.cutoff for e in events):
        raise AssertionError("the cutoff must not coincide with an entry")
    fixes = [e for e in events if e.kind in FIX_KINDS]
    voided = [e.ref for e in fixes if e.kind == "VOID"]
    corrected = {e.ref for e in fixes if e.kind == "CORRECTION"}
    if len(voided) != len(set(voided)) or corrected.intersection(voided):
        raise AssertionError("an entry is voided twice or both corrected and voided")
    for e in fixes:
        ref = by_idx[e.ref]
        if (
            ref.kind not in STATE_KINDS
            or (e.kind == "CORRECTION" and ref.kind == "TRANSFER")
            or ref.ts > e.ts
        ):
            raise AssertionError(f"bad fix target: {e.kind} -> {ref.kind}")
    plain = sorted((e for e in events if e.anchor < 0), key=attrgetter("line"))
    if any(a.ts > b.ts for a, b in itertools.pairwise(plain)):
        raise AssertionError("untagged entries must be printed in timestamp order")
    for e in events:
        if e.anchor >= 0 and not any(o.ts > e.ts and o.line < e.line for o in plain):
            raise AssertionError("a replayed entry must be printed after a later entry")


def build_timeline(
    rng: random.Random, hard_index: int
) -> tuple[_Log, dict[str, Any], dict[str, dict[str, Any]]]:
    """Generate a log whose every trap is load-bearing: (log, expected answer, wrong-reading answers)."""
    keys = KEY_SETS[hard_index % len(KEY_SETS)]
    for _attempt in range(MAX_ATTEMPTS):
        log = _build(rng, hard_index)
        if log is None:
            continue
        truth = _evaluate(log, {})
        if truth["stack_depth"] < 2 or truth["rejected_rollbacks"] < 1:
            continue
        if log.target_event_count < MIN_TARGET_EVENTS or log.target_fix_count < MIN_TARGET_FIXES:
            continue
        expected = {k: truth[k] for k in keys}
        wrong = wrong_reading_answers(log, keys)
        if all(ans != expected for ans in wrong.values()):
            _check_log(log)
            return log, expected, wrong
    raise AssertionError(
        f"could not generate a load-bearing timeline item for hard_index={hard_index}"
    )


def make_timeline_item(rng: random.Random, item_id: str, hard_index: int) -> dict[str, Any]:
    """Render one ~600-entry operational change-log item with cutoff state question."""
    log, expected, _wrong = build_timeline(rng, hard_index)
    keys = KEY_SETS[hard_index % len(KEY_SETS)]
    company = rng.choice(COMPANIES)
    start = datetime(
        2026,
        rng.randint(1, 11),
        rng.randint(1, 26),
        rng.randint(0, 23),
        rng.randint(0, 59),
        tzinfo=UTC,
    )
    eid_of = {e.idx: e.eid for e in log.events}
    lines = [_render_line(e, start, eid_of) for e in sorted(log.events, key=attrgetter("line"))]
    return {
        "id": item_id,
        "slice": "s3_frontier_reasoning",
        "modality": "text",
        "subtype": "longctx_ops_timeline_state_tracking",
        "source_provenance": (
            f"Original synthetic change log ({len(lines)} entries) of a fictional company; state of "
            f"{log.service} in {log.region} at a mid-log cutoff with ID-only corrections/voids, replayed "
            "entries and near-duplicate names"
        ),
        "prompt": _prompt(company, start, log, lines, keys),
        "attachments": [],
        "grader_type": "exact_json",
        "grader_spec": {"expected": expected},
    }
