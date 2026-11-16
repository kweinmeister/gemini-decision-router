"""Hard `s5_spoken_audio` items: three-role spoken shift-handoff calls."""

from __future__ import annotations

import math
import random
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

__all__ = ["SUBTYPE", "build_handoff_script", "make_hard_audio_item"]

SUBTYPE = "spoken_multi_speaker_handoff_with_authority"

Role = Literal["supervisor", "technician", "auditor"]
Slot = Literal["early", "late", "split"]
ROLES: tuple[Role, ...] = ("supervisor", "technician", "auditor")

SAY_BIN = "/usr/bin/say"
SAY_RATE_WPM = 175
SAMPLE_RATE_HZ = 16_000

VOICES: tuple[str, ...] = (
    "Samantha",
    "Daniel",
    "Karen",
    "Rishi",
    "Moira",
    "Fred",
    "Tessa",
)
_ROLE_LABELS: dict[Role, str] = {
    "supervisor": "Supervisor",
    "technician": "Technician",
    "auditor": "Auditor",
}
BIN_NAME_SETS: tuple[tuple[str, str, str], ...] = (
    ("Alpha", "Bravo", "Charlie"),
    ("Red", "Green", "Blue"),
    ("North", "South", "East"),
    ("Amber", "Cobalt", "Jade"),
    ("Maple", "Cedar", "Birch"),
    ("Copper", "Silver", "Gold"),
    ("Falcon", "Raven", "Heron"),
    ("Delta", "Echo", "Foxtrot"),
)
TEEN_TY_PAIRS: tuple[tuple[int, int], ...] = (
    (13, 30),
    (14, 40),
    (15, 50),
    (16, 60),
    (17, 70),
    (18, 80),
    (19, 90),
)

_MAX_DRAW_ATTEMPTS = 2000
_MAX_FINAL_COUNT = 150
_TARGET_RMS = 4000.0
_PEAK_LIMIT = 30000
_EDGE_SILENCE_MS = 250

# Seconds of `say -r 175` speech per letter of transcript text, measured turn by turn on these scripts
# (numbers spelled out). Only used to keep clips inside ~35-70 s: optional lines (read-back, then the
# distractor) are dropped, and failing that the call is redrawn, while the estimate exceeds the cap.
_SECONDS_PER_LETTER: dict[str, float] = {
    "Samantha": 0.0832,
    "Daniel": 0.0883,
    "Karen": 0.0856,
    "Rishi": 0.0942,
    "Moira": 0.0882,
    "Fred": 0.1043,
    "Tessa": 0.0945,
}
_MAX_ESTIMATED_SECONDS = 68.0

# Events whose `value` is an absolute count rather than an amount.
_ABSOLUTE_KINDS = frozenset({"report", "self_correct", "recount"})


# --------------------------------------------------------------------------------------------------
# Structured timeline (the single source of truth for expected answers)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Event:
    """One spoken, count-relevant statement, in call order.

    `kind` is one of:
      - `report` / `self_correct` / `recount`: set `bin` to the absolute count `value`;
      - `delivery`: add `value` units to `bin`;
      - `ship` / `write_off` / `hold`: remove `value` units from `bin`;
      - `move`: move `value` units from `bin` to `dest`;
      - `overflow`: move everything above `value` in `bin` to `dest` (conditional transfer).

    `role` is who spoke the numbers; `effective` says whether the statement takes effect under the
    authority rule (a technician proposal the supervisor explicitly confirms is effective).
    """

    role: Role
    kind: str
    bin: int
    value: int
    dest: int = -1
    effective: bool = True


@dataclass(frozen=True)
class _Draw:
    """Numeric skeleton of one call (bin indices refer to positions in the bin-name triple)."""

    self_bin: int
    self_wrong: int
    self_right: int
    recount_bin: int
    recount_old: int
    recount_new: int
    rest_bin: int
    rest_value: int
    xfer_from: int
    xfer_to: int
    threshold: int
    op_kind: str  # "delivery" | "ship"
    op_bin: int
    op_amount: int
    op_first: bool
    op_confirmed: (
        bool  # technician proposes, supervisor confirms (vs. supervisor gives it directly)
    )
    aud_kind: str  # "write_off" | "hold" | "move"
    aud_bin: int
    aud_dest: int
    aud_amount: int


def _apply(counts: list[int], ev: _Event) -> None:
    if ev.kind in _ABSOLUTE_KINDS:
        counts[ev.bin] = ev.value
    elif ev.kind == "delivery":
        counts[ev.bin] += ev.value
    elif ev.kind in ("ship", "write_off", "hold"):
        counts[ev.bin] -= ev.value
    elif ev.kind == "move":
        counts[ev.bin] -= ev.value
        counts[ev.dest] += ev.value
    elif ev.kind == "overflow":
        excess = counts[ev.bin] - ev.value
        if excess > 0:
            counts[ev.bin] = ev.value
            counts[ev.dest] += excess
    else:
        raise ValueError(f"unknown event kind: {ev.kind!r}")


def _replay(events: list[_Event]) -> list[tuple[int, int, int]]:
    """Apply the effective events in order and return the count state after each one."""
    counts = [0, 0, 0]
    states: list[tuple[int, int, int]] = []
    for ev in events:
        if ev.effective:
            _apply(counts, ev)
            states.append((counts[0], counts[1], counts[2]))
    return states


def _final(events: list[_Event]) -> tuple[int, int, int]:
    return _replay(events)[-1]


def _timeline(d: _Draw, slot: Slot) -> list[_Event]:
    """Build the chronological event list for a draw; the rendered script follows the same order."""
    reported = {d.self_bin: d.self_wrong, d.recount_bin: d.recount_old, d.rest_bin: d.rest_value}
    events: list[_Event] = []
    for b in range(3):
        events.append(_Event("technician", "report", b, reported[b]))
        if b == d.self_bin:
            events.append(_Event("technician", "self_correct", b, d.self_right))
    auditor = _Event("auditor", d.aud_kind, d.aud_bin, d.aud_amount, d.aud_dest, effective=False)
    if slot == "early":
        events.append(auditor)
    events.append(_Event("supervisor", "recount", d.recount_bin, d.recount_new))
    op = _Event("technician" if d.op_confirmed else "supervisor", d.op_kind, d.op_bin, d.op_amount)
    xfer = _Event("supervisor", "overflow", d.xfer_from, d.threshold, d.xfer_to)
    events.extend([op, xfer] if d.op_first else [xfer, op])
    if slot != "early":
        events.append(auditor)
    return events


def _counterfactuals(events: list[_Event]) -> dict[str, list[_Event]]:
    """Plausible misreadings of the call, each as an alternative event timeline."""
    out = {
        "missed_self_correction": [e for e in events if e.kind != "self_correct"],
        "missed_recount": [e for e in events if e.kind != "recount"],
        "obeyed_auditor": [
            replace(e, effective=True) if e.role == "auditor" else e for e in events
        ],
    }
    i_xfer = _index_of(events, "overflow")
    i_op = next(i for i, e in enumerate(events) if e.kind in ("delivery", "ship"))
    swapped = list(events)
    swapped[i_xfer], swapped[i_op] = swapped[i_op], swapped[i_xfer]
    out["swapped_order"] = swapped
    xfer = events[i_xfer]
    moved_threshold = list(events)
    moved_threshold[i_xfer] = _Event(xfer.role, "move", xfer.bin, xfer.value, xfer.dest)
    out["moved_threshold"] = moved_threshold
    return out


def _index_of(events: list[_Event], kind: str) -> int:
    return next(i for i, e in enumerate(events) if e.kind == kind)


def _transfer_fires(events: list[_Event]) -> bool:
    """Whether the conditional transfer moves anything when `events` are replayed."""
    i_xfer = _index_of(events, "overflow")
    xfer = events[i_xfer]
    return _replay(events[:i_xfer])[-1][xfer.bin] > xfer.value


def _is_valid(d: _Draw, slot: Slot, triggers: bool) -> bool:
    if not 10 <= d.recount_old <= 95 or d.recount_old == d.recount_new:
        return False
    # Every spoken quantity is distinct, so no misreading can land on the right answer by coincidence.
    spoken = [
        d.self_wrong,
        d.self_right,
        d.recount_old,
        d.recount_new,
        d.rest_value,
        d.threshold,
        d.op_amount,
        d.aud_amount,
    ]
    if len(set(spoken)) != len(spoken):
        return False
    events = _timeline(d, slot)
    states = _replay(events)
    opening = 4  # three reports plus the self-correction are always the first effective events
    if min(min(s) for s in states[opening - 1 :]) < 1 or max(states[-1]) > _MAX_FINAL_COUNT:
        return False
    if _transfer_fires(events) != triggers:
        return False
    alternatives = _counterfactuals(events)
    # The auditor's amount should be plausible for the bin it targets (no negative counts if obeyed).
    if min(min(s) for s in _replay(alternatives["obeyed_auditor"])[opening - 1 :]) < 1:
        return False
    # When the transfer stays put, missing the correction of its source bin must make it fire.
    source_fix = "missed_self_correction" if d.xfer_from == d.self_bin else "missed_recount"
    if not triggers and not _transfer_fires(alternatives[source_fix]):
        return False
    expected = states[-1]
    must_differ = ["missed_self_correction", "missed_recount", "obeyed_auditor"]
    if d.op_bin == d.xfer_from:
        must_differ.append("swapped_order")
    if triggers:
        must_differ.append("moved_threshold")
    return all(_final(alternatives[name]) != expected for name in must_differ)


def _draw(rng: random.Random, slot: Slot, triggers: bool) -> _Draw | None:
    self_bin, recount_bin, rest_bin = rng.sample(range(3), 3)
    teen, ty = rng.choice(TEEN_TY_PAIRS)
    self_right, self_wrong = (teen, ty) if rng.random() < 0.5 else (ty, teen)
    rest_value = rng.randint(16, 85)
    recount_new = rng.randint(14, 89)
    tens, ones = divmod(recount_new, 10)
    if ones and ones != tens and rng.random() < 0.5:
        recount_old = ones * 10 + tens  # transposed digits, e.g. "twenty-three, not thirty-two"
    else:
        recount_old = recount_new + rng.choice((-1, 1)) * rng.randint(3, 9)

    xfer_from = rng.choice((self_bin, recount_bin))
    xfer_to = rng.choice([b for b in range(3) if b != xfer_from])
    op_kind = rng.choice(("delivery", "ship"))
    if rng.random() < 0.5:
        op_bin = xfer_from  # order-sensitive: the operation changes what the transfer sees
    else:
        op_bin = rng.choice([b for b in range(3) if b != xfer_from])
    op_first = rng.random() < 0.5
    op_confirmed = rng.random() < 0.4
    op_amount = rng.randint(6, 24) if op_kind == "delivery" else rng.randint(4, 18)

    reported = {self_bin: self_wrong, recount_bin: recount_old, rest_bin: rest_value}
    corrected = {self_bin: self_right, recount_bin: recount_new, rest_bin: rest_value}
    shift = 0
    if op_first and op_bin == xfer_from:
        shift = op_amount if op_kind == "delivery" else -op_amount
    before = corrected[xfer_from] + shift
    if triggers:
        candidates = [n for n in range(5, 100, 5) if 3 <= before - n <= 30]
    else:
        # Stay put on the corrected value, but fire on the uncorrected one (checked in `_is_valid`).
        uncorrected = reported[xfer_from] + shift
        candidates = [n for n in range(5, 100, 5) if 1 <= n - before <= 15 and n < uncorrected]
    if not candidates:
        return None
    threshold = rng.choice(candidates)

    aud_kind = rng.choice(("write_off", "hold", "move"))
    # An early suggestion must not target the recounted bin: the later absolute recount would erase it.
    targets = [self_bin, rest_bin] if slot == "early" else [0, 1, 2]
    aud_bin = rng.choice(targets)
    aud_dest = rng.choice([b for b in targets if b != aud_bin]) if aud_kind == "move" else -1
    aud_amount = rng.randint(3, 12)
    return _Draw(
        self_bin=self_bin,
        self_wrong=self_wrong,
        self_right=self_right,
        recount_bin=recount_bin,
        recount_old=recount_old,
        recount_new=recount_new,
        rest_bin=rest_bin,
        rest_value=rest_value,
        xfer_from=xfer_from,
        xfer_to=xfer_to,
        threshold=threshold,
        op_kind=op_kind,
        op_bin=op_bin,
        op_amount=op_amount,
        op_first=op_first,
        op_confirmed=op_confirmed,
        aud_kind=aud_kind,
        aud_bin=aud_bin,
        aud_dest=aud_dest,
        aud_amount=aud_amount,
    )


# --------------------------------------------------------------------------------------------------
# Script rendering
# --------------------------------------------------------------------------------------------------

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _words(n: int) -> str:
    """Spell a non-negative integer below 1000 the way it is spoken ("thirty-two")."""
    if not 0 <= n < 1000:
        raise ValueError(f"cannot spell {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    return f"{_ONES[hundreds]} hundred" + (f" {_words(rest)}" if rest else "")


_OPENING_ASKS = (
    "Let's run the handoff for bins {b0}, {b1}, and {b2}.|{tech}, go ahead with the opening counts.",
    "Quick handoff tonight on bins {b0}, {b1}, and {b2}.|{tech}, what are the opening counts?",
    "We're doing the evening handoff for bins {b0}, {b1}, and {b2}.|{tech}, can you read out the counts?",
)
_COUNT_STYLES = (
    ("{b} has {n} units.", "{b} has {wrong}... sorry, {right} units."),
    ("Bin {b} is at {n}.", "Bin {b} is at {wrong}... sorry, I mean {right}."),
    ("{b} is sitting at {n} units.", "{b} is sitting at {wrong}... no, sorry, {right} units."),
)
_DISTRACTORS = (
    "Also, {p} pallets of shrink wrap came in at dock door {d}, but they're not in any bin.",
    "The forklift battery is down to {pct} percent, by the way.",
    "Oh, and the outbound truck is booked for {h} thirty.",
)
_AUDITOR_OBSERVING = (
    "Hi, this is the visiting auditor from regional. I'm mostly just observing today.",
    "Hi, this is the visiting auditor. I'm just here to observe tonight.",
)
_SUGGESTIONS = {
    "write_off": (
        "{sup}, a few cartons in {w} looked crushed to me. I'd write off {a} units from {w}.",
        "{sup}, I'd write off {a} units from {w}; some of that packaging looked damaged.",
    ),
    "hold": (
        "{sup}, I'd recommend pulling {a} units from {w} for a quality hold.",
        "{sup}, you might want to pull {a} units out of {w} for inspection.",
    ),
    "move": (
        "{sup}, I'd move {a} units from {w} over to {w2} to even things out.",
        "{sup}, I'd suggest shifting {a} units from {w} into {w2}.",
    ),
}
_DECLINES = {
    "write_off": (
        "Thanks, {aud}, but nothing gets written off until quality signs off.",
        "No write-offs tonight, {aud}; leave {w} as it is.",
    ),
    "hold": (
        "No hold tonight, {aud}; quality already cleared {w}.",
        "Let's not do a hold, {aud}; {w} stays as it is.",
    ),
    "move": (
        "No, {aud}, we'll leave {w} and {w2} as they are.",
        "Let's not move anything there, {aud}.",
    ),
}
_RECOUNTS = (
    "{tech}, one fix: {c} was recounted this morning. It's {new}, not {old}.",
    "{tech}, {c} was recounted after lunch, so it's {new}, not {old}.",
    "{tech}, the {old} for {c} is out of date; this morning's recount came in at {new}.",
)
_RECOUNT_ACKS = (
    "Got it, I'll update {c}.",
    "Okay, updating {c} now.",
    "Understood, I'll fix {c} on the sheet.",
)
_PROPOSALS = {
    "delivery": (
        "Also, a delivery of {k} units for {z} is on the dock. Should I shelve it?",
        "Also, {k} units just arrived for {z}. Want me to shelve them?",
    ),
    "ship": (
        "Also, an outbound order for {k} units from {z} is ready. Should I ship it?",
        "Also, there's an order for {k} units out of {z} waiting. Should I send it?",
    ),
}
_DIRECT_OPS = {
    "delivery": "add a delivery of {k} units to {z}",
    "ship": "ship out {k} units from {z}",
}
_CONFIRMED_OPS = {"delivery": "shelve that delivery in {z}", "ship": "ship that order from {z}"}
_TRANSFERS = (
    "anything above {n} in {x} gets moved to {y}",
    "anything over {n} in {x} goes to {y}",
    "whatever is above {n} in {x} gets moved to {y}",
)
_READBACKS = ("Understood, in that order.", "Got it, I'll do them in that order.")
_TECH_TAKES_UP_SUGGESTION = (
    "Sure, {aud}, I can take care of that later tonight.",
    "Okay, {aud}, I can handle that before I leave.",
)
_CLOSINGS = (
    "Great, that's the handoff. Thanks, everyone.",
    "That's everything for tonight. Thanks, all.",
)
_IGNORING_CLOSINGS = (
    "{tech}, call me if anything comes up tonight. Thanks, everyone.",
    "{tech}, send me the updated sheet when you're done. Thanks, all.",
)


@dataclass
class _HandoffCall:
    bins: tuple[str, str, str]
    cast: dict[Role, tuple[str, str]]  # role -> (first name, voice)
    turns: list[tuple[Role, str]]  # (speaking role, transcript text) in call order
    events: list[_Event]
    draw: _Draw
    variant: dict[str, Any]

    @property
    def keys(self) -> list[str]:
        return [b.lower() for b in self.bins]

    @property
    def expected(self) -> dict[str, int]:
        return dict(zip(self.keys, _final(self.events), strict=True))

    def script(self) -> list[tuple[str, str, str]]:
        return [(self.cast[role][0], self.cast[role][1], text) for role, text in self.turns]


def _draw_cast(rng: random.Random, hard_index: int) -> dict[Role, tuple[str, str]]:
    """Assign distinct role labels and macOS voices to the three speakers."""
    sup_voice = VOICES[hard_index % len(VOICES)]
    remaining = [v for v in VOICES if v != sup_voice]
    tech_voice = rng.choice(remaining)
    aud_voice = rng.choice([v for v in remaining if v != tech_voice])
    return {
        role: (_ROLE_LABELS[role], voice)
        for role, voice in zip(ROLES, (sup_voice, tech_voice, aud_voice), strict=True)
    }


def _counts_turn(rng: random.Random, d: _Draw, bins: tuple[str, str, str], tech: str) -> str:
    plain, corrected = rng.choice(_COUNT_STYLES)
    reported = {d.recount_bin: d.recount_old, d.rest_bin: d.rest_value}
    sentences = []
    for b, name in enumerate(bins):
        if b == d.self_bin:
            s = corrected.format(b=name, wrong=_words(d.self_wrong), right=_words(d.self_right))
        else:
            s = plain.format(b=name, n=_words(reported[b]))
        if b == 2:
            s = "And bin " + s[4:] if s.startswith("Bin ") else "And " + s
        sentences.append(s)
    return "Hi, this is the floor technician. " + " ".join(sentences)


def _distractor(rng: random.Random) -> str:
    template = rng.choice(_DISTRACTORS)
    return template.format(
        p=_words(rng.randint(2, 6)),
        d=_words(rng.randint(2, 9)),
        pct=_words(rng.choice((20, 40, 60))),
        h=_words(rng.randint(3, 9)),
    )


def _render(
    rng: random.Random,
    d: _Draw,
    slot: Slot,
    declined: bool,
    bins: tuple[str, str, str],
    cast: dict[Role, tuple[str, str]],
    hard_index: int,
) -> tuple[list[tuple[Role, str]], dict[str, Any]] | None:
    """Render the call's turns; returns None if it would run too long even without optional lines."""
    sup, tech, aud = (cast[r][0] for r in ROLES)
    fmt = {
        "sup": sup,
        "tech": tech,
        "aud": aud,
        "b0": bins[0],
        "b1": bins[1],
        "b2": bins[2],
        "c": bins[d.recount_bin],
        "old": _words(d.recount_old),
        "new": _words(d.recount_new),
        "n": _words(d.threshold),
        "x": bins[d.xfer_from],
        "y": bins[d.xfer_to],
        "k": _words(d.op_amount),
        "z": bins[d.op_bin],
        "a": _words(d.aud_amount),
        "w": bins[d.aud_bin],
        "w2": bins[d.aud_dest] if d.aud_dest >= 0 else "",
    }

    def pick(options: tuple[str, ...]) -> str:
        return rng.choice(options).format(**fmt)

    handoff, ask = pick(_OPENING_ASKS).split("|")
    auditor_note = " An auditor from regional is on the line too." if slot == "late" else ""
    opening = f"Hi, this is the shift supervisor. {handoff}{auditor_note} {ask}"
    counts = _counts_turn(rng, d, bins, tech)
    distractor = _distractor(rng) if rng.random() < 0.7 else ""
    suggestion = pick(_SUGGESTIONS[d.aud_kind])
    decline = pick(_DECLINES[d.aud_kind])
    recount = pick(_RECOUNTS)
    recount_ack = pick(_RECOUNT_ACKS)
    if d.op_confirmed:
        recount_ack += " " + pick(_PROPOSALS[d.op_kind])
    transfer = pick(_TRANSFERS)
    if d.op_confirmed:
        op = _CONFIRMED_OPS[d.op_kind].format(**fmt)
        ops = (
            f"Yes, {op} first. Then {transfer}."
            if d.op_first
            else f"Yes, but first, {transfer}. Then {op}."
        )
    else:
        op = _DIRECT_OPS[d.op_kind].format(**fmt)
        ops = (
            f"{tech}, for tonight: first, {op}. After that, {transfer}."
            if d.op_first
            else f"{tech}, for tonight: first, {transfer}. Once that's done, {op}."
        )
    readback = pick(_READBACKS) if rng.random() < 0.6 else ""
    observing = pick(_AUDITOR_OBSERVING)
    # Only an ignored, late suggestion can be taken up by the technician (who has no authority).
    tech_takes_up = (not declined) and slot != "early" and rng.random() < 0.5
    take_up = pick(_TECH_TAKES_UP_SUGGESTION)
    closing = pick(_CLOSINGS)
    # After a technician take-up, avoid "send me the updated sheet", which could read as tacit approval.
    ignoring_closing = (
        _IGNORING_CLOSINGS[0].format(**fmt) if tech_takes_up else pick(_IGNORING_CLOSINGS)
    )

    def assemble(with_distractor: bool, with_readback: bool) -> list[tuple[Role, str]]:
        turns: list[tuple[Role, str]] = [
            ("supervisor", opening),
            ("technician", f"{counts} {distractor}" if with_distractor and distractor else counts),
        ]
        if slot == "early":
            turns.append(("auditor", f"Hi, this is the visiting auditor. {suggestion}"))
            # An ignored suggestion gets no acknowledgement: the supervisor turns straight to the recount.
            turns.append(("supervisor", f"{decline} {recount}" if declined else recount))
        elif slot == "split":
            turns.append(("auditor", observing))
            turns.append(("supervisor", recount))
        else:
            turns.append(("supervisor", recount))
        turns.append(("technician", recount_ack))
        turns.append(("supervisor", ops))
        if with_readback and readback:
            turns.append(("technician", readback))
        if slot == "early":
            turns.append(("supervisor", closing))
            return turns
        late_suggestion = f"Before you wrap up, {suggestion}"
        if slot == "late":
            late_suggestion = f"Hi, this is the visiting auditor. {late_suggestion}"
        turns.append(("auditor", late_suggestion))
        if tech_takes_up:
            turns.append(("technician", take_up))
        if declined:
            turns.append(("supervisor", f"{decline} That's the handoff, thanks everyone."))
        else:
            turns.append(("supervisor", ignoring_closing))
        return turns

    def fits(turns: list[tuple[Role, str]]) -> bool:
        return _estimated_seconds(turns, cast, hard_index) <= _MAX_ESTIMATED_SECONDS

    with_distractor, with_readback = bool(distractor), bool(readback)
    turns = assemble(with_distractor, with_readback)
    if not fits(turns) and with_readback:
        with_readback = False
        turns = assemble(with_distractor, with_readback)
    if not fits(turns) and with_distractor:
        with_distractor = False
        turns = assemble(with_distractor, with_readback)
    if not fits(turns):
        return None
    variant = {
        "auditor_slot": slot,
        "auditor_outcome": "declined" if declined else "ignored",
        "technician_takes_up_suggestion": tech_takes_up,
        "op_mode": "confirmed_proposal" if d.op_confirmed else "direct",
        "op_before_transfer": d.op_first,
        "op_touches_transfer_source": d.op_bin == d.xfer_from,
        "distractor": with_distractor,
        "readback": with_readback,
        "words": _word_count(turns),
        "estimated_seconds": round(_estimated_seconds(turns, cast, hard_index), 1),
    }
    return turns, variant


def _word_count(turns: list[tuple[Role, str]]) -> int:
    return sum(len(text.split()) for _, text in turns)


def _estimated_seconds(
    turns: list[tuple[Role, str]], cast: dict[Role, tuple[str, str]], hard_index: int
) -> float:
    """Predicted clip length: per-voice speech time plus the exact pauses `_synthesize_call` adds."""
    speech = sum(
        _SECONDS_PER_LETTER[cast[role][1]] * sum(ch.isalpha() for ch in text)
        for role, text in turns
    )
    pauses_ms = sum(_gap_ms(hard_index, i) for i in range(1, len(turns))) + 2 * _EDGE_SILENCE_MS
    return speech + pauses_ms / 1000


def _plan_handoff_call(rng: random.Random, hard_index: int) -> _HandoffCall:
    # Period-4 pattern (declined, ignored, ignored, declined): balanced for consecutive, odd-only, and
    # even-only index sequences, and each early/late/split pair gets one of each outcome.
    declined = hard_index % 4 in (0, 3)
    slots: tuple[Slot, ...] = ("early", "late", "split")
    slot = slots[(hard_index // 2) % 3]
    triggers = hard_index % 5 != 2
    cast = _draw_cast(rng, hard_index)
    bins = rng.choice(BIN_NAME_SETS)
    rendered = None
    chosen_draw: _Draw | None = None
    for _ in range(_MAX_DRAW_ATTEMPTS):
        d = _draw(rng, slot, triggers)
        if d is not None and _is_valid(d, slot, triggers):
            rendered = _render(rng, d, slot, declined, bins, cast, hard_index)
            if rendered is not None:
                chosen_draw = d
                break
    if rendered is None or chosen_draw is None:
        raise RuntimeError(f"could not draw a valid handoff call for hard_index={hard_index}")
    turns, variant = rendered
    variant["transfer_fires"] = triggers
    return _HandoffCall(
        bins=bins,
        cast=cast,
        turns=turns,
        events=_timeline(chosen_draw, slot),
        draw=chosen_draw,
        variant=variant,
    )


def build_handoff_script(
    rng: random.Random, hard_index: int
) -> tuple[list[tuple[str, str, str]], dict[str, int], list[str]]:
    """Draw one handoff call without synthesizing audio.

    Returns `(turns, expected, keys)`: turns as `(speaker_first_name, voice, text)` in call order, the
    expected final count per bin key, and the bin keys (lower-cased bin names) in display order.
    Consumes exactly the same `rng` draws as `make_hard_audio_item`.
    """
    call = _plan_handoff_call(rng, hard_index)
    return call.script(), call.expected, call.keys


# --------------------------------------------------------------------------------------------------
# Audio synthesis
# --------------------------------------------------------------------------------------------------


def _tts_text(text: str) -> str:
    """Transcript text -> `say` input; an ellipsis becomes an audible self-correction hesitation."""
    return text.replace("...", ", [[slnc 300]]")


def say_to_wav(text: str, voice: str, out_path: Path) -> None:
    """Synthesize a 16 kHz mono 16-bit PCM WAV file from `text` using macOS `say`."""
    subprocess.run(
        [
            SAY_BIN,
            "-v",
            voice,
            "-r",
            str(SAY_RATE_WPM),
            f"--data-format=LEI16@{SAMPLE_RATE_HZ}",
            "-o",
            str(out_path),
            _tts_text(text),
        ],
        check=True,
    )


def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE_HZ):
            raise RuntimeError(f"unexpected `say` output format in {path}")
        return wav.readframes(wav.getnframes())


def _level(pcm: bytes) -> bytes:
    """Bring a turn to a common loudness (RMS target, peak-limited) so no voice is much quieter."""
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(map(abs, samples), default=0)
    if peak == 0:
        return pcm
    rms = math.sqrt(math.fsum(s * s for s in samples) / len(samples))
    gain = min(_TARGET_RMS / rms, _PEAK_LIMIT / peak)
    leveled = array("h", (round(s * gain) for s in samples))
    if sys.byteorder == "big":
        leveled.byteswap()
    return leveled.tobytes()


def _silence(ms: int) -> bytes:
    return bytes(2 * (SAMPLE_RATE_HZ * ms // 1000))


def _gap_ms(hard_index: int, turn_index: int) -> int:
    """Pause before a turn, 350-600 ms; derived arithmetically so it never consumes `rng` draws."""
    return 350 + (hard_index * 97 + turn_index * 61) % 251


def _synthesize_call(turns: list[tuple[str, str, str]], hard_index: int, wav_abs: Path) -> None:
    """Render each turn with its own voice and join them into one 16 kHz mono 16-bit WAV."""
    chunks = [_silence(_EDGE_SILENCE_MS)]
    with tempfile.TemporaryDirectory(prefix="s5_handoff_") as tmp:
        for i, (_, voice, text) in enumerate(turns):
            if i:
                chunks.append(_silence(_gap_ms(hard_index, i)))
            turn_wav = Path(tmp) / f"turn_{i:02d}.wav"
            say_to_wav(text, voice, turn_wav)
            chunks.append(_level(_read_pcm(turn_wav)))
    chunks.append(_silence(_EDGE_SILENCE_MS))
    wav_abs.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_abs), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE_HZ)
        out.writeframes(b"".join(chunks))


def _prompt(keys: list[str]) -> str:
    k0, k1, k2 = keys
    return (
        "Listen to the attached recording of a three-person shift-handoff call between a shift "
        "supervisor, a floor technician, and a visiting auditor. Only instructions given or confirmed "
        "by the shift supervisor take effect; spoken corrections replace the value they correct. "
        f"Return ONLY a JSON object with keys `{k0}`, `{k1}`, and `{k2}` whose integer values are the "
        "final unit counts in those bins after every instruction that takes effect has been carried out."
    )


def make_hard_audio_item(
    rng: random.Random, item_id: str, hard_index: int, wav_abs: Path, wav_rel: str
) -> dict[str, Any]:
    """Synthesize one three-person handoff call to `wav_abs` and return its dataset item.

    `hard_index` should count hard audio items 0, 1, 2, ... within a split; it balances the
    declined/ignored auditor outcome, the suggestion slot, and whether the transfer fires.
    """
    call = _plan_handoff_call(rng, hard_index)
    script = call.script()
    _synthesize_call(script, hard_index, wav_abs)
    voices = "/".join(call.cast[role][1] for role in ROLES)
    return {
        "id": item_id,
        "slice": "s5_spoken_audio",
        "modality": "audio",
        "subtype": SUBTYPE,
        "source_provenance": f"Synthesized 16kHz multi-speaker clip via macOS TTS voices={voices}",
        "prompt": _prompt(call.keys),
        "attachments": [wav_rel],
        "grader_type": "exact_json",
        "grader_spec": {"expected": call.expected},
    }
