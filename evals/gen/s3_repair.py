"""Subtype `symptom_only_multibug_repair` for `s3_frontier_reasoning`."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterator
from typing import Any

from evals.gen._edits import apply_edits as _apply

MIN_DETECTIONS = 3
MAX_SCENARIO_ATTEMPTS = 2000

CRON_SOURCE = '''"""cronlite: a dependency-free evaluator for classic 5-field cron expressions.

Times are naive ``(year, month, day, hour, minute)`` tuples; there are no time zones and no DST.

Fields, in order::

    minute         0-59
    hour           0-23
    day-of-month   1-31
    month          1-12   (JAN-DEC also accepted, case-insensitive)
    day-of-week    0-7    (0 and 7 both mean Sunday; SUN-SAT also accepted, case-insensitive)

Each field is a comma-separated list of items. An item is one of::

    *        every value in the field's range
    N        the single value N
    A-B      every value from A through B inclusive (A <= B)
    */S      every S-th value of the field's range, starting at the field minimum
    A-B/S    A, A+S, A+2S, ... while <= B
    N/S      N, N+S, N+2S, ... while <= the field maximum

Day matching follows Vixie cron. A day field is *restricted* when its text does not start
with ``*`` (so ``*/2`` counts as unrestricted). When BOTH day-of-month and day-of-week are
restricted, a day matches if EITHER field matches. Otherwise a day must match both fields
(an unrestricted field matches every day).
"""

from __future__ import annotations

MONTH_NAMES = {
    name: i + 1
    for i, name in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
DOW_NAMES = {name: i for i, name in enumerate(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"])}
MAX_SEARCH_YEARS = 8

Time = tuple[int, int, int, int, int]


def is_leap(year: int) -> bool:
    """Gregorian leap-year rule."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def weekday(year: int, month: int, day: int) -> int:
    """Day of week for a Gregorian date: 0 = Sunday ... 6 = Saturday (Sakamoto's method)."""
    offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if month < 3:
        year -= 1
    return (year + year // 4 - year // 100 + year // 400 + offsets[month - 1] + day) % 7


def _value(token: str, names: dict[str, int] | None) -> int:
    token = token.strip().upper()
    if names is not None and token in names:
        return names[token]
    return int(token)


def parse_field(text: str, lo: int, hi: int, names: dict[str, int] | None = None) -> frozenset[int]:
    """Expand one cron field into the set of values it selects (see the module docstring)."""
    values: set[int] = set()
    for item in text.split(","):
        base, _, step_text = item.partition("/")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise ValueError(f"step must be >= 1 in {item!r}")
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            first, last = base.split("-", 1)
            start, end = _value(first, names), _value(last, names)
        else:
            start = _value(base, names)
            end = hi if step_text else start
        if not lo <= start <= end <= hi:
            raise ValueError(f"value out of range in {item!r}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


class CronSchedule:
    """A parsed cron expression."""

    def __init__(self, expr: str) -> None:
        fields = expr.split()
        if len(fields) != 5:
            raise ValueError("a cron expression has exactly 5 fields")
        self.minutes = parse_field(fields[0], 0, 59)
        self.hours = parse_field(fields[1], 0, 23)
        self.days = parse_field(fields[2], 1, 31)
        self.months = parse_field(fields[3], 1, 12, MONTH_NAMES)
        self.weekdays = frozenset(d % 7 for d in parse_field(fields[4], 0, 7, DOW_NAMES))
        self.dom_restricted = not fields[2].startswith("*")
        self.dow_restricted = not fields[4].startswith("*")

    def day_matches(self, year: int, month: int, day: int) -> bool:
        dom_ok = day in self.days
        dow_ok = weekday(year, month, day) in self.weekdays
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def matches(self, when: Time) -> bool:
        """True if the schedule fires at minute ``when``."""
        year, month, day, hour, minute = when
        return (
            month in self.months
            and hour in self.hours
            and minute in self.minutes
            and self.day_matches(year, month, day)
        )

    def next_fire(self, after: Time) -> Time:
        """Earliest firing time strictly after ``after``.

        Raises ValueError if the schedule does not fire within MAX_SEARCH_YEARS years of ``after``.
        """
        year, month, day, hour, minute = after
        minute += 1
        last_year = year + MAX_SEARCH_YEARS
        while True:
            # Normalize carries left over from the previous step.
            if minute > 59:
                minute, hour = 0, hour + 1
            if hour > 23:
                hour, day = 0, day + 1
            if month > 12:
                month, year = 1, year + 1
            if day > days_in_month(year, month):
                day, month = 1, month + 1
                if month > 12:
                    month, year = 1, year + 1
            if year > last_year:
                raise ValueError("schedule does not fire within the search window")
            if month not in self.months:
                month, day, hour, minute = month + 1, 1, 0, 0
                continue
            if not self.day_matches(year, month, day):
                day, hour, minute = day + 1, 0, 0
                continue
            if hour not in self.hours:
                hour, minute = hour + 1, 0
                continue
            if minute not in self.minutes:
                minute += 1
                continue
            return (year, month, day, hour, minute)
'''

# Injected bugs: (exact text in CRON_SOURCE, buggy replacement, reviewer-facing description).
CRON_BUGS: dict[str, tuple[str, str, str]] = {
    "dow_or_scope": (
        "        if self.dom_restricted and self.dow_restricted:\n",
        "        if self.dom_restricted or self.dow_restricted:\n",
        "OR-semantics applied when only ONE day field is restricted (`0 9 * * MON` fires daily)",
    ),
    "range_step_anchor": (
        "        values.update(range(start, end + 1, step))\n",
        "        values.update(v for v in range(lo, end + 1, step) if v >= start)\n",
        "`A-B/S` and `N/S` steps anchored at the field minimum instead of at A/N",
    ),
    "leap_century": (
        "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n",
        "    return year % 4 == 0 and year % 100 != 0\n",
        "leap-year rule drops the 400-year exception (Feb 29, 2000 does not exist)",
    ),
    "sunday_seven": (
        "        self.weekdays = frozenset(d % 7 for d in parse_field(fields[4], 0, 7, DOW_NAMES))\n",
        "        self.weekdays = parse_field(fields[4], 0, 7, DOW_NAMES)\n",
        "day-of-week 7 is not normalized to Sunday (0)",
    ),
    "day_skip_minute_reset": (
        "                day, hour, minute = day + 1, 0, 0\n",
        "                day, hour = day + 1, 0\n",
        "skipping a non-matching day keeps the stale minute, so hour-0 slots on the next day are missed",
    ),
}

# Plausible-but-wrong "fixes" of correct code (red herrings) that the suite must catch.
CRON_RED_HERRING_EDITS: dict[str, tuple[str, str]] = {
    "dom_star_step_restricted": (
        '        self.dom_restricted = not fields[2].startswith("*")\n',
        '        self.dom_restricted = fields[2] != "*"\n',
    ),
    "dow_star_step_restricted": (
        '        self.dow_restricted = not fields[4].startswith("*")\n',
        '        self.dow_restricted = fields[4] != "*"\n',
    ),
    "n_slash_s_single_value": (
        "            end = hi if step_text else start\n",
        "            end = start\n",
    ),
    "python_weekday_monday_zero": (
        "    if month < 3:\n        year -= 1\n    return (year + year // 4 - year // 100 + year // 400 + offsets[month - 1] + day) % 7\n",
        "    import datetime\n\n    return datetime.date(year, month, day).weekday()\n",
    ),
}

DOW_LABELS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]


def buggy_source(bug_keys: tuple[str, ...]) -> str:
    return _apply(CRON_SOURCE, [(CRON_BUGS[k][0], CRON_BUGS[k][1]) for k in bug_keys])


def load_schedule_class(source: str) -> type:
    namespace: dict[str, Any] = {"__name__": "cronlite"}
    exec(compile(source, "cronlite.py", "exec"), namespace)  # noqa: S102 - trusted, generated source
    return namespace["CronSchedule"]


def _weekday(year: int, month: int, day: int) -> int:
    offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if month < 3:
        year -= 1
    return (year + year // 4 - year // 100 + year // 400 + offsets[month - 1] + day) % 7


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def _random_date(rng: random.Random, y_lo: int = 2019, y_hi: int = 2031) -> tuple[int, int, int]:
    year = rng.randint(y_lo, y_hi)
    month = rng.randint(1, 12)
    return year, month, rng.randint(1, _days_in_month(year, month))


def _date_with_weekday(rng: random.Random, dow: int) -> tuple[int, int, int]:
    while True:
        y, m, d = _random_date(rng)
        if _weekday(y, m, d) == dow and d <= 26:
            return y, m, d


def _run(cls: type, expr: str, after: tuple[int, ...]) -> tuple[int, ...] | str:
    try:
        return tuple(cls(expr).next_fire(after))
    except ValueError:
        return "ValueError"


# ---------------------------------------------------------------------------
# Bug-report scenarios (one generator per bug; each yields candidate (expr, after) pairs).
# Several shapes per bug: `make_repair_item` keeps only candidates whose symptom is caused by
# the reported bug alone, so every bug needs at least one shape that the others cannot mask.
# ---------------------------------------------------------------------------


def _scen_dow_or_scope(rng: random.Random) -> Iterator[tuple[str, tuple[int, ...]]]:
    while True:
        dow = rng.randint(1, 5)
        y, m, d = _date_with_weekday(rng, (dow + 2) % 7)
        hour = rng.choice([7, 8, 9, 10])
        if rng.random() < 0.5:
            yield f"{rng.choice([0, 15, 30])} {hour} * * {DOW_LABELS[dow]}", (y, m, d, hour + 2, 0)
        else:
            # Restricted day-of-month, unrestricted day-of-week: must fire on that date only.
            yield f"0 {hour} {rng.randint(1, 28)} * *", (y, m, d, hour + 2, 0)


def _scen_range_step_anchor(rng: random.Random) -> Iterator[tuple[str, tuple[int, ...]]]:
    while True:
        step = rng.choice([7, 9, 11, 13])
        start = rng.randint(2, step - 1)
        y, m, d = _random_date(rng)
        hour = rng.randint(6, 18)
        if rng.random() < 0.5:
            end = rng.randint(start + 2 * step, 59)
            yield f"{start}-{end}/{step} {hour} * * *", (y, m, d, hour - 1, 30)
        else:
            hstart = rng.randint(1, 4)
            yield f"0 {hstart}-23/{rng.choice([5, 6, 7])} * * *", (y, m, d, 0, 0)


def _scen_leap_century(rng: random.Random) -> Iterator[tuple[str, tuple[int, ...]]]:
    while True:
        hour, minute = rng.randint(1, 22), rng.choice([0, 10, 20, 30, 45])
        shape = rng.randrange(3)
        if shape == 0:
            yield (
                f"{minute} {hour} 29 2 *",
                (rng.choice([1997, 1998, 1999]), rng.randint(3, 12), 1, 0, 0),
            )
        elif shape == 1:
            yield f"{minute} {hour} * * *", (2000, 2, 28, hour, minute)
        else:
            yield f"{minute} {hour} */2 * *", (2000, 2, rng.choice([27, 28]), hour, minute)


def _scen_sunday_seven(rng: random.Random) -> Iterator[tuple[str, tuple[int, ...]]]:
    while True:
        hour = rng.randint(6, 20)
        shape = rng.randrange(3)
        if shape == 0:
            y, m, d = _date_with_weekday(rng, rng.choice([2, 3, 4]))
            yield f"0 {hour} * * 7", (y, m, d, 12, 0)
        elif shape == 1:
            # Both day fields restricted (OR semantics): Sunday must still fire before the 1st.
            y, m, d = _date_with_weekday(rng, rng.choice([2, 3, 4]))
            yield f"0 {hour} 1 * 7", (y, m, d, 12, 0)
        else:
            minute = rng.choice([0, 15, 30])
            y, m, d = _date_with_weekday(rng, 6)
            yield f"{minute} {hour} {rng.randint(1, 28)} * 5-7", (y, m, d, hour, minute + 5)


def _scen_day_skip_minute_reset(rng: random.Random) -> Iterator[tuple[str, tuple[int, ...]]]:
    while True:
        dow = rng.randint(1, 5)
        minute = rng.choice([5, 10, 15])
        y, m, d = _date_with_weekday(rng, (dow + 6) % 7)
        after = (y, m, d, 0, minute + rng.randint(10, 40))
        if rng.random() < 0.5:
            yield f"{minute} 0 * * {DOW_LABELS[dow]}", after
        else:
            # Both day fields restricted, so the skip happens regardless of the OR/AND rule.
            dom = rng.choice([v for v in range(1, 29) if v not in (d, d + 1)])
            yield f"{minute} 0 {dom} * {DOW_LABELS[dow]}", after


SCENARIOS: dict[str, Callable[[random.Random], Iterator[tuple[str, tuple[int, ...]]]]] = {
    "dow_or_scope": _scen_dow_or_scope,
    "range_step_anchor": _scen_range_step_anchor,
    "leap_century": _scen_leap_century,
    "sunday_seven": _scen_sunday_seven,
    "day_skip_minute_reset": _scen_day_skip_minute_reset,
}


# ---------------------------------------------------------------------------
# Hidden differential suite.
# ---------------------------------------------------------------------------


def _random_minute_field(rng: random.Random) -> str:
    kind = rng.randrange(6)
    if kind == 0:
        return f"*/{rng.choice([5, 10, 15, 20, 30])}"
    if kind == 1:
        return str(rng.randint(0, 59))
    if kind == 2:
        step = rng.choice([7, 9, 11, 13])
        start = rng.randint(1, 20)
        return f"{start}-{rng.randint(start + step, 59)}/{step}"
    if kind == 3:
        return f"{rng.randint(1, 25)}/{rng.choice([15, 20, 25])}"
    if kind == 4:
        return ",".join(str(v) for v in sorted(rng.sample(range(60), 3)))
    start = rng.randint(0, 50)
    return f"{start}-{rng.randint(start, 59)}"


def _random_hour_field(rng: random.Random) -> str:
    kind = rng.randrange(6)
    if kind == 0:
        return "*"
    if kind == 1:
        return str(rng.randint(0, 23))
    if kind == 2:
        return f"*/{rng.choice([4, 6, 8])}"
    if kind == 3:
        start = rng.randint(0, 6)
        return f"{start}-{rng.randint(start + 5, 23)}/{rng.choice([3, 5])}"
    if kind == 4:
        return ",".join(str(v) for v in sorted(rng.sample(range(24), 2)))
    return "0"


def _random_dom_field(rng: random.Random) -> str:
    return rng.choice(
        [
            "*",
            "*",
            "*/2",
            "*/10",
            str(rng.randint(1, 28)),
            f"{rng.randint(1, 10)}-{rng.randint(15, 31)}",
            f"{rng.randint(1, 14)},{rng.randint(15, 31)}",
            "31",
            "29",
        ]
    )


def _random_month_field(rng: random.Random) -> str:
    return rng.choice(["*", "*", "*", "*/3", "FEB", "jan-mar", "6-8", "DEC", "2,4,6", "Nov"])


def _random_dow_field(rng: random.Random) -> str:
    return rng.choice(
        [
            "*",
            "*",
            "*",
            "MON-FRI",
            "sun",
            "7",
            "5-7",
            "1,7",
            "SAT,SUN",
            "*/2",
            str(rng.randint(0, 6)),
            "0",
        ]
    )


def _random_after(rng: random.Random) -> tuple[int, int, int, int, int]:
    y, m, d = _random_date(rng, 1996, 2032)
    return (y, m, d, rng.randint(0, 23), rng.randint(0, 59))


def build_suite(rng: random.Random) -> list[tuple[str, tuple[int, ...]]]:
    """Targeted cases for every documented feature and every known trap, plus random expressions."""
    cases: list[tuple[str, tuple[int, ...]]] = []
    for gen in SCENARIOS.values():
        it = gen(rng)
        cases.extend(next(it) for _ in range(4))
    # Red-herring traps: `*/S` day fields are unrestricted; `N/S` runs to the field maximum.
    for _ in range(3):
        y, m, d = _random_date(rng)
        cases.append(
            (f"0 6 */2 * {rng.choice(['MON', 'TUE', 'WED', 'THU', 'FRI'])}", (y, m, d, 12, 0))
        )
        y, m, d = _random_date(rng)
        cases.append((f"30 7 {rng.randint(1, 28)} * */2", (y, m, d, 12, 0)))
        y, m, d = _random_date(rng)
        n, h = rng.randint(1, 9), rng.randint(1, 22)
        cases.append((f"{n}/20 {h} * * *", (y, m, d, h, n + rng.randint(1, 15))))
    # Both day fields restricted: OR semantics.
    for _ in range(3):
        y, m, d = _random_date(rng)
        cases.append(
            (
                f"0 12 {rng.randint(1, 28)} * {rng.choice(['MON', 'FRI', 'SUN', '7'])}",
                (y, m, d, 13, 0),
            )
        )
    # Impossible schedules raise ValueError.
    cases.append(("0 0 30 2 *", _random_after(rng)))
    cases.append(("15 10 31 4,6,9,11 *", _random_after(rng)))
    # Randomized expressions.
    while len(cases) < 80:
        expr = " ".join(
            [
                _random_minute_field(rng),
                _random_hour_field(rng),
                _random_dom_field(rng),
                _random_month_field(rng),
                _random_dow_field(rng),
            ]
        )
        cases.append((expr, _random_after(rng)))
    return cases


def _detections(cls: type, ref_results: list[Any], suite: list[tuple[str, tuple[int, ...]]]) -> int:
    return sum(
        1
        for (expr, after), ref in zip(suite, ref_results, strict=True)
        if _run(cls, expr, after) != ref
    )


def _describe(value: tuple[int, ...] | str) -> str:
    return "raised `ValueError`" if value == "ValueError" else f"returned `{tuple(value)!r}`"


def _report_case(
    rng: random.Random, bug: str, silent_bug: str, classes: dict[str, type]
) -> tuple[str, tuple[int, ...], tuple[int, ...] | str, tuple[int, ...] | str]:
    """Find a scenario whose symptom comes from `bug` alone and does not expose `silent_bug`."""
    for attempt, (expr, after) in enumerate(SCENARIOS[bug](rng)):
        if attempt >= MAX_SCENARIO_ATTEMPTS:
            raise AssertionError(f"no clean bug-report scenario found for {bug}")
        want = _run(classes["reference"], expr, after)
        if want == "ValueError":
            continue
        got = _run(classes["buggy"], expr, after)
        if got == want or got != _run(classes[bug], expr, after):
            continue
        if _run(classes[silent_bug], expr, after) != want:
            continue
        return expr, after, got, want
    raise AssertionError("unreachable")


def make_repair_item(
    rng: random.Random, item_id: str, hard_index: int, combo_offset: int = 0
) -> dict[str, Any]:
    """Build one `cronlite.py` repair item with 3 injected bugs (1 silent) and differential suite.

    Verifies at generation time that every injected bug and every red-herring "fix" fails at least
    `MIN_DETECTIONS` cases against the reference implementation.
    """
    combos = list(itertools.combinations(sorted(CRON_BUGS), 3))
    bug_keys = combos[(hard_index + combo_offset) % len(combos)]
    silent_bug = bug_keys[(hard_index + combo_offset) % 3]
    reported = [k for k in bug_keys if k != silent_bug]

    ref_cls = load_schedule_class(CRON_SOURCE)
    buggy_cls = load_schedule_class(buggy_source(bug_keys))
    classes: dict[str, type] = {"reference": ref_cls, "buggy": buggy_cls}
    classes.update({key: load_schedule_class(buggy_source((key,))) for key in CRON_BUGS})

    report_lines: list[str] = []
    report_cases: list[tuple[str, tuple[int, ...]]] = []
    for n, bug in enumerate(reported, start=1):
        expr, after, got, want = _report_case(rng, bug, silent_bug, classes)
        report_lines.append(
            f'{n}. `CronSchedule("{expr}").next_fire({after!r})` {_describe(got)}; expected `{tuple(want)!r}`.'
        )
        report_cases.append((expr, after))

    suite = report_cases + build_suite(rng)
    expected = [_run(ref_cls, expr, after) for expr, after in suite]
    if _detections(buggy_cls, expected, suite) < MIN_DETECTIONS:
        raise AssertionError("combined buggy module is not detected by the suite")
    for key in CRON_BUGS:
        if _detections(classes[key], expected, suite) < MIN_DETECTIONS:
            raise AssertionError(f"bug {key} is under-detected by the suite")
    for key, edit in CRON_RED_HERRING_EDITS.items():
        mutated = load_schedule_class(_apply(CRON_SOURCE, [edit]))
        if _detections(mutated, expected, suite) < 1:
            raise AssertionError(f"red-herring edit {key} is not detected by the suite")

    # `matches()` spot checks on firing times returned by the reference.
    match_cases: list[tuple[str, tuple[int, ...], bool]] = []
    for expr, _after in suite[:: max(1, len(suite) // 12)]:
        when = _run(ref_cls, expr, _after)
        if isinstance(when, str):
            continue
        when_tuple = tuple(int(x) for x in when)
        match_cases.append((expr, when_tuple, True))
        y, mo, d, h, mi = when_tuple
        shifted = (y, mo, d, h, (mi + 1) % 60)
        match_cases.append((expr, shifted, bool(ref_cls(expr).matches(shifted))))

    prompt = (
        "You maintain `cronlite.py`, a dependency-free cron evaluator used by our job scheduler "
        "(listed below). Users filed the bug report that follows. The report is not exhaustive: "
        "the module may contain other defects that nobody has reported yet.\n\n"
        "Fix the module so that it behaves exactly as its docstrings document. Do not change "
        "behavior that already matches the documentation, and keep the public API "
        "(`CronSchedule`, `CronSchedule.matches`, `CronSchedule.next_fire`, `CronSchedule.day_matches`, "
        "`parse_field`, `is_leap`, `days_in_month`, `weekday`) and the tuple-based time format unchanged.\n\n"
        "### Bug report\n" + "\n".join(report_lines) + "\n\n"
        "### cronlite.py\n```python\n" + buggy_source(bug_keys) + "```\n\n"
        "Return the complete corrected module in a single Python code block."
    )

    cases_literal = [(expr, after, exp) for (expr, after), exp in zip(suite, expected, strict=True)]
    test_code = f"""
_GDR_CASES = {cases_literal!r}
_GDR_MATCH_CASES = {match_cases!r}
_gdr_failures = []
for _gdr_expr, _gdr_after, _gdr_exp in _GDR_CASES:
    try:
        _gdr_got = tuple(CronSchedule(_gdr_expr).next_fire(tuple(_gdr_after)))
    except ValueError:
        _gdr_got = "ValueError"
    if _gdr_got != _gdr_exp:
        _gdr_failures.append(f"next_fire({{_gdr_expr!r}}, {{_gdr_after!r}}): expected {{_gdr_exp!r}}, got {{_gdr_got!r}}")
for _gdr_expr, _gdr_when, _gdr_exp in _GDR_MATCH_CASES:
    _gdr_got = bool(CronSchedule(_gdr_expr).matches(tuple(_gdr_when)))
    if _gdr_got != _gdr_exp:
        _gdr_failures.append(f"matches({{_gdr_expr!r}}, {{_gdr_when!r}}): expected {{_gdr_exp!r}}, got {{_gdr_got!r}}")
_gdr_total = len(_GDR_CASES) + len(_GDR_MATCH_CASES)
assert not _gdr_failures, f"{{len(_gdr_failures)}}/{{_gdr_total}} checks failed; first: " + "; ".join(_gdr_failures[:3])
"""
    return {
        "id": item_id,
        "slice": "s3_frontier_reasoning",
        "modality": "code",
        "subtype": "symptom_only_multibug_repair",
        "source_provenance": (
            "SWE-bench-style symptom-only repair of an original cron evaluator; 3 injected bugs "
            f"({', '.join(bug_keys)}), 1 unreported ({silent_bug}); hidden differential suite"
        ),
        "prompt": prompt,
        "attachments": [],
        "grader_type": "python_unit_tests",
        "grader_spec": {"test_code": test_code, "timeout_sec": 15.0},
    }
