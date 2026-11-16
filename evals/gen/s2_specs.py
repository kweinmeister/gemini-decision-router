"""Single-function coding specs for `s2_everyday_code`, graded by hidden differential suites."""

from __future__ import annotations

import contextlib
import io
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evals.gen._edits import apply_edits as _apply

MIN_DETECTIONS = 2
TIMEOUT_SEC = 10.0
SUITE_SIZE = 64
TOPUP_ATTEMPTS = 4000
# Printed as the last line of a passing hidden suite and checked by `_test_failure` at build time
# (at runtime, `evals/sandbox.py` appends its own per-run cryptographic sentinel).
_SUITE_OK_SENTINEL = "GDR_S2_HIDDEN_SUITE_OK"

SUBTYPES: dict[int, str] = {
    0: "rfc_quoted_row_tokenizer",
    1: "sliding_window_alert_deduplicator",
    2: "nested_macro_ast_validator",
    3: "span_capped_interval_merger",
    4: "semver2_prerelease_resolver",
}
FUNCTION_NAMES: dict[int, str] = {
    0: "parse_escaped_row",
    1: "deduplicate_alerts",
    2: "validate_macro_template",
    3: "merge_capped_intervals",
    4: "select_highest_semver",
}
PROVENANCE = (
    "evalplus/mbppplus & bigcode/bigcodebench-style parameterized function spec; "
    "hidden differential suite from a reference implementation"
)

# ---------------------------------------------------------------------------
# Reference implementations (templates) and mutants (exact-once edits of the template).
# ---------------------------------------------------------------------------

CSV_REFERENCE = """DELIM = __DELIM__
SENTINEL = __SENTINEL__


def parse_escaped_row(line: str) -> list[str]:
    if line == "":
        return []
    fields = []
    n = len(line)
    i = 0
    while True:
        j = i
        while j < n and line[j] == " ":
            j += 1
        if j < n and line[j] == '"':
            j += 1
            buf = []
            while True:
                if line[j] == '"':
                    if j + 1 < n and line[j + 1] == '"':
                        buf.append('"')
                        j += 2
                        continue
                    j += 1
                    break
                buf.append(line[j])
                j += 1
            value = "".join(buf) or SENTINEL
            while j < n and line[j] == " ":
                j += 1
        else:
            j = line.find(DELIM, i)
            if j == -1:
                j = n
            value = line[i:j].strip(" ") or SENTINEL
        fields.append(value)
        if j >= n:
            return fields
        i = j + 1
"""

ALERT_REFERENCE = """WINDOW = __WINDOW__
MAX_BURST = __MAX_BURST__
PENALTY = __PENALTY__


def deduplicate_alerts(events: list[tuple[int, str]]) -> list[tuple[int, str, int]]:
    emitted: dict[str, list[int]] = {}
    last_suppressed: dict[str, int] = {}
    pending: dict[str, int] = {}
    out = []
    for t, key in events:
        times = emitted.setdefault(key, [])
        in_window = sum(1 for e in times if t - WINDOW <= e <= t)
        last = last_suppressed.get(key)
        burst = in_window >= MAX_BURST
        cooling = last is not None and t - last <= PENALTY
        if burst or cooling:
            last_suppressed[key] = t
            pending[key] = pending.get(key, 0) + 1
        else:
            out.append((t, key, pending.get(key, 0)))
            pending[key] = 0
            times.append(t)
    return out
"""

MACRO_REFERENCE = """import re

MAX_DEPTH = __MAX_DEPTH__
_VALID_TAG = re.compile(r"\\[\\[(open|close):([a-z]{1,__MAX_NAME_LEN__})\\]\\]")


def validate_macro_template(text: str) -> int:
    stack = []
    prev_close = None
    pos = 0
    while True:
        start = text.find("[[", pos)
        if start == -1:
            break
        close_at = text.find("]]", start + 2)
        if close_at == -1:
            return start  # unterminated tag
        end = close_at + 2
        m = _VALID_TAG.fullmatch(text[start:end])
        if m is None:
            return start
        kind, name = m.group(1), m.group(2)
        if kind == "close":
            if not stack or stack[-1][0] != name:
                return start
            stack.pop()
            prev_close = (name, end)
        else:
            if len(stack) >= MAX_DEPTH:
                return start
            if prev_close == (name, start):
                return start
            stack.append((name, start))
            prev_close = None
        pos = end
    if stack:
        return stack[-1][1]
    return -1
"""

INTERVAL_REFERENCE = """GRACE = __GRACE__
CAP_SPAN = __CAP_SPAN__


def merge_capped_intervals(intervals: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    result = []
    cur = None
    for nxt_s, nxt_e, nxt_w in sorted(intervals, key=lambda iv: (iv[0], iv[1])):
        if cur is None:
            cur = (nxt_s, nxt_e, nxt_w)
            continue
        cur_s, cur_e, cur_w = cur
        merged_e = max(cur_e, nxt_e)
        if nxt_s - cur_e <= GRACE and merged_e - cur_s <= CAP_SPAN:
            cur = (cur_s, merged_e, cur_w + nxt_w)
        else:
            result.append(cur)
            cur = (nxt_s, nxt_e, nxt_w)
    if cur is not None:
        result.append(cur)
    return result
"""

SEMVER_REFERENCE = """def _precedence_key(version: str) -> tuple:
    core, _, pre = version.partition("-")
    major, minor, patch = (int(x) for x in core.split("."))
    if not pre:
        return (major, minor, patch, 1, ())
    ids = []
    for ident in pre.split("."):
        if ident.isdigit():
            ids.append((0, int(ident), ""))
        else:
            ids.append((1, 0, ident))
    return (major, minor, patch, 0, tuple(ids))


def select_highest_semver(versions: list[str], major_req: int = __DEFAULT_MAJOR__) -> str | None:
    best = None
    best_key = None
    for v in versions:
        key = _precedence_key(v)
        if key[0] != major_req:
            continue
        if best_key is None or key > best_key:
            best, best_key = v, key
    return best
"""

REFERENCES: dict[int, str] = {
    0: CSV_REFERENCE,
    1: ALERT_REFERENCE,
    2: MACRO_REFERENCE,
    3: INTERVAL_REFERENCE,
    4: SEMVER_REFERENCE,
}

MUTANTS: dict[int, dict[str, tuple[str, str]]] = {
    0: {
        "strip_tabs_too": (
            'value = line[i:j].strip(" ") or SENTINEL',
            "value = line[i:j].strip() or SENTINEL",
        ),
        "drop_trailing_empty_field": (
            "        i = j + 1\n",
            "        i = j + 1\n        if i >= n:\n            return fields\n",
        ),
        "no_doubled_quote_unescape": ("buf.append('\"')", "buf.append('\"\"')"),
        "blank_line_returns_empty_list": (
            '    if line == "":\n',
            '    if line.strip(" ") == "":\n',
        ),
        "strip_quoted_content": (
            'value = "".join(buf) or SENTINEL',
            'value = "".join(buf).strip(" ") or SENTINEL',
        ),
        "quoted_empty_not_sentinel": ('value = "".join(buf) or SENTINEL', 'value = "".join(buf)'),
    },
    1: {
        "exclusive_window_start": ("t - WINDOW <= e <= t", "t - WINDOW < e <= t"),
        "count_suppressed_in_window": (
            "            last_suppressed[key] = t\n",
            "            last_suppressed[key] = t\n            times.append(t)\n",
        ),
        "penalty_refreshed_only_by_burst": (
            "            last_suppressed[key] = t\n",
            "            if burst:\n                last_suppressed[key] = t\n",
        ),
        "strict_penalty_boundary": ("t - last <= PENALTY", "t - last < PENALTY"),
        "burst_off_by_one": ("in_window >= MAX_BURST", "in_window > MAX_BURST"),
        "pending_not_reset": ("            pending[key] = 0\n", ""),
    },
    2: {
        "depth_off_by_one": ("len(stack) >= MAX_DEPTH", "len(stack) > MAX_DEPTH"),
        "adjacency_ignores_text": (
            "if prev_close == (name, start):",
            "if prev_close is not None and prev_close[0] == name:",
        ),
        "adjacency_across_levels": (
            "            if prev_close == (name, start):\n                return start\n"
            "            stack.append((name, start))\n            prev_close = None\n",
            "            if prev_close is not None and prev_close[0] == name:\n                return start\n"
            "            stack.append((name, start))\n",
        ),
        "outermost_unclosed": ("return stack[-1][1]", "return stack[0][1]"),
        "tag_starts_at_last_bracket": (
            "        end = close_at + 2\n",
            '        end = close_at + 2\n        start = text.rfind("[[", start, close_at)\n',
        ),
        "unterminated_tag_ignored": ("return start  # unterminated tag", "break"),
        "name_length_unbounded": ("([a-z]{1,__MAX_NAME_LEN__})", "([a-z]+)"),
    },
    3: {
        "cap_measured_from_nxt_s": ("merged_e - cur_s <= CAP_SPAN", "merged_e - nxt_s <= CAP_SPAN"),
        "cap_ignores_cur_e": ("merged_e - cur_s <= CAP_SPAN", "nxt_e - cur_s <= CAP_SPAN"),
        "strict_gap": ("nxt_s - cur_e <= GRACE", "nxt_s - cur_e < GRACE"),
        "strict_cap": ("merged_e - cur_s <= CAP_SPAN", "merged_e - cur_s < CAP_SPAN"),
        "end_from_next_interval": (
            "cur = (cur_s, merged_e, cur_w + nxt_w)",
            "cur = (cur_s, nxt_e, cur_w + nxt_w)",
        ),
        "sort_by_start_only": ("key=lambda iv: (iv[0], iv[1])", "key=lambda iv: iv[0]"),
    },
    4: {
        "string_compare_minor_patch": (
            '    major, minor, patch = (int(x) for x in core.split("."))\n',
            '    major, minor, patch = core.split(".")\n    major = int(major)\n',
        ),
        "prefix_filter_major": ("if key[0] != major_req:", "if not v.startswith(str(major_req)):"),
        "release_below_prerelease": (
            "return (major, minor, patch, 1, ())",
            "return (major, minor, patch, 0, ())",
        ),
        "numeric_above_alphanumeric": (
            'ids.append((0, int(ident), ""))',
            'ids.append((2, int(ident), ""))',
        ),
        "numeric_identifiers_as_text": (
            'ids.append((0, int(ident), ""))',
            "ids.append((0, 0, ident))",
        ),
        "shorter_prerelease_wins": (
            "return (major, minor, patch, 0, tuple(ids))",
            'return (major, minor, patch, 0, tuple(ids) + ((2, 0, ""),))',
        ),
        "default_major_off_by_one": (
            "major_req: int = __DEFAULT_MAJOR__)",
            "major_req: int = __DEFAULT_MAJOR__ + 1)",
        ),
    },
}

# Mutants that are provably equivalent to the reference under some parameters: if no suppressed
# event can lie in `[t - W, t]` without also triggering the cooldown (W <= P), counting suppressed
# events in the burst window changes nothing.
EQUIVALENT_WHEN: dict[int, dict[str, Callable[[dict[str, Any]], bool]]] = {
    1: {"count_suppressed_in_window": lambda p: p["window_sec"] <= p["cooldown_penalty"]},
}

_PLACEHOLDERS: dict[int, dict[str, str]] = {
    0: {"__DELIM__": "delim", "__SENTINEL__": "empty_sentinel"},
    1: {
        "__WINDOW__": "window_sec",
        "__MAX_BURST__": "max_burst",
        "__PENALTY__": "cooldown_penalty",
    },
    2: {"__MAX_DEPTH__": "max_depth", "__MAX_NAME_LEN__": "max_name_len"},
    3: {"__GRACE__": "grace", "__CAP_SPAN__": "cap_span"},
    4: {"__DEFAULT_MAJOR__": "default_major"},
}


def render_source(variant: int, params: dict[str, Any], mutant: str | None = None) -> str:
    """Self-contained source of the reference (or one mutant) with the item's parameters filled in."""
    source = REFERENCES[variant]
    if mutant is not None:
        source = _apply(source, [MUTANTS[variant][mutant]])
    for placeholder, key in _PLACEHOLDERS[variant].items():
        source = source.replace(placeholder, repr(params[key]))
    return source


def load_function(source: str, name: str) -> Callable[..., Any]:
    namespace: dict[str, Any] = {"__name__": "s2_reference"}
    exec(compile(source, "s2_reference.py", "exec"), namespace)  # noqa: S102 - trusted, generated source
    return namespace[name]


# ---------------------------------------------------------------------------
# Hidden-suite harness (runs in the candidate's namespace, so every name is `_gdr_`-prefixed).
# ---------------------------------------------------------------------------

_CALL_EXPR: dict[int, str] = {
    0: "parse_escaped_row(_gdr_args)",
    1: "deduplicate_alerts(list(_gdr_args))",
    2: "validate_macro_template(_gdr_args)",
    3: "merge_capped_intervals(list(_gdr_args))",
    4: (
        "select_highest_semver(list(_gdr_args[0])) if _gdr_args[1] is None "
        "else select_highest_semver(list(_gdr_args[0]), _gdr_args[1])"
    ),
}


def _call(variant: int, fn: Callable[..., Any], args: Any) -> Any:
    """Generation-time mirror of `_CALL_EXPR`; exceptions become a (never expected) marker string."""
    try:
        if variant in (1, 3):
            return fn(list(args))
        if variant == 4:
            return fn(list(args[0])) if args[1] is None else fn(list(args[0]), args[1])
        return fn(args)
    except Exception as exc:  # noqa: BLE001 - mutants may crash; that counts as a detection
        return f"raised {type(exc).__name__}: {exc}"


def _harness(variant: int, cases: list[tuple[Any, Any]]) -> str:
    name = FUNCTION_NAMES[variant]
    return f"""
_GDR_CASES = {cases!r}
_gdr_failures = []
for _gdr_i, (_gdr_args, _gdr_exp) in enumerate(_GDR_CASES):
    try:
        _gdr_got = {_CALL_EXPR[variant]}
    except Exception as _gdr_exc:
        _gdr_got = f"raised {{type(_gdr_exc).__name__}}: {{_gdr_exc}}"
    if _gdr_got != _gdr_exp:
        _gdr_failures.append(f"case {{_gdr_i}} input={{_gdr_args!r:.160}}: expected {{_gdr_exp!r:.120}}, got {{_gdr_got!r:.120}}")
assert not _gdr_failures, f"{name}: {{len(_gdr_failures)}}/{{len(_GDR_CASES)}} hidden cases failed; first: " + " | ".join(_gdr_failures[:3])
print("{_SUITE_OK_SENTINEL}", len(_GDR_CASES))
"""


# ---------------------------------------------------------------------------
# Variant 0: quoted delimited-row tokenizer.
# ---------------------------------------------------------------------------

_CSV_PLAIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/'|#"


def _params_v0(rng: random.Random, i: int) -> dict[str, Any]:
    delim = ";" if (i // 5) % 2 == 1 else ","
    return {"delim": delim, "empty_sentinel": f"NULL_{rng.randint(100, 9999)}"}


def _prompt_v0(p: dict[str, Any], ref: Callable[..., Any]) -> str:
    d, s = p["delim"], p["empty_sentinel"]
    other = "," if d == ";" else ";"
    example = f' a {d}"b{d}""c"" " {d} \t{other}{d}'
    return (
        f"Write a Python function `parse_escaped_row(line: str) -> list[str]` that splits one row of "
        f"delimited text into its field values. The delimiter is `'{d}'`.\n\n"
        f"The input is always well-formed and never contains a newline:\n"
        f"- Fields are separated by `'{d}'`. Apart from the empty line (rule 4), a row with k delimiters "
        f"outside quotes has exactly k + 1 fields, so a delimiter at the very end of the line is followed "
        f"by one final empty field.\n"
        f'- A quoted field is: optional leading spaces, an opening `"`, the content, a closing `"`, '
        f"optional trailing spaces, then the next delimiter or the end of the line. Inside the content, "
        f'each pair of double quotes `""` stands for one literal `"`; every other character (including '
        f"`'{d}'`, spaces and tabs) is literal content.\n"
        f"- Every other field is an unquoted field, which never contains `\"` or `'{d}'`.\n\n"
        f"Return the list of field values, in order:\n"
        f"1. Unquoted field: its text with leading and trailing space characters (`' '`) removed. Only "
        f"spaces are removed; tabs and every other character, including inner spaces, are kept.\n"
        f'2. Quoted field: its content with each `""` replaced by `"`. Spaces and delimiters inside the '
        f"quotes are kept exactly; the spaces outside the quotes are discarded.\n"
        f"3. A field whose value is the empty string (an empty or all-space unquoted field, or the quoted "
        f'field `""`) is returned as `\'{s}\'` instead. Non-empty values, such as a quoted `"  "` '
        f"(two spaces) or an unquoted tab, are kept as they are.\n"
        f"4. If `line` is the empty string, return `[]`. A non-empty line made only of spaces is a single "
        f"empty unquoted field, so it returns `['{s}']`.\n\n"
        f"Example: `parse_escaped_row({example!r})` returns `{ref(example)!r}`.\n\n"
        f"Return only valid Python code defining `parse_escaped_row`."
    )


def _handwritten_v0(p: dict[str, Any]) -> str:
    delim, empty_sentinel = p["delim"], p["empty_sentinel"]
    return f"""
assert parse_escaped_row("") == []
assert parse_escaped_row(" alpha {delim}  {delim} \\" beta {delim} \\"\\"quoted\\"\\" \\" {delim} \\"\\" ") == [
    "alpha", "{empty_sentinel}", ' beta {delim} "quoted" ', "{empty_sentinel}"
]
assert parse_escaped_row("x{delim}y{delim}") == ["x", "y", "{empty_sentinel}"]
"""


def _csv_row(crng: random.Random, p: dict[str, Any]) -> tuple[str, list[str]]:
    """A random well-formed row and the field values it encodes (an independent oracle)."""
    d, s = p["delim"], p["empty_sentinel"]
    plain = _CSV_PLAIN + ("," if d == ";" else ";")
    while True:
        parts: list[str] = []
        values: list[str] = []
        for _ in range(crng.randint(1, 6)):
            pad_l = " " * crng.choice([0, 0, 1, 2, 3])
            pad_r = " " * crng.choice([0, 0, 1, 2])
            roll = crng.random()
            if crng.random() < 0.45:
                if roll < 0.12:
                    content = ""
                elif roll < 0.22:
                    content = " " * crng.randint(1, 3)
                else:
                    pool = plain + d + d + '"' + '"' + "  \t"
                    content = "".join(crng.choice(pool) for _ in range(crng.randint(1, 8)))
                parts.append(pad_l + '"' + content.replace('"', '""') + '"' + pad_r)
                values.append(content or s)
            else:
                if roll < 0.15:
                    core = ""
                else:
                    edge = plain + ("\t\t\t" if roll < 0.4 else "")
                    chars = [crng.choice(plain + "  \t") for _ in range(crng.randint(1, 7))]
                    chars[0], chars[-1] = crng.choice(edge), crng.choice(edge)
                    core = "".join(chars)
                parts.append(pad_l + core + pad_r)
                values.append(core or s)
        line = d.join(parts)
        if line:
            return line, values


def _cases_v0(crng: random.Random, p: dict[str, Any], ref: Callable[..., Any]) -> list[Any]:
    d = p["delim"]
    other = "," if d == ";" else ";"
    cases: list[Any] = [
        "",
        " ",
        "   ",
        "      ",
        d,
        d + d,
        "a" + d,
        ' "x" ' + d,
        f"{d} {d}\t{d}",
        "\tab\t",
        " \t ",
        f"a{d}\t{d} b\t ",
        "\t",
        f"  x\t {d}\ty",
        '"a""b"',
        '""""',
        ' """x""" ',
        f'"{d}""{d}"',
        '"  "',
        f' " a " {d}b',
        f'"a "  {d}  " b"',
        f'" {d} " {d} "\t"',
        '""',
        f' "" {d}""',
        f'x{d}"" {d}',
        f'"a{d}b"{d}c',
        f"a{other}b",
        "  a  b  ",
        f' alpha {d}  {d} " beta {d} ""quoted"" " {d} "" ',
        f"x{d}y{d}",
    ]
    for _ in range(SUITE_SIZE - len(cases)):
        line, values = _csv_row(crng, p)
        if ref(line) != values:
            raise AssertionError(f"CSV reference disagrees with the row generator on {line!r}")
        cases.append(line)
    return cases


def _random_v0(crng: random.Random, p: dict[str, Any]) -> Any:
    return _csv_row(crng, p)[0]


# ---------------------------------------------------------------------------
# Variant 1: sliding-window alert deduplicator with a refreshed cooldown.
# ---------------------------------------------------------------------------


def _params_v1(rng: random.Random, i: int) -> dict[str, Any]:
    window_sec = rng.randint(10, 45)
    max_burst = rng.randint(2, 3)
    cooldown_penalty = rng.randint(5, 15)
    return {"window_sec": window_sec, "max_burst": max_burst, "cooldown_penalty": cooldown_penalty}


def _prompt_v1(p: dict[str, Any], ref: Callable[..., Any]) -> str:
    w, b, pen = p["window_sec"], p["max_burst"], p["cooldown_penalty"]
    return (
        f"Write a Python function `deduplicate_alerts(events: list[tuple[int, str]]) -> list[tuple[int, str, int]]`.\n"
        f"Each element of `events` is `(timestamp, key)` with an int `timestamp`; the list is sorted by "
        f"non-decreasing timestamp, and a timestamp may repeat (also for the same key).\n\n"
        f"Process the events one at a time, in list order. Every key has its own independent state "
        f"(events of one key never affect another key):\n"
        f"- the timestamps of the alerts EMITTED so far for that key,\n"
        f"- `last_suppressed`: the timestamp of the most recent SUPPRESSED event of that key (initially none),\n"
        f"- `pending`: the number of events of that key suppressed since its last emitted alert (initially 0).\n\n"
        f"An event `(t, key)` is SUPPRESSED if at least one of these holds:\n"
        f"  a) at least {b} alerts of `key` were already emitted with timestamps in the inclusive range "
        f"`[t - {w}, t]` (count only emitted alerts, never suppressed events), or\n"
        f"  b) `key` has a previous suppressed event and `t - last_suppressed <= {pen}`.\n"
        f"On every suppression, whichever condition caused it, set that key's `last_suppressed` to `t` and "
        f"add 1 to its `pending`. Suppressed events produce no output.\n"
        f"Otherwise the event is EMITTED: append `(t, key, pending)` to the output, record `t` as an emitted "
        f"timestamp of that key, and reset that key's `pending` to 0.\n\n"
        f"Return the emitted tuples in the order they were emitted (`[]` for an empty input).\n"
        f"Return only valid Python code defining `deduplicate_alerts`."
    )


def _handwritten_v1(p: dict[str, Any]) -> str:
    window_sec, max_burst, cooldown_penalty = p["window_sec"], p["max_burst"], p["cooldown_penalty"]
    return f"""
ev = [
    (0, "db"),
    (1, "db"),
    (2, "db"),
    (3, "db"),
    (3 + {cooldown_penalty}, "db"),
    (50 + {window_sec} + {cooldown_penalty}, "db"),
]
res = deduplicate_alerts(ev)
if {max_burst} == 2:
    assert res[0] == (0, "db", 0)
    assert res[1] == (1, "db", 0)
    assert res[2] == (50 + {window_sec} + {cooldown_penalty}, "db", 3)
else:
    assert res[0] == (0, "db", 0)
    assert res[1] == (1, "db", 0)
    assert res[2] == (2, "db", 0)
    assert res[3] == (50 + {window_sec} + {cooldown_penalty}, "db", 2)
"""


_ALERT_KEYS = ["db", "api", "cache", "auth", "queue", "disk"]


def _alert_burst_then(
    base: int, key: str, p: dict[str, Any], offsets: list[int]
) -> list[tuple[int, str]]:
    """`max_burst` emitted alerts at base, base+1, ...; then events at base + each offset."""
    return [(base + k, key) for k in range(p["max_burst"])] + [(base + off, key) for off in offsets]


def _cases_v1(crng: random.Random, p: dict[str, Any], ref: Callable[..., Any]) -> list[Any]:
    w, b, pen = p["window_sec"], p["max_burst"], p["cooldown_penalty"]
    far = 2 * w + 2 * pen + 10
    cases: list[Any] = [
        [],
        [(7, "db")],
        [(0, "db"), (1, "db"), (2, "db"), (3, "db"), (3 + pen, "db"), (50 + w + pen, "db")],
        [(5, "api")] * (b + 2),
    ]
    for n, key in enumerate(crng.sample(_ALERT_KEYS, 3)):
        base = crng.randint(0, 30) + 100 * n
        cases.append(_alert_burst_then(base, key, p, [w, far]))  # window start is inclusive
        cases.append(
            _alert_burst_then(base, key, p, [w, w + pen, far])
        )  # cooldown boundary is inclusive
        cases.append(
            _alert_burst_then(base, key, p, [w, w + pen, w + 2 * pen, far])
        )  # refresh on every suppression
        cases.append(_alert_burst_then(base, key, p, [b, far, 2 * far]))  # `at least` max_burst
        cases.append(_alert_burst_then(base, key, p, [w + 1, w + 1, w + pen + 2, far]))
    # Two keys interleaved: one key's suppressions must not affect the other.
    k1, k2 = crng.sample(_ALERT_KEYS, 2)
    cases.append(
        sorted(
            _alert_burst_then(0, k1, p, [w, w + pen])
            + [(t, k2) for t in (1, w, w + pen, w + 2 * pen)]
        )
    )
    while len(cases) < SUITE_SIZE:
        cases.append(_random_v1(crng, p))
    return cases


def _random_v1(crng: random.Random, p: dict[str, Any]) -> Any:
    w, pen = p["window_sec"], p["cooldown_penalty"]
    keys = crng.sample(_ALERT_KEYS, crng.choice([1, 1, 2, 2, 3]))
    last: dict[str, int] = {}
    t = crng.randint(0, 5)
    events: list[tuple[int, str]] = []
    for _ in range(crng.randint(4, 18)):
        key = crng.choice(keys)
        roll = crng.random()
        if roll < 0.15:
            step = 0
        elif roll < 0.45:
            step = crng.randint(1, 3)
        elif roll < 0.62:
            step = pen + crng.choice([-1, 0, 0, 1])
        elif roll < 0.8:
            step = w + crng.choice([-2, -1, 0, 0, 1])
        elif roll < 0.9:
            step = crng.randint(4, max(5, w // 2))
        else:
            step = w + pen + crng.randint(1, 10)
        t = max(t, last.get(key, t) + step) if crng.random() < 0.7 else t + step
        last[key] = t
        events.append((t, key))
    return events


# ---------------------------------------------------------------------------
# Variant 2: nested macro-tag validator.
# ---------------------------------------------------------------------------

_MACRO_NAMES = ["a", "b", "x", "sec", "item", "loop", "abcdefghijkl"]
_MACRO_TEXT = [
    "",
    "",
    "",
    " ",
    "hi",
    "a b",
    "]",
    "]]",
    "x]",
    ": ",
    "open",
    "[x]",
    "]]y",
    "[ y",
    "z[",
]
_MACRO_BAD = [
    "[[open:]]",
    "[[Open:a]]",
    "[[open:a1]]",
    "[[open: a]]",
    "[[open:a ]]",
    "[[]]",
    "[[close:]]",
    "[[open:a:b]]",
    "[[opened:a]]",
    "[[ open:a]]",
    "[[open:abcdefghijklm]]",
    "[[close:B]]",
    "[[open:x-y]]",
    "[[open:[[close:a]]",
    "[[[open:a]]",
    "[[close:a]",
]


def _params_v2(rng: random.Random, i: int) -> dict[str, Any]:
    return {
        "max_depth": rng.randint(2, 6),
        "max_name_len": rng.randint(6, 16),
        "ex_tag": rng.choice(["sec", "blk", "item", "row", "node", "span", "cell", "part"]),
        "ex_id": rng.randint(10, 99),
    }


def _prompt_v2(p: dict[str, Any], ref: Callable[..., Any]) -> str:
    d, mlen = p["max_depth"], p["max_name_len"]
    ex_text = f"hdr#{p['ex_id']} [[open:{p['ex_tag']}]]ok[[close:{p['ex_tag']}]]"
    return (
        f"Write a Python function `validate_macro_template(text: str) -> int` that validates nested macro "
        f"blocks written as `[[open:NAME]] ... [[close:NAME]]`.\n\n"
        f"Tags: scan `text` from left to right. The next tag starts at the next occurrence of `[[` and ends with "
        f"the first `]]` after it; scanning then resumes right after that `]]`. If no `]]` follows, the tag is "
        f"unterminated, which makes it malformed. Everything outside tags is plain text and is ignored, "
        f"including single `[` or `]` and stray `]]`. A `[[` inside a tag does not start a new tag: "
        f"`[[open:[[close:a]]` is one single (malformed) tag.\n"
        f"A tag is valid only if it is exactly `[[open:NAME]]` or `[[close:NAME]]`, where `NAME` is 1 to {mlen} "
        f"lowercase ASCII letters (`a`-`z`). Any other tag is malformed, e.g. `[[open:]]`, `[[Open:a]]`, "
        f"`[[open:a1]]`, `[[open: a]]` or `[[[open:a]]`.\n\n"
        f"Process the tags in order with a stack of open blocks. The offending tag is the first tag for which "
        f"one of these holds:\n"
        f"1. It is malformed.\n"
        f"2. It is `[[close:NAME]]` but no block is open or the innermost open block has a different name "
        f"(blocks close in strict LIFO order).\n"
        f"3. It is `[[open:NAME]]` but {d} blocks are already open (the nesting depth may never exceed {d}).\n"
        f"4. It is `[[open:X]]` and it starts exactly at the index where a `[[close:X]]` with the same name "
        f"ended (a directly adjacent identical sibling at the same nesting level, like the second open in "
        f"`[[open:x]]..[[close:x]][[open:x]]`). If any character separates the close from the open, they are "
        f"not adjacent.\n\n"
        f"Return the 0-based index of the first character of the offending tag. If there is no offending tag "
        f"but blocks are still open at the end of `text`, return the index of the innermost (most recently "
        f"opened) unclosed `[[open:NAME]]`. Otherwise return `-1`.\n"
        f"Example: `validate_macro_template({ex_text!r})` returns `{ref(ex_text)!r}`.\n"
        f"Return only valid Python code defining `validate_macro_template`."
    )


def _handwritten_v2(p: dict[str, Any]) -> str:
    max_depth = p["max_depth"]
    return f"""
assert validate_macro_template("hello [[open:sec]]world[[close:sec]]!") == -1
assert validate_macro_template("[[open:a]][[close:b]]") == 10
assert validate_macro_template("[[open:a]]unclosed") == 0
assert validate_macro_template("[[open:a]]ok[[close:a]][[open:a]]dup[[close:a]]") == 23
assert validate_macro_template("bad [[open:123]] tag") == 4
nested = "".join("[[open:x]]" for _ in range({max_depth} + 1)) + "".join("[[close:x]]" for _ in range({max_depth} + 1))
assert validate_macro_template(nested) == 10 * {max_depth}
"""


def _nest(names: list[str], inner: str = "", sep: str = "") -> str:
    return (
        sep.join(f"[[open:{n}]]" for n in names)
        + inner
        + sep.join(f"[[close:{n}]]" for n in reversed(names))
    )


def _cases_v2(crng: random.Random, p: dict[str, Any], ref: Callable[..., Any]) -> list[Any]:
    d = p["max_depth"]
    mlen = p["max_name_len"]
    exact_name = ("abcdefghijklmnopqrstuvwxyz" * 2)[:mlen]
    over_1 = ("abcdefghijklmnopqrstuvwxyz" * 2)[: mlen + 1]
    over_2 = ("abcdefghijklmnopqrstuvwxyz" * 2)[: mlen + 2]
    over_5 = ("abcdefghijklmnopqrstuvwxyz" * 2)[: mlen + 5]
    deep = [crng.choice(_MACRO_NAMES) for _ in range(d + 1)]
    cases: list[Any] = [
        "hello [[open:sec]]world[[close:sec]]!",
        "[[open:a]][[close:b]]",
        "[[open:a]]unclosed",
        "[[open:a]]ok[[close:a]][[open:a]]dup[[close:a]]",
        "bad [[open:123]] tag",
        "",
        "plain text only",
        "a ] b [ c ]] d [open:a]",
        # depth limit: exactly max_depth is fine, one more is an error
        _nest(deep[:d], "core"),
        _nest(deep, "core", " "),
        "t" + _nest(deep) + "[[close:zz]]",
        _nest(["x"] * (d + 1)),
        # adjacency: only a directly adjacent same-name sibling is an error
        "[[open:a]][[close:a]] [[open:a]][[close:a]]",
        "x[[open:b]]y[[close:b]]z[[open:b]]w[[close:b]]",
        "[[open:item]][[close:item]]][[open:item]][[close:item]]",
        "[[open:loop]][[close:loop]][[open:loop]][[close:loop]]",
        "[[open:a]][[close:a]][[open:b]][[close:b]]",
        "[[open:a]][[open:x]][[close:x]][[close:a]][[open:x]][[close:x]]",
        "[[open:x]][[close:x]][[open:a]][[open:x]][[close:x]][[close:a]]",
        "[[open:p]][[open:x]][[close:x]][[open:x]][[close:x]][[close:p]]",
        # unclosed blocks: report the innermost one
        "[[open:a]][[open:b]]text",
        "zz[[open:a]][[open:b]][[close:b]] [[open:c]]",
        "[[open:a]] [[open:b]][[close:b]]",
        "[[open:sec]]  [[open:item]]",
        # tag boundaries: a tag runs from `[[` to the first following `]]`
        "[[open:[[close:a]]",
        "[[[open:a]][[close:a]]",
        "q[[open:a]][[[close:a]]",
        "[[open:a]]]][[close:a]]",
        "[[open:a]][[close:a]]]",
        "[[open:a]][[close:a]][[open:b",
        "[[open:a]] [[close:a]",
        "[[open:a]][[",
        # names: 1 to max_name_len lowercase letters
        f"[[open:{exact_name}]][[close:{exact_name}]]",
        f"[[open:{over_1}]][[close:{over_1}]]",
        f"ok [[open:a]][[open:{over_2}]][[close:{over_2}]][[close:a]]",
        f"x [[open:{over_5}]]y[[close:{over_5}]]",
        "[[open:]]",
        "[[Open:a]][[close:a]]",
        "[[open:a1]]",
        "[[open: a]]",
        "[[]]",
        "[[open:a:b]]",
        # close errors
        "[[close:a]]",
        "x[[open:a]][[close:a]][[close:a]]",
        "[[open:a]][[open:b]][[close:a]]",
    ]
    while len(cases) < SUITE_SIZE:
        cases.append(_random_v2(crng, p))
    return cases


def _random_v2(crng: random.Random, p: dict[str, Any]) -> Any:
    d = p["max_depth"]
    tokens: list[str] = []
    stack: list[str] = []
    for _ in range(crng.randint(3, 14)):
        tokens.append(crng.choice(_MACRO_TEXT))
        roll = crng.random()
        if roll < 0.42 and (len(stack) < d or crng.random() < 0.3):
            stack.append(crng.choice(_MACRO_NAMES))
            tokens.append(f"[[open:{stack[-1]}]]")
        elif roll < 0.8 and stack:
            name = stack.pop() if crng.random() < 0.93 else crng.choice(_MACRO_NAMES)
            tokens.append(f"[[close:{name}]]")
            if crng.random() < 0.35:
                tokens.append(crng.choice(["", "", " ", "t", "]"]))
                stack.append(name)
                tokens.append(f"[[open:{name}]]")
        elif roll < 0.86:
            tokens.append(crng.choice(_MACRO_BAD))
    if crng.random() < 0.6:
        tokens.extend(f"[[close:{n}]]" for n in reversed(stack))
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Variant 3: span-capped interval merger.
# ---------------------------------------------------------------------------


def _params_v3(rng: random.Random, i: int) -> dict[str, Any]:
    grace = rng.randint(1, 5)
    cap_span = rng.randint(14, 48)
    ex_w = rng.randint(11, 95)
    return {"grace": grace, "cap_span": cap_span, "ex_w": ex_w}


def _prompt_v3(p: dict[str, Any], ref: Callable[..., Any]) -> str:
    g, c, ex_w = p["grace"], p["cap_span"], p["ex_w"]
    ex_in = [(1, 4, ex_w), (4 + g, 8, 10)]
    return (
        f"Write a Python function `merge_capped_intervals(intervals: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]`.\n"
        f"Each input tuple is `(start, end, weight)` of ints with `start <= end`. The list may be in any order, "
        f"and no two tuples share the same `(start, end)` pair.\n"
        f"1. Sort the intervals by `(start, end)` ascending.\n"
        f"2. Walk the sorted list keeping a current active interval `(cur_s, cur_e, cur_w)`, initially the first "
        f"interval. Merge each following interval `(nxt_s, nxt_e, nxt_w)` into the current active interval "
        f"IF AND ONLY IF BOTH conditions hold (always evaluated against the current active interval, which may "
        f"already be the result of earlier merges):\n"
        f"   a) `nxt_s - cur_e <= {g}` (gap tolerance; overlapping intervals have a negative gap), AND\n"
        f"   b) `max(cur_e, nxt_e) - cur_s <= {c}` (maximum merged span cap).\n"
        f"3. Merging replaces the active interval with `(cur_s, max(cur_e, nxt_e), cur_w + nxt_w)`.\n"
        f"4. If the conditions do not both hold, emit `(cur_s, cur_e, cur_w)` and start a new active interval "
        f"`(nxt_s, nxt_e, nxt_w)`. After the last interval, emit the active interval.\n"
        f"Return the list of emitted `(start, end, weight)` tuples in order (`[]` for an empty input).\n"
        f"Example: `merge_capped_intervals({ex_in!r})` returns `{ref(ex_in)!r}`.\n"
        f"Return only valid Python code defining `merge_capped_intervals`."
    )


def _handwritten_v3(p: dict[str, Any]) -> str:
    grace, cap_span = p["grace"], p["cap_span"]
    return f"""
assert merge_capped_intervals([]) == []
assert merge_capped_intervals([(1, 5, 10), (5 + {grace}, 10, 20), (10, 10 + {cap_span}, 5)]) == [
    (1, 10, 30),
    (10, 10 + {cap_span}, 5),
]
"""


def _cases_v3(crng: random.Random, p: dict[str, Any], ref: Callable[..., Any]) -> list[Any]:
    g, c = p["grace"], p["cap_span"]
    cases: list[Any] = [
        [],
        [(4, 9, 2)],
        [(3, 3, 1), (3, 4, 1)],
        [(1, 5, 10), (5 + g, 10, 20), (10, 10 + c, 5)],
        # gap tolerance boundary (inclusive)
        [(0, 5, 1), (5 + g, 7 + g, 2)],
        [(0, 5, 1), (6 + g, 8 + g, 2)],
        [(14 + 2 * g, 15 + 2 * g, 7), (10, 12, 4), (12 + g, 13 + g, 1)],
        # span cap boundary (inclusive), measured from the active interval's start
        [(0, 4, 1), (3, c, 2)],
        [(0, 4, 1), (3, c + 1, 2)],
        [(6, 8, 3), (8 + g, 6 + c, 4), (7 + c, 9 + c, 5)],
        [(0, 10, 1), (5, c + 1, 1)],
        [(6, 6, 3), (2, 6, 2), (6, c + 3, 2)],
        # an oversized active interval absorbs nothing, not even nested intervals
        [(0, c + 5, 1), (1, 3, 2)],
        [(6, 6, 1), (5, 6 + c, 1)],
        # nested intervals keep the larger end, and later gaps are measured from it
        [(0, 10, 1), (2, 5, 1)],
        [(2, 5, 9), (0, 10, 1), (10 + g, 12 + g, 4)],
        # sort by (start, end), not by start alone
        [(10, 11 + c, 1), (10, 12, 2)],
        [(30, 32 + c, 5), (30, 31, 1), (0, 2, 3)],
        [(20, 21 + c, 1), (20, 20, 2)],
    ]
    while len(cases) < SUITE_SIZE:
        cases.append(_random_v3(crng, p))
    return cases


def _random_v3(crng: random.Random, p: dict[str, Any]) -> Any:
    g, c = p["grace"], p["cap_span"]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, int]] = []
    cursor = crng.randint(0, 10)
    for _ in range(crng.randint(2, 9)):
        roll = crng.random()
        if roll < 0.5:
            start = cursor + crng.choice([-3, -1, 0, 1, g - 1, g, g, g + 1, 5])
        else:
            start = crng.randint(0, 60)
        start = max(0, start)
        length = crng.choice([0, 1, 2, 3, 5, 8, c - 6, c - start % 5, c, c + 1, c + 4])
        end = start + max(0, length)
        if (start, end) in seen:
            continue
        seen.add((start, end))
        out.append((start, end, crng.randint(1, 20)))
        cursor = max(cursor, end)
    crng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Variant 4: SemVer 2.0.0 pre-release resolver.
# ---------------------------------------------------------------------------

_SEMVER_NUMERIC = ["0", "1", "2", "9", "10", "11", "100"]
_SEMVER_ALNUM = ["alpha", "beta", "rc", "Beta", "RC", "a1", "1a", "alpha-1", "x-y", "-", "--x", "z"]


def _params_v4(rng: random.Random, i: int) -> dict[str, Any]:
    return {
        "default_major": rng.randint(1, 14),
        "ex_minor": rng.randint(1, 12),
        "ex_patch": rng.randint(0, 19),
    }


def _prompt_v4(p: dict[str, Any], ref: Callable[..., Any]) -> str:
    m = p["default_major"]
    ex_versions = [
        f"{m}.{p['ex_minor']}.{p['ex_patch']}-rc.2",
        f"{m}.{p['ex_minor']}.{p['ex_patch']}-rc.10",
        f"{m + 1}.0.0",
    ]
    return (
        f"Write a Python function `select_highest_semver(versions: list[str], major_req: int = {m}) -> str | None`.\n"
        f"Every string in `versions` is a distinct, valid Semantic Versioning 2.0.0 version without build "
        f"metadata: `MAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCH-PRERELEASE`. `MAJOR`, `MINOR` and `PATCH` are "
        f"non-negative decimal integers without leading zeros. `PRERELEASE` is one or more dot-separated "
        f"identifiers made of ASCII letters, digits and hyphens; an identifier consisting only of digits is "
        f"numeric (and has no leading zeros), every other identifier is alphanumeric (e.g. `alpha`, `x-1`, `1a`).\n"
        f"Follow SemVer 2.0.0 precedence:\n"
        f"1. Consider only versions whose `MAJOR` is numerically equal to `major_req`.\n"
        f"2. Compare `(MAJOR, MINOR, PATCH)` numerically.\n"
        f"3. With equal `MAJOR.MINOR.PATCH`, a normal release (`1.2.0`) has HIGHER precedence than any "
        f"pre-release (`1.2.0-rc.1`).\n"
        f"4. Compare two pre-releases with equal `MAJOR.MINOR.PATCH` identifier by identifier, left to right: "
        f"numeric identifiers are compared as integers, alphanumeric identifiers are compared lexically in "
        f"ASCII order, and a numeric identifier always has LOWER precedence than an alphanumeric one. If all "
        f"compared identifiers are equal, the version with more identifiers has higher precedence.\n"
        f"Return the matching version with the highest precedence, exactly as it appears in `versions`, or "
        f"`None` if no version matches (including an empty list).\n"
        f"Example: `select_highest_semver({ex_versions!r})` returns `{ref(ex_versions)!r}`.\n"
        f"Return only valid Python code defining `select_highest_semver`."
    )


def _handwritten_v4(p: dict[str, Any]) -> str:
    default_major = p["default_major"]
    return f"""
cands = [
    f"{default_major}.2.0-alpha",
    f"{default_major}.2.0-alpha.1",
    f"{default_major}.2.0-alpha.beta",
    f"{default_major}.2.0-beta",
    f"{default_major}.2.0-beta.2",
    f"{default_major}.2.0-beta.11",
    f"{default_major}.2.0-rc.1",
]
assert select_highest_semver(cands, {default_major}) == f"{default_major}.2.0-rc.1"
assert select_highest_semver(cands + [f"{default_major}.2.0"], {default_major}) == f"{default_major}.2.0"
assert select_highest_semver([f"{default_major}.1.9", f"{default_major}.2.0-1", f"{default_major}.2.0-alpha"], {default_major}) == f"{default_major}.2.0-alpha"
assert select_highest_semver(["99.0.0"], {default_major}) is None
"""


def _cases_v4(crng: random.Random, p: dict[str, Any], ref: Callable[..., Any]) -> list[Any]:
    dm = p["default_major"]
    m = crng.choice([x for x in (1, 2, 3, 4) if x != dm])
    hw = [
        f"{dm}.2.0-{s}"
        for s in ("alpha", "alpha.1", "alpha.beta", "beta", "beta.2", "beta.11", "rc.1")
    ]
    cases: list[Any] = [
        (hw, dm),
        ([*hw, f"{dm}.2.0"], dm),
        ([f"{dm}.1.9", f"{dm}.2.0-1", f"{dm}.2.0-alpha"], dm),
        (["99.0.0"], dm),
        ([], dm),
        ([], None),
        # the default major_req
        ([f"{dm}.1.0", f"{dm + 1}.0.0", f"{dm - 1}.9.9"], None),
        ([f"{dm + 1}.0.0", f"{dm + 1}.2.0-rc.1"], None),
        ([f"{dm}.0.0-rc.1", f"{dm}.0.0-beta.11", f"{dm + 1}.0.0"], None),
        # MAJOR must match numerically, not by string prefix
        ([f"{m}.5.0", f"{m}1.0.0", f"{m}0.1.0"], m),
        ([f"{m}2.3.4", f"{m}0.0.0-rc.1"], m),
        ([f"{dm}.0.1", f"{dm}{dm}.0.0"], None),
        ([f"1{m}.0.0", f"{m}.0.0-alpha"], m),
        # MINOR / PATCH compare as integers
        ([f"{m}.9.0", f"{m}.10.0"], m),
        ([f"{m}.2.10", f"{m}.2.2", f"{m}.2.9"], m),
        ([f"{m}.11.0-rc.1", f"{m}.2.0"], m),
        # release beats pre-release of the same core
        ([f"{m}.3.0-rc.9", f"{m}.3.0"], m),
        ([f"{m}.3.0", f"{m}.3.0-beta.11", f"{m}.2.99"], m),
        # numeric identifiers: lower than alphanumeric, compared as integers
        ([f"{m}.0.0-1", f"{m}.0.0-alpha"], m),
        ([f"{m}.0.0-rc.a", f"{m}.0.0-rc.1"], m),
        ([f"{m}.0.0-2", f"{m}.0.0-11"], m),
        ([f"{m}.0.0-alpha.11", f"{m}.0.0-alpha.2"], m),
        ([f"{m}.0.0-rc.100", f"{m}.0.0-rc.9", f"{m}.0.0-rc.10"], m),
        ([f"{m}.0.0-1a", f"{m}.0.0-100"], m),
        # more identifiers win when all compared identifiers are equal
        ([f"{m}.0.0-alpha", f"{m}.0.0-alpha.1"], m),
        ([f"{m}.0.0-rc.1.1", f"{m}.0.0-rc.1"], m),
        ([f"{m}.0.0-alpha.beta", f"{m}.0.0-alpha"], m),
        # ASCII order for alphanumeric identifiers
        ([f"{m}.0.0-Beta", f"{m}.0.0-alpha"], m),
        ([f"{m}.0.0-alpha-1", f"{m}.0.0-alpha.1"], m),
        ([f"{m}.0.0-RC.2", f"{m}.0.0---x"], m),
        (["0.1.0", "0.0.9", f"{m}.0.0"], 0),
    ]
    while len(cases) < SUITE_SIZE:
        cases.append(_random_v4(crng, p))
    return cases


def _random_v4(crng: random.Random, p: dict[str, Any]) -> Any:
    dm = p["default_major"]
    req = crng.choice([None, None, dm, dm, 1, 2, 3, 0, 10, 11])
    m = dm if req is None else req
    majors = [m, m, m, m * 10 + crng.randint(0, 9), m + 1, max(0, m - 1), 1, 11]
    versions: list[str] = []
    for _ in range(crng.randint(2, 9)):
        core = f"{crng.choice(majors)}.{crng.choice([0, 1, 2, 9, 10, 11])}.{crng.choice([0, 2, 9, 10])}"
        if crng.random() < 0.6:
            ids = [
                crng.choice(_SEMVER_NUMERIC if crng.random() < 0.45 else _SEMVER_ALNUM)
                for _ in range(crng.choice([1, 1, 2, 2, 3]))
            ]
            core += "-" + ".".join(ids)
        if core not in versions:
            versions.append(core)
    return (versions, req)


# ---------------------------------------------------------------------------
# Item assembly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VariantSpec:
    draw: Callable[[random.Random, int], dict[str, Any]]
    prompt: Callable[[dict[str, Any], Callable[..., Any]], str]
    handwritten: Callable[[dict[str, Any]], str]
    cases: Callable[[random.Random, dict[str, Any], Callable[..., Any]], list[Any]]
    random_case: Callable[[random.Random, dict[str, Any]], Any]


_VARIANTS: dict[int, _VariantSpec] = {
    0: _VariantSpec(_params_v0, _prompt_v0, _handwritten_v0, _cases_v0, _random_v0),
    1: _VariantSpec(_params_v1, _prompt_v1, _handwritten_v1, _cases_v1, _random_v1),
    2: _VariantSpec(_params_v2, _prompt_v2, _handwritten_v2, _cases_v2, _random_v2),
    3: _VariantSpec(_params_v3, _prompt_v3, _handwritten_v3, _cases_v3, _random_v3),
    4: _VariantSpec(_params_v4, _prompt_v4, _handwritten_v4, _cases_v4, _random_v4),
}


@dataclass(frozen=True)
class S2Build:
    """A generated item plus everything needed to re-validate it."""

    item: dict[str, Any]
    variant: int
    params: dict[str, Any]
    reference_source: str
    mutant_sources: dict[str, str]
    cases: list[tuple[Any, Any]]
    detections: dict[str, int]
    equivalent_mutants: tuple[str, ...]
    handwritten_asserts: str


def _equivalent(variant: int, mutant: str, params: dict[str, Any]) -> bool:
    predicate = EQUIVALENT_WHEN.get(variant, {}).get(mutant)
    return bool(predicate and predicate(params))


def _test_failure(source: str, test_code: str) -> str | None:
    """Run `test_code` against `source` in-process, as the sandbox does; return the failure or None."""
    script = source + "\n\n# --- UNIT TEST HARNESS ---\n" + test_code
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(script, "s2_check.py", "exec"), {"__name__": "s2_check"})  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - any exception fails the sandbox run as well
        return f"{type(exc).__name__}: {exc}"
    if _SUITE_OK_SENTINEL in test_code and _SUITE_OK_SENTINEL not in out.getvalue():
        return "the hidden suite did not print its completion marker"
    return None


def build_s2_item(rng: random.Random, item_id: str, i: int) -> S2Build:
    """Build one `s2_everyday_code` item (`i % 5` subtype) and verify reference + mutants.

    Draws parameters from `rng`, generates a hidden differential suite from a private RNG seeded by
    `(item_id, variant, params)`, and asserts in-process that the reference passes while every
    non-equivalent mutant fails at least `MIN_DETECTIONS` cases.
    """
    variant = i % 5
    spec = _VARIANTS[variant]
    params = spec.draw(rng, i)
    case_rng = random.Random(f"s2_specs|{item_id}|{variant}|{sorted(params.items())!r}")
    name = FUNCTION_NAMES[variant]

    reference_source = render_source(variant, params)
    ref = load_function(reference_source, name)
    handwritten = spec.handwritten(params)
    failure = _test_failure(reference_source, handwritten)
    if failure is not None:
        raise AssertionError(f"{item_id}: the reference fails the hand-written asserts: {failure}")

    mutant_sources = {key: render_source(variant, params, key) for key in MUTANTS[variant]}
    mutants = {key: load_function(src, name) for key, src in mutant_sources.items()}

    args_list: list[Any] = []
    seen: set[str] = set()

    def add(args: Any) -> bool:
        key = repr(args)
        if key in seen:
            return False
        seen.add(key)
        args_list.append(args)
        return True

    for args in spec.cases(case_rng, params, ref):
        add(args)
    expected = [_call(variant, ref, a) for a in args_list]
    for exp in expected:
        if isinstance(exp, str) and exp.startswith("raised "):
            raise AssertionError(f"{item_id}: reference crashed on a suite case: {exp}")

    detections: dict[str, int] = {}
    for key, fn in mutants.items():
        detections[key] = sum(
            1 for a, e in zip(args_list, expected, strict=True) if _call(variant, fn, a) != e
        )
        if detections[key] > MIN_DETECTIONS or _equivalent(variant, key, params):
            continue
        # Top up with randomized cases that separate this mutant from the reference.
        for _ in range(TOPUP_ATTEMPTS):
            cand = spec.random_case(case_rng, params)
            want = _call(variant, ref, cand)
            if _call(variant, fn, cand) != want and add(cand):
                expected.append(want)
                detections[key] += 1
                if detections[key] > MIN_DETECTIONS:
                    break

    # Final accounting over the complete suite (top-ups may also catch other mutants).
    equivalent: list[str] = []
    for key, fn in mutants.items():
        detections[key] = sum(
            1 for a, e in zip(args_list, expected, strict=True) if _call(variant, fn, a) != e
        )
        if _equivalent(variant, key, params):
            if detections[key]:
                raise AssertionError(f"{item_id}: mutant {key} declared equivalent but detected")
            equivalent.append(key)
        elif detections[key] < MIN_DETECTIONS:
            raise AssertionError(
                f"{item_id}: mutant {key} detected by only {detections[key]} suite cases"
            )

    cases = list(zip(args_list, expected, strict=True))
    test_code = handwritten + _harness(variant, cases)
    # End-to-end check of the emitted test code itself (literal round-trip, harness, hand-written asserts).
    failure = _test_failure(reference_source, test_code)
    if failure is not None:
        raise AssertionError(
            f"{item_id}: the reference fails the emitted test code: {failure[:300]}"
        )
    for key, src in mutant_sources.items():
        caught = _test_failure(src, test_code) is not None
        if caught == (key in equivalent):
            raise AssertionError(
                f"{item_id}: mutant {key} {'fails' if caught else 'passes'} the emitted test code"
            )

    item = {
        "id": item_id,
        "slice": "s2_everyday_code",
        "modality": "code",
        "subtype": SUBTYPES[variant],
        "source_provenance": PROVENANCE,
        "prompt": spec.prompt(params, ref),
        "attachments": [],
        "grader_type": "python_unit_tests",
        "grader_spec": {"test_code": test_code, "timeout_sec": TIMEOUT_SEC},
    }
    return S2Build(
        item=item,
        variant=variant,
        params=params,
        reference_source=reference_source,
        mutant_sources=mutant_sources,
        cases=cases,
        detections=detections,
        equivalent_mutants=tuple(equivalent),
        handwritten_asserts=handwritten,
    )


def make_s2_item(rng: random.Random, item_id: str, i: int) -> dict[str, Any]:
    """Build one `s2_everyday_code` item (variant `i % 5`); same schema as the original generator."""
    return build_s2_item(rng, item_id, i).item
