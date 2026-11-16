"""Deterministic graders for all 5 evals slices (zero LLM-as-a-judge)."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from evals.sandbox import SandboxBackend, run_python_in_sandbox

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"(?:FINAL_ANSWER|Final Answer|ANSWER)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _find_outer_json_span(text: str) -> str | None:
    """Return the outermost `{...}` or `[...]` substring in O(N) without regex backtracking."""
    spans: list[tuple[int, int]] = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            spans.append((start, end))
    if not spans:
        return None
    start, end = min(spans, key=lambda s: s[0])
    return text[start : end + 1]


def _extract_json_obj(text: str) -> Any:
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        span = _find_outer_json_span(text)
        if span is not None:
            return json.loads(span)
        raise


def grade_exact_json(response_text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Compare extracted JSON object against `spec['expected']` (with optional float tolerance)."""
    try:
        parsed = _extract_json_obj(response_text)
    except Exception as exc:
        return False, f"json_parse_error:{exc}"

    expected = spec["expected"]
    tol = float(spec.get("float_abs_tol", 1e-3))

    def _match(a: Any, b: Any) -> bool:
        if isinstance(b, float) and isinstance(a, (int, float)):
            return math.isclose(float(a), b, abs_tol=tol, rel_tol=1e-4)
        if isinstance(b, int) and not isinstance(b, bool) and isinstance(a, (int, float)):
            return math.isclose(float(a), float(b), abs_tol=tol)
        if isinstance(b, str) and isinstance(a, str):
            return a.strip().lower() == b.strip().lower()
        if isinstance(b, dict) and isinstance(a, dict):
            return all(k in a and _match(a[k], v) for k, v in b.items())
        if isinstance(b, list) and isinstance(a, list):
            return len(a) == len(b) and all(_match(x, y) for x, y in zip(a, b, strict=False))
        return a == b

    ok = _match(parsed, expected)
    return ok, "ok" if ok else f"mismatch:got={parsed!r},expected={expected!r}"


def grade_ifeval_constraints(response_text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Verify programmatic formatting and lexical constraints (modeled after IFEval)."""
    stripped = response_text.strip()
    constraints: dict[str, Any] = spec["constraints"]

    if "exact_bullet_count" in constraints:
        bullets = [line for line in stripped.splitlines() if re.match(r"^\s*-\s+\S+", line)]
        if len(bullets) != int(constraints["exact_bullet_count"]):
            return False, f"bullet_count:{len(bullets)}!={constraints['exact_bullet_count']}"

    if "forbidden_letters" in constraints:
        lower = stripped.lower()
        for ch in constraints["forbidden_letters"]:
            if ch.lower() in lower:
                return False, f"forbidden_letter_present:{ch}"

    if "forbidden_words" in constraints:
        words = set(re.findall(r"[a-zA-Z]+", stripped.lower()))
        for w in constraints["forbidden_words"]:
            if w.lower() in words:
                return False, f"forbidden_word_present:{w}"

    if "required_keywords" in constraints:
        lower = stripped.lower()
        for kw in constraints["required_keywords"]:
            if kw.lower() not in lower:
                return False, f"missing_keyword:{kw}"

    if "max_words" in constraints:
        word_cnt = len(stripped.split())
        if word_cnt > int(constraints["max_words"]):
            return False, f"too_many_words:{word_cnt}>{constraints['max_words']}"

    if "min_words" in constraints:
        word_cnt = len(stripped.split())
        if word_cnt < int(constraints["min_words"]):
            return False, f"too_few_words:{word_cnt}<{constraints['min_words']}"

    if "ends_with" in constraints:
        target_end = str(constraints["ends_with"])
        if not stripped.endswith(target_end):
            return False, f"missing_suffix:{target_end!r}"

    return True, "ok"


def grade_numeric_tolerance(response_text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Extract the final numerical answer and compare within `abs_tol` / `rel_tol`."""
    expected = float(spec["expected"])
    abs_tol = float(spec.get("abs_tol", 1e-2))
    rel_tol = float(spec.get("rel_tol", 1e-3))

    m = _FINAL_ANSWER_RE.search(response_text)
    search_target = m.group(1) if m else response_text
    nums = _NUMBER_RE.findall(search_target)
    if not nums:
        return False, "no_number_found"

    candidate_val = float(nums[-1].replace(",", ""))
    ok = math.isclose(candidate_val, expected, abs_tol=abs_tol, rel_tol=rel_tol)
    return ok, "ok" if ok else f"numeric_mismatch:got={candidate_val},expected={expected}"


def grade_canonical_string(response_text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    """Extract `FINAL_ANSWER: <str>` (or last non-empty line) and compare canonical normalized text."""
    expected = str(spec["expected"]).strip().lower()
    m = _FINAL_ANSWER_RE.search(response_text)
    if m:
        got = m.group(1).strip().strip("`'\".").lower()
    else:
        lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
        got = lines[-1].strip("`'\".").lower() if lines else ""

    ok = got == expected or expected in got
    return ok, "ok" if ok else f"string_mismatch:got={got!r},expected={expected!r}"


def grade_item(
    item: dict[str, Any],
    response_text: str,
    *,
    sandbox_backend: SandboxBackend | Literal["auto"] = "auto",
) -> tuple[bool, str]:
    """Dispatch an item to its deterministic grader."""
    grader_type = item["grader_type"]
    spec = item["grader_spec"]

    if grader_type == "exact_json":
        return grade_exact_json(response_text, spec)
    if grader_type == "ifeval_constraints":
        return grade_ifeval_constraints(response_text, spec)
    if grader_type == "numeric_tolerance":
        return grade_numeric_tolerance(response_text, spec)
    if grader_type == "canonical_string":
        return grade_canonical_string(response_text, spec)
    if grader_type == "python_unit_tests":
        res = run_python_in_sandbox(
            response_text,
            spec["test_code"],
            timeout_sec=float(spec.get("timeout_sec", 5.0)),
            backend=sandbox_backend,
        )
        if res.passed:
            return True, "ok"
        if res.exit_code == 0 and not res.harness_completed:
            # Clean exit without the completion marker: the candidate exited early
            # (`sys.exit(0)`, `os._exit(0)`, `unittest.main()`, ...) before the tests finished.
            return False, "exit=0:harness_did_not_complete"
        # Keep the tail of stderr: the traceback / assertion message is at the end.
        return False, f"exit={res.exit_code}:{res.stderr.strip()[-1500:]}"

    raise ValueError(f"Unknown grader_type: {grader_type!r}")
