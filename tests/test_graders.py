"""Pin the current behavior of the deterministic graders in `evals.graders`.

These tests document what the graders do today, including lenient or surprising behavior
(marked "PINNED CURRENT BEHAVIOR"), so that tightening a grader is a deliberate, visible change
rather than a silent shift in benchmark scores. Code items use a fake sandbox; only the last
test touches a real container backend and it is skipped when none is running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import graders
from evals.graders import (
    _FINAL_ANSWER_RE,
    _NUMBER_RE,
    _extract_json_obj,
    grade_canonical_string,
    grade_exact_json,
    grade_ifeval_constraints,
    grade_item,
    grade_numeric_tolerance,
)
from evals.sandbox import SandboxResult, extract_python_code

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASETS = ["data/pilot.jsonl", "data/eval.jsonl", "data/calibration.jsonl"]
_GRADER_TYPES = {
    "exact_json",
    "ifeval_constraints",
    "numeric_tolerance",
    "canonical_string",
    "python_unit_tests",
}


# --- JSON extraction (`_extract_json_obj`) ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The FIRST fenced block wins, even when a later block holds a different answer.
        ('```json\n{"a": 1}\n```\nOr maybe:\n```json\n{"a": 2}\n```', {"a": 1}),
        # Untagged fences and a case-insensitive `JSON` tag are accepted.
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('```JSON\n{"a": 1}\n```', {"a": 1}),
        # Unfenced: the greedy first-brace-to-last-brace fallback drops surrounding prose.
        ('Sure! {"a": 1} Hope this helps.', {"a": 1}),
        ('Here you go:\n{"outer": {"inner": [1, 2]}}\nDone.', {"outer": {"inner": [1, 2]}}),
        # Bare JSON parses directly (any JSON value, not only objects).
        ('  {"a": 1}  ', {"a": 1}),
        ("42", 42),
        # A fence that is not valid JSON falls back to the brace search over the WHOLE text.
        ('```json\nsee below\n```\n{"a": 2}', {"a": 2}),
    ],
)
def test_extract_json_obj(text: str, expected: Any) -> None:
    assert _extract_json_obj(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Greedy brace span covers both objects -> "Extra data".
        'A {"a": 1} B {"b": 2}',
        # PINNED CURRENT BEHAVIOR: braces in an earlier non-JSON code block swallow the answer.
        '```python\nd = {1: 2}\n```\nAnswer: {"a": 1}',
        "no json here",
        '{"a": 1,}',
        # Pathological unmatched braces must not trigger O(N^2) regex backtracking.
        "{" * 50_000,
    ],
)
def test_extract_json_obj_raises(text: str) -> None:
    with pytest.raises(json.JSONDecodeError):
        _extract_json_obj(text)


def test_extract_json_obj_prose_bracket_hijacks_fallback() -> None:
    # PINNED CURRENT BEHAVIOR (surprising): the fallback regex starts at the first `{` OR `[`,
    # so a bracketed citation in the prose wins over the actual JSON object.
    text = 'See [1]. Result: {"a": 1}'
    assert _extract_json_obj(text) == [1]
    assert grade_exact_json(text, {"expected": {"a": 1}}) == (
        False,
        "mismatch:got=[1],expected={'a': 1}",
    )


# --- `grade_exact_json` matching rules (incl. bool vs int) ---


@pytest.mark.parametrize(
    ("got", "expected", "ok"),
    [
        # PINNED CURRENT BEHAVIOR (surprising): bool is an int subclass and is NOT distinguished
        # from numbers in either direction (`true` satisfies 1 / 1.0; 1 satisfies True).
        (True, 1, True),
        (True, 1.0, True),
        (False, 0, True),
        (1, True, True),
        (0, False, True),
        (0.0, False, True),
        (2, True, False),
        # Numeric strings are never coerced.
        ("1", 1, False),
        (1, "1", False),
        # Expected ints compare with the float tolerance (default abs 1e-3).
        (3.0004, 3, True),
        (3.01, 3, False),
        (2.0005, 2.0, True),
        # Strings: case-insensitive and whitespace-stripped, otherwise exact.
        ("  Paris ", "paris", True),
        ("Paris!", "paris", False),
        # Dicts ignore extra keys; lists are order- and length-sensitive.
        ({"a": 1, "extra": 9}, {"a": 1}, True),
        ({"b": 1}, {"a": 1}, False),
        ([1, 2], [2, 1], False),
        ([1, 2, 3], [1, 2], False),
        ({"xs": [{"k": "V"}]}, {"xs": [{"k": "v"}]}, True),
        (None, None, True),
    ],
)
def test_grade_exact_json_match_rules(got: Any, expected: Any, ok: bool) -> None:
    passed, reason = grade_exact_json(json.dumps(got), {"expected": expected})
    assert passed is ok
    assert reason == ("ok" if ok else f"mismatch:got={got!r},expected={expected!r}")


def test_grade_exact_json_custom_tolerance_and_parse_error_reason() -> None:
    assert grade_exact_json("3.4", {"expected": 3, "float_abs_tol": 0.5}) == (True, "ok")
    passed, reason = grade_exact_json("no json here", {"expected": {"a": 1}})
    assert passed is False
    assert reason.startswith("json_parse_error:")


# --- Number parsing (`_NUMBER_RE`) and `grade_numeric_tolerance` ---


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("1,234", ["1,234"]),
        ("1,234,567.89", ["1,234,567.89"]),
        ("-12.5", ["-12.5"]),
        ("+3", ["+3"]),
        ("0.75", ["0.75"]),
        # PINNED CURRENT BEHAVIOR (surprising): no leading-dot decimals, no scientific notation,
        # and commas that are not 3-digit thousands groups split the number.
        (".5", ["5"]),
        ("1.5e3", ["1.5", "3"]),
        ("1,5", ["1", "5"]),
        ("1,23", ["1", "23"]),
        ("1,2,3", ["1", "2", "3"]),
        # A hyphen directly before digits is read as a minus sign.
        ("3-4", ["3", "-4"]),
        ("2026-11-12", ["2026", "-11", "-12"]),
        ("12.5 m^2", ["12.5", "2"]),
    ],
)
def test_number_re_tokens(text: str, tokens: list[str]) -> None:
    assert _NUMBER_RE.findall(text) == tokens


@pytest.mark.parametrize(
    ("response", "expected", "ok", "reason"),
    [
        ("FINAL_ANSWER: 1,234", 1234, True, "ok"),
        ("final answer: -12.5", -12.5, True, "ok"),
        ("Working...\nFINAL_ANSWER:\n42", 42, True, "ok"),
        # PINNED CURRENT BEHAVIOR (surprising): the LAST number on the FINAL_ANSWER line is
        # graded, so trailing qualifiers, units with exponents, or sci-notation break it.
        ("FINAL_ANSWER: 42 (approx 40)", 42, False, "numeric_mismatch:got=40.0,expected=42.0"),
        ("FINAL_ANSWER: 12.5 m^2", 12.5, False, "numeric_mismatch:got=2.0,expected=12.5"),
        ("FINAL_ANSWER: 1.5e3", 1500, False, "numeric_mismatch:got=3.0,expected=1500.0"),
        # Without a FINAL_ANSWER line, the last number anywhere in the response is graded.
        ("I think it is 3, no wait, 7.", 7, True, "ok"),
        ("The total is 7, not 9.", 7, False, "numeric_mismatch:got=9.0,expected=7.0"),
        # A FINAL_ANSWER line without digits does not fall back to the rest of the text.
        ("The value is 5.\nFINAL_ANSWER: five", 5, False, "no_number_found"),
        ("no digits at all", 5, False, "no_number_found"),
    ],
)
def test_grade_numeric_tolerance(response: str, expected: float, ok: bool, reason: str) -> None:
    assert grade_numeric_tolerance(response, {"expected": expected}) == (ok, reason)


def test_grade_numeric_tolerance_uses_spec_tolerances() -> None:
    spec = {"expected": 10.0, "abs_tol": 0.5, "rel_tol": 1e-9}
    assert grade_numeric_tolerance("FINAL_ANSWER: 10.4", spec) == (True, "ok")
    assert grade_numeric_tolerance("FINAL_ANSWER: 10.6", spec)[0] is False
    # Defaults: abs_tol=1e-2, rel_tol=1e-3.
    assert grade_numeric_tolerance("FINAL_ANSWER: 1000.9", {"expected": 1000})[0] is True


# --- `_FINAL_ANSWER_RE` ---


@pytest.mark.parametrize(
    ("text", "captured"),
    [
        ("FINAL_ANSWER: 5", "5"),
        ("final_answer: 5", "5"),
        ("FINAL ANSWER: 5", "5"),
        ("Final Answer = 5", "5"),
        ("answer: 5", "5"),
        ("ANSWER=5", "5"),
        # The bare `ANSWER` alternative also fires inside ordinary prose.
        ("My answer: 5", "5"),
        # `\s*` after the separator crosses a newline.
        ("FINAL_ANSWER:\n5", "5"),
        # Markdown bold around the label leaks into the capture; trailing spaces are kept.
        ("**FINAL_ANSWER:** 5", "** 5"),
        ("FINAL_ANSWER : 7 \nmore", "7 "),
        # FIRST match wins.
        ("Answer: 1\nFINAL_ANSWER: 2", "1"),
        ("I will end with FINAL_ANSWER: <number>.\nFINAL_ANSWER: 65.4", "<number>."),
    ],
)
def test_final_answer_re_first_match(text: str, captured: str) -> None:
    m = _FINAL_ANSWER_RE.search(text)
    assert m is not None
    assert m.group(1) == captured


@pytest.mark.parametrize("text", ["Final answer is 5", "**Final Answer**: 5", "Answers: 5"])
def test_final_answer_re_no_match(text: str) -> None:
    assert _FINAL_ANSWER_RE.search(text) is None


def test_restated_format_instruction_hides_the_real_answer() -> None:
    # PINNED CURRENT BEHAVIOR: first-match semantics grade the restated instruction, not the
    # answer that the prompt asks to put at the END of the response.
    response = "I will end with `FINAL_ANSWER: <number>`.\nArea = 65.4\nFINAL_ANSWER: 65.4"
    assert grade_numeric_tolerance(response, {"expected": 65.4}) == (False, "no_number_found")


# --- `grade_canonical_string` ---


@pytest.mark.parametrize(
    ("response", "expected", "ok"),
    [
        ("FINAL_ANSWER: 2026-07-12", "2026-07-12", True),
        # Surrounding backticks / quotes / dots are stripped; comparison is case-insensitive.
        ("final answer: `2026-07-12`.", "2026-07-12", True),
        ('FINAL_ANSWER: "Tuesday"', "tuesday", True),
        ("FINAL_ANSWER: TUESDAY", "Tuesday", True),
        # PINNED CURRENT BEHAVIOR (over-lenient): `expected in got` accepts any answer that merely
        # CONTAINS the expected string -- hedges, negations and longer numbers all pass.
        ("FINAL_ANSWER: 2026-07-12 or 2026-12-07", "2026-07-12", True),
        ("FINAL_ANSWER: not 2026-07-12", "2026-07-12", True),
        ("FINAL_ANSWER: 12", "1", True),
        ("**FINAL_ANSWER:** 2026-07-12", "2026-07-12", True),
        # Without FINAL_ANSWER, only the last non-empty line is compared.
        ("Let me think.\n\n2026-07-12\n\n", "2026-07-12", True),
        ("2026-07-12\nI hope that helps!", "2026-07-12", False),
        ("FINAL_ANSWER: 2026-12-07", "2026-07-12", False),
        ("", "x", False),
    ],
)
def test_grade_canonical_string(response: str, expected: str, ok: bool) -> None:
    assert grade_canonical_string(response, {"expected": expected})[0] is ok


def test_grade_canonical_string_mismatch_reason() -> None:
    response = "2026-07-12\nI hope that helps!"
    assert grade_canonical_string(response, {"expected": "2026-07-12"}) == (
        False,
        "string_mismatch:got='i hope that helps!',expected='2026-07-12'",
    )


# --- `grade_ifeval_constraints` (the constraint keys used by the datasets) ---


@pytest.mark.parametrize(
    ("response", "constraints", "result"),
    [
        ("- a\n- b\nDone.", {"exact_bullet_count": 2, "ends_with": "Done."}, (True, "ok")),
        ("- a\nDone.", {"exact_bullet_count": 2}, (False, "bullet_count:1!=2")),
        # Forbidden words match whole alphabetic tokens only ...
        ("The cat sat.", {"forbidden_words": ["cat"]}, (False, "forbidden_word_present:cat")),
        ("Concatenate.", {"forbidden_words": ["cat"]}, (True, "ok")),
        # ... while required keywords are plain substrings (lenient).
        ("Concatenate.", {"required_keywords": ["cat"]}, (True, "ok")),
        ("Hello.", {"required_keywords": ["cat"]}, (False, "missing_keyword:cat")),
        ("one two three", {"max_words": 2}, (False, "too_many_words:3>2")),
        ("one two", {"min_words": 3}, (False, "too_few_words:2<3")),
        # `ends_with` is checked on the stripped response.
        ("All done.  \n", {"ends_with": "done."}, (True, "ok")),
        ("All done!", {"ends_with": "done."}, (False, "missing_suffix:'done.'")),
    ],
)
def test_grade_ifeval_constraints(
    response: str, constraints: dict[str, Any], result: tuple[bool, str]
) -> None:
    assert grade_ifeval_constraints(response, {"constraints": constraints}) == result


# --- `extract_python_code` (what the code grader actually executes) ---


@pytest.mark.parametrize(
    ("response", "code"),
    [
        # PINNED CURRENT BEHAVIOR: ALL python/py/untagged fences are concatenated, so a usage
        # example after the solution runs before the harness.
        ("```python\na = 1\n```\nand\n```python\nb = 2\n```", "a = 1\n\nb = 2"),
        ("```py\na = 1\n```", "a = 1"),
        ("```Python\na = 1\n```", "a = 1"),
        ("```\na = 1\n```", "a = 1"),
        # No fence: the whole response (stripped) is executed.
        ("  a = 1\n", "a = 1"),
        # PINNED CURRENT BEHAVIOR (bug): a non-python fence before the python one makes the
        # regex pair the wrong backticks -- only the prose between the blocks is extracted.
        ("```bash\npip install x\n```\nThen:\n```python\na = 1\n```", "Then:"),
        # Unterminated or unrecognized fences fall back to the raw text, fences included.
        ("```python\na = 1\n", "```python\na = 1"),
        ("```python3\na = 1\n```", "```python3\na = 1\n```"),
    ],
)
def test_extract_python_code(response: str, code: str) -> None:
    assert extract_python_code(response) == code


# --- `grade_item` dispatch ---


class _FakeSandbox:
    """Stand-in for `run_python_in_sandbox` that records calls and returns a canned result."""

    def __init__(self, result: SandboxResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, candidate_response: str, test_harness: str, **kwargs: Any) -> SandboxResult:
        self.calls.append(
            {"candidate_response": candidate_response, "test_harness": test_harness, **kwargs}
        )
        return self.result


def _result(**overrides: Any) -> SandboxResult:
    fields: dict[str, Any] = {
        "passed": True,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "harness_completed": True,
    }
    return SandboxResult(**{**fields, **overrides})


@pytest.fixture
def install_sandbox(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(result: SandboxResult) -> _FakeSandbox:
        fake = _FakeSandbox(result)
        monkeypatch.setattr(graders, "run_python_in_sandbox", fake)
        return fake

    return _install


_CODE_ITEM: dict[str, Any] = {
    "grader_type": "python_unit_tests",
    "grader_spec": {"test_code": "assert solve() == 1"},
}


@pytest.mark.parametrize(
    ("item", "good", "bad", "bad_reason_prefix"),
    [
        (
            {"grader_type": "exact_json", "grader_spec": {"expected": {"a": 1}}},
            'Answer:\n```json\n{"a": 1}\n```',
            '{"a": 2}',
            "mismatch:",
        ),
        (
            {
                "grader_type": "ifeval_constraints",
                "grader_spec": {"constraints": {"exact_bullet_count": 1, "ends_with": "Done."}},
            },
            "- one\nDone.",
            "- one\n- two\nDone.",
            "bullet_count:",
        ),
        (
            {
                "grader_type": "numeric_tolerance",
                "grader_spec": {"expected": 3.14, "abs_tol": 0.01},
            },
            "FINAL_ANSWER: 3.14",
            "FINAL_ANSWER: 3.2",
            "numeric_mismatch:",
        ),
        (
            {"grader_type": "canonical_string", "grader_spec": {"expected": "B"}},
            "FINAL_ANSWER: B",
            "FINAL_ANSWER: C",
            "string_mismatch:",
        ),
    ],
)
def test_grade_item_dispatches_non_code_graders_without_sandbox(
    install_sandbox: Any, item: dict[str, Any], good: str, bad: str, bad_reason_prefix: str
) -> None:
    fake = install_sandbox(_result())
    assert grade_item(item, good) == (True, "ok")
    passed, reason = grade_item(item, bad)
    assert passed is False
    assert reason.startswith(bad_reason_prefix)
    assert fake.calls == []


def test_grade_item_code_pass_forwards_spec_and_backend(install_sandbox: Any) -> None:
    fake = install_sandbox(_result())
    response = "```python\ndef solve():\n    return 1\n```"
    assert grade_item(_CODE_ITEM, response) == (True, "ok")
    item = {**_CODE_ITEM, "grader_spec": {**_CODE_ITEM["grader_spec"], "timeout_sec": 10}}
    assert grade_item(item, response, sandbox_backend="container") == (True, "ok")
    assert fake.calls == [
        {
            # The raw response is forwarded; code extraction happens inside the sandbox module.
            "candidate_response": response,
            "test_harness": "assert solve() == 1",
            "timeout_sec": 5.0,
            "backend": "auto",
        },
        {
            "candidate_response": response,
            "test_harness": "assert solve() == 1",
            "timeout_sec": 10.0,
            "backend": "container",
        },
    ]


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (
            _result(passed=False, exit_code=0, harness_completed=False),
            "exit=0:harness_did_not_complete",
        ),
        (
            _result(passed=False, exit_code=1, stderr="Traceback ...\nAssertionError: boom\n\n"),
            "exit=1:Traceback ...\nAssertionError: boom",
        ),
        (
            _result(passed=False, exit_code=1, harness_completed=False, stderr="NameError: solve"),
            "exit=1:NameError: solve",
        ),
        (
            _result(
                passed=False,
                exit_code=-1,
                stderr="Execution timed out inside container sandbox.",
                timed_out=True,
                harness_completed=False,
            ),
            "exit=-1:Execution timed out inside container sandbox.",
        ),
    ],
    ids=["early-exit", "assertion", "crash", "timeout"],
)
def test_grade_item_code_failure_reasons(
    install_sandbox: Any, result: SandboxResult, reason: str
) -> None:
    install_sandbox(result)
    assert grade_item(_CODE_ITEM, "def solve():\n    return 2") == (False, reason)


def test_grade_item_code_failure_keeps_stderr_tail(install_sandbox: Any) -> None:
    stderr = "x" * 2000 + "\nAssertionError: the useful part"
    install_sandbox(_result(passed=False, exit_code=1, stderr=stderr))
    passed, reason = grade_item(_CODE_ITEM, "def solve(): ...")
    assert passed is False
    assert reason == "exit=1:" + stderr[-1500:]
    assert reason.endswith("AssertionError: the useful part")


def test_grade_item_unknown_grader_type() -> None:
    with pytest.raises(ValueError, match="Unknown grader_type"):
        grade_item({"grader_type": "llm_judge", "grader_spec": {}}, "anything")


@pytest.mark.parametrize("rel_path", _DATASETS)
def test_every_dataset_item_is_dispatchable_and_empty_answers_fail(
    install_sandbox: Any, rel_path: str
) -> None:
    path = _REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert items
    install_sandbox(_result(passed=False, exit_code=0, harness_completed=False))
    for item in items:
        assert item["grader_type"] in _GRADER_TYPES, item["id"]
        assert grade_item(item, "")[0] is False, item["id"]


# --- Real sandbox (skipped when no container backend is running) ---


@pytest.mark.slow
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python\ndef solve():\n    return 1\n```", (True, "ok")),
        (
            "```python\nimport sys\n\ndef solve():\n    return 2\n\nsys.exit(0)\n```",
            (False, "exit=0:harness_did_not_complete"),
        ),
    ],
    ids=["correct", "early-exit"],
)
def test_grade_item_code_real_sandbox(
    real_backend: Any, response: str, expected: tuple[bool, str]
) -> None:
    item = {**_CODE_ITEM, "grader_spec": {**_CODE_ITEM["grader_spec"], "timeout_sec": 60}}
    assert grade_item(item, response, sandbox_backend=real_backend) == expected
