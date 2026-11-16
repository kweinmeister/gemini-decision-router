"""Subtype `vm_spec_interpreter` for `s3_frontier_reasoning`."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evals.gen._edits import apply_edits as _apply

MIN_DETECTIONS = 2

VM_SOURCE = '''"""Reference TOYVM interpreter."""

WORD_BITS = 12
STACK_CAPACITY = 8
MAX_CALL_DEPTH = 4
STEP_LIMIT = 200
DIV_FLOOR = False
SHR_ARITHMETIC = True

_MOD = 1 << WORD_BITS
_HALF = 1 << (WORD_BITS - 1)
_NEEDS = {
    "POP": 1, "DUP": 1, "NOT": 1, "JZ": 1, "JNEG": 1, "OUT": 1,
    "SWAP": 2, "OVER": 2, "ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2, "MOD": 2,
    "AND": 2, "OR": 2, "XOR": 2, "SHL": 2, "SHR": 2, "ROT": 3,
}
_GROWS = ("PUSH", "DUP", "OVER", "IN")


def _wrap(x):
    return (x + _HALF) % _MOD - _HALF


def _parse(program):
    code, labels = [], {}
    for raw in program:
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1].strip()] = len(code)
            continue
        parts = line.split()
        code.append((parts[0], parts[1] if len(parts) > 1 else None))
    return code, labels


def _binary(op, a, b):
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op in ("DIV", "MOD"):
        if DIV_FLOOR:
            q = a // b
        else:
            q = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                q = -q
        return q if op == "DIV" else a - b * q
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    n = b % WORD_BITS
    if op == "SHL":
        return a << n
    return a >> n if SHR_ARITHMETIC else (a % _MOD) >> n


def run(program, inputs):
    code, labels = _parse(program)
    stack, calls, outputs = [], [], []
    queue = list(inputs)
    pc = steps = 0
    while True:
        if pc >= len(code):
            return "HALT", outputs, steps
        if steps == STEP_LIMIT:
            return "TIMEOUT", outputs, steps
        op, arg = code[pc]
        pc += 1
        steps += 1
        if len(stack) < _NEEDS.get(op, 0):
            return "UNDERFLOW", outputs, steps
        if op == "IN" and not queue:
            return "NO_INPUT", outputs, steps
        if op in _GROWS and len(stack) >= STACK_CAPACITY:
            return "OVERFLOW", outputs, steps
        if op == "PUSH":
            stack.append(_wrap(int(arg)))
        elif op == "IN":
            stack.append(_wrap(queue.pop(0)))
        elif op == "POP":
            stack.pop()
        elif op == "DUP":
            stack.append(stack[-1])
        elif op == "OVER":
            stack.append(stack[-2])
        elif op == "SWAP":
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == "ROT":
            stack.append(stack.pop(-3))
        elif op == "NOT":
            stack.append(_wrap(~stack.pop()))
        elif op == "OUT":
            outputs.append(stack.pop())
        elif op == "JMP":
            pc = labels[arg]
        elif op == "JZ":
            if stack.pop() == 0:
                pc = labels[arg]
        elif op == "JNEG":
            if stack.pop() < 0:
                pc = labels[arg]
        elif op == "CALL":
            if len(calls) >= MAX_CALL_DEPTH:
                return "CALL_DEPTH", outputs, steps
            calls.append(pc)
            pc = labels[arg]
        elif op == "RET":
            if not calls:
                return "HALT", outputs, steps
            pc = calls.pop()
        elif op == "HALT":
            return "HALT", outputs, steps
        else:
            b = stack.pop()
            a = stack.pop()
            if op in ("DIV", "MOD") and b == 0:
                return "DIV_ZERO", outputs, steps
            stack.append(_wrap(_binary(op, a, b)))
'''


@dataclass(frozen=True)
class VMConfig:
    word_bits: int
    capacity: int
    call_depth: int
    step_limit: int
    div_floor: bool
    shr_arithmetic: bool

    @property
    def lo(self) -> int:
        return -(1 << (self.word_bits - 1))

    @property
    def hi(self) -> int:
        return (1 << (self.word_bits - 1)) - 1


def draw_config(rng: random.Random, hard_index: int) -> VMConfig:
    return VMConfig(
        word_bits=(8, 12, 16)[hard_index % 3],
        capacity=rng.choice((6, 8, 10, 12)),
        call_depth=rng.choice((3, 4, 5, 6)),
        step_limit=rng.choice((150, 200, 250, 300)),
        div_floor=hard_index % 2 == 1,
        shr_arithmetic=(hard_index // 2) % 2 == 0,
    )


def reference_source(cfg: VMConfig) -> str:
    return _apply(
        VM_SOURCE,
        [
            ("WORD_BITS = 12\n", f"WORD_BITS = {cfg.word_bits}\n"),
            ("STACK_CAPACITY = 8\n", f"STACK_CAPACITY = {cfg.capacity}\n"),
            ("MAX_CALL_DEPTH = 4\n", f"MAX_CALL_DEPTH = {cfg.call_depth}\n"),
            ("STEP_LIMIT = 200\n", f"STEP_LIMIT = {cfg.step_limit}\n"),
            ("DIV_FLOOR = False\n", f"DIV_FLOOR = {cfg.div_floor}\n"),
            ("SHR_ARITHMETIC = True\n", f"SHR_ARITHMETIC = {cfg.shr_arithmetic}\n"),
        ],
    )


_RET_FAULTS = ("UNDERFLOW", "NO_INPUT", "OVERFLOW", "DIV_ZERO", "CALL_DEPTH")

# Mutants: (applies(cfg), edits on the configured reference source). Each is one plausible misreading.
VM_MUTANTS: dict[
    str, tuple[Callable[[VMConfig], bool], Callable[[VMConfig], list[tuple[str, str]]]]
] = {
    # Results kept as 64-bit machine integers instead of W-bit values. (A truly unbounded variant can
    # square its way into multi-megabyte integers inside a timed-out loop and stall generation.)
    "results_not_wrapped": (
        lambda c: True,
        lambda c: [
            (
                "stack.append(_wrap(_binary(op, a, b)))",
                "stack.append((_binary(op, a, b) + (1 << 63)) % (1 << 64) - (1 << 63))",
            )
        ],
    ),
    "literals_not_wrapped": (
        lambda c: True,
        lambda c: [("stack.append(_wrap(int(arg)))", "stack.append(int(arg))")],
    ),
    "inputs_not_wrapped": (
        lambda c: True,
        lambda c: [("stack.append(_wrap(queue.pop(0)))", "stack.append(queue.pop(0))")],
    ),
    "unsigned_wrap": (
        lambda c: True,
        lambda c: [("    return (x + _HALF) % _MOD - _HALF\n", "    return x % _MOD\n")],
    ),
    "wrong_word_size": (
        lambda c: True,
        lambda c: [
            (f"WORD_BITS = {c.word_bits}\n", f"WORD_BITS = {16 if c.word_bits != 16 else 12}\n")
        ],
    ),
    "division_rounding_flipped": (
        lambda c: True,
        lambda c: [("        if DIV_FLOOR:\n", "        if not DIV_FLOOR:\n")],
    ),
    "mod_uses_floor_sign": (
        lambda c: not c.div_floor,
        lambda c: [
            (
                '        return q if op == "DIV" else a - b * q\n',
                '        return q if op == "DIV" else a % b\n',
            )
        ],
    ),
    "mod_uses_trunc_sign": (
        lambda c: c.div_floor,
        lambda c: [
            (
                '        return q if op == "DIV" else a - b * q\n',
                '        return q if op == "DIV" else a - b * int(a / b)\n',
            )
        ],
    ),
    "shr_kind_flipped": (
        lambda c: True,
        lambda c: [
            (
                "    return a >> n if SHR_ARITHMETIC else (a % _MOD) >> n\n",
                "    return a >> n if not SHR_ARITHMETIC else (a % _MOD) >> n\n",
            )
        ],
    ),
    "shift_count_abs": (
        lambda c: True,
        lambda c: [("    n = b % WORD_BITS\n", "    n = abs(b) % WORD_BITS\n")],
    ),
    "shift_count_clamped": (
        lambda c: True,
        lambda c: [("    n = b % WORD_BITS\n", "    n = min(max(b, 0), WORD_BITS)\n")],
    ),
    "sub_operands_swapped": (
        lambda c: True,
        lambda c: [("        return a - b\n", "        return b - a\n")],
    ),
    "not_is_logical": (
        lambda c: True,
        lambda c: [("stack.append(_wrap(~stack.pop()))", "stack.append(0 if stack.pop() else 1)")],
    ),
    "rot_reversed": (
        lambda c: True,
        lambda c: [("stack.append(stack.pop(-3))", "stack.insert(-2, stack.pop())")],
    ),
    "jz_peeks": (
        lambda c: True,
        lambda c: [("            if stack.pop() == 0:\n", "            if stack[-1] == 0:\n")],
    ),
    "jneg_includes_zero": (
        lambda c: True,
        lambda c: [("            if stack.pop() < 0:\n", "            if stack.pop() <= 0:\n")],
    ),
    "input_consumed_from_back": (
        lambda c: True,
        lambda c: [("stack.append(_wrap(queue.pop(0)))", "stack.append(_wrap(queue.pop()))")],
    ),
    "capacity_off_by_one": (
        lambda c: True,
        lambda c: [("len(stack) >= STACK_CAPACITY", "len(stack) > STACK_CAPACITY")],
    ),
    "call_depth_off_by_one": (
        lambda c: True,
        lambda c: [("len(calls) >= MAX_CALL_DEPTH", "len(calls) > MAX_CALL_DEPTH")],
    ),
    "timeout_off_by_one": (
        lambda c: True,
        lambda c: [("        if steps == STEP_LIMIT:\n", "        if steps > STEP_LIMIT:\n")],
    ),
    "timeout_checked_before_end": (
        lambda c: True,
        lambda c: [
            (
                '        if pc >= len(code):\n            return "HALT", outputs, steps\n'
                '        if steps == STEP_LIMIT:\n            return "TIMEOUT", outputs, steps\n',
                '        if steps == STEP_LIMIT:\n            return "TIMEOUT", outputs, steps\n'
                '        if pc >= len(code):\n            return "HALT", outputs, steps\n',
            )
        ],
    ),
    "end_of_program_counts_a_step": (
        lambda c: True,
        lambda c: [
            (
                '        if pc >= len(code):\n            return "HALT", outputs, steps\n',
                '        if pc >= len(code):\n            return "HALT", outputs, steps + 1\n',
            )
        ],
    ),
    "faulting_instruction_not_counted": (
        lambda c: True,
        lambda c: [
            (f'return "{f}", outputs, steps\n', f'return "{f}", outputs, steps - 1\n')
            for f in _RET_FAULTS
        ],
    ),
    "halt_not_counted": (
        lambda c: True,
        lambda c: [
            (
                '        elif op == "HALT":\n            return "HALT", outputs, steps\n',
                '        elif op == "HALT":\n            return "HALT", outputs, steps - 1\n',
            )
        ],
    ),
    "top_level_ret_underflows": (
        lambda c: True,
        lambda c: [
            (
                '            if not calls:\n                return "HALT", outputs, steps\n',
                '            if not calls:\n                return "UNDERFLOW", outputs, steps\n',
            )
        ],
    ),
    "overflow_checked_before_no_input": (
        lambda c: True,
        lambda c: [
            (
                '        if op == "IN" and not queue:\n            return "NO_INPUT", outputs, steps\n'
                '        if op in _GROWS and len(stack) >= STACK_CAPACITY:\n            return "OVERFLOW", outputs, steps\n',
                '        if op in _GROWS and len(stack) >= STACK_CAPACITY:\n            return "OVERFLOW", outputs, steps\n'
                '        if op == "IN" and not queue:\n            return "NO_INPUT", outputs, steps\n',
            )
        ],
    ),
    "label_detected_by_colon_anywhere": (
        lambda c: True,
        lambda c: [('        if line.endswith(":"):\n', '        if ":" in raw:\n')],
    ),
}


def load_run(source: str) -> Callable[[list[str], list[int]], tuple[str, list[int], int]]:
    namespace: dict[str, Any] = {"__name__": "toyvm"}
    exec(compile(source, "toyvm.py", "exec"), namespace)  # noqa: S102 - trusted, generated source
    return namespace["run"]


def mutant_sources(cfg: VMConfig) -> dict[str, str]:
    base = reference_source(cfg)
    return {
        name: _apply(base, edits(cfg))
        for name, (applies, edits) in VM_MUTANTS.items()
        if applies(cfg)
    }


# ---------------------------------------------------------------------------
# Test programs.
# ---------------------------------------------------------------------------

Program = tuple[list[str], list[int]]


def _call_chain(depth: int) -> list[str]:
    lines = ["CALL f0", "PUSH 99", "OUT", "HALT"]
    for k in range(depth):
        lines += [f"f{k}:", f"PUSH {k}", "OUT"]
        if k + 1 < depth:
            lines.append(f"CALL f{k + 1}")
        lines.append("RET")
    return lines


def _padding(steps: int) -> list[str]:
    """Exactly `steps` executed, stack-neutral instructions."""
    lines: list[str] = []
    if steps % 2:
        lines += ["PUSH 1", "NOT", "POP"]
        steps -= 3
    lines += ["PUSH 0", "POP"] * (steps // 2)
    return lines


def _exact_limit_program(limit: int, iterations: int, explicit_halt: bool) -> Program:
    """A countdown whose last instruction is executed exactly as step number `limit`."""
    pad = limit - 1 - (5 * iterations - 1) - 1
    assert pad >= 3 or pad in (0, 2), pad
    lines = [
        "IN",
        *_padding(pad),
        "loop:",
        "PUSH -1",
        "ADD",
        "DUP",
        "JZ done",
        "JMP loop",
        "done:",
        "POP",
    ]
    if explicit_halt:
        lines.append("HALT")
    return lines, [iterations]


def _edge_programs(cfg: VMConfig, rng: random.Random) -> list[Program]:
    w, cap, depth, limit, lo, hi = (
        cfg.word_bits,
        cfg.capacity,
        cfg.call_depth,
        cfg.step_limit,
        cfg.lo,
        cfg.hi,
    )
    progs: list[Program] = []
    progs.append(
        (
            [
                f"PUSH {hi + 1}",
                "OUT",
                f"PUSH {lo - 1}",
                "OUT",
                f"PUSH {3 * (hi + 1) + 5}",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    # Out-of-range literals that reach JNEG or DIV/MOD before any wrapping arithmetic could mask them.
    progs.append(
        (
            [
                f"PUSH {hi + 5}",
                "DUP",
                "OUT",
                "JNEG neg",
                "PUSH 1",
                "OUT",
                "HALT",
                "neg:",
                "PUSH -1",
                "OUT",
            ],
            [],
        )
    )
    progs.append(
        (
            [
                f"PUSH {hi + 7}",
                "PUSH 4",
                "DIV",
                "OUT",
                f"PUSH {lo - 6}",
                "PUSH 5",
                "MOD",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append(
        (
            ["IN", "DUP", "OUT", "JNEG neg", "PUSH 1", "OUT", "HALT", "neg:", "PUSH -1", "OUT"],
            [hi + 3],
        )
    )
    progs.append((["IN", "IN", "DIV", "OUT", "IN", "OUT", "HALT"], [lo - 9, 4, 2 * (hi + 1) + 11]))
    progs.append(
        (
            [
                "PUSH 10",
                "PUSH 20",
                "PUSH 30",
                "PUSH 40",
                "ROT",
                "ROT",
                "OUT",
                "OUT",
                "OUT",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append(
        (["PUSH 6 ; ratio: 2:1", "PUSH 3", "DIV", "OUT", "end: ; label with a comment", "HALT"], [])
    )
    for a, b in [
        (-7, 2),
        (7, -2),
        (-7, -2),
        (lo, -1),
        (lo + 1, 3),
        (rng.randint(lo, -10), rng.randint(3, 9)),
    ]:
        progs.append(
            (
                [
                    f"PUSH {a}",
                    f"PUSH {b}",
                    "DIV",
                    "OUT",
                    f"PUSH {a}",
                    f"PUSH {b}",
                    "MOD",
                    "OUT",
                    "HALT",
                ],
                [],
            )
        )
    for a, k in [
        (-16, 2),
        (lo, 1),
        (1, w),
        (1, -1),
        (-1, w + 1),
        (5, -w),
        (5, -3),
        (-100, -w - 2),
        (hi, w - 1),
        (-3, 0),
    ]:
        progs.append(
            (
                [
                    f"PUSH {a}",
                    f"PUSH {k}",
                    "SHL",
                    "OUT",
                    f"PUSH {a}",
                    f"PUSH {k}",
                    "SHR",
                    "OUT",
                    "HALT",
                ],
                [],
            )
        )
    progs.append(([f"PUSH {i}" for i in range(cap)] + ["OUT"] * cap + ["HALT"], []))
    progs.append(([f"PUSH {i}" for i in range(cap)] + ["DUP", "OUT", "HALT"], []))
    progs.append(([f"PUSH {i + 1}" for i in range(cap - 1)] + ["OVER", "OUT", "OVER", "OUT"], []))
    progs.append((_call_chain(depth), []))
    progs.append((_call_chain(depth + 1), []))
    recursion = [
        "IN",
        "CALL rec",
        "OUT",
        "HALT",
        "rec:",
        "DUP",
        "JZ base",
        "PUSH -1",
        "ADD",
        "CALL rec",
        "PUSH 2",
        "ADD",
        "RET",
        "base:",
        "RET",
    ]
    progs.append((recursion, [depth - 1]))
    progs.append((recursion, [depth]))
    progs.append((["PUSH 4", "PUSH 5", "MUL", "OUT", "RET", "PUSH 1", "OUT"], []))
    progs.append((["CALL f", "f:", "PUSH 1", "OUT", "RET"], []))
    progs.append(([f"PUSH {i}" for i in range(cap)] + ["IN"], []))
    progs.append(([f"PUSH {i}" for i in range(cap)] + ["IN"], [7]))
    progs.append((["IN"] + ["DUP"] * (cap - 1) + ["OUT", "IN", "IN"], [3, -4]))
    progs.append((["IN", "OUT", "IN", "OUT", "IN"], [rng.randint(1, 50), -rng.randint(1, 50)]))
    progs.append((["PUSH 1", "PUSH 2", "ROT"], []))
    progs.append((["PUSH 1", "OUT", "OUT"], []))
    progs.append((["PUSH 3", "JZ end", "JZ end", "end:"], []))
    progs.append((["PUSH 1", "JMP end", "PUSH 2", "end:"], []))
    progs.append(
        (
            [
                "; countdown with comments",
                "  PUSH 3   ; start: 3",
                "loop: ; top of loop",
                "again:",
                "\tDUP",
                "\tOUT",
                "  PUSH\t-1",
                "  ADD",
                "  DUP",
                "  JNEG done ; exit when negative",
                "  JMP loop",
                "",
                "done:",
                "  POP",
                "  HALT ; bye",
            ],
            [],
        )
    )
    progs.append(
        (
            [
                "PUSH 0",
                "NOT",
                "OUT",
                "PUSH -1",
                "NOT",
                "OUT",
                f"PUSH {lo}",
                "NOT",
                "OUT",
                "PUSH 5",
                "NOT",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append(
        (
            [
                "PUSH -6",
                "PUSH 11",
                "AND",
                "OUT",
                "PUSH -6",
                "PUSH 11",
                "OR",
                "OUT",
                "PUSH -6",
                "PUSH 11",
                "XOR",
                "OUT",
                f"PUSH {lo}",
                "PUSH -1",
                "XOR",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append(
        (
            [
                f"PUSH {hi}",
                f"PUSH {hi}",
                "MUL",
                "OUT",
                f"PUSH {lo}",
                "PUSH -1",
                "MUL",
                "OUT",
                f"PUSH {hi}",
                "PUSH 1",
                "ADD",
                "OUT",
                f"PUSH {lo}",
                "PUSH 1",
                "SUB",
                "OUT",
                "PUSH 3",
                "PUSH 10",
                "SUB",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append((["PUSH 5", "PUSH 0", "JZ skip", "PUSH 7", "skip:", "OUT", "HALT"], []))
    progs.append((["PUSH 9", "PUSH 0", "JNEG skip", "PUSH 7", "skip:", "OUT", "OUT", "HALT"], []))
    progs.append(
        (
            [
                "PUSH 1",
                "PUSH 2",
                "PUSH 3",
                "ROT",
                "OUT",
                "OUT",
                "OUT",
                "PUSH 4",
                "PUSH 5",
                "OVER",
                "OUT",
                "OUT",
                "OUT",
                "PUSH 6",
                "PUSH 7",
                "SWAP",
                "OUT",
                "OUT",
                "HALT",
            ],
            [],
        )
    )
    progs.append(
        (["IN", "loop:", "PUSH 3", "ADD", "DUP", "OUT", "JMP loop"], [rng.randint(-20, 20)])
    )
    progs.append((["loop:", "JMP loop"], []))
    progs.append((["PUSH 9", "OUT", "PUSH 1", "PUSH 0", "MOD", "PUSH 2", "OUT"], []))
    gcd = [
        "IN",
        "IN",
        "loop:",
        "DUP",
        "JZ done",
        "SWAP",
        "OVER",
        "MOD",
        "JMP loop",
        "done:",
        "POP",
        "OUT",
        "HALT",
    ]
    progs.extend((gcd, list(pair)) for pair in [(-12, 18), (35, -21), (-40, -64)])
    base_iters = (limit - 12) // 5
    exact, _ = _exact_limit_program(limit, base_iters, explicit_halt=False)
    progs.append((exact, [base_iters]))
    progs.append((exact, [base_iters + 1]))
    progs.append((exact, [base_iters - 1]))
    progs.append(_exact_limit_program(limit, base_iters - 1, explicit_halt=False))
    progs.append(_exact_limit_program(limit, base_iters, explicit_halt=True))
    return progs


def _literal(rng: random.Random, cfg: VMConfig) -> int:
    lo, hi = cfg.lo, cfg.hi
    kind = rng.randrange(8)
    if kind == 0:
        return rng.choice([lo, hi, -1, 0, 1])
    if kind == 1:
        return rng.randint(hi + 1, 4 * hi)
    if kind == 2:
        return -rng.randint(hi + 2, 4 * hi)
    if kind in (3, 4):
        return rng.randint(-20, 20)
    return rng.randint(lo, hi)


_BINARY_OPS = ("ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR")


def _straight_line(rng: random.Random, cfg: VMConfig) -> Program:
    lines: list[str] = []
    inputs: list[int] = []
    depth = 0
    for _ in range(rng.randint(10, 22)):
        options: list[str] = []
        if depth < cfg.capacity - 1:
            options += ["PUSH", "PUSH", "IN"]
        if depth >= 1:
            options += ["DUP", "NOT", "OUT"] if depth < cfg.capacity else ["NOT", "OUT"]
        if depth >= 2:
            options += (
                list(_BINARY_OPS) * 2 + ["SWAP", "POP"] + (["OVER"] if depth < cfg.capacity else [])
            )
        if depth >= 3:
            options.append("ROT")
        op = rng.choice(options)
        if op == "PUSH":
            lines.append(f"PUSH {_literal(rng, cfg)}")
            depth += 1
        elif op == "IN":
            inputs.append(_literal(rng, cfg))
            lines.append("IN")
            depth += 1
        elif op in ("SHL", "SHR") and depth < cfg.capacity:
            lines += [
                f"PUSH {rng.choice([0, 1, 2, 3, cfg.word_bits - 1, cfg.word_bits, cfg.word_bits + 2, -1, -2])}",
                op,
            ]
        elif op in ("DIV", "MOD") and depth < cfg.capacity and rng.random() < 0.7:
            lines += [f"PUSH {rng.choice([-1, 1]) * rng.randint(1, 9)}", op]
        elif op in _BINARY_OPS:
            lines.append(op)
            depth -= 1
        elif op in ("DUP", "OVER"):
            lines.append(op)
            depth += 1
        elif op in ("OUT", "POP"):
            lines.append(op)
            depth -= 1
        else:
            lines.append(op)
    lines += ["OUT"] * min(depth, rng.randint(1, 3))
    ending = rng.random()
    if ending < 0.5:
        lines.append("HALT")
    elif ending < 0.75:
        lines.append("RET")
    if rng.random() < 0.3:
        inputs.append(_literal(rng, cfg))
    return lines, inputs


def _acc_body(rng: random.Random, cfg: VMConfig) -> list[str]:
    """Stack-neutral transformation of the top value."""
    body: list[str] = []
    for _ in range(rng.randint(1, 3)):
        kind = rng.randrange(6)
        if kind == 0:
            body += [f"PUSH {rng.randint(-9, 9)}", rng.choice(["ADD", "SUB", "XOR", "MUL"])]
        elif kind == 1:
            body += ["DUP", "MUL"]
        elif kind == 2:
            body += [
                f"PUSH {rng.choice([1, 2, 3, -1, cfg.word_bits + 1])}",
                rng.choice(["SHL", "SHR"]),
            ]
        elif kind == 3:
            body += [f"PUSH {rng.choice([-1, 1]) * rng.randint(2, 7)}", rng.choice(["DIV", "MOD"])]
        elif kind == 4:
            body.append("NOT")
        else:
            body += [f"PUSH {rng.randint(1, 3 * cfg.hi)}", "ADD"]
    return body


def _loop_program(rng: random.Random, cfg: VMConfig) -> Program:
    exit_op = rng.choice(["JZ", "JNEG"])
    lines = [
        "IN",
        "IN",
        "loop:",
        "SWAP",
        "DUP",
        f"{exit_op} done",
        "PUSH -1",
        "ADD",
        "SWAP",
        *_acc_body(rng, cfg),
        "DUP",
        "OUT",
        "JMP loop",
        "done:",
        "POP",
        "OUT",
        "HALT",
    ]
    count = rng.choice([0, 1, 2, rng.randint(3, 9), rng.randint(3, 9), -rng.randint(1, 3)])
    return lines, [count, _literal(rng, cfg)]


def _call_program(rng: random.Random, cfg: VMConfig) -> Program:
    lines = [
        "IN",
        "CALL f",
        "OUT",
        "IN",
        "CALL f",
        "OUT",
        rng.choice(["HALT", "RET", "JMP end"]),
        "f:",
        *_acc_body(rng, cfg),
        "CALL g",
        "RET",
        "g:",
        *_acc_body(rng, cfg),
        "RET",
        "end:",
    ]
    return lines, [_literal(rng, cfg), _literal(rng, cfg)]


def _fault_probe(rng: random.Random, cfg: VMConfig) -> Program:
    ops = [
        "PUSH",
        "PUSH",
        "PUSH",
        "IN",
        "DUP",
        "OVER",
        "POP",
        "OUT",
        "OUT",
        "SWAP",
        "ROT",
        "NOT",
        *_BINARY_OPS,
    ]
    lines = []
    for _ in range(rng.randint(4, 3 * cfg.capacity)):
        op = rng.choice(ops)
        lines.append(f"PUSH {_literal(rng, cfg)}" if op == "PUSH" else op)
    return lines, [_literal(rng, cfg) for _ in range(rng.randint(0, 3))]


def build_programs(rng: random.Random, cfg: VMConfig) -> list[Program]:
    progs = _edge_programs(cfg, rng)
    progs += [_straight_line(rng, cfg) for _ in range(22)]
    progs += [_loop_program(rng, cfg) for _ in range(12)]
    progs += [_call_program(rng, cfg) for _ in range(6)]
    progs += [_fault_probe(rng, cfg) for _ in range(12)]
    return progs


def _safe(fn: Callable[[list[str], list[int]], Any], prog: Program) -> Any:
    try:
        status, outputs, steps = fn(list(prog[0]), list(prog[1]))
        return (status, list(outputs), steps)
    except Exception as exc:  # noqa: BLE001 - mutants may crash; that is a detection
        return f"raised {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Prompt.
# ---------------------------------------------------------------------------


def _spec_text(cfg: VMConfig) -> str:
    w, lo, hi = cfg.word_bits, cfg.lo, cfg.hi
    if cfg.div_floor:
        div_rule = "q = floor(a / b) (rounded toward negative infinity) and r = a - b*q (so r is 0 or has the sign of b)"
    else:
        div_rule = (
            "q = a / b truncated toward zero and r = a - b*q (so r is 0 or has the sign of a)"
        )
    if cfg.shr_arithmetic:
        shr_rule = (
            f"`a b -- floor(a / 2^n)` (arithmetic shift: the sign bit is replicated), n = b mod {w}"
        )
    else:
        shr_rule = (
            f"`a b -- s(floor(u / 2^n))` (logical shift: zeros enter from the left), n = b mod {w}, where "
            f"u = a mod 2^{w} is the unsigned {w}-bit pattern of a and s() reinterprets a {w}-bit pattern as signed"
        )
    return f"""# TOYVM-{w}

## Values
Every value on the data stack is a {w}-bit two's-complement signed integer in [{lo}, {hi}].
**wrap(x)** maps a mathematical integer x to the unique value in that range congruent to x modulo 2^{w}.
PUSH literals, IN values, and the result of every arithmetic and bitwise instruction are wrapped.
"x mod m" below is mathematical modulo: the result is always in 0..m-1, also for negative x.

## Program text
`program` is a list of lines. On each line, everything from the first `;` onward is a comment; then
leading and trailing whitespace is removed. Lines that are now empty are ignored. A line ending in `:`
defines a label (the name before the `:`) that refers to the next instruction, or to the end of the
program if no instruction follows. Every other line is one instruction: an upper-case mnemonic,
optionally followed by whitespace and a single operand. Programs are always well-formed: required
operands are present, every referenced label is defined exactly once, and PUSH operands are decimal
integer literals (possibly negative, possibly outside the value range).

## Machine state
- a data stack holding at most {cfg.capacity} values;
- a separate return stack holding at most {cfg.call_depth} return addresses;
- the input queue (`inputs`, consumed from the front) and the output list;
- the program counter (starts at the first instruction) and a step counter (starts at 0).

## Instructions
Notation: `a b -- r` means pop b (the top value), then pop a, then push r.

| Instruction | Effect |
|---|---|
| `PUSH n` | push wrap(n) |
| `POP` | `a --` |
| `DUP` | `a -- a a` |
| `SWAP` | `a b -- b a` |
| `OVER` | `a b -- a b a` |
| `ROT` | `a b c -- b c a` |
| `ADD`, `SUB`, `MUL` | `a b -- wrap(a + b)`, `wrap(a - b)`, `wrap(a * b)` |
| `DIV`, `MOD` | `a b -- wrap(q)` and `a b -- wrap(r)`, where {div_rule} |
| `AND`, `OR`, `XOR` | bitwise operation on the {w}-bit two's-complement patterns of a and b |
| `NOT` | `a -- wrap(-a - 1)` (bitwise complement) |
| `SHL` | `a b -- wrap(a * 2^n)`, n = b mod {w} |
| `SHR` | {shr_rule} |
| `JMP L` | continue at label L |
| `JZ L` | `a --`, then continue at label L if a == 0 |
| `JNEG L` | `a --`, then continue at label L if a < 0 |
| `CALL L` | push the address of the instruction after the CALL onto the return stack; continue at L |
| `RET` | pop an address from the return stack and continue there; if the return stack is empty, the program ends with status `HALT` |
| `IN` | remove the first value x from the input queue and push wrap(x) |
| `OUT` | `a --`, then append a to the outputs |
| `HALT` | the program ends with status `HALT` |

## Execution
Before each instruction:
1. If the program counter is past the last instruction, the program ends with status `HALT`.
2. Otherwise, if {cfg.step_limit} instructions have already been executed, the program ends with status `TIMEOUT`.

Otherwise the instruction executes and counts as one step, including an instruction that faults or
ends the program. An instruction faults, ending the program immediately with the fault's name as its
status, under these conditions (checked in this order):
1. `UNDERFLOW`: the data stack holds fewer values than the instruction pops or reads (POP, DUP, NOT,
   JZ, JNEG, OUT need 1; SWAP, OVER and every two-operand arithmetic/bitwise instruction need 2; ROT needs 3).
2. `NO_INPUT`: IN when the input queue is empty.
3. `OVERFLOW`: PUSH, DUP, OVER, or IN when the data stack already holds {cfg.capacity} values.
4. `DIV_ZERO`: DIV or MOD with b == 0.
5. `CALL_DEPTH`: CALL when the return stack already holds {cfg.call_depth} addresses.

## Result
`run` returns `(status, outputs, steps)`: status is one of `HALT`, `TIMEOUT`, `UNDERFLOW`, `NO_INPUT`,
`OVERFLOW`, `DIV_ZERO`, `CALL_DEPTH`; outputs is the list of all values output before the program
ended (in order); steps is the number of instructions executed.

Examples (valid for every TOYVM variant):
- `run(["PUSH 2", "PUSH 3", "ADD", "OUT"], [])` returns `("HALT", [5], 4)`
- `run(["IN", "OUT", "IN"], [7])` returns `("NO_INPUT", [7], 3)`
"""


def build_vm_item(
    rng: random.Random, item_id: str, hard_index: int
) -> tuple[dict[str, Any], VMConfig]:
    """Build one TOYVM spec item and verify the differential suite against `VM_MUTANTS`."""
    cfg = draw_config(rng, hard_index)
    ref = load_run(reference_source(cfg))
    programs = build_programs(rng, cfg)
    expected = [_safe(ref, prog) for prog in programs]
    if any(isinstance(e, str) for e in expected):
        raise AssertionError("reference interpreter crashed on a generated program")
    statuses = {e[0] for e in expected}
    missing = {
        "HALT",
        "TIMEOUT",
        "UNDERFLOW",
        "NO_INPUT",
        "OVERFLOW",
        "DIV_ZERO",
        "CALL_DEPTH",
    } - statuses
    if missing:
        raise AssertionError(f"suite does not exercise statuses {sorted(missing)}")
    for name, source in mutant_sources(cfg).items():
        mutant = load_run(source)
        hits = sum(
            1 for prog, exp in zip(programs, expected, strict=True) if _safe(mutant, prog) != exp
        )
        if hits < MIN_DETECTIONS:
            raise AssertionError(f"mutant {name} is detected by only {hits} programs ({cfg})")

    prompt = (
        "Implement the TOYVM stack machine specified below as a Python function\n\n"
        "```python\ndef run(program: list[str], inputs: list[int]) -> tuple[str, list[int], int]:\n```\n\n"
        "Follow the specification exactly. Hidden tests run many programs and compare all three returned "
        "values (status, outputs, steps).\n\n"
        + _spec_text(cfg)
        + "\nReturn only valid Python code (standard library only) defining `run`."
    )
    cases_literal = [(prog[0], prog[1], exp) for prog, exp in zip(programs, expected, strict=True)]
    test_code = f"""
_GDR_CASES = {cases_literal!r}
_gdr_failures = []
for _gdr_i, (_gdr_prog, _gdr_inp, _gdr_exp) in enumerate(_GDR_CASES):
    try:
        _gdr_res = run(list(_gdr_prog), list(_gdr_inp))
        _gdr_got = (_gdr_res[0], list(_gdr_res[1]), _gdr_res[2])
    except Exception as _gdr_exc:
        _gdr_got = "raised " + type(_gdr_exc).__name__ + ": " + str(_gdr_exc)[:80]
    if _gdr_got != tuple(_gdr_exp):
        _gdr_failures.append(f"case {{_gdr_i}} program={{_gdr_prog!r}} inputs={{_gdr_inp!r}}: "
                             f"expected {{tuple(_gdr_exp)!r}}, got {{_gdr_got!r}}")
assert not _gdr_failures, f"{{len(_gdr_failures)}}/{{len(_GDR_CASES)}} programs failed; first: " + " | ".join(
    _gdr_f[:400] for _gdr_f in _gdr_failures[:2]
)
"""
    item = {
        "id": item_id,
        "slice": "s3_frontier_reasoning",
        "modality": "code",
        "subtype": "vm_spec_interpreter",
        "source_provenance": (
            f"original TOYVM-{cfg.word_bits} spec (capacity {cfg.capacity}, call depth {cfg.call_depth}, step limit "
            f"{cfg.step_limit}, {'floor' if cfg.div_floor else 'trunc'} division, "
            f"{'arithmetic' if cfg.shr_arithmetic else 'logical'} SHR); differential suite of {len(programs)} "
            "programs vs. a reference interpreter"
        ),
        "prompt": prompt,
        "attachments": [],
        "grader_type": "python_unit_tests",
        "grader_spec": {"test_code": test_code, "timeout_sec": 15.0},
    }
    return item, cfg


def make_vm_item(rng: random.Random, item_id: str, hard_index: int) -> dict[str, Any]:
    return build_vm_item(rng, item_id, hard_index)[0]
