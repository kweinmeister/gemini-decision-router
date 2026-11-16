"""Command-line interface (`gdr`) for the Gemini Decision Router."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, get_args

from gemini_decision_router.envelope import extract_interaction_text, load_media_block
from gemini_decision_router.ladder import LADDER_ENV_VAR, resolve_ladder
from gemini_decision_router.router import DecisionRouter, PolicyMode, RouterPolicyConfig


def _conformal_threshold_arg(raw: str) -> float:
    try:
        val = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected float in (0, 1], got {raw!r}") from exc
    if not (0.0 < val <= 1.0 and math.isfinite(val)):
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {raw!r}")
    return val


def add_tiers_argument(parser: argparse.ArgumentParser) -> None:
    """Add the `--tiers PATH` option."""
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help=(
            "Path to a TOML tier ladder (models, thinking levels, compute weights). "
            f"Default: ${LADDER_ENV_VAR} if set, else the bundled default_tiers.toml."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdr",
        description="Route multimodal prompts across a configurable tier ladder using gemini-decision-1.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    route_p = sub.add_parser(
        "route", help="Route a prompt (and optional attachments) to an optimal tier."
    )
    route_p.add_argument("prompt", type=str, help="Prompt text to route.")
    route_p.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Path to an image, audio, video, or PDF attachment (must reside inside --allowed-dir).",
    )
    route_p.add_argument(
        "--allowed-dir",
        type=str,
        default=".",
        help="Root directory within which --attach paths are permitted (default: current directory).",
    )
    route_p.add_argument(
        "--policy",
        choices=get_args(PolicyMode),
        default="score_tail",
        help="Routing decision policy (default: score_tail).",
    )
    route_p.add_argument(
        "--conformal-cdf-threshold",
        type=_conformal_threshold_arg,
        default=None,
        help="Optional calibrated CDF threshold in (0, 1] for --policy choice_crc.",
    )
    route_p.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable low-res image and page-1 PDF compression on the router pass.",
    )
    route_p.add_argument(
        "--execute",
        action="store_true",
        help="Also execute the request against the selected downstream model via Interactions API.",
    )
    add_tiers_argument(route_p)

    tiers_p = sub.add_parser("tiers", help="Validate and print the resolved tier ladder as JSON.")
    add_tiers_argument(tiers_p)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "route":
        allowed_root = Path(args.allowed_dir).resolve()
        if args.attach:
            blocks: list[dict[str, Any]] = [
                load_media_block(att, allowed_root) for att in args.attach
            ]
            blocks.append({"type": "text", "text": args.prompt})
            payload: str | list[dict[str, Any]] = blocks
        else:
            payload = args.prompt

        ladder = resolve_ladder(args.tiers)
        cfg = RouterPolicyConfig(
            policy=args.policy,
            compress_media_for_router=not args.no_compress,
            conformal_cdf_threshold=args.conformal_cdf_threshold,
            ladder=ladder,
        )
        router = DecisionRouter(config=cfg)

        if args.execute:
            decision, interaction = router.route_and_execute(payload)
            result = {
                "decision": dataclasses.asdict(decision),
                "downstream_output": extract_interaction_text(interaction),
            }
        else:
            decision = router.route(payload)
            result = dataclasses.asdict(decision)

        print(json.dumps(result, indent=2))
        return 0

    if args.command == "tiers":
        print(json.dumps(resolve_ladder(args.tiers).to_summary(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
