"""Per-slice dataset item generators used by `evals.build_dataset`.

- `s1_trivial`: JSON extraction, IFEval formatting constraints, and ISO-8601 date normalization.
- `s2_specs`: single-function specs graded by hidden differential test suites.
- `s3_repair`: symptom-only multi-bug repair of a realistic module (SWE-style).
- `s3_timeline`: long-context operations-timeline state tracking with corrections.
- `s3_vm`: long-spec stack-VM interpreter graded on generated programs.
- `s4_vision`: perception-gated diagrams, unlabeled charts, and scanned PDFs.
- `s5_dialogue`: multi-speaker spoken handoff calls with corrections and authority rules.
"""

from evals.gen._schema import DatasetItem, Modality, SliceName, item_rng

__all__ = ["DatasetItem", "Modality", "SliceName", "item_rng"]
