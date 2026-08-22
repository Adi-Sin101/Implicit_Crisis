"""The project's label and stratum vocabularies.

The two vocabularies are deliberately kept apart:

* ``STRATA``  — sampling buckets used to choose what a human should read.
* ``LABELS``  — the four gold categories, assigned by human annotation only.

Nothing in this codebase may map a stratum onto a label.
"""
from __future__ import annotations

LABELS: tuple[str, ...] = (
    "explicit_crisis",
    "implicit_crisis",
    "hard_negative",
    "non_crisis",
)

STRATA: tuple[str, ...] = (
    "A_likely_explicit",
    "B_likely_implicit",
    "C_hard_negative",
    "D_likely_noncrisis",
)

#: Slices reported separately in the evaluation (Section 7 of the report).
EVALUATION_SLICES: tuple[str, ...] = (
    "explicit_crisis",
    "implicit_crisis",
    "hard_negative",
)

#: Columns present in the internal candidate pool.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "source",
    "src_risk",
    "src_meta",
    "stratum",
    "explicit_lex",
    "n_words",
)

#: Columns an annotator is allowed to see. Everything else is withheld to
#: avoid anchoring the annotator on a source label or on the lexicon hit.
ANNOTATOR_COLUMNS: tuple[str, ...] = ("id", "text", "label", "confidence", "notes")
