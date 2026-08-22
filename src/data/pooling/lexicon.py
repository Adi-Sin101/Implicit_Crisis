"""Explicit crisis-terminology detector.

This detector exists for ONE purpose: routing candidates into sampling strata
so that annotators see a useful mix of explicit, implicit and hard-negative
material. It is a lexical heuristic, and by construction it cannot tell
"I want to kill myself" from "the article was about suicide prevention".

It must never be used to assign a gold label, and it is never a model feature.
"""
from __future__ import annotations

import re

from src.utils.config import load_config

_DEFAULT_TERMS = (
    "suicid",
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "take my own life",
    "taking my own life",
    "end it all",
    "kms",
    "unalive",
    "off myself",
    "hang myself",
    "overdose on",
)


def load_terms(config: dict | None = None) -> tuple[str, ...]:
    cfg = config or load_config("candidate_pool.yaml")
    terms = cfg.get("explicit_lexicon") or list(_DEFAULT_TERMS)
    return tuple(t.lower() for t in terms)


def _compile(terms) -> re.Pattern:
    # Word-boundary anchored on the left so 'kms' does not fire inside 'kmsomething'.
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)


def matches(text: str, terms=None) -> list[str]:
    """Return the explicit terms present in ``text`` (possibly empty)."""
    terms = terms or load_terms()
    pattern = _compile(terms)
    return sorted({m.group(0).lower() for m in pattern.finditer(str(text))})


def has_explicit_term(text: str, terms=None) -> bool:
    return bool(matches(text, terms))


def annotate_frame(df, text_col: str = "text", terms=None):
    """Attach a boolean ``explicit_lex`` column and the matched terms."""
    terms = terms or load_terms()
    pattern = _compile(terms)
    out = df.copy()
    found = out[text_col].map(
        lambda t: sorted({m.group(0).lower() for m in pattern.finditer(str(t))})
    )
    out["explicit_lex"] = found.map(bool)
    out["explicit_terms"] = found.map(lambda xs: ";".join(xs))
    return out
