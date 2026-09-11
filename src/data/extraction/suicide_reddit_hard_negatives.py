"""High-recall hard-negative-like routing for the Suicide Reddit Dataset.

Every output row remains an unlabelled annotation candidate.  The heuristics
only describe why a row was selected for human review; they never infer a
project label.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.cleaning import dedup
from src.data.pooling import lexicon

_REASONS: tuple[tuple[str, re.Pattern], ...] = (
    ("prevention", re.compile(r"\b(prevention|prevent|awareness|hotline|helpline|crisis line|reach out|support someone)\b", re.I)),
    ("research", re.compile(r"\b(statistic|statistics|study|studies|research|survey|rate[s]?|data|report)\b", re.I)),
    ("news", re.compile(r"\b(news|article|journalist|reported|reporting|headline|press|documentary)\b", re.I)),
    ("historical", re.compile(r"\b(history|historical|years? ago|decades? ago|anniversary|memorial)\b", re.I)),
    ("quote_or_fiction", re.compile(r"\b(quote|quoted|lyrics?|song|movie|film|book|novel|fiction|character|game)\b", re.I)),
    ("third_party", re.compile(r"\b(my (friend|brother|sister|son|daughter|partner|wife|husband)|someone|somebody|he|she|they|their)\b", re.I)),
    ("hyperbole", re.compile(r"\b(kill myself|kms|end my life)\b.{0,70}\b(game|exam|homework|work|traffic|queue|lag|computer)\b|\b(game|exam|homework|work|traffic|queue|lag|computer)\b.{0,70}\b(kill myself|kms|end my life)\b", re.I)),
)

_SELF_RISK = re.compile(
    r"\b(?:i(?:\s+am|'m)?|im|me|my|myself)\b.{0,55}\b(?:want(?:\s+to)?|going\s+to|gonna|plan(?:ning)?\s+to|think(?:ing)?\s+about|feel(?:ing)?\s+like|wish(?:ed)?\s+to|need\s+to|tried\s+to|attempt(?:ed)?|overdos(?:e|ed))\b.{0,55}\b(?:suicid|kill myself|end my life|take my own life|end it all|off myself|hang myself|overdose)\b|\b(?:suicid|kill myself|end my life|take my own life|end it all|off myself|hang myself|overdose)\b.{0,55}\b(?:tonight|today|now|right now)\b",
    re.I,
)


def candidate_reason(text: str) -> str:
    """Return an auditable routing reason, not a label."""
    for name, pattern in _REASONS:
        if pattern.search(str(text)):
            return name
    return "crisis_term_context"


def is_obvious_self_risk(text: str) -> bool:
    """Conservatively identify obvious first-person current-risk language."""
    return bool(_SELF_RISK.search(str(text)))


def _existing_hashes(*frames: pd.DataFrame) -> set[str]:
    hashes: set[str] = set()
    for frame in frames:
        if frame is not None and "text" in frame:
            hashes.update(frame["text"].dropna().map(dedup.text_hash))
    return hashes


def extract_candidates(
    records: pd.DataFrame,
    existing_pool: pd.DataFrame | None = None,
    existing_candidates: pd.DataFrame | None = None,
    terms: tuple[str, ...] | None = None,
    limit: int = 250,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean, exact-deduplicate, route and cap unlabelled candidates.

    Existing pool/candidate matches are removed by the project's normalised
    text hash.  Selection is deterministic: reason priority then source/id.
    """
    from src.data.cleaning import text_cleaning

    stats = {"raw_rows": len(records)}
    cleaned = text_cleaning.clean_frame(records)
    stats["usable_text_rows"] = len(cleaned)
    local = dedup.drop_exact_duplicates(cleaned)
    stats["within_source_duplicates_removed"] = len(cleaned) - len(local)
    existing = _existing_hashes(existing_pool, existing_candidates)
    local["_hash"] = local["text"].map(dedup.text_hash)
    unseen = local[~local["_hash"].isin(existing)].copy()
    stats["existing_duplicates_removed"] = len(local) - len(unseen)

    annotated = lexicon.annotate_frame(unseen, terms=terms)
    vocab = annotated[annotated["explicit_lex"]].copy()
    stats["crisis_vocabulary_rows"] = len(vocab)
    vocab["_obvious_self_risk"] = vocab["text"].map(is_obvious_self_risk)
    stats["obvious_self_risk_excluded"] = int(vocab["_obvious_self_risk"].sum())
    routed = vocab[~vocab["_obvious_self_risk"]].copy()
    routed["matched_terms"] = routed["explicit_terms"]
    routed["candidate_reason"] = routed["text"].map(candidate_reason)
    # ``src_id`` is the internal pipeline spelling; retain the more explicit
    # alias in this hand-off file for external/manual annotation workflows.
    routed["source_id"] = routed["src_id"]
    priority = {name: i for i, (name, _) in enumerate(_REASONS)} | {"crisis_term_context": len(_REASONS)}
    routed["_priority"] = routed["candidate_reason"].map(priority)
    routed = routed.sort_values(["_priority", "source_label", "src_id"], kind="stable").head(limit)
    stats["selected_candidates"] = len(routed)
    stats["remaining_after_deduplication"] = len(unseen)
    columns = [
        "text", "source", "source_id", "src_id", "source_label", "src_meta",
        "matched_terms", "candidate_reason",
    ]
    return routed[columns].reset_index(drop=True), stats
