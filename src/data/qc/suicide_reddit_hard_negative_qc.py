"""Conservative QC routing for Suicide Reddit hard-negative-like candidates.

The statuses produced here are selection-quality findings, not project gold
labels.  In particular, ``obvious_current_self_crisis`` means the candidate
does not belong in this *hard-negative-oriented* hand-off pool; it does not
assign the record any of the project's four annotation labels.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.cleaning import dedup

_CRISIS = r"(?:suicid\w*|kill(?:ing)? myself|end(?:ing)? my life|take my own life|end it all|kms|unalive|off myself|hang myself|overdos\w*)"
_CURRENT_SELF_RISK = re.compile(
    rf"\b(?:i|i'm|im|i am|myself)\b.{{0,100}}\b(?:want(?:\s+to)?|plan(?:ning)?(?:\s+to|\s+on)?|going\s+to|gonna|think(?:ing)?\s+about|contemplat(?:e|ing)|feel(?:ing)?|am|attempt(?:ed|ing)?|tr(?:y|ied)(?:ing)?\s+to|wish(?:ed)?\s+(?:i|to))\b.{{0,70}}{_CRISIS}|\b(?:i(?:'m| am)?|im)\s+(?:currently\s+)?(?:suicidal|at risk)\b|\b(?:i|i'm|im|i am|myself)\b.{{0,40}}(?:want to die|wish (?:i (?:was|were)|to be) dead|no reason to live|no point in living|can't go on|cannot go on)\b|\b(?:i|i'm|im|i am|myself)\b.{{0,60}}\b(?:kill myself|end my life|take my own life|hang myself|off myself)\b",
    re.I | re.S,
)
_FIRST_PERSON_CRISIS_CONTEXT = re.compile(
    rf"\b(?:i|i'm|im|i am|i've|ive|me|myself)\b.{{0,100}}{_CRISIS}|{_CRISIS}.{{0,100}}\b(?:i|i'm|im|i am|i've|ive|me|myself)\b",
    re.I | re.S,
)

_CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("prevention", re.compile(r"\b(suicide prevention|prevention (?:campaign|resource|program)|hotline|helpline|crisis line)\b", re.I)),
    ("research", re.compile(rf"\b(?:statistics?|research|study|studies|survey|rates?|data|report)\b.{{0,80}}{_CRISIS}|{_CRISIS}.{{0,80}}\b(?:statistics?|research|study|studies|survey|rates?|data|report)\b", re.I | re.S)),
    ("news", re.compile(rf"\b(?:news|article|journalist|reported|reporting|headline|press|documentary)\b.{{0,80}}{_CRISIS}|{_CRISIS}.{{0,80}}\b(?:news|article|journalist|reported|reporting|headline|press|documentary)\b", re.I | re.S)),
    ("historical", re.compile(rf"\b(?:history|historical|years? ago|decades? ago|anniversary|memorial)\b.{{0,100}}{_CRISIS}|{_CRISIS}.{{0,100}}\b(?:history|historical|years? ago|decades? ago|anniversary|memorial)\b", re.I | re.S)),
    ("quote_or_fiction", re.compile(rf"\b(?:quote|quoted|lyrics?|song|movie|film|book|novel|fiction|character|game)\b.{{0,100}}{_CRISIS}|{_CRISIS}.{{0,100}}\b(?:quote|quoted|lyrics?|song|movie|film|book|novel|fiction|character|game)\b", re.I | re.S)),
    ("third_party", re.compile(rf"\b(?:friend|brother|sister|son|daughter|partner|wife|husband|someone|somebody|he|she|they|their)\b.{{0,100}}{_CRISIS}|{_CRISIS}.{{0,100}}\b(?:friend|brother|sister|son|daughter|partner|wife|husband|someone|somebody|he|she|they|their)\b", re.I | re.S)),
    ("hyperbole", re.compile(r"\b(?:kill myself|kms|end my life)\b.{0,80}\b(?:game|exam|homework|work|traffic|queue|lag|computer)\b|\b(?:game|exam|homework|work|traffic|queue|lag|computer)\b.{0,80}\b(?:kill myself|kms|end my life)\b", re.I | re.S)),
)


def is_obvious_current_self_crisis(text: str) -> bool:
    """Flag direct first-person current/intent language for pool exclusion."""
    return bool(_CURRENT_SELF_RISK.search(str(text)))


def contextual_reason(text: str) -> str | None:
    """Return a verified non-self contextual reason, if one is evident."""
    for reason, pattern in _CONTEXT_PATTERNS:
        if pattern.search(str(text)):
            return reason
    return None


def has_first_person_crisis_context(text: str) -> bool:
    """Detect self-referential crisis discussion that merits human review."""
    return bool(_FIRST_PERSON_CRISIS_CONTEXT.search(str(text)))


def near_duplicate_of(frame: pd.DataFrame, threshold: float = 0.9) -> dict[int, int]:
    """Map later near-duplicate row indices to the first retained row index."""
    prior: list[tuple[int, set[str]]] = []
    duplicates: dict[int, int] = {}
    for index, text in frame["text"].items():
        current = dedup.shingles(text)
        match = next((other for other, shingles in prior if dedup.jaccard(current, shingles) >= threshold), None)
        if match is None:
            prior.append((index, current))
        else:
            duplicates[index] = match
    return duplicates


def review(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a new QC frame and mutually exclusive primary QC counts."""
    required = {"text", "source_id", "source", "source_label", "matched_terms", "candidate_reason"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate file missing QC columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    near = near_duplicate_of(out)
    out["qc_context_reason"] = out["text"].map(contextual_reason)
    out["qc_current_self_risk_signal"] = out["text"].map(is_obvious_current_self_crisis)
    out["qc_first_person_crisis_context"] = out["text"].map(has_first_person_crisis_context)
    out["qc_duplicate_of_source_id"] = [out.loc[near[i], "source_id"] if i in near else "" for i in out.index]
    statuses: list[str] = []
    explanations: list[str] = []
    for index, row in out.iterrows():
        if index in near:
            statuses.append("duplicate_or_near_duplicate")
            explanations.append("Near-duplicate of an earlier candidate; retain only the first for annotation review.")
        elif row["qc_current_self_risk_signal"]:
            statuses.append("obvious_current_self_crisis")
            explanations.append("Direct first-person current-risk or intent language; exclude from this hard-negative-oriented pool.")
        elif row["qc_first_person_crisis_context"]:
            statuses.append("borderline_needs_annotator_judgment")
            explanations.append("First-person crisis discussion is present without a direct current-risk signal; require annotator judgment.")
        elif isinstance(row["qc_context_reason"], str):
            statuses.append("strong_hard_negative_candidate")
            explanations.append(f"Clear non-self crisis-discussion context: {row['qc_context_reason']}.")
        else:
            statuses.append("borderline_needs_annotator_judgment")
            explanations.append("Crisis vocabulary remains, but no clear non-self context was verified by QC.")
    out["qc_status"] = statuses
    out["qc_explanation"] = explanations
    stats = {status: int((out["qc_status"] == status).sum()) for status in (
        "strong_hard_negative_candidate", "borderline_needs_annotator_judgment",
        "obvious_current_self_crisis", "clearly_irrelevant_non_crisis",
        "duplicate_or_near_duplicate",
    )}
    # This safety-oriented signal can overlap primary duplicate status.
    stats["current_self_risk_signal_total"] = int(out["qc_current_self_risk_signal"].sum())
    return out, stats
