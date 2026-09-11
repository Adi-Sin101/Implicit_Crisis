"""Safe integration of QC-retained Suicide Reddit candidates.

This module appends unlabelled records only.  QC status and extraction reasons
remain provenance inside ``src_meta`` and are never project labels.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.data.cleaning import dedup, text_cleaning
from src.data.pooling.build_pool import make_id

RETAINED_STATUSES = {
    "strong_hard_negative_candidate",
    "borderline_needs_annotator_judgment",
}
CANDIDATE_COLUMNS = [
    "id", "text", "source", "src_risk", "src_meta", "stratum",
    "explicit_lex", "n_words", "explicit_terms", "src_id",
]


def completed_calibration_ids(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame) -> set[str]:
    """Identify completed calibration items by nonblank labels in both sheets."""
    def labelled_ids(sheet: pd.DataFrame) -> set[str]:
        required = {"id", "label"}
        missing = required - set(sheet.columns)
        if missing:
            raise ValueError(f"Annotation sheet missing columns: {sorted(missing)}")
        labels = sheet["label"].fillna("").astype(str).str.strip()
        return set(sheet.loc[labels.ne(""), "id"].astype(str))

    a_ids, b_ids = labelled_ids(sheet_a), labelled_ids(sheet_b)
    if a_ids != b_ids:
        raise ValueError("Completed calibration sheet ID sets differ; integration stopped.")
    return a_ids


def candidate_fingerprint(frame: pd.DataFrame) -> str:
    """Stable fingerprint used to prove pre-existing rows were untouched."""
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def prepare_retained(qc: pd.DataFrame) -> pd.DataFrame:
    """Convert retained QC rows to the existing candidate-file schema."""
    required = {"text", "source", "source_id", "source_label", "matched_terms", "candidate_reason", "qc_status"}
    missing = required - set(qc.columns)
    if missing:
        raise ValueError(f"QC file missing required columns: {sorted(missing)}")
    retained = qc[qc["qc_status"].isin(RETAINED_STATUSES)].copy()
    if len(retained) != 105:
        raise ValueError(f"Expected exactly 105 QC-retained records, found {len(retained)}; integration stopped.")
    retained["text"] = retained["text"].map(text_cleaning.clean_text)
    retained["src_id"] = retained["source_id"].astype(str)
    retained["src_risk"] = "none"
    retained["src_meta"] = (
        "source_label=" + retained["source_label"].astype(str)
        + ";candidate_reason=" + retained["candidate_reason"].astype(str)
        + ";qc_status=" + retained["qc_status"].astype(str)
    )
    retained["stratum"] = "C_hard_negative"
    retained["explicit_lex"] = True
    retained["n_words"] = retained["text"].map(text_cleaning.word_count)
    retained["explicit_terms"] = retained["matched_terms"].fillna("").astype(str)
    retained["id"] = [
        make_id(source, source_id, text)
        for source, source_id, text in zip(retained["source"], retained["src_id"], retained["text"])
    ]
    if retained["id"].duplicated().any():
        raise ValueError("QC-retained records produce duplicate candidate IDs; integration stopped.")
    if retained["text"].map(dedup.text_hash).duplicated().any():
        raise ValueError("QC-retained records contain exact duplicate text; integration stopped.")
    return retained[CANDIDATE_COLUMNS].reset_index(drop=True)


def duplicate_report(existing: pd.DataFrame, additions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Report collisions without silently discarding a candidate."""
    existing_hashes = set(existing["text"].map(dedup.text_hash))
    existing_source_ids = set(zip(existing["source"].astype(str), existing["src_id"].astype(str)))
    out = additions.copy()
    out["_exact_text_duplicate"] = out["text"].map(dedup.text_hash).isin(existing_hashes)
    out["_source_id_duplicate"] = [
        pair in existing_source_ids for pair in zip(out["source"].astype(str), out["src_id"].astype(str))
    ]
    duplicates = out[out["_exact_text_duplicate"] | out["_source_id_duplicate"]].copy()
    return duplicates, {
        "exact_text_duplicates": int(out["_exact_text_duplicate"].sum()),
        "source_id_duplicates": int(out["_source_id_duplicate"].sum()),
        "genuinely_new": int((~out["_exact_text_duplicate"] & ~out["_source_id_duplicate"]).sum()),
    }


def append_additions(existing: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    """Append validated candidates, preserving every original row and order."""
    if list(existing.columns) != CANDIDATE_COLUMNS:
        raise ValueError("Existing candidate schema changed; integration stopped.")
    combined = pd.concat([existing, additions], ignore_index=True)
    if combined["id"].duplicated().any() or combined["text"].map(dedup.text_hash).duplicated().any():
        raise ValueError("Integration would create duplicate ID or exact text; integration stopped.")
    return combined
