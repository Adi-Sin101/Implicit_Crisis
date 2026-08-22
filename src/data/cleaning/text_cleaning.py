"""Conservative text cleaning for the candidate pool.

The cleaning here is deliberately light. Candidates are read by human
annotators, so the text must stay close to what the author actually wrote;
aggressive normalisation would destroy exactly the subtle cues that separate
implicit crisis language from ordinary sadness.
"""
from __future__ import annotations

import html
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")

PLACEHOLDERS = {"", "nan", "none", "[removed]", "[deleted]", "removed", "deleted"}


def clean_text(text: str, mask_urls: bool = True) -> str:
    text = html.unescape(str(text))
    if mask_urls:
        text = URL_RE.sub("<URL>", text)
    text = WS_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_placeholder(text: str) -> bool:
    return str(text).strip().lower() in PLACEHOLDERS


def word_count(text: str) -> int:
    return len(str(text).split())


def clean_frame(
    df: pd.DataFrame,
    min_words: int = 5,
    max_words: int = 400,
    mask_urls: bool = True,
    drop_placeholders: bool = True,
) -> pd.DataFrame:
    """Clean, filter by length, and attach ``n_words``.

    Returns a new frame; the input is not modified.
    """
    out = df.copy()
    out["text"] = out["text"].map(lambda t: clean_text(t, mask_urls=mask_urls))
    if drop_placeholders:
        out = out[~out["text"].map(is_placeholder)]
    out["n_words"] = out["text"].map(word_count)
    out = out[(out["n_words"] >= min_words) & (out["n_words"] <= max_words)]
    return out.reset_index(drop=True)
