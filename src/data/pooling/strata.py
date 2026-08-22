"""Assignment of candidates to sampling strata, and stratified sampling.

    A_likely_explicit   crisis-oriented source signal AND an explicit lexicon hit
    B_likely_implicit   crisis-oriented source signal AND no explicit lexicon hit
    C_hard_negative     explicit lexicon hit AND a non-crisis source signal
    D_likely_noncrisis  neither signal

Strata are a *reading order*, not labels. A post in ``B_likely_implicit`` is a
post worth having a human read for possible implicit crisis language; it is not
an implicit-crisis post until an annotator says so.
"""
from __future__ import annotations

import pandas as pd

from src.utils.labels import STRATA

#: Source signals that make a post worth reading as a possible crisis case.
CRISIS_LEANING = {"positive", "risk_factor", "cause"}
#: Source signals that assert the source considered the post non-crisis.
NON_CRISIS_LEANING = {"negative", "none"}


def assign_stratum(src_risk: str, explicit_lex: bool) -> str:
    crisis_leaning = str(src_risk) in CRISIS_LEANING
    if crisis_leaning and explicit_lex:
        return "A_likely_explicit"
    if crisis_leaning and not explicit_lex:
        return "B_likely_implicit"
    if explicit_lex:
        return "C_hard_negative"
    return "D_likely_noncrisis"


def assign_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stratum"] = [
        assign_stratum(r, bool(e)) for r, e in zip(out["src_risk"], out["explicit_lex"])
    ]
    return out


def sample_strata(
    df: pd.DataFrame, targets: dict[str, int], seed: int = 42
) -> pd.DataFrame:
    """Draw ``targets[stratum]`` items per stratum, balanced across sources.

    Sampling is source-balanced within a stratum so a single large corpus does
    not dominate the annotation batch. If a stratum holds fewer items than its
    target, everything available is taken and the shortfall is reported by the
    caller.
    """
    picks = []
    for stratum in STRATA:
        want = int(targets.get(stratum, 0))
        if want <= 0:
            continue
        block = df[df["stratum"] == stratum]
        if block.empty:
            continue
        sources = list(block["source"].unique())
        per_source = max(1, want // len(sources))
        taken = []
        for source in sources:
            sub = block[block["source"] == source]
            taken.append(sub.sample(min(len(sub), per_source), random_state=seed))
        got = pd.concat(taken)
        if len(got) < want:  # top up from whatever is left in this stratum
            rest = block.drop(index=got.index)
            extra = min(len(rest), want - len(got))
            if extra:
                got = pd.concat([got, rest.sample(extra, random_state=seed)])
        picks.append(got.head(want))
    if not picks:
        return df.head(0)
    return (
        pd.concat(picks)
        .sample(frac=1.0, random_state=seed)  # shuffle so strata are not visible by order
        .reset_index(drop=True)
    )


def stratum_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["stratum", "source"]).size().rename("n").reset_index().sort_values(
            ["stratum", "n"], ascending=[True, False]
        )
    )
