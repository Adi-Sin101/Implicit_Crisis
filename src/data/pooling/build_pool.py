"""Build the unified candidate pool from the external corpora."""
from __future__ import annotations

import hashlib

import pandas as pd

from src.data.cleaning import dedup, text_cleaning
from src.data.loaders import cams, goemotions, irf, komati, sdcnl
from src.data.pooling import lexicon, strata
from src.utils.labels import CANDIDATE_COLUMNS

LOADERS = {
    "komati": komati.load,
    "irf": irf.load,
    "sdcnl": sdcnl.load,
    "cams": cams.load,
    "goemotions": goemotions.load,
}


def make_id(source: str, src_id: str, text: str) -> str:
    """Stable, content-derived id so re-running the pipeline keeps ids fixed."""
    digest = hashlib.sha1(f"{source}|{src_id}|{text[:200]}".encode("utf-8")).hexdigest()
    return f"{source}-{digest[:12]}"


def load_sources(config: dict, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for name, settings in (config.get("sources") or {}).items():
        if not settings.get("enabled", True):
            continue
        loader = LOADERS.get(name)
        if loader is None:
            raise KeyError(f"No loader registered for source '{name}'")
        df = loader()
        cap = settings.get("max_sample")
        if cap and len(df) > cap:
            df = df.sample(int(cap), random_state=config.get("seed", 42))
        if verbose:
            print(f"  {name:12s} {len(df):>7,} rows")
        frames.append(df)
    if not frames:
        raise RuntimeError("No sources enabled in configs/candidate_pool.yaml")
    return pd.concat(frames, ignore_index=True)


def build(config: dict, verbose: bool = True) -> pd.DataFrame:
    """Load -> clean -> deduplicate -> lexicon-route -> assign strata."""
    seed = config.get("seed", 42)
    clean_cfg = config.get("cleaning", {})

    pool = load_sources(config, verbose=verbose)
    if verbose:
        print(f"  loaded       {len(pool):>7,} rows")

    pool = text_cleaning.clean_frame(
        pool,
        min_words=clean_cfg.get("min_words", 5),
        max_words=clean_cfg.get("max_words", 400),
        mask_urls=clean_cfg.get("mask_urls", True),
        drop_placeholders=clean_cfg.get("drop_removed_placeholders", True),
    )
    if verbose:
        print(f"  cleaned      {len(pool):>7,} rows")

    dd = clean_cfg.get("dedup", {})
    if dd.get("exact", True):
        pool = dedup.drop_exact_duplicates(pool)
        if verbose:
            print(f"  exact-dedup  {len(pool):>7,} rows")
    if dd.get("near_duplicate", False):
        pool = dedup.drop_near_duplicates(
            pool, threshold=dd.get("minhash_threshold", 0.9)
        )
        if verbose:
            print(f"  near-dedup   {len(pool):>7,} rows")

    pool = lexicon.annotate_frame(pool, terms=lexicon.load_terms(config))
    pool = strata.assign_frame(pool)
    pool["id"] = [
        make_id(s, i, t) for s, i, t in zip(pool["source"], pool["src_id"], pool["text"])
    ]
    pool = pool.drop_duplicates(subset="id").reset_index(drop=True)

    keep = list(CANDIDATE_COLUMNS) + ["explicit_terms", "src_id"]
    return pool[[c for c in keep if c in pool.columns]]


def sample_candidates(pool: pd.DataFrame, config: dict) -> pd.DataFrame:
    targets = {k: v.get("target_n", 0) for k, v in (config.get("strata") or {}).items()}
    return strata.sample_strata(pool, targets, seed=config.get("seed", 42))
