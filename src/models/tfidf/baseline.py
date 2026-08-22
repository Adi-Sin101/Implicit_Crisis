"""TF-IDF + Logistic Regression baseline.

Word and character n-grams are combined in a feature union, matching Step 5 of
the methodology. This is the lexical model the transformer is compared against:
it can only see surface forms, which is precisely why it is the right control
for the implicit-crisis question.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_pipeline(config: dict) -> Pipeline:
    cfg = config.get("tfidf", {})
    lr_cfg = cfg.get("logistic_regression", {})
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(cfg.get("word_ngram_range", [1, 2])),
                    min_df=cfg.get("min_df", 2),
                    max_features=cfg.get("max_features", 100000),
                    sublinear_tf=cfg.get("sublinear_tf", True),
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(cfg.get("char_ngram_range", [3, 5])),
                    min_df=cfg.get("min_df", 2),
                    max_features=cfg.get("max_features", 100000),
                    sublinear_tf=cfg.get("sublinear_tf", True),
                    lowercase=True,
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 2000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        random_state=config.get("seed", 42),
    )
    return Pipeline([("features", features), ("clf", clf)])


def top_features(pipeline: Pipeline, class_index: int, k: int = 20) -> list[tuple[str, float]]:
    """Highest-weighted features for one class - used in the error analysis."""
    names = pipeline.named_steps["features"].get_feature_names_out()
    weights = pipeline.named_steps["clf"].coef_[class_index]
    order = weights.argsort()[::-1][:k]
    return [(str(names[i]), float(weights[i])) for i in order]
