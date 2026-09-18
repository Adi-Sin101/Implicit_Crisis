"""Guards on the TF-IDF + Logistic Regression baseline.

These tests verify the properties the baseline's credibility rests on: that it
reads the project's final dataset, maps severity the way the BERT notebook
does, splits 70/15/15 under seed 42 with no overlap, fits the vectorizer on
training text alone, and that the saved artifacts reload and reproduce the
recorded test numbers.

The split test is the important one for comparability: it reruns the BERT
notebook's own `train_test_split` calls and checks the baseline lands on the
same partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "tfidf_baseline"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
MODEL_DIR = REPO_ROOT / "models" / "tfidf_logistic_regression"

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "final_datasets"
    / "merged_severity_dataset_not_keyword_keyword.csv"
)

SEED = 42
TEXT_COL = "content"
SEVERITY_COL = "severity"
LABEL_NAMES = ["Non-Crisis", "Implicit Crisis", "Explicit Crisis"]

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="final dataset not present"
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def df():
    raw = pd.read_csv(DATA_PATH)
    out = raw.dropna(subset=[TEXT_COL, SEVERITY_COL]).reset_index(drop=True)
    out["label"] = out[SEVERITY_COL].apply(
        lambda s: 0 if int(s) == 0 else (1 if int(s) == 1 else 2)
    )
    return out


@pytest.fixture(scope="module")
def config():
    path = EXP_DIR / "config.json"
    assert path.exists(), "run experiments/tfidf_baseline/run_tfidf_baseline.py first"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split_df():
    path = RESULTS_DIR / "data_split.csv"
    assert path.exists(), "run experiments/tfidf_baseline/run_tfidf_baseline.py first"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def artifacts():
    vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    return vec, clf


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------
def test_dataset_has_expected_columns(df):
    for col in [TEXT_COL, SEVERITY_COL]:
        assert col in df.columns


def test_config_points_at_the_final_dataset(config):
    assert config["dataset"] == (
        "data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv"
    )
    assert config["text_column"] == TEXT_COL
    assert config["target_column"] == SEVERITY_COL


def test_no_missing_content_or_severity_after_load(df):
    assert df[TEXT_COL].isna().sum() == 0
    assert df[SEVERITY_COL].isna().sum() == 0


def test_label_mapping_matches_bert_notebook(df):
    """severity 0 -> 0, 1 -> 1, 2..6 -> 2, for every severity value present."""
    for sev in sorted(df[SEVERITY_COL].unique()):
        labels = set(df.loc[df[SEVERITY_COL] == sev, "label"])
        expected = 0 if int(sev) == 0 else (1 if int(sev) == 1 else 2)
        assert labels == {expected}, f"severity {sev} mapped to {labels}"


def test_exactly_three_classes(df):
    assert sorted(df["label"].unique()) == [0, 1, 2]


def test_no_duplicate_content(df):
    """The baseline's split is leakage-free only because this holds."""
    assert df[TEXT_COL].astype(str).duplicated().sum() == 0


# ----------------------------------------------------------------------
# split
# ----------------------------------------------------------------------
def test_split_reproduces_bert_notebook_partition(df, split_df):
    """Rerun the BERT notebook's exact split calls; the baseline must match.

    The notebook passes only (texts, labels). The baseline additionally passes
    an index array, which must not perturb the partition -- this test is what
    proves it does not, and therefore that both experiments share a test set.
    """
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df["label"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, _, _ = train_test_split(
        temp_texts, temp_labels, test_size=0.50, random_state=SEED,
        stratify=temp_labels,
    )

    saved = {
        name: set(split_df.loc[split_df["split"] == name, "content"].astype(str))
        for name in ["train", "val", "test"]
    }
    assert saved["train"] == set(train_texts)
    assert saved["val"] == set(val_texts)
    assert saved["test"] == set(test_texts)


def test_split_proportions_are_70_15_15(split_df):
    n = len(split_df)
    counts = split_df["split"].value_counts()
    assert abs(counts["train"] / n - 0.70) < 0.01
    assert abs(counts["val"] / n - 0.15) < 0.01
    assert abs(counts["test"] / n - 0.15) < 0.01
    assert counts.sum() == n


def test_split_is_stratified(split_df):
    """Each split's class mix must track the overall mix within 2 points."""
    overall = split_df["label"].value_counts(normalize=True).sort_index()
    for name in ["train", "val", "test"]:
        sub = split_df.loc[split_df["split"] == name, "label"]
        share = sub.value_counts(normalize=True).sort_index()
        for lab in [0, 1, 2]:
            assert abs(share[lab] - overall[lab]) < 0.02


def test_splits_do_not_overlap(split_df):
    sets = {
        name: set(split_df.loc[split_df["split"] == name, "content"].astype(str))
        for name in ["train", "val", "test"]
    }
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


def test_split_seed_recorded(config):
    assert config["seed"] == SEED
    assert config["split"]["random_state"] == SEED
    assert config["split"]["stratified"] is True


# ----------------------------------------------------------------------
# leakage
# ----------------------------------------------------------------------
def test_vectorizer_vocabulary_comes_from_training_text_only(
    split_df, config, artifacts
):
    """Refit the recorded config on the training split; vocabulary must match.

    If the shipped vectorizer had seen validation or test text, its vocabulary
    would differ from a train-only fit.
    """
    vec, _ = artifacts
    tf = config["tfidf"]
    train_texts = split_df.loc[split_df["split"] == "train", "content"].astype(str)

    refit = TfidfVectorizer(
        analyzer=tf["analyzer"],
        ngram_range=tuple(tf["ngram_range"]),
        min_df=tf["min_df"],
        max_df=tf["max_df"],
        sublinear_tf=tf["sublinear_tf"],
        max_features=tf["max_features"],
        lowercase=tf["lowercase"],
        strip_accents=tf["strip_accents"],
    )
    refit.fit(train_texts)

    assert set(refit.get_feature_names_out()) == set(vec.get_feature_names_out())
    assert config["tfidf"]["fitted_on"] == "training split only"


def test_fitting_on_all_data_would_differ(split_df, config, artifacts):
    """Sanity check that the train-only test above has teeth."""
    vec, _ = artifacts
    tf = config["tfidf"]

    all_fit = TfidfVectorizer(
        analyzer=tf["analyzer"],
        ngram_range=tuple(tf["ngram_range"]),
        min_df=tf["min_df"],
        max_df=tf["max_df"],
        sublinear_tf=tf["sublinear_tf"],
        max_features=tf["max_features"],
        lowercase=tf["lowercase"],
        strip_accents=tf["strip_accents"],
    )
    all_fit.fit(split_df["content"].astype(str))

    assert set(all_fit.get_feature_names_out()) != set(vec.get_feature_names_out())


def test_label_columns_are_excluded_from_features(config):
    for col in [
        "severity", "gpt_label", "claude_label", "gemini_label",
        "llama_label", "mistral_label", "url", "author", "created",
    ]:
        assert col in config["excluded_from_features"]


# ----------------------------------------------------------------------
# artifacts and recorded metrics
# ----------------------------------------------------------------------
def test_artifacts_exist():
    for name in ["tfidf_vectorizer.joblib", "logistic_regression.joblib", "config.json"]:
        assert (MODEL_DIR / name).exists(), f"missing artifact: {name}"


def test_reloaded_artifacts_reproduce_recorded_test_metrics(split_df, artifacts):
    """Predict from `content` alone and match the numbers on disk."""
    vec, clf = artifacts
    test = split_df[split_df["split"] == "test"]

    X = vec.transform(test["content"].astype(str))
    y_true = test["label"].to_numpy()
    y_pred = clf.predict(X)

    recorded = pd.read_csv(RESULTS_DIR / "test_results.csv").set_index("metric")["value"]

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    assert acc == pytest.approx(float(recorded["accuracy"]), abs=1e-4)
    assert mp == pytest.approx(float(recorded["macro_precision"]), abs=1e-4)
    assert mr == pytest.approx(float(recorded["macro_recall"]), abs=1e-4)
    assert mf1 == pytest.approx(float(recorded["macro_f1"]), abs=1e-4)
    assert wp == pytest.approx(float(recorded["weighted_precision"]), abs=1e-4)
    assert wr == pytest.approx(float(recorded["weighted_recall"]), abs=1e-4)
    assert wf1 == pytest.approx(float(recorded["weighted_f1"]), abs=1e-4)


def test_predict_proba_is_well_formed(split_df, artifacts):
    vec, clf = artifacts
    test = split_df[split_df["split"] == "test"]
    probs = clf.predict_proba(vec.transform(test["content"].astype(str)))
    assert probs.shape == (len(test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_confusion_matrix_is_consistent_with_report():
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0)
    report = pd.read_csv(RESULTS_DIR / "classification_report.csv")
    test_results = pd.read_csv(RESULTS_DIR / "test_results.csv").set_index("metric")

    assert cm.to_numpy().sum() == int(test_results.loc["n_test", "value"])
    # row sums are the per-class supports
    assert list(cm.sum(axis=1)) == list(report["support"])
    # diagonal over row sum is recall
    diag = np.diag(cm.to_numpy())
    recall = diag / cm.to_numpy().sum(axis=1)
    assert np.allclose(recall, report["recall"], atol=1e-3)


def test_implicit_crisis_metrics_match_the_confusion_matrix():
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0).to_numpy()
    imp = pd.read_csv(RESULTS_DIR / "implicit_crisis_metrics.csv").set_index("metric")[
        "value"
    ]

    correct = cm[1, 1]
    missed = cm[1, :].sum() - correct
    false_pos = cm[:, 1].sum() - correct

    assert int(imp["implicit_correctly_detected"]) == correct
    assert int(imp["implicit_missed_total"]) == missed
    assert int(imp["false_implicit_total"]) == false_pos
    assert int(imp["implicit_support"]) == cm[1, :].sum()


def test_normalized_confusion_matrix_rows_sum_to_one():
    cmn = pd.read_csv(
        RESULTS_DIR / "normalized_confusion_matrix.csv", index_col=0
    ).to_numpy()
    assert np.allclose(cmn.sum(axis=1), 1.0, atol=1e-3)


def test_model_comparison_file_holds_only_the_tfidf_row():
    mc = pd.read_csv(RESULTS_DIR / "model_comparison_ready.csv")
    assert len(mc) == 1
    assert mc.loc[0, "model"] == "TF-IDF + Logistic Regression"
    assert "bert" not in " ".join(mc["model"].astype(str)).lower()
    for col in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "implicit_precision", "implicit_recall", "implicit_f1",
    ]:
        assert col in mc.columns


# ----------------------------------------------------------------------
# selection protocol
# ----------------------------------------------------------------------
def test_selected_config_is_the_validation_macro_f1_winner(config):
    val = pd.read_csv(RESULTS_DIR / "validation_results.csv")
    best = val.loc[val["val_macro_f1"].idxmax()]

    assert config["tfidf"]["analyzer"] == best["analyzer"]
    assert str(tuple(config["tfidf"]["ngram_range"])) == best["ngram_range"]
    assert config["logistic_regression"]["C"] == pytest.approx(float(best["C"]))
    recorded_cw = config["logistic_regression"]["class_weight"]
    assert (recorded_cw if recorded_cw is not None else "None") == best["class_weight"]
    assert config["selection_metric"] == "validation macro F1"


def test_every_configuration_converged():
    val = pd.read_csv(RESULTS_DIR / "validation_results.csv")
    assert val["converged"].all(), "some configurations hit max_iter"
    assert (val["n_iter_used"] < val["max_iter"]).all()


def test_all_expected_outputs_exist():
    for name in [
        "dataset_audit.csv", "class_distribution.csv", "text_statistics.csv",
        "validation_results.csv", "test_results.csv", "classification_report.csv",
        "confusion_matrix.csv", "normalized_confusion_matrix.csv",
        "top_features.csv", "model_comparison_ready.csv", "data_split.csv",
        "duplicate_report.csv", "implicit_crisis_metrics.csv",
        "split_distribution.csv",
    ]:
        assert (RESULTS_DIR / name).exists(), f"missing result: {name}"

    for name in [
        "class_distribution.png", "text_length_distribution.png",
        "validation_hyperparameters.png", "per_class_metrics.png",
        "confusion_matrix.png", "normalized_confusion_matrix.png",
        "top_features.png",
    ]:
        assert (FIGURES_DIR / name).exists(), f"missing figure: {name}"


def test_top_features_cover_all_three_classes():
    top = pd.read_csv(RESULTS_DIR / "top_features.csv")
    assert sorted(top["class_label"].unique()) == [0, 1, 2]
    for lab in [0, 1, 2]:
        assert (top["class_label"] == lab).sum() >= 15
