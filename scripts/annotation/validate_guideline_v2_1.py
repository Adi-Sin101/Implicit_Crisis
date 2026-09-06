"""Validate the v2.1 guideline and supporting freeze documents without editing files."""
from __future__ import annotations

import hashlib
from pathlib import Path

LABELS = ("explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis")
PROTECTED_HASHES = {
    "data/gold/annotation/calibration_150_A.csv": "b6bc82a34b71ccffdcca5fa3ab9200eddf347a0dc36fee2852ec0a60bf969600",
    "data/gold/annotation/calibration_150_B.csv": "58b77f4a72c4c4878b6d4b9dbe30c67dfbb95ec8042a54bd96918ac8245e19eb",
    "data/gold/annotation/agreement/calibration_150_adjudicated.csv": "454b7f5c069f252275fa3124d1e28e983c679b09906fc9c0e37b82502fd6c090",
    "data/gold/annotation/agreement/calibration_150_disagreements.csv": "8997a26f0b6c1ba1575c7c5594f0a44acf8809352cad33a54f0ddc7a0a2fa975",
    "data/gold/annotation/agreement/calibration_150_disagreements_2.csv": "e411e8457571812e6cc497f66387281c44799fac5afb63cf1d5481cb89bfe34f",
    "data/gold/annotation/agreement/calibration_150_disagreements_3.csv": "e411e8457571812e6cc497f66387281c44799fac5afb63cf1d5481cb89bfe34f",
    "data/gold/annotation/agreement/guideline_sanity_check_20_30.csv": "7db4844046a93dd6ab4a5fb91fdfbbcc39a80ac5b70daa7d4da098723b3337a0",
    "data/gold/annotation/agreement/guideline_sanity_check_report.md": "1bdf3a1de3af4ee6c6df9efbf6ae630c8e780005b6e6a57e7419f50d67b20b3a",
    "docs/annotation_guidelines_v2.md": "e7979916bc0bcd8a77e9f93e75a1ccc710533880413410c5cd97dd6a178b5432",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    canonical_v2 = root / "data/gold/annotation/annotation_guidelines_v2.md"
    fallback_v2 = root / "docs/annotation_guidelines_v2.md"
    v21 = root / "data/gold/annotation/annotation_guidelines_v2.1.md"
    changelog = root / "data/gold/annotation/annotation_guidelines_v2.1_changelog.md"
    checklist = root / "data/gold/annotation/annotation_guideline_freeze_checklist.md"
    report = root / "data/gold/annotation/guideline_v2.1_finalization_report.md"

    source_v2 = canonical_v2 if canonical_v2.exists() else fallback_v2
    checks = {
        "canonical_v2_exists": canonical_v2.exists(),
        "v2_source_exists": source_v2.exists(),
        "v2.1_exists": v21.exists(),
        "v2.1_changelog_exists": changelog.exists(),
        "freeze_checklist_exists": checklist.exists(),
        "finalization_report_exists": report.exists(),
    }
    if not source_v2.exists() or not v21.exists():
        raise FileNotFoundError("Required guideline source or v2.1 file is missing")
    text = v21.read_text(encoding="utf-8")
    checks.update({
        "all_four_labels": all(label in text for label in LABELS),
        "no_unexpected_label_heading": "### `crisis`" not in text,
        "temporal_scope": "## 2. Temporal Scope" in text,
        "decision_order": "## 3. Operational decision order" in text,
        "paired_examples": "## 10. Paired Boundary Examples" in text,
        "confidence_section": "## 11. Confidence and notes" in text,
        "version_history": "## 13. Traceability and version history" in text,
        "final_temporal_policy": "current or contextually ongoing" in text,
        "project_author_decision": "project author has approved and finalized" in text,
        "no_advisor_requirement": "research advisor" not in text and "advisor approval" not in text,
        "frozen": "Guideline v2.1 is frozen" in text,
    })
    required_checks = {name: result for name, result in checks.items() if name != "canonical_v2_exists"}
    if not all(required_checks.values()):
        raise AssertionError(f"Validation failed: {checks}")
    protected_results = {}
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Protected file is missing: {relative_path}")
        actual_hash = sha256(path)
        protected_results[relative_path] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            raise AssertionError(f"Protected file hash changed: {relative_path}")
    checklist_text = checklist.read_text(encoding="utf-8")
    report_text = report.read_text(encoding="utf-8")
    if "[x] Project-author temporal-policy decision obtained" not in checklist_text:
        raise AssertionError("Freeze checklist is incomplete")
    if "[x] Guideline frozen before full annotation" not in checklist_text:
        raise AssertionError("Freeze checklist does not mark v2.1 frozen")
    if "READY FOR FULL ANNOTATION — GUIDELINE FROZEN" not in report_text:
        raise AssertionError("Finalization report does not contain frozen readiness status")
    if "No full annotation, relabeling, dataset splitting, or model training was performed." not in report_text:
        raise AssertionError("Finalization report is missing scope protection statement")
    print("GUIDELINE V2.1 VALIDATION PASS")
    print(f"v2 source used: {source_v2}")
    print(f"canonical v2 path exists: {checks['canonical_v2_exists']}")
    print(f"v2 source SHA-256: {sha256(source_v2)}")
    print(f"v2.1 SHA-256: {sha256(v21)}")
    for name, result in checks.items():
        print(f"{name}: {result}")
    print(f"protected_hashes: {all(protected_results.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())