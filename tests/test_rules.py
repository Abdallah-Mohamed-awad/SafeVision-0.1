from collections import Counter
from app.inference import summarize_counts, VIOLATION_CLASSES


def test_violation_classes_are_explicit():
    assert "no_helmet" in VIOLATION_CLASSES
    assert "no_vest" in VIOLATION_CLASSES


def test_summary_counts_are_stable():
    counts = Counter({"Person": 3, "helmet": 2, "no_helmet": 1, "no_vest": 2, "no_gloves": 4})
    summary = summarize_counts(counts)
    assert summary["people"] == 3
    assert summary["missing_helmet_regions"] == 1
    assert summary["missing_vest_regions"] == 2
    assert summary["other_missing_ppe_regions"] == 4
