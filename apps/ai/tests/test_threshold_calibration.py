import pytest
import numpy as np
from types import SimpleNamespace

from app.training.threshold_calibration import metrics_at_threshold, select_threshold
from scripts.train_detector import _operating_metrics


def test_metrics_at_threshold_uses_nearest_curve_point() -> None:
    metrics = metrics_at_threshold(
        [0.0, 0.25, 0.5],
        [0.2, 0.7, 0.9],
        [0.9, 0.6, 0.2],
        [0.33, 0.65, 0.33],
        0.24,
    )

    assert metrics.threshold == 0.25
    assert metrics.precision == 0.7
    assert metrics.recall == 0.6


def test_select_threshold_favors_recall_after_quality_constraints() -> None:
    selected, constraints_met = select_threshold(
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.72, 0.8, 0.9],
        [0.8, 0.65, 0.5, 0.2],
        [0.53, 0.68, 0.62, 0.33],
        minimum_precision=0.7,
        minimum_recall=0.3,
    )

    assert constraints_met is True
    assert selected.threshold == 0.2


def test_select_threshold_marks_best_f1_fallback_as_failed() -> None:
    selected, constraints_met = select_threshold(
        [0.1, 0.2, 0.3],
        [0.2, 0.4, 0.6],
        [0.8, 0.6, 0.1],
        [0.32, 0.48, 0.17],
        minimum_precision=0.8,
        minimum_recall=0.2,
    )

    assert constraints_met is False
    assert selected.threshold == 0.2


def test_curve_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        select_threshold(
            [0.1],
            [0.5, 0.6],
            [0.5],
            [0.5],
            minimum_precision=0.5,
            minimum_recall=0.5,
        )


def test_training_gate_reads_named_class_at_runtime_threshold() -> None:
    confidence = np.array([0.1, 0.2, 0.3])
    metrics = SimpleNamespace(
        names={0: "fire", 1: "smoke"},
        box=SimpleNamespace(ap_class_index=np.array([0, 1])),
        curves_results=[
            (confidence, np.array([[0.4, 0.7, 0.9], [0.2, 0.3, 0.5]]), "Confidence", "Precision"),
            (confidence, np.array([[0.8, 0.5, 0.2], [0.4, 0.2, 0.1]]), "Confidence", "Recall"),
            (confidence, np.array([[0.53, 0.58, 0.33], [0.27, 0.24, 0.17]]), "Confidence", "F1"),
        ],
    )

    selected = _operating_metrics(metrics, "fire", 0.2)

    assert selected == {
        "class": "fire",
        "threshold": 0.2,
        "precision": 0.7,
        "recall": 0.5,
        "f1": 0.58,
    }


def test_training_gate_requires_class_for_multiclass_model() -> None:
    confidence = np.array([0.1])
    metrics = SimpleNamespace(
        names={0: "fire", 1: "smoke"},
        box=SimpleNamespace(ap_class_index=np.array([0, 1])),
        curves_results=[
            (confidence, np.array([[0.4], [0.2]]), "Confidence", "Precision"),
            (confidence, np.array([[0.8], [0.4]]), "Confidence", "Recall"),
            (confidence, np.array([[0.53], [0.27]]), "Confidence", "F1"),
        ],
    )

    with pytest.raises(ValueError, match="operating-class"):
        _operating_metrics(metrics, None, 0.1)
