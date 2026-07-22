from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float


def metrics_at_threshold(
    confidence: Sequence[float],
    precision: Sequence[float],
    recall: Sequence[float],
    f1: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    """Return the curve point nearest to a deployable confidence threshold."""
    if not confidence:
        raise ValueError("The confidence curve cannot be empty.")
    if not (len(confidence) == len(precision) == len(recall) == len(f1)):
        raise ValueError("Confidence, precision, recall, and F1 curves must have equal lengths.")
    index = min(range(len(confidence)), key=lambda item: abs(float(confidence[item]) - threshold))
    return ThresholdMetrics(
        threshold=round(float(confidence[index]), 6),
        precision=round(float(precision[index]), 6),
        recall=round(float(recall[index]), 6),
        f1=round(float(f1[index]), 6),
    )


def select_threshold(
    confidence: Sequence[float],
    precision: Sequence[float],
    recall: Sequence[float],
    f1: Sequence[float],
    *,
    minimum_precision: float,
    minimum_recall: float,
) -> tuple[ThresholdMetrics, bool]:
    """Select the highest-recall point meeting alert-quality constraints.

    If the model cannot meet both constraints, return its best F1 point while
    reporting that the requested quality gate was not satisfied. Callers must
    not silently treat that fallback as a production-ready recommendation.
    """
    if not confidence:
        raise ValueError("The confidence curve cannot be empty.")
    if not (len(confidence) == len(precision) == len(recall) == len(f1)):
        raise ValueError("Confidence, precision, recall, and F1 curves must have equal lengths.")

    eligible = [
        index
        for index in range(len(confidence))
        if float(precision[index]) >= minimum_precision
        and float(recall[index]) >= minimum_recall
    ]
    constraints_met = bool(eligible)
    if eligible:
        index = max(
            eligible,
            key=lambda item: (
                float(recall[item]),
                float(f1[item]),
                float(precision[item]),
                -float(confidence[item]),
            ),
        )
    else:
        viable = [index for index in range(len(confidence)) if float(recall[index]) > 0]
        index = max(
            viable or list(range(len(confidence))),
            key=lambda item: (float(f1[item]), float(precision[item]), float(recall[item])),
        )

    return (
        ThresholdMetrics(
            threshold=round(float(confidence[index]), 6),
            precision=round(float(precision[index]), 6),
            recall=round(float(recall[index]), 6),
            f1=round(float(f1[index]), 6),
        ),
        constraints_met,
    )
