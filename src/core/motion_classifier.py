"""Pure NumPy inference for the exported motion-routing forest."""

from __future__ import annotations

import json
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .motion_features import DESCRIPTOR_SIZE, FEATURE_SCHEMA_HASH


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    window_count: int
    reason: str


@dataclass(frozen=True)
class _Tree:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    probabilities: np.ndarray


class NumpyRandomForest:
    """Read-only RandomForest evaluator with no scikit-learn dependency."""

    def __init__(
        self,
        classes: Sequence[str],
        thresholds: Mapping[str, float],
        trees: Sequence[_Tree],
        deployment_status: str,
    ):
        self.classes = tuple(classes)
        self.thresholds = {name: float(value) for name, value in thresholds.items()}
        self._trees = tuple(trees)
        self.deployment_status = deployment_status

    @classmethod
    def load(cls, path: Path, allow_experimental: bool = False) -> "NumpyRandomForest":
        model_path = Path(path)
        try:
            raw = model_path.read_bytes()
            checksum_path = model_path.with_suffix(".sha256")
            if checksum_path.exists():
                expected_checksum = checksum_path.read_text(encoding="utf-8").strip().lower()
                actual_checksum = hashlib.sha256(raw).hexdigest()
                if not hmac.compare_digest(expected_checksum, actual_checksum):
                    raise ValueError("Motion model checksum does not match JSON content")
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid motion model JSON: {exc}") from exc

        if payload.get("model_format_version") != 1:
            raise ValueError("Unsupported motion model format version")
        if payload.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
            raise ValueError("Motion model feature schema does not match runtime schema")
        if payload.get("descriptor_size") != DESCRIPTOR_SIZE:
            raise ValueError("Motion model descriptor size does not match runtime")

        status = str(payload.get("deployment_status", "experimental"))
        if status not in {"production", "experimental"}:
            raise ValueError(f"Invalid deployment status: {status}")
        if status == "experimental" and not allow_experimental:
            raise ValueError("Motion model is experimental; pass allow_experimental=True to load it")

        classes = payload.get("classes")
        if not isinstance(classes, list) or set(classes) != {"walk", "run", "other"} or len(classes) != 3:
            raise ValueError("Motion model classes must contain walk, run, and other exactly once")
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != {"walk", "run"}:
            raise ValueError("Motion model thresholds must contain walk and run")
        if any(not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
            raise ValueError("Motion model thresholds must be finite probabilities")

        raw_trees = payload.get("trees")
        if not isinstance(raw_trees, list) or not raw_trees:
            raise ValueError("Motion model must contain at least one tree")
        try:
            trees = [cls._parse_tree(tree, len(classes)) for tree in raw_trees]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid motion model tree: {exc}") from exc
        return cls(classes, thresholds, trees, status)

    @staticmethod
    def _parse_tree(payload: dict, class_count: int) -> _Tree:
        left = np.asarray(payload["children_left"], dtype=np.int64)
        right = np.asarray(payload["children_right"], dtype=np.int64)
        feature = np.asarray(payload["feature"], dtype=np.int64)
        threshold = np.asarray(payload["threshold"], dtype=np.float64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        node_count = len(left)
        if node_count == 0 or any(array.shape != (node_count,) for array in (right, feature, threshold)):
            raise ValueError("tree node arrays have inconsistent lengths")
        if probabilities.shape != (node_count, class_count):
            raise ValueError("tree probability array has an invalid shape")
        if not np.isfinite(threshold).all() or not np.isfinite(probabilities).all():
            raise ValueError("tree contains non-finite values")
        if np.any(probabilities < 0.0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("tree probabilities are invalid")
        for node in range(node_count):
            is_leaf = left[node] == -1 and right[node] == -1
            if is_leaf:
                continue
            if not (0 <= left[node] < node_count and 0 <= right[node] < node_count):
                raise ValueError("tree child index is out of bounds")
            if not 0 <= feature[node] < DESCRIPTOR_SIZE:
                raise ValueError("tree feature index is out of bounds")
        state = np.zeros(node_count, dtype=np.int8)

        def visit(node: int) -> None:
            if state[node] == 1:
                raise ValueError("tree contains a cycle")
            if state[node] == 2:
                return
            state[node] = 1
            if left[node] != -1:
                visit(int(left[node]))
                visit(int(right[node]))
            state[node] = 2

        visit(0)
        if np.any(state != 2):
            raise ValueError("tree contains unreachable nodes")
        return _Tree(left, right, feature, threshold, probabilities)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != DESCRIPTOR_SIZE:
            raise ValueError(f"features must have shape (samples, {DESCRIPTOR_SIZE})")
        if not np.isfinite(values).all():
            raise ValueError("features must contain only finite values")

        result = np.zeros((len(values), len(self.classes)), dtype=np.float64)
        for tree in self._trees:
            for sample_index, sample in enumerate(values):
                node = 0
                while tree.children_left[node] != -1:
                    node = (
                        int(tree.children_left[node])
                        if sample[tree.feature[node]] <= tree.threshold[node]
                        else int(tree.children_right[node])
                    )
                result[sample_index] += tree.probabilities[node]
        return result / len(self._trees)


def route_probabilities(
    probabilities: Sequence[float],
    classes: Sequence[str],
    thresholds: Mapping[str, float],
) -> tuple[str, str]:
    probability_by_class = {name: float(value) for name, value in zip(classes, probabilities)}
    walk = probability_by_class["walk"]
    run = probability_by_class["run"]
    if walk >= float(thresholds["walk"]) and walk > run:
        return "walk", "walk_threshold_passed"
    if run >= float(thresholds["run"]) and run > walk:
        return "run", "run_threshold_passed"
    return "other", "specialized_threshold_not_met"


def classify_feature_windows(
    forest: NumpyRandomForest,
    features: np.ndarray,
) -> ClassificationResult:
    values = np.asarray(features)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("features must contain at least one descriptor window")
    mean_probabilities = forest.predict_proba(values).mean(axis=0)
    label, reason = route_probabilities(mean_probabilities, forest.classes, forest.thresholds)
    probability_by_class = {
        name: float(mean_probabilities[index]) for index, name in enumerate(forest.classes)
    }
    return ClassificationResult(
        label=label,
        confidence=probability_by_class[label],
        probabilities=probability_by_class,
        window_count=len(values),
        reason=reason,
    )
