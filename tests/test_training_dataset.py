import json

import numpy as np

from training.dataset import (
    coarse_label,
    descriptor_from_window,
    select_training_windows,
)


def _row(source: str, category: str, start: int, window_id: str) -> dict:
    return {
        "source_path": source,
        "category": category,
        "window_start_frame": start,
        "window_id": window_id,
        "output_path": f"windows/{window_id}.npz",
        "split": "train",
    }


def test_coarse_label_mapping():
    assert coarse_label("walk") == "walk"
    assert coarse_label("run") == "run"
    for category in ("combat", "idle", "dance", "jump_acrobatic", "unknown_mixed"):
        assert coarse_label(category) == "other"


def test_source_class_cap_is_even_and_keeps_endpoints():
    rows = [_row("recording.npz", "walk", frame * 15, f"w{frame:02d}") for frame in range(31)]
    rows += [_row("recording.npz", "run", frame * 15, f"r{frame:02d}") for frame in range(4)]
    selected = select_training_windows(rows, max_per_source_class=10)

    walk = [row for row in selected if row["coarse_class"] == "walk"]
    run = [row for row in selected if row["coarse_class"] == "run"]
    assert len(walk) == 10
    assert len(run) == 4
    assert walk[0]["window_start_frame"] == 0
    assert walk[-1]["window_start_frame"] == 450
    gaps = np.diff([row["window_start_frame"] for row in walk])
    assert gaps.max() - gaps.min() <= 15


def test_other_subcategories_share_one_source_cap():
    rows = []
    for frame in range(8):
        rows.append(_row("same.npz", "dance", frame * 30, f"d{frame}"))
        rows.append(_row("same.npz", "idle", frame * 30 + 15, f"i{frame}"))
    selected = select_training_windows(rows, max_per_source_class=10)
    assert len(selected) == 10
    assert {row["coarse_class"] for row in selected} == {"other"}


def test_window_npz_converts_6d_local_rotations_and_integrates_root(tmp_path):
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rotations = np.tile(identity, (45, 22, 1)).reshape(45, 132)
    velocity = np.tile(np.array([2.0, 0.0, 0.0], dtype=np.float32), (45, 1))
    features = np.concatenate((rotations, velocity), axis=1)
    path = tmp_path / "window.npz"
    np.savez(path, features=features, fps=np.float32(30.0))

    descriptor = descriptor_from_window(path)
    assert descriptor.shape == (175,)
    assert np.isfinite(descriptor).all()
    # The constant 2 m/s root speed is the first root statistic after 126 joint features.
    assert np.isclose(descriptor[126], 2.0)
