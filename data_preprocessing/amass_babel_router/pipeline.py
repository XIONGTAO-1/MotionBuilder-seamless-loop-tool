from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .intervals import Segment, resolve_segment_conflicts
from .features import build_router_features, resample_pose_trans, window_starts
from .labels import CLASS_NAMES, CLASS_TO_ID, classify_babel_label


@dataclass(frozen=True)
class ClipCandidate:
    record: dict[str, Any]
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class PipelineConfig:
    min_clip_seconds: float = 0.5
    target_fps: float = 30.0
    window_frames: int = 45
    stride_frames: int = 15
    val_subjects: tuple[str, ...] = ("tr",)
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_clip_seconds) or self.min_clip_seconds <= 0:
            raise ValueError("min_clip_seconds must be finite and positive")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        if self.window_frames <= 0 or self.stride_frames <= 0:
            raise ValueError("Window and stride frame counts must be positive")
        if not self.val_subjects or any(not subject for subject in self.val_subjects):
            raise ValueError("At least one non-empty validation subject is required")


def subject_from_feat_path(feat_p: str) -> str:
    parts = PurePosixPath(feat_p).parts
    if len(parts) < 4 or parts[0] != "MPIHDM05" or parts[1] != "MPI_HDM05":
        raise ValueError(f"Unsupported HDM05 feature path: {feat_p!r}")
    subject = parts[2]
    if not subject or subject in {".", ".."}:
        raise ValueError(f"Invalid HDM05 subject in path: {feat_p!r}")
    return subject


def assign_split(subject: str, val_subjects: Collection[str]) -> str:
    return "val" if subject in val_subjects else "train"


def resolve_amass_path(amass_root: Path, feat_p: str) -> Path:
    prefix = "MPIHDM05/"
    if not feat_p.startswith(prefix):
        raise ValueError(f"Unsupported BABEL dataset path: {feat_p!r}")
    relative = PurePosixPath(feat_p[len(prefix) :])
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"Unsafe BABEL feature path: {feat_p!r}")
    root = Path(amass_root).resolve()
    candidate = (root / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Feature path escapes AMASS root: {feat_p!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def validate_motion(data: Mapping[str, np.ndarray]) -> tuple[int, float]:
    for required in ("poses", "trans", "mocap_framerate"):
        if required not in data:
            raise ValueError(f"Motion archive is missing {required!r}")
    arrays = {name: np.asarray(value) for name, value in data.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are not supported")

    poses = arrays["poses"]
    trans = arrays["trans"]
    if poses.ndim != 2 or poses.shape[1] < 66 or poses.shape[0] == 0:
        raise ValueError("poses must have shape (frames, >=66)")
    n_frames = int(poses.shape[0])
    if trans.ndim != 2 or trans.shape != (n_frames, 3):
        raise ValueError("trans must have shape (frames, 3)")
    if "dmpls" in arrays and (
        arrays["dmpls"].ndim == 0 or arrays["dmpls"].shape[0] != n_frames
    ):
        raise ValueError("dmpls must share the source frame count")

    fps_array = arrays["mocap_framerate"]
    if fps_array.size != 1:
        raise ValueError("mocap_framerate must be scalar")
    fps = float(fps_array.reshape(-1)[0])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("mocap_framerate must be finite and positive")
    return n_frames, fps


def time_to_frame_range(
    start_t: float,
    end_t: float,
    fps: float,
    n_frames: int,
) -> tuple[int, int]:
    values = (float(start_t), float(end_t), float(fps))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Time bounds and FPS must be finite")
    if fps <= 0 or n_frames <= 0:
        raise ValueError("FPS and frame count must be positive")
    start_frame = min(n_frames, max(0, int(round(start_t * fps))))
    end_frame = min(n_frames, max(0, int(round(end_t * fps))))
    if end_frame <= start_frame:
        raise ValueError("Frame range is empty")
    return start_frame, end_frame


def crop_motion(
    data: Mapping[str, np.ndarray],
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    n_frames, _ = validate_motion(data)
    if start_frame < 0 or end_frame > n_frames or end_frame <= start_frame:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for {n_frames} frames"
        )
    cropped: dict[str, np.ndarray] = {}
    for name, value in data.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == n_frames:
            cropped[name] = array[start_frame:end_frame].copy()
        else:
            cropped[name] = array.copy()
    return cropped


def _rejection_context(
    reason: str,
    *,
    babel_sid: object,
    feat_p: str,
    original_split: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "babel_sid": babel_sid,
        "feat_p": feat_p,
        "original_babel_split": original_split,
        **details,
    }


def build_sequence_clips(
    row: Mapping[str, Any],
    *,
    original_split: str,
    motion: Mapping[str, np.ndarray],
    val_subjects: Collection[str],
    min_clip_seconds: float,
) -> tuple[list[ClipCandidate], list[dict[str, Any]]]:
    if not math.isfinite(min_clip_seconds) or min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be finite and positive")
    n_frames, fps = validate_motion(motion)
    feat_p = str(row.get("feat_p", ""))
    subject = subject_from_feat_path(feat_p)
    split = assign_split(subject, val_subjects)
    babel_sid = row.get("babel_sid")
    source_duration = n_frames / fps
    annotated_duration = float(row.get("dur", source_duration))
    if not math.isfinite(annotated_duration) or annotated_duration <= 0:
        annotated_duration = source_duration
    valid_end = min(source_duration, annotated_duration)

    frame_ann = row.get("frame_ann")
    labels = frame_ann.get("labels", []) if isinstance(frame_ann, Mapping) else []
    segments: list[Segment] = []
    rejected: list[dict[str, Any]] = []
    for index, label_value in enumerate(labels or []):
        if not isinstance(label_value, Mapping):
            rejected.append(
                _rejection_context(
                    "invalid_label",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    label_index=index,
                )
            )
            continue
        try:
            start_t = float(label_value["start_t"])
            end_t = float(label_value["end_t"])
        except (KeyError, TypeError, ValueError):
            rejected.append(
                _rejection_context(
                    "invalid_time_bounds",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    label_index=index,
                )
            )
            continue
        if not math.isfinite(start_t) or not math.isfinite(end_t) or end_t <= start_t:
            rejected.append(
                _rejection_context(
                    "invalid_time_bounds",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    label_index=index,
                    start_t=start_t,
                    end_t=end_t,
                )
            )
            continue
        start_t = max(0.0, start_t)
        end_t = min(valid_end, end_t)
        if end_t <= start_t:
            rejected.append(
                _rejection_context(
                    "time_bounds_outside_motion",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    label_index=index,
                )
            )
            continue
        decision = classify_babel_label(label_value)
        seg_id = str(label_value.get("seg_id") or f"{babel_sid}_{index:04d}")
        segments.append(
            Segment(
                start_t=start_t,
                end_t=end_t,
                category=decision.category,
                seg_ids=(seg_id,),
                labels=(dict(label_value),),
                matched_by=(decision.matched_by,),
            )
        )

    pure, unknown, conflict_rejections = resolve_segment_conflicts(segments)
    for item in conflict_rejections:
        rejected.append(
            _rejection_context(
                item.pop("reason"),
                babel_sid=babel_sid,
                feat_p=feat_p,
                original_split=original_split,
                **item,
            )
        )

    sid_text = str(babel_sid)
    if sid_text.isdigit():
        sid_text = sid_text.zfill(6)
    candidates: list[ClipCandidate] = []
    for segment in sorted(pure + unknown, key=lambda item: (item.start_t, item.end_t, item.category)):
        if segment.end_t - segment.start_t + 1e-9 < min_clip_seconds:
            rejected.append(
                _rejection_context(
                    "below_min_duration",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    category=segment.category,
                    start_t=segment.start_t,
                    end_t=segment.end_t,
                    seg_ids=list(segment.seg_ids),
                )
            )
            continue
        try:
            start_frame, end_frame = time_to_frame_range(
                segment.start_t, segment.end_t, fps, n_frames
            )
        except ValueError:
            rejected.append(
                _rejection_context(
                    "empty_frame_range",
                    babel_sid=babel_sid,
                    feat_p=feat_p,
                    original_split=original_split,
                    category=segment.category,
                    start_t=segment.start_t,
                    end_t=segment.end_t,
                    seg_ids=list(segment.seg_ids),
                )
            )
            continue
        clip_id = (
            f"{split}_{sid_text}_{segment.category}_"
            f"{start_frame:06d}_{end_frame:06d}"
        )
        record = {
            "clip_id": clip_id,
            "split": split,
            "original_babel_split": original_split,
            "category": segment.category,
            "label_id": CLASS_TO_ID[segment.category],
            "subject": subject,
            "source_path": feat_p[len("MPIHDM05/") :],
            "babel_sid": babel_sid,
            "seg_ids": list(segment.seg_ids),
            "raw_labels": list(segment.labels),
            "matched_by": list(segment.matched_by),
            "start_t": segment.start_t,
            "end_t": segment.end_t,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": end_frame - start_frame,
            "fps": fps,
            "duration": (end_frame - start_frame) / fps,
        }
        candidates.append(ClipCandidate(record, start_frame, end_frame))

    candidates.sort(key=lambda item: item.record["clip_id"])
    rejected.sort(
        key=lambda item: (
            str(item.get("babel_sid", "")),
            float(item.get("start_t", -1.0)),
            str(item.get("reason", "")),
        )
    )
    return candidates, rejected


def _load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    validate_motion(motion)
    return motion


def _load_babel_rows(babel_root: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "val"):
        path = babel_root / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"BABEL {path.name} must contain an object")
        values = [value for value in payload.values() if isinstance(value, dict)]
        values.sort(key=lambda row: (str(row.get("babel_sid", "")), str(row.get("feat_p", ""))))
        rows.extend((split, row) for row in values)
    return rows


def _safe_output_root(
    output_root: Path,
    amass_root: Path,
    babel_root: Path,
) -> Path:
    output = output_root.resolve()
    inputs = (amass_root.resolve(), babel_root.resolve())
    filesystem_root = Path(output.anchor)
    if output in {Path.cwd().resolve(), filesystem_root, *inputs}:
        raise ValueError(f"Unsafe output root: {output}")
    if any(input_path.is_relative_to(output) for input_path in inputs):
        raise ValueError("Output root may not contain an input root")
    if any(output.is_relative_to(input_path) for input_path in inputs):
        raise ValueError("Output root may not be inside an input root")
    if output.exists() and not output.is_dir():
        raise ValueError("Output root exists and is not a directory")
    return output


def _json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(f"{_json_line(record)}\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})


def _empty_category_counts() -> dict[str, dict[str, int]]:
    return {
        split: {category: 0 for category in CLASS_NAMES}
        for split in ("train", "val")
    }


def _counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = _empty_category_counts()
    for record in records:
        counts[record["split"]][record["category"]] += 1
    return counts


def _seconds(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals = {
        split: {category: 0.0 for category in CLASS_NAMES}
        for split in ("train", "val")
    }
    for record in records:
        totals[record["split"]][record["category"]] += float(record["duration"])
    return {
        split: {category: round(value, 6) for category, value in categories.items()}
        for split, categories in totals.items()
    }


def _coverage(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("train", "val"):
        result[split] = {}
        for category in CLASS_NAMES:
            selected = [
                record
                for record in records
                if record["split"] == split and record["category"] == category
            ]
            result[split][category] = {
                "source_sequences": len({record["source_path"] for record in selected}),
                "subjects": len({record["subject"] for record in selected}),
            }
    return result


def _class_weights(window_records: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(
        record["category"] for record in window_records if record["split"] == "train"
    )
    total = sum(counts.values())
    return {
        category: round(total / (len(CLASS_NAMES) * counts[category]), 8)
        if counts[category]
        else 0.0
        for category in CLASS_NAMES
    }


def _summary(
    clip_records: list[dict[str, Any]],
    window_records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any]:
    coverage = _coverage(clip_records)
    warnings: list[str] = []
    for category in CLASS_NAMES[:-1]:
        train_sources = coverage["train"][category]["source_sequences"]
        val_sources = coverage["val"][category]["source_sequences"]
        if train_sources < 10:
            warnings.append(
                f"{category}: only {train_sources} training source sequences; recommended minimum is 10"
            )
        if val_sources < 5:
            warnings.append(
                f"{category}: only {val_sources} validation source sequences; recommended minimum is 5"
            )
    return {
        "class_to_id": CLASS_TO_ID,
        "config": {
            "min_clip_seconds": config.min_clip_seconds,
            "target_fps": config.target_fps,
            "window_frames": config.window_frames,
            "stride_frames": config.stride_frames,
            "val_subjects": list(config.val_subjects),
        },
        "clips": _counts(clip_records),
        "clip_seconds": _seconds(clip_records),
        "windows": _counts(window_records),
        "coverage": coverage,
        "train_class_weights": _class_weights(window_records),
        "rejections": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "warnings": warnings,
    }


def _build_into(
    amass_root: Path,
    babel_root: Path,
    temporary_root: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    clip_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for original_split, row in _load_babel_rows(babel_root):
        feat_p = str(row.get("feat_p", ""))
        if not feat_p.startswith("MPIHDM05/"):
            continue
        try:
            source_path = resolve_amass_path(amass_root, feat_p)
            motion = _load_motion(source_path)
            candidates, sequence_rejected = build_sequence_clips(
                row,
                original_split=original_split,
                motion=motion,
                val_subjects=config.val_subjects,
                min_clip_seconds=config.min_clip_seconds,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            rejected.append(
                {
                    "reason": "source_or_sequence_error",
                    "babel_sid": row.get("babel_sid"),
                    "feat_p": feat_p,
                    "original_babel_split": original_split,
                    "error": str(error),
                }
            )
            continue
        rejected.extend(sequence_rejected)

        for candidate in candidates:
            record = dict(candidate.record)
            category = record["category"]
            split = record["split"]
            clip_relative = Path("clips") / split / category / f"{record['clip_id']}.npz"
            clip_path = temporary_root / clip_relative
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            cropped = crop_motion(motion, candidate.start_frame, candidate.end_frame)
            np.savez_compressed(clip_path, **cropped)
            record["output_path"] = clip_relative.as_posix()
            clip_records.append(record)

            sampled_poses, sampled_trans = resample_pose_trans(
                cropped["poses"],
                cropped["trans"],
                source_fps=float(record["fps"]),
                target_fps=config.target_fps,
            )
            starts = window_starts(
                len(sampled_poses), config.window_frames, config.stride_frames
            )
            if not starts:
                rejected.append(
                    {
                        "reason": "below_window_length",
                        "clip_id": record["clip_id"],
                        "category": category,
                        "split": split,
                        "resampled_frames": len(sampled_poses),
                    }
                )
            for start in starts:
                end = start + config.window_frames
                features = build_router_features(
                    sampled_poses[start:end],
                    sampled_trans[start:end],
                    config.target_fps,
                )
                window_id = f"{record['clip_id']}_w{start:06d}_{end:06d}"
                window_relative = Path("windows") / split / category / f"{window_id}.npz"
                window_path = temporary_root / window_relative
                window_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    window_path,
                    features=features,
                    label_id=np.array(record["label_id"], dtype=np.int64),
                    fps=np.array(config.target_fps, dtype=np.float32),
                    source_clip_id=np.array(record["clip_id"]),
                )
                window_records.append(
                    {
                        "window_id": window_id,
                        "source_clip_id": record["clip_id"],
                        "split": split,
                        "original_babel_split": record["original_babel_split"],
                        "category": category,
                        "label_id": record["label_id"],
                        "subject": record["subject"],
                        "source_path": record["source_path"],
                        "output_path": window_relative.as_posix(),
                        "window_start_frame": start,
                        "window_end_frame": end,
                        "n_frames": config.window_frames,
                        "feature_dim": 135,
                        "fps": config.target_fps,
                        "duration": config.window_frames / config.target_fps,
                    }
                )

    clip_records.sort(key=lambda record: record["clip_id"])
    window_records.sort(key=lambda record: record["window_id"])
    rejected.sort(
        key=lambda record: (
            str(record.get("babel_sid", "")),
            str(record.get("clip_id", "")),
            str(record.get("reason", "")),
            float(record.get("start_t", -1.0)),
        )
    )
    weights = _class_weights(window_records)
    for record in window_records:
        record["class_weight"] = weights[record["category"]]

    summary = _summary(clip_records, window_records, rejected, config)
    _write_jsonl(temporary_root / "clips.jsonl", clip_records)
    _write_csv(temporary_root / "clips.csv", clip_records)
    _write_jsonl(temporary_root / "windows.jsonl", window_records)
    _write_csv(temporary_root / "windows.csv", window_records)
    _write_jsonl(temporary_root / "rejected.jsonl", rejected)
    (temporary_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_dataset(
    amass_root: Path,
    babel_root: Path,
    output_root: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    amass = Path(amass_root).resolve()
    babel = Path(babel_root).resolve()
    if not amass.is_dir() or not babel.is_dir():
        raise FileNotFoundError("AMASS and BABEL roots must exist")
    output = _safe_output_root(Path(output_root), amass, babel)
    if output.exists() and not config.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        summary = _build_into(amass, babel, temporary_root, config)
        if output.exists():
            shutil.rmtree(output)
        temporary_root.replace(output)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return summary
