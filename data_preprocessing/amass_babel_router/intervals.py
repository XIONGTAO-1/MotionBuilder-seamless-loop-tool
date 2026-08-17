from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_EPSILON = 1e-9


@dataclass(frozen=True)
class Segment:
    start_t: float
    end_t: float
    category: str
    seg_ids: tuple[str, ...]
    labels: tuple[dict[str, Any], ...]
    matched_by: tuple[str, ...]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _combine(segments: list[Segment], category: str | None = None) -> Segment:
    return Segment(
        start_t=min(item.start_t for item in segments),
        end_t=max(item.end_t for item in segments),
        category=category or segments[0].category,
        seg_ids=_unique(seg_id for item in segments for seg_id in item.seg_ids),
        labels=tuple(label for item in segments for label in item.labels),
        matched_by=_unique(value for item in segments for value in item.matched_by),
    )


def _sort_key(segment: Segment) -> tuple[float, float, str, tuple[str, ...]]:
    return (segment.start_t, segment.end_t, segment.category, segment.seg_ids)


def _positive_overlap(left: Segment, right: Segment) -> bool:
    return min(left.end_t, right.end_t) - max(left.start_t, right.start_t) > _EPSILON


def merge_same_category(segments: Iterable[Segment]) -> list[Segment]:
    result: list[Segment] = []
    source = list(segments)
    categories = sorted({item.category for item in source})
    for category in categories:
        ordered = sorted((item for item in source if item.category == category), key=_sort_key)
        if not ordered:
            continue
        current = ordered[0]
        for item in ordered[1:]:
            if item.start_t <= current.end_t + _EPSILON:
                current = _combine([current, item])
            else:
                result.append(current)
                current = item
        result.append(current)
    return sorted(result, key=_sort_key)


def resolve_segment_conflicts(
    segments: Iterable[Segment],
) -> tuple[list[Segment], list[Segment], list[dict[str, Any]]]:
    merged = merge_same_category(segments)
    known = [item for item in merged if item.category != "unknown_mixed"]
    original_unknown = [item for item in merged if item.category == "unknown_mixed"]

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(known))}
    for left_index, left in enumerate(known):
        for right_index in range(left_index + 1, len(known)):
            right = known[right_index]
            if left.category != right.category and _positive_overlap(left, right):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    conflict_indices = {index for index, neighbors in adjacency.items() if neighbors}
    pure = [item for index, item in enumerate(known) if index not in conflict_indices]
    rejected: list[dict[str, Any]] = []
    conflict_unknown: list[Segment] = []
    visited: set[int] = set()
    for start_index in sorted(conflict_indices):
        if start_index in visited:
            continue
        stack = [start_index]
        component: list[int] = []
        while stack:
            index = stack.pop()
            if index in visited:
                continue
            visited.add(index)
            component.append(index)
            stack.extend(sorted(adjacency[index] - visited, reverse=True))
        members = [known[index] for index in sorted(component)]
        for item in members:
            rejected.append(
                {
                    "reason": "cross_category_overlap",
                    "category": item.category,
                    "start_t": item.start_t,
                    "end_t": item.end_t,
                    "seg_ids": list(item.seg_ids),
                }
            )
        mixed = _combine(members, category="unknown_mixed")
        conflict_unknown.append(
            Segment(
                start_t=mixed.start_t,
                end_t=mixed.end_t,
                category=mixed.category,
                seg_ids=mixed.seg_ids,
                labels=mixed.labels,
                matched_by=("cross_category_overlap",),
            )
        )

    retained_unknown: list[Segment] = []
    for item in original_unknown:
        if any(_positive_overlap(item, known_item) for known_item in pure):
            rejected.append(
                {
                    "reason": "unknown_overlaps_known",
                    "category": item.category,
                    "start_t": item.start_t,
                    "end_t": item.end_t,
                    "seg_ids": list(item.seg_ids),
                }
            )
        else:
            retained_unknown.append(item)

    unknown = merge_same_category(retained_unknown + conflict_unknown)
    return sorted(pure, key=_sort_key), unknown, sorted(
        rejected,
        key=lambda item: (item["start_t"], item["end_t"], item["reason"], item["category"]),
    )
