from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


CLASS_NAMES = (
    "walk",
    "run",
    "combat",
    "idle",
    "dance",
    "jump_acrobatic",
    "unknown_mixed",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

ACT_CAT_MAP = {
    "walk": "walk",
    "run": "run",
    "jog": "run",
    "kick": "combat",
    "punch": "combat",
    "martial art": "combat",
    "stand": "idle",
    "wait": "idle",
    "dance": "dance",
    "jump": "jump_acrobatic",
    "cartwheel": "jump_acrobatic",
    "jumping jacks": "jump_acrobatic",
    "jump rope": "jump_acrobatic",
}

_OVERRIDE_PATTERNS = (
    re.compile(r"\btransition\b"),
    re.compile(r"\bstand(?:ing)?\s+up\b"),
    re.compile(r"\bget(?:ting)?\s+up\b"),
    re.compile(r"\brise\b"),
)

_LEXICAL_MAP = {
    "walk": "walk",
    "walks": "walk",
    "walking": "walk",
    "stride": "walk",
    "strides": "walk",
    "striding": "walk",
    "march": "walk",
    "marches": "walk",
    "marching": "walk",
    "run": "run",
    "runs": "run",
    "running": "run",
    "jog": "run",
    "jogs": "run",
    "jogging": "run",
    "sprint": "run",
    "sprints": "run",
    "sprinting": "run",
    "kick": "combat",
    "kicks": "combat",
    "kicking": "combat",
    "punch": "combat",
    "punches": "combat",
    "punching": "combat",
    "fight": "combat",
    "fights": "combat",
    "fighting": "combat",
    "boxing": "combat",
    "stand": "idle",
    "stands": "idle",
    "standing": "idle",
    "idle": "idle",
    "wait": "idle",
    "waits": "idle",
    "waiting": "idle",
    "dance": "dance",
    "dances": "dance",
    "dancing": "dance",
    "waltz": "dance",
    "waltzes": "dance",
    "waltzing": "dance",
    "jump": "jump_acrobatic",
    "jumps": "jump_acrobatic",
    "jumping": "jump_acrobatic",
    "cartwheel": "jump_acrobatic",
    "cartwheels": "jump_acrobatic",
}


@dataclass(frozen=True)
class LabelDecision:
    category: str
    matched_by: str
    matched_values: tuple[str, ...]


def _normalize(value: object) -> str:
    return " ".join(re.findall(r"[a-z]+", str(value).lower()))


def _values(label: Mapping[str, object], field: str) -> list[str]:
    value = label.get(field)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value if item is not None]
    return [_normalize(value)]


def classify_babel_label(label: Mapping[str, object]) -> LabelDecision:
    act_values = _values(label, "act_cat")
    fallback_values = _values(label, "proc_label") + _values(label, "raw_label")
    all_values = act_values + fallback_values

    override_hits = tuple(
        value
        for value in all_values
        if any(pattern.search(value) for pattern in _OVERRIDE_PATTERNS)
    )
    if override_hits:
        return LabelDecision("unknown_mixed", "override", override_hits)

    act_hits = [(value, ACT_CAT_MAP[value]) for value in act_values if value in ACT_CAT_MAP]
    act_categories = {category for _, category in act_hits}
    if len(act_categories) == 1:
        return LabelDecision(
            next(iter(act_categories)),
            "act_cat",
            tuple(value for value, _ in act_hits),
        )
    if len(act_categories) > 1:
        return LabelDecision(
            "unknown_mixed",
            "multi_route",
            tuple(value for value, _ in act_hits),
        )

    lexical_hits: list[tuple[str, str]] = []
    for value in fallback_values:
        for token in value.split():
            category = _LEXICAL_MAP.get(token)
            if category is not None:
                lexical_hits.append((token, category))
    lexical_categories = {category for _, category in lexical_hits}
    if len(lexical_categories) == 1:
        return LabelDecision(
            next(iter(lexical_categories)),
            "lexical",
            tuple(token for token, _ in lexical_hits),
        )
    if len(lexical_categories) > 1:
        return LabelDecision(
            "unknown_mixed",
            "multi_route",
            tuple(token for token, _ in lexical_hits),
        )
    return LabelDecision("unknown_mixed", "unmapped", tuple(all_values))

