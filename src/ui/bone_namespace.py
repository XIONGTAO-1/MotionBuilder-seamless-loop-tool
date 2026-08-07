"""Helpers for applying MotionBuilder namespaces to configured bone names."""


def normalize_namespace(value: str) -> str:
    namespace = (value or "").strip().rstrip(":")
    return f"{namespace}:" if namespace else ""


def extract_namespace(model_name: str) -> str:
    value = (model_name or "").strip()
    namespace, separator, _local_name = value.rpartition(":")
    return normalize_namespace(namespace) if separator else ""


def qualify_bone_name(
    bone_name: str,
    namespace: str,
    default_name: str,
) -> str:
    local_name = ((bone_name or "").strip() or default_name).rsplit(":", 1)[-1]
    return f"{normalize_namespace(namespace)}{local_name}"
