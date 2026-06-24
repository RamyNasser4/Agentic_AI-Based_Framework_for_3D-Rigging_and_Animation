from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


_STEP_PATTERN = re.compile(
    r"Step\s+(\d+)\s*:\s*(.*?)(?=Step\s+\d+\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def patch_plan(plan_text: str, plan_fixes: Sequence[Dict[str, Any]]) -> str:
    patched_plan, _ = patch_plan_with_metadata(plan_text, plan_fixes)
    return patched_plan


def patch_plan_with_metadata(
    plan_text: str,
    plan_fixes: Sequence[Dict[str, Any]],
) -> Tuple[str, List[int]]:
    steps = _parse_steps(plan_text)
    if not steps:
        return str(plan_text or ""), []

    patched_steps: List[int] = []

    for fix in plan_fixes or []:
        step_number = _safe_int(fix.get("step"))
        if step_number is None:
            continue

        step = next((item for item in steps if item["number"] == step_number), None)
        if step is None:
            continue

        old_text = _strip_step_prefix(str(fix.get("old_text") or "").strip())
        new_text = _strip_step_prefix(str(fix.get("new_text") or "").strip())
        if not old_text or not new_text:
            continue

        updated_body = _apply_text_edit(step["body"], old_text, new_text)
        if updated_body == step["body"]:
            print(f"[Refinement] Skipped fix for step {step_number}; old_text was not found.")
            continue

        step["body"] = updated_body
        patched_steps.append(step_number)
        print(f"[Refinement] Patched step {step_number}")

    patched_plan = "\n".join(
        f"Step {step['number']}: {_clean_step_body(step['body'])}"
        for step in steps
        if str(step["body"]).strip()
    )
    return patched_plan, patched_steps


def _parse_steps(plan_text: str) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for match in _STEP_PATTERN.finditer(str(plan_text or "").strip()):
        body = _clean_step_body(match.group(2))
        if body:
            steps.append({"number": int(match.group(1)), "body": body})
    return steps


def _apply_text_edit(step_body: str, old_text: str, new_text: str) -> str:
    cleaned_body = _clean_step_body(step_body)
    cleaned_old = _clean_step_body(old_text)
    cleaned_new = _clean_step_body(new_text)

    if _normalize(cleaned_body) == _normalize(cleaned_old):
        return cleaned_new

    if cleaned_old in cleaned_body:
        return _clean_step_body(cleaned_body.replace(cleaned_old, cleaned_new, 1))

    pattern = _whitespace_flexible_pattern(cleaned_old)
    match = pattern.search(cleaned_body)
    if match is None:
        return cleaned_body

    return _clean_step_body(
        f"{cleaned_body[:match.start()]}{cleaned_new}{cleaned_body[match.end():]}"
    )


def _whitespace_flexible_pattern(text: str) -> re.Pattern[str]:
    escaped_tokens = [re.escape(token) for token in text.split()]
    return re.compile(r"\s+".join(escaped_tokens), re.IGNORECASE)


def _strip_step_prefix(text: str) -> str:
    return re.sub(r"^\s*Step\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def _clean_step_body(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned).strip()
    return cleaned


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
