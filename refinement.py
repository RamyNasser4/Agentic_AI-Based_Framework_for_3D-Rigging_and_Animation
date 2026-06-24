from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

try:
    from .keyframe_agent import KeyFrameAgent
    from .plan_fix_generator import generate_plan_fixes
    from .plan_patcher import patch_plan_with_metadata
except ImportError:  # pragma: no cover - direct script fallback
    from keyframe_agent import KeyFrameAgent
    from plan_fix_generator import generate_plan_fixes
    from plan_patcher import patch_plan_with_metadata


_STEP_PATTERN = re.compile(
    r"Step\s+(\d+)\s*:\s*(.*?)(?=Step\s+\d+\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def refine(
    plan,
    keyframes,
    feedback,
    object_name: str | None = None,
    object_json: str | None = None,
    prompt: str | None = None,
) -> Tuple[str, str]:
    if not object_name or not object_json or not prompt:
        raise ValueError(
            "refine requires `object_name`, `object_json`, and `prompt` to wrap the existing agents."
        )

    observation_lines = _collect_feedback_observation_lines(feedback)

    print("[Refinement] Generating plan fixes...")
    plan_fixes = generate_plan_fixes(
        original_prompt=prompt,
        current_plan=str(plan),
        critic_feedback=feedback,
    )
    print(f"[Refinement] Generated {len(plan_fixes)} fixes")

    print("[Refinement] Applying fixes...")
    new_plan, patched_steps = patch_plan_with_metadata(str(plan), plan_fixes)
    if plan_fixes and not patched_steps:
        print("[Refinement] No plan steps were patched; keeping the existing plan.")
        new_plan = str(plan)

    keyframe_instruction = "\n".join(
        [
            prompt.strip(),
            "",
            "Realism refinement guidance:",
            "Generate smoother, more realistic keyframes for the patched plan while preserving timing continuity.",
            "Use the patched plan steps as the source of truth.",
            "",
            "Previous keyframe reference:",
            str(keyframes).strip() or "None.",
            "",
            "Visible critic observations:",
            _format_issue_block(observation_lines),
        ]
    ).strip()
    print("[Refinement] Regenerating keyframes from patched plan")
    new_keyframes = generate_keyframes_for_plan(
        object_name=object_name,
        object_json=object_json,
        prompt=keyframe_instruction,
        plan_text=new_plan,
    )

    return new_plan, new_keyframes


def generate_keyframes_for_plan(
    object_name: str,
    object_json: str,
    prompt: str,
    plan_text: str,
) -> str:
    steps = split_animation_plan(plan_text)
    if not steps:
        raise ValueError("No executable plan steps were found while generating keyframes.")

    keyframe_agent = KeyFrameAgent()
    keyframe_agent.initialize_chain()

    plan_history: List[str] = []
    previous_animation = ""
    outputs: List[str] = []

    for step_index, raw_step in enumerate(steps, start=1):
        current_plan_step = format_plan_step(raw_step, step_index)
        response = keyframe_agent.invoke_chain(
            {
                "object": object_name,
                "object_json": object_json,
                "user_instruction": prompt,
                "current_plan_step": current_plan_step,
                "plan_history": list(plan_history),
                "last_step_index": len(plan_history),
                "previous_animation": previous_animation,
            }
        )
        outputs.append(response)
        previous_animation = "\n".join(output for output in outputs if str(output).strip())
        plan_history.append(current_plan_step)

    return previous_animation


def split_animation_plan(plan_text: str) -> List[str]:
    steps: List[str] = []
    for match in _STEP_PATTERN.finditer(str(plan_text or "").strip()):
        body = re.sub(r"\s+", " ", match.group(2)).strip()
        if body:
            steps.append(body)
    return steps


def format_plan_step(step_text: str, step_index: int) -> str:
    cleaned_step = re.sub(
        r"^\s*Step\s*\d+\s*:\s*",
        "",
        str(step_text or "").strip(),
        flags=re.IGNORECASE,
    )
    cleaned_step = re.sub(r"\s+", " ", cleaned_step).strip()
    return f"Step {int(step_index)}: {cleaned_step}"


def _collect_issue_lines(issues: Sequence[Any]) -> List[str]:
    lines: List[str] = []
    for issue in issues or []:
        if isinstance(issue, str):
            cleaned = issue.strip()
            if cleaned:
                lines.append(cleaned)
            continue

        if not isinstance(issue, dict):
            continue

        parts = []
        frame = issue.get("frame")
        if frame is not None:
            parts.append(f"frame={frame}")

        joint = issue.get("joint")
        if joint:
            parts.append(f"joint={joint}")

        issue_type = issue.get("type")
        if issue_type:
            parts.append(f"type={issue_type}")

        description = issue.get("description")
        if description:
            parts.append(f"description={description}")

        reason = issue.get("reason")
        if reason:
            parts.append(f"reason={reason}")

        missing_action = issue.get("missing_or_incorrect_action")
        if missing_action:
            parts.append(f"missing_or_incorrect_action={missing_action}")

        suggestion = issue.get("suggestion")
        if suggestion:
            parts.append(f"suggestion={suggestion}")

        if parts:
            lines.append("; ".join(parts))

    return lines


def _collect_feedback_observation_lines(feedback: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for section_name in ("faithfulness", "realism"):
        section = feedback.get(section_name, {})
        if not isinstance(section, dict):
            continue
        section_lines = _collect_issue_lines(section.get("issues", []))
        for line in section_lines:
            lines.append(f"{section_name}: {line}")
    return lines


def _format_issue_block(lines: Sequence[str]) -> str:
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return "None."
    return "\n".join(f"- {line}" for line in cleaned)
