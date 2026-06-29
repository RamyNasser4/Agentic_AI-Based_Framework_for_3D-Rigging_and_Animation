from __future__ import annotations

from os import getenv
from typing import Any, Dict, List

import numpy as np

try:
    from .multimodal_utils import extract_response_text, image_to_data_url, parse_response_json
except ImportError:  # pragma: no cover - direct script fallback
    from multimodal_utils import extract_response_text, image_to_data_url, parse_response_json

_AI_DEPENDENCY_MESSAGE = (
    "The AI critic dependencies are not installed. Install the Grad planner/keyframe "
    "AI dependencies, including langchain-core and langchain-google-genai, then run "
    "the AI planner feature again. UniRig API features do not require these packages."
)


def _import_critic_ai_dependencies():
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as error:
        raise RuntimeError(_AI_DEPENDENCY_MESSAGE) from error

    return HumanMessage, SystemMessage, ChatGoogleGenerativeAI


CRITIC_SYSTEM_PROMPT = """You are a strict animation critic and evaluator.

You will receive:
- the user's motion prompt
- a small ordered set of skeleton-render images on black backgrounds

Each image includes a visible frame number label. Use those visible frame numbers to localize
every issue with either:
- frame_start and frame_end, when the issue spans a visible range
- frames, when the issue appears in one or more specific sampled frames

IMPORTANT:
The rendered skeleton images are character-centered.
World-space translation is NOT visible.
Do NOT evaluate:
- root translation
- travel distance
- locomotion distance
- forward displacement
- movement through world space

Do NOT generate issues related to translation.
Evaluate ONLY:
- visible body motion
- pose changes
- limb coordination
- gait quality
- realism
- faithfulness of visible body motion

Evaluate the motion on two axes:
1. faithfulness to the prompt
2. realism of the motion

The critic diagnoses only. Do NOT propose:
- exact angles
- parameter changes
- operations
- direct plan edits
- replacement text
- suggestions for how to modify the plan

Return STRICT JSON only with this exact shape:
{
  "faithfulness": {
    "score": float,
    "issues": [
      {
        "description": "...",
        "frame_start": int,
        "frame_end": int,
        "joint": "...",
        "reason": "..."
      }
    ]
  },
  "realism": {
    "score": float,
    "issues": [
      {
        "joint": "...",
        "frames": [int],
        "type": "constraint_violation | discontinuity | unnatural_motion",
        "description": "...",
        "reason": "..."
      }
    ]
  }
}

Rules:
- Scores must be floats in the range [0.0, 1.0].
- Use empty lists when there are no issues.
- Every issue must include either integer frame_start and frame_end fields, or a non-empty
  frames list of integers.
- If an issue appears in only one sampled frame, use frames: [frame_number].
- Each issue must describe what looks wrong, where it appears wrong, and why it appears wrong.
- Do not include keys named suggestion, edit, old_text, new_text, angle, degrees, increase_rotation, or parameter_change.
- Do not mention missing root movement, missing forward displacement, missing travel distance, or lack of world-space movement.
- Focus on realism, coordination, pose quality, motion quality, timing consistency, and faithfulness to the prompt.
- Return valid JSON only. No markdown. No prose outside JSON.
- Be deterministic and conservative."""


def evaluate_motion(prompt: str, images: List[np.ndarray]) -> dict:
    if not images:
        raise ValueError("evaluate_motion requires at least one rendered image.")

    HumanMessage, SystemMessage, ChatGoogleGenerativeAI = _import_critic_ai_dependencies()

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "User motion prompt:\n"
                f"{prompt}\n\n"
                "Evaluate the sampled motion frames and return strict JSON only. "
                "Use frame_start/frame_end or frames for every issue."
            ),
        }
    ]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image),
                },
            }
        )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    last_error: Exception | None = None
    retry_messages = list(messages)
    for _ in range(3):
        try:
            response = llm.invoke(retry_messages)
            parsed = parse_response_json(
                extract_response_text(response.content),
                "Critic returned an empty response.",
            )
            _validate_feedback(parsed)
            _remove_translation_issues(parsed)
            return parsed
        except Exception as error:
            last_error = error
            retry_messages = retry_messages + [
                HumanMessage(
                    content="Return only valid JSON matching the required schema. Fix formatting errors."
                )
            ]

    raise ValueError(f"Critic failed to return valid JSON: {last_error}") from last_error


def _validate_feedback(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Critic payload must be a JSON object.")

    for section_name in ("faithfulness", "realism"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Missing `{section_name}` section in critic payload.")

        score = section.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"`{section_name}.score` must be numeric.")
        if float(score) < 0.0 or float(score) > 1.0:
            raise ValueError(f"`{section_name}.score` must be between 0.0 and 1.0.")

        issues = section.get("issues")
        if not isinstance(issues, list):
            raise ValueError(f"`{section_name}.issues` must be a list.")

        for issue in issues:
            if not isinstance(issue, dict):
                raise ValueError(f"`{section_name}.issues` items must be objects.")
            _reject_plan_edit_fields(issue)
            description = issue.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"`{section_name}.issues[].description` must be non-empty.")
            reason = issue.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"`{section_name}.issues[].reason` must be non-empty.")
            _validate_issue_frames(issue, section_name)

    if "priority_fixes" in payload:
        raise ValueError("Critic payload must not include `priority_fixes`; observations only.")


def _reject_plan_edit_fields(issue: Dict[str, Any]) -> None:
    forbidden_keys = {
        "suggestion",
        "edit",
        "old_text",
        "new_text",
        "angle",
        "angles",
        "degrees",
        "increase_rotation",
        "decrease_rotation",
        "parameter_change",
    }
    present = forbidden_keys.intersection(issue)
    if present:
        raise ValueError(f"Critic issue contains plan-edit fields: {sorted(present)}")


def _validate_issue_frames(issue: Dict[str, Any], section_name: str) -> None:
    frames = issue.get("frames")
    frame_start = issue.get("frame_start")
    frame_end = issue.get("frame_end")

    if isinstance(frames, list) and frames:
        if not all(isinstance(frame, int) for frame in frames):
            raise ValueError(f"`{section_name}.issues[].frames` must contain only integers.")
        issue.pop("frame", None)
        return

    if frame_start is not None or frame_end is not None:
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise ValueError(
                f"`{section_name}.issues[]` must include integer frame_start and frame_end."
            )
        if frame_end < frame_start:
            raise ValueError(f"`{section_name}.issues[].frame_end` must be >= frame_start.")
        issue.pop("frame", None)
        return

    legacy_frame = issue.get("frame")
    if isinstance(legacy_frame, int):
        issue["frames"] = [legacy_frame]
        issue.pop("frame", None)
        return

    raise ValueError(
        f"`{section_name}.issues[]` must include either frames or frame_start/frame_end."
    )


def _remove_translation_issues(payload: dict) -> None:
    for section_name in ("faithfulness", "realism"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        issues = section.get("issues")
        if not isinstance(issues, list):
            continue
        section["issues"] = [
            issue
            for issue in issues
            if not _is_translation_issue(issue)
        ]


def _is_translation_issue(issue: Dict[str, Any]) -> bool:
    text = " ".join(str(value).lower() for value in issue.values())
    blocked_terms = (
        "root translation",
        "travel distance",
        "locomotion distance",
        "forward displacement",
        "world-space",
        "world space",
        "move forward",
        "moving forward",
        "does not travel",
        "doesn't travel",
    )
    return any(term in text for term in blocked_terms)
