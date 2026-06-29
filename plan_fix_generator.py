from __future__ import annotations

import json
from os import getenv
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from .multimodal_utils import extract_response_text, image_to_data_url, parse_response_json
except ImportError:  # pragma: no cover - direct script fallback
    from multimodal_utils import extract_response_text, image_to_data_url, parse_response_json


PLAN_FIX_SYSTEM_PROMPT = """You are a multimodal Plan Refiner for a Blender animation pipeline.

You receive:
- the original user motion prompt
- the current numbered animation plan
- critic observations about rendered skeleton frames
- the same sampled rendered skeleton images that were given to the critic

Your job:
- Verify critic observations against the rendered images.
- Use the images to understand pose quality, timing, coordination, and motion phase.
- Determine which existing plan step caused the visible issue.
- Convert critic observations into concrete, local edits to existing plan steps.
- Modify only the step or steps needed to address the visible observations.
- Preserve the user's original intent.
- Preserve all unrelated plan steps and actions.
- Do not regenerate the whole plan.
- Produce the minimum number of edits required.

The critic diagnoses what is wrong. The rendered images explain why it is wrong. The current
plan explains how the motion was generated. You are the only module responsible for deciding
how to modify the plan.

Return STRICT JSON only with this exact shape:
{
  "plan_fixes": [
    {
      "step": 1,
      "reason": "...",
      "old_text": "...",
      "new_text": "..."
    }
  ]
}

Rules:
- Each plan_fixes item must target one existing Step N.
- old_text must quote either the full existing step body or an exact existing action phrase from that step.
- new_text must be the direct replacement for old_text.
- Use concrete animation operations only: rotate/move, joint name, axis token, numeric magnitude, and units.
- Prefer changing existing rotation magnitudes, timing/phase actions, or visible limb coordination.
- Modify only the existing step that caused the visual issue whenever possible.
- Preserve all unrelated actions in that step exactly.
- Do not add new steps.
- Do not replace the whole plan.
- Do not create vague suggestions such as "increase realism" or "make it better".
- Do not introduce edits that only change root/world translation.
- Do not edit root/world translation to address travel distance, locomotion distance, forward displacement, or movement through world space.
- If the observations do not justify a concrete visible body-motion edit, return {"plan_fixes": []}.
- Return valid JSON only. No markdown. No prose outside JSON."""


_STEP_PATTERN = re.compile(
    r"Step\s+(\d+)\s*:\s*(.*?)(?=Step\s+\d+\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def generate_plan_fixes(
    original_prompt: str,
    current_plan: str,
    critic_feedback: Dict[str, Any],
    rendered_images: Optional[List[np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    if not _has_feedback_issues(critic_feedback):
        return []

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Original user prompt:\n"
                f"{str(original_prompt or '').strip()}\n\n"
                "Current numbered animation plan:\n"
                f"{str(current_plan or '').strip()}\n\n"
                "Critic observations JSON:\n"
                f"{json.dumps(critic_feedback or {}, ensure_ascii=True, indent=2)}\n\n"
                "The following images are the same sampled skeleton renders that were given "
                "to the critic. Use them to verify the observations and localize the minimal "
                "plan edit. Return strict JSON plan fixes only."
            ),
        }
    ]
    for index, image in enumerate(rendered_images or [], start=1):
        content.append(
            {
                "type": "text",
                "text": f"Sampled skeleton image {index}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image),
                },
            }
        )

    messages = [
        SystemMessage(content=PLAN_FIX_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    last_error: Exception | None = None
    retry_messages = list(messages)
    valid_steps = _plan_step_numbers(current_plan)

    for _ in range(3):
        try:
            response = llm.invoke(retry_messages)
            payload = parse_response_json(
                extract_response_text(response.content),
                "Plan refiner returned an empty response.",
            )
            return _validate_plan_fixes(payload, valid_steps)
        except Exception as error:
            last_error = error
            retry_messages = retry_messages + [
                HumanMessage(
                    content=(
                        "Return only valid JSON matching the required schema. "
                        "Every fix must include step, reason, old_text, and new_text."
                    )
                )
            ]

    raise ValueError(f"Plan refiner failed to return valid JSON: {last_error}") from last_error


def _has_feedback_issues(feedback: Dict[str, Any]) -> bool:
    if not isinstance(feedback, dict):
        return False
    for section_name in ("faithfulness", "realism"):
        section = feedback.get(section_name, {})
        if isinstance(section, dict) and section.get("issues"):
            return True
    return bool(feedback.get("observations"))


def _plan_step_numbers(plan_text: str) -> set[int]:
    return {int(match.group(1)) for match in _STEP_PATTERN.finditer(str(plan_text or ""))}


def _validate_plan_fixes(payload: dict, valid_steps: Sequence[int] | set[int]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError("Plan fix payload must be a JSON object.")

    plan_fixes = payload.get("plan_fixes")
    if not isinstance(plan_fixes, list):
        raise ValueError("Plan fix payload must include a `plan_fixes` list.")

    valid_step_set = set(valid_steps or [])
    validated: List[Dict[str, Any]] = []
    for index, fix in enumerate(plan_fixes, start=1):
        if not isinstance(fix, dict):
            raise ValueError(f"plan_fixes[{index}] must be an object.")

        step = fix.get("step")
        if not isinstance(step, int):
            raise ValueError(f"plan_fixes[{index}].step must be an integer.")
        if valid_step_set and step not in valid_step_set:
            raise ValueError(f"plan_fixes[{index}].step references unknown Step {step}.")

        reason = str(fix.get("reason") or "").strip()
        old_text = str(fix.get("old_text") or "").strip()
        new_text = str(fix.get("new_text") or "").strip()
        if not reason or not old_text or not new_text:
            raise ValueError(
                f"plan_fixes[{index}] must include non-empty reason, old_text, and new_text."
            )
        if _is_translation_only_edit(old_text, new_text):
            raise ValueError(
                f"plan_fixes[{index}] must not introduce a root/world translation-only edit."
            )

        validated.append(
            {
                "step": step,
                "reason": reason,
                "old_text": old_text,
                "new_text": new_text,
            }
        )

    return validated


def _is_translation_only_edit(old_text: str, new_text: str) -> bool:
    combined = f"{old_text} {new_text}".lower()
    if "rotate " in combined:
        return False
    if "move " not in combined:
        return False
    blocked_terms = (
        "root",
        "world",
        "travel distance",
        "locomotion distance",
        "forward displacement",
        "movement through world",
    )
    return any(term in combined for term in blocked_terms)
