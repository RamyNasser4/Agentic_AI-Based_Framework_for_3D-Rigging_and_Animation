from __future__ import annotations

import json
from os import getenv
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ConfigDict, Field

try:
    from .mesh_renderer import VisualEvidence
    from .multimodal_utils import compress_image_for_llm, extract_response_text, parse_response_json
except ImportError:  # pragma: no cover - direct script fallback
    from mesh_renderer import VisualEvidence
    from multimodal_utils import compress_image_for_llm, extract_response_text, parse_response_json


class PlanFix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    reason: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str = Field(min_length=1)


class PlanFixOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_fixes: List[PlanFix]


PLAN_FIX_SYSTEM_PROMPT = """You are a multimodal Plan Refiner for a Blender animation pipeline.

You receive:
- the original user motion prompt
- the current numbered animation plan
- critic observations about rendered mesh frames
- the same mesh_collages that were given to the critic

Your job:
- Verify critic observations against the mesh_collages only.
- Use the mesh_collages to understand pose quality, timing, coordination, and motion phase.
- Determine which existing plan step caused the visible issue.
- Convert critic observations into concrete, local edits to existing plan steps.
- Modify only the step or steps needed to address the visible observations.
- Preserve the user's original intent.
- Preserve all unrelated plan steps and actions.
- Do not regenerate the whole plan.
- Produce the minimum number of edits required.

The critic diagnoses what is wrong. The mesh_collages explain why it is wrong. The current
plan explains how the motion was generated. You are the only module responsible for deciding
how to modify the plan. Do not request, trigger, recompute, regenerate, or assume any new
renders.

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


def generate_plan_fixes(
    original_prompt: str,
    current_plan: str,
    critic_feedback: Dict[str, Any],
    visual: Optional[VisualEvidence] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not _has_feedback_issues(critic_feedback):
        return {"plan_fixes": []}
    mesh_collages = _mesh_collages(visual)
    if not mesh_collages:
        raise ValueError("generate_plan_fixes requires VisualEvidence with mesh_collages.")

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
                "Use only the mesh_collages from the VisualEvidence that was given to the "
                "critic. Do not use skeleton evidence as primary evidence. Do not trigger "
                "or recompute rendering. Return strict JSON plan fixes only."
            ),
        }
    ]
    frame_indices = list((visual or {}).get("mesh_frame_indices") or [])
    for index, image in enumerate(mesh_collages):
        collage = _collage_image(image)
        if collage is None:
            continue
        frame_index = frame_indices[index] if index < len(frame_indices) else index
        content.append(
            {
                "type": "text",
                "text": f"Mesh collage frame {frame_index}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": compress_image_for_llm(collage)},
            }
        )

    messages = [
        SystemMessage(content=PLAN_FIX_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    last_error: Exception | None = None
    retry_messages = list(messages)

    for _ in range(3):
        try:
            response = llm.invoke(retry_messages)
            payload = parse_response_json(
                extract_response_text(response.content),
                "Plan refiner returned an empty response.",
            )
            validated = PlanFixOutput.model_validate(payload)
            return validated.model_dump()
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


def _mesh_collages(visual: Optional[VisualEvidence]) -> List[Any]:
    if not isinstance(visual, dict):
        return []
    return list(visual.get("mesh_collages") or [])


def _collage_image(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("collage")
    return getattr(item, "collage", item)
