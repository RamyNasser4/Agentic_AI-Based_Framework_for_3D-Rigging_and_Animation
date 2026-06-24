from __future__ import annotations

import base64
import json
from os import getenv
from typing import Any, Dict, List

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64



CRITIC_SYSTEM_PROMPT = """You are a strict animation critic and evaluator.

You will receive:
- the user's motion prompt
- a small ordered set of skeleton-render images on black backgrounds

Each image includes a visible frame number label. Use that visible frame number in all issue.frame fields.

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
        "frame": int,
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
        "frame": int,
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
- Each issue must describe what looks wrong, where it appears wrong, and why it appears wrong.
- Do not include keys named suggestion, edit, old_text, new_text, angle, degrees, increase_rotation, or parameter_change.
- Do not mention missing root movement, missing forward displacement, missing travel distance, or lack of world-space movement.
- Focus on realism, coordination, pose quality, motion quality, timing consistency, and faithfulness to the prompt.
- Return valid JSON only. No markdown. No prose outside JSON.
- Be deterministic and conservative."""


def evaluate_motion(prompt: str, images: List[np.ndarray]) -> dict:
    if not images:
        raise ValueError("evaluate_motion requires at least one rendered image.")

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
                "Evaluate the sampled motion frames and return strict JSON only."
            ),
        }
    ]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(image),
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
            parsed = _parse_response_json(_extract_response_text(response.content))
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


def _image_to_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode skeleton image as PNG.")
    encoded_bytes = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_bytes}"


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _parse_response_json(text: str) -> dict:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Critic returned an empty response.")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])


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
            frame = issue.get("frame")
            if not isinstance(frame, int):
                raise ValueError(f"`{section_name}.issues[].frame` must be an integer.")

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
