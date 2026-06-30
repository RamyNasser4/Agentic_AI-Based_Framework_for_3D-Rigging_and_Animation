from __future__ import annotations

from os import getenv
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

try:
    from .mesh_renderer import VisualEvidence
    from .multimodal_utils import compress_image_for_llm, extract_response_text, parse_response_json
except ImportError:  # pragma: no cover - direct script fallback
    from mesh_renderer import VisualEvidence
    from multimodal_utils import compress_image_for_llm, extract_response_text, parse_response_json


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    frames: Optional[List[int]] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None

    @model_validator(mode="after")
    def validate_frame_reference(self) -> "Issue":
        if self.frames:
            return self
        if self.frame_start is not None and self.frame_end is not None:
            if self.frame_end < self.frame_start:
                raise ValueError("frame_end must be greater than or equal to frame_start.")
            return self
        raise ValueError("Issue must include frames or frame_start/frame_end.")


class CriticSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    issues: List[Issue]


class CriticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: CriticSection
    realism: CriticSection


CRITIC_SYSTEM_PROMPT = """You are a strict animation critic and evaluator.

You receive:
- a motion prompt
- mesh renders of a character animation across time (primary evidence)
- optional skeleton sequence (secondary evidence)
- optional critic history across iterations

IMPORTANT:
Mesh renders are generated using mesh_renderer.py and are the canonical source of truth.

You are evaluating a temporal multi-view animation sequence.

Each frame is a labeled four-view collage containing FRONT, RIGHT, TOP, and PERSPECTIVE.

TASK:
Evaluate motion quality across time.

Focus on:
- pose correctness
- limb coordination
- temporal smoothness
- physical plausibility
- motion continuity across frames

DO NOT evaluate:
- root/world translation
- locomotion distance

MEMORY RULE:
If critic history exists:
- compare against previous issues
- classify as new / persistent / resolved
- do NOT duplicate resolved issues

OUTPUT STRICT JSON ONLY:
{
  "faithfulness": {"score": float, "issues": []},
  "realism": {"score": float, "issues": []}
}

Each issue object MUST have one of these exact structures:

{
  "description": "...",
  "reason": "...",
  "frames": [12, 24]
}

or

{
  "description": "...",
  "reason": "...",
  "frame_start": 12,
  "frame_end": 24
}

description:
- Describe WHAT is visually wrong.

reason:
- Explain WHY this is incorrect, unrealistic, or inconsistent with the prompt.

Both description and reason are REQUIRED.
Never omit the reason field.

Rules:
- Scores must be floats in the range [0.0, 1.0].
- Use empty lists when there are no issues.
- Every issue object must contain description.
- Every issue object must contain reason.
- Every issue object must contain either:
  - frames, or
  - frame_start and frame_end.
- If an issue appears in only one sampled frame, use frames: [frame_number].
- Each issue must describe what looks wrong, where it appears wrong, and why it appears wrong.
- Do not invent additional keys.
- Do not include keys named suggestion, edit, old_text, new_text, angle, degrees, increase_rotation, or parameter_change.
- Do not mention missing root movement, missing forward displacement, missing travel distance, or lack of world-space movement.
- Focus on realism, coordination, pose quality, motion quality, timing consistency, and faithfulness to the prompt.
- Return valid JSON only. No markdown. No prose outside JSON.
- Be deterministic and conservative.

Example valid output:
{
  "faithfulness": {
    "score": 0.35,
    "issues": [
      {
        "description": "The character stops walking after frame 27.",
        "reason": "The prompt requests a continuous walking motion, but the animation freezes.",
        "frames": [27, 97]
      }
    ]
  },
  "realism": {
    "score": 0.55,
    "issues": [
      {
        "description": "The arms remain rigid throughout the gait.",
        "reason": "Natural walking requires coordinated arm swing to balance the body.",
        "frame_start": 15,
        "frame_end": 90
      }
    ]
  }
}"""


def evaluate_motion(
    prompt: str,
    visual: VisualEvidence,
    critic_state: Optional[dict] = None,
) -> dict:
    if not isinstance(visual, dict):
        raise TypeError("evaluate_motion requires a VisualEvidence dictionary.")
    mesh_collages = _collages_from_visual(visual, "mesh_collages")
    if not mesh_collages:
        raise ValueError("evaluate_motion requires visual['mesh_collages'] with at least one frame.")
    skeleton_collages = _collages_from_visual(visual, "skeleton_collages")

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-3.5-flash",
    #     google_api_key=getenv("GOOGLE_API_KEY"),
    #     temperature=0
    # )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "User motion prompt:\n"
                f"{prompt}\n\n"
                "Evaluate this temporal multi-view animation sequence. Mesh renders are "
                "primary evidence. Skeleton renders, when present, are secondary evidence. "
                "Use frame_start/frame_end or frames for every issue.\n\n"
                "Visual evidence summary:\n"
                f"{_visual_summary(visual)}\n\n"
                "Critic history:\n"
                f"{_history_summary(critic_state)}"
            ),
        }
    ]
    _append_collage_sequence(
        content,
        "Mesh",
        mesh_collages,
        list(visual.get("mesh_frame_indices") or []),
    )
    _append_collage_sequence(
        content,
        "Skeleton",
        skeleton_collages,
        list(visual.get("skeleton_frame_indices") or []),
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
            validated = CriticOutput.model_validate(parsed)
            payload = validated.model_dump()
            _remove_translation_issues(payload)
            _apply_critic_memory(payload, critic_state)
            validated = CriticOutput.model_validate(payload)
            return validated.model_dump()
        except Exception as error:
            last_error = error
            retry_messages = retry_messages + [
                HumanMessage(
                    content=_validation_retry_prompt(error)
                )
            ]

    raise ValueError(f"Critic failed to return valid JSON: {last_error}") from last_error


def _append_collage_sequence(
    content: List[Dict[str, Any]],
    label: str,
    sequence: List[Any],
    frame_indices: List[int],
) -> None:
    for index, image in enumerate(sequence):
        collage = _collage_image(image)
        if collage is None:
            continue
        frame_index = frame_indices[index] if index < len(frame_indices) else index
        content.append(
            {
                "type": "text",
                "text": f"{label} collage frame {frame_index}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": compress_image_for_llm(collage)},
            }
        )


def _visual_summary(visual: VisualEvidence) -> str:
    mesh_frame_ids = list(visual.get("mesh_frame_indices") or [])
    skeleton_frame_ids = list(visual.get("skeleton_frame_indices") or [])
    return (
        f"mesh_frames={mesh_frame_ids}; "
        f"skeleton_frames={skeleton_frame_ids}; "
        f"mesh_collages={len(visual.get('mesh_collages') or [])}; "
        f"skeleton_collages={len(visual.get('skeleton_collages') or [])}"
    )


def _history_summary(critic_state: Optional[dict]) -> str:
    if not critic_state:
        return "None."
    return str(
        {
            "iteration": critic_state.get("iteration"),
            "previous_issues": critic_state.get("previous_issues", []),
            "resolved_issues": critic_state.get("resolved_issues", []),
            "previous_score": critic_state.get("previous_score"),
        }
    )


def _collages_from_visual(visual: VisualEvidence, key: str) -> List[Any]:
    return list(visual.get(key) or [])


def _collage_image(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("collage")
    return getattr(item, "collage", item)


def _validation_retry_prompt(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = _format_validation_errors(error)
    else:
        details = str(error)

    return (
        "The previous response did not satisfy the required JSON schema.\n\n"
        "Validation errors:\n\n"
        f"{details}\n\n"
        "Fix only the schema violations.\n"
        "Do not change the semantic content unless necessary.\n"
        "Every issue object must contain description, reason, and either frames or "
        "frame_start/frame_end.\n"
        "Do not invent additional keys.\n"
        "Return valid JSON only."
    )


def _format_validation_errors(error: ValidationError) -> str:
    lines: List[str] = []
    for item in error.errors():
        location = _format_validation_location(item.get("loc", ()))
        message = str(item.get("msg") or "Invalid value")
        if location:
            lines.append(location)
        lines.append(message)
        lines.append("")
    return "\n".join(lines).strip() or str(error)


def _format_validation_location(location: Any) -> str:
    if not isinstance(location, (list, tuple)):
        return str(location)

    formatted = ""
    for part in location:
        if isinstance(part, int):
            formatted += f"[{part}]"
        elif formatted:
            formatted += f".{part}"
        else:
            formatted = str(part)
    return formatted


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


def _apply_critic_memory(payload: dict, critic_state: Optional[dict]) -> None:
    current_issues = _flatten_issues(payload)
    if critic_state is None:
        return

    previous_records = list(critic_state.get("previous_issues") or [])
    resolved_records = list(critic_state.get("resolved_issues") or [])
    previous_by_key = {_record_key(record): record for record in previous_records}
    resolved_keys = {_record_key(record) for record in resolved_records}

    retained_current: List[Dict[str, Any]] = []
    current_keys = set()
    for record in current_issues:
        key = record["fingerprint"]
        current_keys.add(key)
        if key in resolved_keys:
            continue
        record["status"] = "persistent" if key in previous_by_key else "new"
        retained_current.append(record)

    previous_keys = set(previous_by_key)
    newly_resolved = [
        dict(previous_by_key[key], status="resolved")
        for key in sorted(previous_keys - current_keys)
    ]
    resolved_by_key = {
        _record_key(record): record
        for record in [*resolved_records, *newly_resolved]
    }

    _replace_payload_issues(payload, retained_current)
    critic_state["iteration"] = int(critic_state.get("iteration") or 0) + 1
    critic_state["previous_issues"] = [
        dict(_issue_record(record["section"], record["issue"]), status=record.get("status"))
        for record in retained_current
    ]
    critic_state["resolved_issues"] = list(resolved_by_key.values())
    critic_state["previous_score"] = {
        "faithfulness": payload.get("faithfulness", {}).get("score"),
        "realism": payload.get("realism", {}).get("score"),
    }


def _flatten_issues(payload: dict) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for section_name in ("faithfulness", "realism"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for issue in section.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            record = _issue_record(section_name, issue)
            records.append(
                {
                    "section": section_name,
                    "issue": issue,
                    "fingerprint": record["fingerprint"],
                }
            )
    return records


def _replace_payload_issues(payload: dict, records: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {"faithfulness": [], "realism": []}
    for record in records:
        grouped.setdefault(record["section"], []).append(record["issue"])

    for section_name in ("faithfulness", "realism"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            section["issues"] = grouped.get(section_name, [])


def _issue_record(section_name: str, issue: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "section": section_name,
        "fingerprint": _issue_fingerprint(section_name, issue),
        "joint": issue.get("joint"),
        "type": issue.get("type"),
        "description": issue.get("description"),
        "frames": issue.get("frames"),
        "frame_start": issue.get("frame_start"),
        "frame_end": issue.get("frame_end"),
        "status": issue.get("status"),
    }


def _record_key(record: Dict[str, Any]) -> str:
    fingerprint = record.get("fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    return _issue_fingerprint(str(record.get("section") or ""), record)


def _issue_fingerprint(section_name: str, issue: Dict[str, Any]) -> str:
    frame_key = _issue_frame_key(issue)
    text_parts = [
        section_name,
        str(issue.get("joint") or ""),
        str(issue.get("type") or ""),
        str(issue.get("description") or ""),
        str(issue.get("reason") or ""),
    ]
    normalized_text = " ".join(_normalize_text(part) for part in text_parts if part)
    return f"{normalized_text}|{frame_key}"


def _issue_frame_key(issue: Dict[str, Any]) -> str:
    frames = issue.get("frames")
    if isinstance(frames, list) and frames:
        return ",".join(str(frame) for frame in frames)
    frame_start = issue.get("frame_start")
    frame_end = issue.get("frame_end")
    if frame_start is not None or frame_end is not None:
        return f"{frame_start}-{frame_end}"
    return ""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()