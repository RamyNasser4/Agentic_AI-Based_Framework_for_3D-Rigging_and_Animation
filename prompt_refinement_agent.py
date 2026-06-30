from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .planner_agent import FREE_MODELS, get_llm


MAX_REFINEMENT_QUESTIONS = 10
REFINEMENT_DECISION_ATTEMPTS = 3
DEFAULT_REFINEMENT_MODEL = FREE_MODELS[0] if FREE_MODELS else "openrouter/free"

RIG_CAPABILITY_ALIASES = {
    "arm": ("arm", "shoulder", "elbow", "forearm", "wrist", "hand", "palm"),
    "ear": ("ear",),
    "eye": ("eye",),
    "finger": ("finger", "thumb", "index", "middle", "ring", "pinky"),
    "fin": ("fin", "flipper"),
    "flipper": ("flipper", "fin"),
    "foot": ("foot", "toe", "ankle"),
    "hand": ("hand", "palm", "wrist"),
    "head": ("head", "neck"),
    "leg": ("leg", "hip", "knee", "ankle", "foot", "thigh", "shin", "toe"),
    "tail": ("tail",),
    "wing": ("wing",),
}

REFINEMENT_SYSTEM_PROMPT = """You are a Prompt Refinement Agent for a Blender animation generator.

Your job is to reduce ambiguity in the current or evolving user animation request before the animation planner runs.

You receive:
- the current prompt, which may already include resolved clarifications
- object name
- object JSON rig hierarchy
- clarification history
- question count

Rig-awareness rules:
- You must inspect the object JSON before deciding whether to ask a question.
- Use the object JSON only to understand what body parts, joints, appendages, and structural capabilities the object has.
- Base questions on both the current prompt and the supplied rig hierarchy.
- Ask about a body part or capability only if the object JSON includes a matching joint or hierarchy branch.
- If the rig contains wings, wing-related clarification questions may be appropriate.
- If the rig contains a tail, tail-related clarification questions may be appropriate.
- If the rig does not contain a body part, do not ask about that body part.
- Do not perform animation planning, joint rotation planning, step planning, keyframe planning, or motion generation.
- Do not mention exact rotation axes, angle values, keyframes, or planner instructions.

Ask another question only when the answer would produce a different animation plan.

Allowed question topics:
- body action selection
- action order
- direction of travel or turn
- whether motion is in place or moves through space
- which limb/side/object performs an action
- whether repeated actions should cycle
- whether a vague action should include a concrete movement that changes the plan

Forbidden question topics:
- style, realism, quality, smoothness, speed preference, emotion, mood
- camera, lighting, rendering, colors, materials, appearance, outfit, environment
- animation settings, duration, frame rate, keyframe density, resolution, export options
- preferences that do not change the actual body-motion plan

Question rules:
- Ask at most one question.
- The question must be answerable by Yes, No, or Skip.
- The question must affect the animation plan.
- Do not ask open-ended questions.
- Do not ask for style, realism, camera, lighting, or appearance.
- If the prompt is already specific enough, stop.
- If question_count is 10 or more, stop.

Return STRICT JSON only with one of these exact shapes:
{{"status": "QUESTION", "question": "..."}}
{{"status": "STOP"}}"""


refinement_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REFINEMENT_SYSTEM_PROMPT),
        (
            "human",
            (
                "Current prompt:\n{current_prompt}\n\n"
                "Object name:\n{object_name}\n\n"
                "Object JSON rig hierarchy:\n{object_json}\n\n"
                "Clarification history:\n{clarification_history}\n\n"
                "Question count: {question_count}\n"
                "Maximum questions: {max_questions}\n\n"
                "{format_feedback}"
                "Return strict JSON only."
            ),
        ),
    ]
)


final_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite a Blender animation request after clarification questions. "
                "Return only the final user prompt, with no markdown, labels, or Q/A history.\n"
                "Use the original request as the base. Incorporate only Yes or No answers that "
                "change the body-motion animation plan. Ignore Skip answers. Do not add style, "
                "realism, camera, lighting, appearance, timing settings, or rendering details. "
                "Do not invent new motion details that are not implied by the original request "
                "and clarification answers. If there are no useful non-Skip clarifications, "
                "return the original request unchanged."
            ),
        ),
        (
            "human",
            (
                "Original request:\n{original_prompt}\n\n"
                "Clarification history:\n{clarification_history}\n\n"
                "Final prompt only:"
            ),
        ),
    ]
)


@lru_cache(maxsize=4)
def _get_refinement_llm(model: str):
    return get_llm(model)


class PromptRefinementAgent:
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_REFINEMENT_MODEL
        llm = _get_refinement_llm(self.model)
        self.chain = refinement_prompt | llm | StrOutputParser()
        self.final_prompt_chain = final_prompt_template | llm | StrOutputParser()

    def decide(
        self,
        current_prompt: str,
        object_name: str = "",
        object_json: str = "",
        clarification_history: Sequence[Dict[str, str]] | str | None = None,
        question_count: int = 0,
    ) -> Dict[str, str]:
        if question_count >= MAX_REFINEMENT_QUESTIONS:
            return {"status": "STOP"}

        format_feedback = ""
        object_json_text = str(object_json or "").strip()

        for _ in range(REFINEMENT_DECISION_ATTEMPTS):
            response_text = self.chain.invoke(
                {
                    "current_prompt": str(current_prompt or "").strip(),
                    "object_name": str(object_name or "").strip() or "Unknown object",
                    "object_json": object_json_text or "No object JSON supplied.",
                    "clarification_history": format_clarification_history(
                        clarification_history
                    ),
                    "question_count": int(question_count),
                    "max_questions": MAX_REFINEMENT_QUESTIONS,
                    "format_feedback": format_feedback,
                }
            )
            try:
                return _validate_decision(
                    _parse_response_json(response_text),
                    object_json=object_json_text,
                )
            except Exception as error:
                format_feedback = _build_retry_feedback(error)

        return {"status": "STOP"}

    def synthesize_final_prompt(
        self,
        original_prompt: str,
        clarification_history: Sequence[Dict[str, str]] | str | None = None,
    ) -> str:
        prompt = str(original_prompt or "").strip()
        if not _has_useful_clarifications(clarification_history):
            return prompt

        response = self.final_prompt_chain.invoke(
            {
                "original_prompt": prompt,
                "clarification_history": format_clarification_history(clarification_history),
            }
        )
        final_prompt = _extract_response_text(response).strip()
        return final_prompt or build_refined_prompt(prompt, clarification_history)


def build_refined_prompt(
    original_prompt: str,
    clarification_history: Sequence[Dict[str, str]] | None,
) -> str:
    prompt = str(original_prompt or "").strip()
    answered_items: List[str] = []

    for item in clarification_history or []:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip().title()
        if not question or answer == "Skip":
            continue

        answered_items.append(f"- {question} Answer: {answer}.")

    if not answered_items:
        return prompt

    return (
        f"{prompt}\n\n"
        "Resolved animation clarifications that are part of this request:\n"
        + "\n".join(answered_items)
    ).strip()


def format_clarification_history(
    clarification_history: Sequence[Dict[str, str]] | str | None,
) -> str:
    if not clarification_history:
        return "None."

    if isinstance(clarification_history, str):
        text = clarification_history.strip()
        return text or "None."

    lines = []
    for index, item in enumerate(clarification_history, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            lines.append(f"{index}. Q: {question} A: {answer}")

    return "\n".join(lines) if lines else "None."


def _has_useful_clarifications(
    clarification_history: Sequence[Dict[str, str]] | str | None,
) -> bool:
    if isinstance(clarification_history, str):
        return bool(clarification_history.strip() and clarification_history.strip() != "None.")

    for item in clarification_history or []:
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer") or "").strip().title()
        if answer in {"Yes", "No"}:
            return True

    return False


def _parse_response_json(text: Any) -> Dict[str, Any]:
    candidate = _extract_response_text(text).strip()
    if not candidate:
        raise ValueError("Prompt refinement returned an empty response.")

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])

    if not isinstance(payload, dict):
        raise TypeError("Prompt refinement response must be a JSON object.")
    return payload


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _build_retry_feedback(error: Exception) -> str:
    reason = str(error).strip() or "The previous clarification question was invalid."
    return (
        "Previous response was rejected by the refinement validator and will not be shown "
        "to the user.\n"
        f"Rejection reason: {reason}\n"
        "Discard that question. Generate a different high-level Yes/No/Skip clarification "
        "only if another remaining ambiguity would produce a substantially different "
        "animation plan. Do not ask about joint rotations, exact angles, axes, keyframes, "
        "or other planner-only details. Do not ask about body parts that are absent from "
        "the object JSON. If no valid high-level clarification remains, return exactly "
        '{"status": "STOP"}.\n\n'
    )


def _validate_decision(payload: Dict[str, Any], object_json: str = "") -> Dict[str, str]:
    status = str(payload.get("status") or "").strip().upper()
    if status == "STOP":
        return {"status": "STOP"}

    if status != "QUESTION":
        raise ValueError("Prompt refinement status must be QUESTION or STOP.")

    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("QUESTION response must include a non-empty question.")

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    if _contains_forbidden_question_topic(question):
        raise ValueError(f"Question uses a forbidden topic: {question}")

    if _contains_planning_detail(question):
        raise ValueError(f"Question includes planner-only detail: {question}")

    missing_capability = _missing_rig_capability(question, object_json)
    if missing_capability:
        raise ValueError(
            f"Question asks about '{missing_capability}', which is not present in the object JSON."
        )

    if _looks_open_ended(question):
        raise ValueError(f"Question is not a Yes/No/Skip question: {question}")

    return {"status": "QUESTION", "question": question}


def _contains_forbidden_question_topic(question: str) -> bool:
    return bool(
        re.search(
            r"\b(style|realism|realistic|camera|lighting|appearance|color|colour|"
            r"material|texture|render|frame rate|duration|resolution|export|"
            r"quality|smoothness|outfit|environment)\b",
            question,
            re.IGNORECASE,
        )
    )


def _contains_planning_detail(question: str) -> bool:
    return bool(
        re.search(
            r"\b(keyframe|keyframes|rotate|rotation|axis|axes|degrees?|units?)\b|[+-][XYZ]\b",
            question,
            re.IGNORECASE,
        )
    )


def _missing_rig_capability(question: str, object_json: str) -> str:
    if not object_json:
        return ""

    question_text = question.lower()
    rig_text = object_json.lower()

    for capability, aliases in RIG_CAPABILITY_ALIASES.items():
        if not re.search(rf"\b{re.escape(capability)}s?\b", question_text):
            continue
        if not any(
            re.search(rf"(?<![a-z]){re.escape(alias)}[.\w-]*\b", rig_text)
            for alias in aliases
        ):
            return capability

    return ""


def _looks_open_ended(question: str) -> bool:
    return bool(
        re.match(
            r"\s*(what|which|who|where|when|why|how|describe|choose|select)\b",
            question,
            re.IGNORECASE,
        )
    )
