from __future__ import annotations

import base64
import json
from typing import Any, List

import cv2
import numpy as np


def compress_image_for_llm(image: np.ndarray, max_size_kb: int = 200) -> str:
    if image is None:
        raise ValueError("Cannot compress an empty image for LLM input.")

    max_bytes = max(1, int(max_size_kb)) * 1024
    jpeg_image = _normalize_for_jpeg(image)

    while True:
        for quality in range(95, 4, -5):
            encoded_bytes = _encode_jpeg(jpeg_image, quality)
            if len(encoded_bytes) <= max_bytes:
                return _jpeg_data_url(encoded_bytes)

        height, width = jpeg_image.shape[:2]
        if height <= 64 or width <= 64:
            return _jpeg_data_url(_encode_jpeg(jpeg_image, 5))
        next_width = max(64, int(width * 0.85))
        next_height = max(64, int(height * 0.85))
        jpeg_image = cv2.resize(jpeg_image, (next_width, next_height), interpolation=cv2.INTER_AREA)


def image_to_data_url(image: np.ndarray) -> str:
    return compress_image_for_llm(image)


def _normalize_for_jpeg(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    raise ValueError(f"Unsupported image shape for JPEG compression: {array.shape}")


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        raise ValueError("Failed to encode image as JPEG.")
    return encoded.tobytes()


def _jpeg_data_url(encoded_bytes: bytes) -> str:
    encoded_text = base64.b64encode(encoded_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_text}"


def extract_response_text(content: Any) -> str:
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


def parse_response_json(text: str, empty_message: str) -> dict:
    candidate = text.strip()
    if not candidate:
        raise ValueError(empty_message)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])
