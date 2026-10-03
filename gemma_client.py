"""StudyMate AI - Gemma 4 client.

Turns a topic (or a photo of study material) into a structured "study pack":
explanation + analogy + key points + multiple-choice questions + revision keywords.

The AI component is **Gemma 4** (`gemma-4-26b-a4b-it`) served through the
Gemini API via the official `google-genai` SDK. It is multimodal, so the model
reads the uploaded notes/diagram image directly instead of us doing OCR.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
FALLBACK_PATH = Path(__file__).with_name("fallback_study_pack.json")

DIFFICULTIES = ["Beginner", "Intermediate", "Exam-ready"]
LANGUAGES = ["English", "Hinglish"]

SYSTEM_INSTRUCTION = (
    "You are StudyMate, a precise and friendly tutor. You convert study material "
    "(a topic, or a photo of handwritten notes / a textbook page / a diagram) into a "
    "compact, exam-focused study pack. Rules: be accurate and concrete; never invent "
    "content that is not in the source image - if something is unreadable, say so in one "
    "short line; keep sentences short; output nothing outside the requested JSON."
)

_SCHEMA = """{
  "topic": "<the main topic, short>",
  "detected_text": "<if an image was given: 1-3 lines on what you could read; else empty string>",
  "explanation": "<clear explanation, 4-8 short sentences or bullet lines>",
  "analogy": "<one real-life analogy that makes it click>",
  "key_points": ["<important point>", "..."],
  "keywords": ["<exam keyword>", "..."],
  "mcqs": [
    {
      "question": "<question>",
      "options": ["<option A>", "<option B>", "<option C>", "<option D>"],
      "answer_index": 0,
      "explanation": "<why that option is correct, one line>"
    }
  ]
}"""


# --------------------------------------------------------------------------- env

def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader so the app works with or without python-dotenv."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str | None:
    load_env()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_model() -> str:
    load_env()
    return os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)


# --------------------------------------------------------------------------- data

@dataclass
class MCQ:
    question: str
    options: list[str]
    answer_index: int
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": self.options,
            "answer_index": self.answer_index,
            "explanation": self.explanation,
        }


@dataclass
class StudyPack:
    """Validated, structured result of one Gemma 4 call."""

    topic: str
    explanation: str
    analogy: str
    key_points: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    mcqs: list[MCQ] = field(default_factory=list)
    detected_text: str = ""
    model: str = DEFAULT_MODEL

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, model: str = DEFAULT_MODEL) -> "StudyPack":
        mcqs: list[MCQ] = []
        for item in (data.get("mcqs") or [])[:12]:
            if not isinstance(item, dict):
                continue
            options = [str(o) for o in (item.get("options") or []) if str(o).strip()][:4]
            if not item.get("question") or len(options) < 2:
                continue
            try:
                idx = int(item.get("answer_index", 0))
            except (TypeError, ValueError):
                idx = 0
            idx = max(0, min(idx, len(options) - 1))
            mcqs.append(
                MCQ(
                    question=str(item["question"]).strip(),
                    options=options,
                    answer_index=idx,
                    explanation=str(item.get("explanation", "")).strip(),
                )
            )
        return cls(
            topic=str(data.get("topic") or "Study pack").strip(),
            explanation=str(data.get("explanation") or "").strip(),
            analogy=str(data.get("analogy") or "").strip(),
            key_points=[str(x).strip() for x in (data.get("key_points") or []) if str(x).strip()],
            keywords=[str(x).strip() for x in (data.get("keywords") or []) if str(x).strip()],
            mcqs=mcqs,
            detected_text=str(data.get("detected_text") or "").strip(),
            model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "detected_text": self.detected_text,
            "explanation": self.explanation,
            "analogy": self.analogy,
            "key_points": self.key_points,
            "keywords": self.keywords,
            "mcqs": [m.to_dict() for m in self.mcqs],
        }


# --------------------------------------------------------------------------- parsing

def build_prompt(
    *,
    has_image: bool,
    topic: str,
    difficulty: str,
    language: str,
    num_mcqs: int,
) -> str:
    source = (
        "Analyse the attached image and use the study material visible in it as the source. "
        "First read the text/diagram, then build the study pack from it."
        if has_image
        else f"Use this topic as the source: {topic}."
    )
    return (
        f"{source}\n\n"
        f"Difficulty: {difficulty}\n"
        f"Language for ALL output text: {language}\n"
        f"Number of MCQs: {num_mcqs}\n\n"
        "Build a study pack containing:\n"
        "1. A simple explanation tuned to the difficulty.\n"
        "2. One real-life analogy.\n"
        "3. 4-6 key points.\n"
        "4. 6-10 exam keywords.\n"
        f"5. Exactly {num_mcqs} multiple-choice questions, each with exactly 4 options, "
        "a 0-based answer_index (0-3) and a one-line explanation of the correct option.\n\n"
        "Return ONLY valid JSON matching this schema (no markdown fences, no extra text):\n"
        f"{_SCHEMA}"
    )


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    if not text:
        raise ValueError("empty model response")
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)  # drop trailing commas
    return json.loads(raw)


def load_fallback() -> StudyPack:
    """Cached demo pack so a live API failure never kills the demo."""
    data = json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
    return StudyPack.from_dict(data, model=get_model())


# --------------------------------------------------------------------------- main call

def generate_study_pack(
    *,
    topic: str | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    difficulty: str = "Intermediate",
    language: str = "English",
    num_mcqs: int = 5,
    model: str | None = None,
    temperature: float = 0.7,
) -> StudyPack:
    """Call Gemma 4 (multimodal when an image is given) and return a validated StudyPack."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key.")

    model = model or get_model()
    prompt = build_prompt(
        has_image=bool(image_bytes),
        topic=topic or "",
        difficulty=difficulty,
        language=language,
        num_mcqs=num_mcqs,
    )

    contents: list[Any] = []
    if image_bytes:
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type or "image/png")
        )
    contents.append(prompt)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    last_error: Exception | None = None
    for _ in range(2):  # one retry: model occasionally returns malformed JSON
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return StudyPack.from_dict(extract_json(response.text or ""), model=model)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            last_error = exc
    raise RuntimeError(f"Gemma 4 call failed: {last_error}")
