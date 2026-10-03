"""StudyMate AI - Gemma 4 client.

One structured call to Gemma 4 (multimodal, via the Gemini API) turns study material
into a validated "study pack". Five modes are supported:

    explain    - explain a topic / a photo of notes
    quiz       - generate multiple-choice questions
    summarise  - condense pasted or photographed notes
    doubt      - work through something the student is stuck on
    plan       - build a day-by-day study plan

The AI component is Gemma 4 (`gemma-4-26b-a4b-it`). It reads images directly, so no
separate OCR step is needed, and it returns JSON that we validate before showing it.
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
FALLBACK_PATH = Path(__file__).with_name("fallback_study_packs.json")

DIFFICULTIES = ["Beginner", "Intermediate", "Exam-ready"]
LANGUAGES = ["English", "Hinglish"]

MODES = ["explain", "quiz", "summarise", "doubt", "plan"]

# The schema shows every field, so the model sometimes fills fields a mode does not use.
# Keep only the fields each mode actually renders.
_MODE_FIELDS: dict[str, set[str]] = {
    "explain": {"explanation", "analogy", "key_points", "keywords"},
    "quiz": {"explanation", "keywords", "mcqs"},
    "summarise": {"summary", "key_points", "keywords"},
    "doubt": {"explanation", "analogy", "example", "code", "dry_run", "key_points", "keywords"},
    "plan": {"plan", "key_points", "keywords"},
}

MODE_LABELS = {
    "explain": "Explain a topic",
    "quiz": "Generate a quiz",
    "summarise": "Summarise notes",
    "doubt": "Solve a doubt",
    "plan": "Build a study plan",
}

SYSTEM_INSTRUCTION = (
    "You are StudyMate, a precise and friendly tutor. You turn study material - a topic, a photo "
    "of handwritten notes, a textbook page, a diagram, or a student's question - into a compact, "
    "exam-focused study pack. Rules: be accurate and concrete; never invent content that is not in "
    "the source image - if something is unreadable, say so in one short line; keep sentences short; "
    "output nothing outside the requested JSON."
)

_SCHEMA = """{
  "topic": "<short title for the material>",
  "detected_text": "<only when an image was given: 1-3 lines on what you could read; else \\"\\">",
  "explanation": "<clear explanation, 4-8 short sentences, split with \\n>",
  "analogy": "<one real-life analogy>",
  "example": "<one concrete worked example>",
  "summary": "<a tight 3-5 sentence summary>",
  "key_points": ["<important point>"],
  "keywords": ["<exam keyword>"],
  "code": "<a short, correct code snippet, or \\"\\" when the topic has no code>",
  "dry_run": ["<one numbered step of a trace through the example>"],
  "mcqs": [
    {
      "question": "<question>",
      "options": ["<option A>", "<option B>", "<option C>", "<option D>"],
      "answer_index": 0,
      "explanation": "<why that option is correct, one line>"
    }
  ],
  "plan": [
    {
      "day": "<Day 1>",
      "focus": "<the topic for that day>",
      "tasks": ["<what to study>", "<what to practise>"],
      "check": "<a quick self-test for the end of the day>"
    }
  ]
}"""

_FILL = {
    "explain": (
        "Explain the source material for a student.\n"
        "Fill: topic, detected_text (only if an image was given), explanation, analogy, "
        "key_points (4-6), keywords (6-10).\n"
        'Set every other field to "" or [].'
    ),
    "quiz": (
        "Write a quiz from the source material.\n"
        "Fill: topic, detected_text (only if an image was given), keywords (6-10), explanation "
        "(one short line introducing the quiz), and mcqs.\n"
        'Each mcq needs exactly 4 options, a 0-based answer_index and a one-line explanation. '
        'Set every other field to "" or [].'
    ),
    "summarise": (
        "Condense the study material so it can be revised quickly.\n"
        "Fill: topic, detected_text (only if an image was given), summary (3-5 sentences), "
        "key_points (4-6), keywords (6-12).\n"
        'Set every other field to "" or [].'
    ),
    "doubt": (
        "The student is stuck on this. Walk them through it, in this order: the concept, a real-life "
        "analogy, one concrete worked example, a short code snippet if the topic is programming, "
        "and then a step-by-step dry run of the example.\n"
        "Fill: topic, explanation, analogy, example, code (or \"\" if no code applies), dry_run "
        "(4-8 short steps), key_points (4-6), keywords (6-10).\n"
        'Set every other field to "" or [].'
    ),
    "plan": (
        "Build a day-by-day study plan for the listed topics.\n"
        "Fill: topic and plan (exactly one entry per day, each with day, focus, tasks (2-4) and a "
        "check that self-tests that day). Add key_points (3-5 pieces of general advice).\n"
        'Set every other field to "" or [].'
    ),
}


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
class PlanDay:
    day: str
    focus: str
    tasks: list[str] = field(default_factory=list)
    check: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"day": self.day, "focus": self.focus, "tasks": self.tasks, "check": self.check}


@dataclass
class StudyPack:
    """Validated, structured result of one Gemma 4 call."""

    topic: str
    mode: str = "explain"
    explanation: str = ""
    analogy: str = ""
    example: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    code: str = ""
    dry_run: list[str] = field(default_factory=list)
    mcqs: list[MCQ] = field(default_factory=list)
    plan: list[PlanDay] = field(default_factory=list)
    detected_text: str = ""
    model: str = DEFAULT_MODEL

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, model: str = DEFAULT_MODEL, mode: str = "explain"
    ) -> "StudyPack":
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

        plan: list[PlanDay] = []
        for item in (data.get("plan") or [])[:14]:
            if not isinstance(item, dict):
                continue
            tasks = [str(t).strip() for t in (item.get("tasks") or []) if str(t).strip()]
            if not item.get("focus") and not tasks:
                continue
            plan.append(
                PlanDay(
                    day=str(item.get("day") or f"Day {len(plan) + 1}").strip(),
                    focus=str(item.get("focus") or "").strip(),
                    tasks=tasks,
                    check=str(item.get("check") or "").strip(),
                )
            )

        def strings(key: str) -> list[str]:
            return [str(x).strip() for x in (data.get(key) or []) if str(x).strip()]

        pack = cls(
            topic=str(data.get("topic") or "Study pack").strip(),
            mode=mode,
            explanation=str(data.get("explanation") or "").strip(),
            analogy=str(data.get("analogy") or "").strip(),
            example=str(data.get("example") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            key_points=strings("key_points"),
            keywords=strings("keywords"),
            code=str(data.get("code") or "").strip(),
            dry_run=strings("dry_run"),
            mcqs=mcqs,
            plan=plan,
            detected_text=str(data.get("detected_text") or "").strip(),
            model=model,
        )

        for name in (
            "explanation",
            "analogy",
            "example",
            "summary",
            "key_points",
            "keywords",
            "code",
            "dry_run",
            "mcqs",
            "plan",
        ):
            if name not in _MODE_FIELDS.get(mode, set()):
                setattr(pack, name, [] if isinstance(getattr(pack, name), list) else "")
        return pack

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "topic": self.topic,
            "detected_text": self.detected_text,
            "explanation": self.explanation,
            "analogy": self.analogy,
            "example": self.example,
            "summary": self.summary,
            "key_points": self.key_points,
            "keywords": self.keywords,
            "code": self.code,
            "dry_run": self.dry_run,
            "mcqs": [m.to_dict() for m in self.mcqs],
            "plan": [d.to_dict() for d in self.plan],
        }


# --------------------------------------------------------------------------- prompt

def build_prompt(
    *,
    mode: str,
    has_image: bool,
    topic: str = "",
    notes: str = "",
    question: str = "",
    plan_topics: str = "",
    days: int = 5,
    difficulty: str = "Intermediate",
    language: str = "English",
    num_mcqs: int = 5,
) -> str:
    if has_image:
        source = (
            "Analyse the attached image and use the study material visible in it as the source. "
            "First read the text and any diagram, then work from that."
        )
    elif notes:
        source = "Use the notes below as the source material:\n\n" + notes
    elif mode == "doubt":
        source = "The student says: " + question
    elif mode == "plan":
        source = (
            f"The student has {days} days before the exam and needs to cover these topics:\n"
            + plan_topics
        )
    else:
        source = f"Use this topic as the source: {topic}"

    extra = ""
    if mode == "plan":
        extra = f"\nNumber of days: {days} (produce exactly {days} plan entries)"
    elif mode == "quiz":
        extra = f"\nNumber of MCQs: {num_mcqs} (produce exactly {num_mcqs})"

    return (
        f"{source}\n\n"
        f"Difficulty: {difficulty}\n"
        f"Language for ALL output text: {language}\n"
        f"{_FILL[mode]}{extra}\n\n"
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


def load_fallback(mode: str = "explain") -> StudyPack:
    """Cached demo pack so a live API failure never kills a walkthrough."""
    data = json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
    pack = data.get(mode) or data.get("explain")
    return StudyPack.from_dict(pack, model=get_model(), mode=mode)


# --------------------------------------------------------------------------- call

def generate_study_pack(
    *,
    mode: str = "explain",
    topic: str | None = None,
    notes: str | None = None,
    question: str | None = None,
    plan_topics: str | None = None,
    days: int = 5,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    difficulty: str = "Intermediate",
    language: str = "English",
    num_mcqs: int = 5,
    model: str | None = None,
    temperature: float = 0.7,
) -> StudyPack:
    """Call Gemma 4 (multimodal when an image is given) and return a validated StudyPack."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key.")

    model = model or get_model()
    prompt = build_prompt(
        mode=mode,
        has_image=bool(image_bytes),
        topic=topic or "",
        notes=notes or "",
        question=question or "",
        plan_topics=plan_topics or "",
        days=days,
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

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=45_000,  # ms: keep a rate-limited call from hanging the page
            retry_options=types.HttpRetryOptions(attempts=1),  # our own loop handles retries
        ),
    )
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    def call() -> str:
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return response.text or ""

    try:
        text = call()
    except Exception as exc:  # noqa: BLE001 - transport/API error: fail fast so the caller can fall back
        raise RuntimeError(f"Gemma 4 call failed: {exc}") from exc

    try:
        return StudyPack.from_dict(extract_json(text), model=model, mode=mode)
    except Exception:  # malformed JSON: retry once
        try:
            return StudyPack.from_dict(extract_json(call()), model=model, mode=mode)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise RuntimeError(f"Gemma 4 returned unusable JSON: {exc}") from exc
