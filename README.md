# 📚 StudyMate AI

**Snap your notes → get an explanation, a quiz and revision keywords — powered by Gemma 4.**

Built for **React Hyderabad × MLH Hack Day 2026 (Hacktoberfest)** — **Track 2: Best Use of Gemma 4**.

---

## Problem

Students revise from **photos of handwritten notes, textbook pages and diagrams**. Turning that raw
material into something they can actually *study from* — a simple explanation, practice questions
and the keywords that matter — is slow manual work. Generic chatbots give a wall of text and no
structure you can practise against.

## Solution

StudyMate AI takes **one input** (a photo of notes/diagram, or a typed topic) plus a difficulty
level, and produces **one structured output** ("study pack"):

| Section | What it gives the student |
|---|---|
| 🧠 **Explanation** | A simple explanation tuned to the chosen difficulty |
| 💡 **Analogy** | One real-life analogy so the concept clicks |
| 📌 **Key points** | The important points, short |
| 🔑 **Revision keywords** | Exam keywords as quick-revision pills |
| ❓ **Quiz** | Multiple-choice questions with instant scoring + explanations |

The core loop is: **user shares → Gemma 4 understands → app responds → user acts (reads, attempts, revises)**.

## How It Works

```
📷 Photo of notes / diagram   (or)   ⌨️ Topic + Difficulty
                    │
                    ▼
        Gemma 4 (gemma-4-26b-a4b-it)  ── multimodal: reads the image directly
                    │
                    ▼
        Structured JSON  →  validated  →  StudyPack object
                    │
                    ▼
        🧠 Explanation · 💡 Analogy · 📌 Key points · 🔑 Keywords · ❓ Quiz (+score)
```

1. **Input** — the student uploads an image or types a topic and picks a difficulty
   (Beginner / Intermediate / Exam-ready) and language (English / Hinglish).
2. **Gemma 4** — the model reads the image (multimodal) or the topic and returns **JSON only**.
3. **Validation** — the app parses and validates the JSON into a `StudyPack` (MCQ count, 4 options
   each, answer index clamped to range). Invalid output is retried once; if the API fails, a cached
   pack is shown so the demo never breaks.
4. **Result** — the UI renders Explanation / Quiz / Revision tabs; the quiz scores the student live.

## Architecture

```
app.py                 Streamlit UI (input, tabs, quiz scoring)
gemma_client.py        Gemma 4 client: prompt + JSON schema, multimodal call, validation
fallback_study_pack.json  Cached demo pack used only if the live API call fails
check.py               CLI end-to-end check (topic mode + image mode)
samples/sample-notes.png  Sample input image (notes on Binary Search)
```

## AI / Partner Technology Used

**Google Gemma (Track 2 — Best Use of Gemma 4).**

- **Model:** `gemma-4-26b-a4b-it` (Gemma 4, open-weights, served through the **Gemini API**)
- **API:** Gemini API — `generativelanguage.googleapis.com` via the official `google-genai` SDK
- **Where the AI lives in the code:** `gemma_client.generate_study_pack()` — this is the AI
  component, and it is **central to the workflow**: without Gemma 4 there is no study pack at all.
- **Why Gemma 4 matters here / where multimodality adds value:** the student's real input is a
  **photo of notes**, not typed text. Gemma 4 reads the image directly, so the app needs no separate
  OCR step and can understand handwriting, layout and small diagrams. The same model then turns
  that understanding into structured, validated study material.

## Setup Instructions

Requires **Python 3.10+**.

```bash
git clone https://github.com/mdkarim6599/gemma4-hackday-2026.git
cd gemma4-hackday-2026

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file (copy `.env.example`) and paste your Google AI Studio key:

```
GEMINI_API_KEY=your_key_here
GEMMA_MODEL=gemma-4-26b-a4b-it
```

Get a key at <https://aistudio.google.com/apikey>. **Never commit `.env`** (it is gitignored).

## How to Run

```bash
streamlit run app.py
```

Then open <http://localhost:8501>, choose **📷 Photo of notes / diagram**, upload
`samples/sample-notes.png` (or your own photo), and press **✨ Generate study pack**.

Quick CLI check without the UI:

```bash
python check.py                          # topic mode
python check.py samples/sample-notes.png # image mode (multimodal)
```

## Current Status

**Works:** image + topic input, difficulty/language controls, explanation + analogy + key points +
keywords, MCQ generation with live scoring, JSON validation with retry, cached fallback pack.

**Limitations / next steps:** no user accounts or saved history yet; quiz is self-scored on one page;
study-plan and summarizer modes (from the original idea) are future work; rate limits are whatever
the Gemini API free tier allows.

## Team Members

| Name | GitHub |
|---|---|
| Md karim | [@mdkarim6599](https://github.com/mdkarim6599) |
| Abbani Vaishnavi | [@abbanivaishnavi](https://github.com/abbanivaishnavi) |
| minnatullah | [@Minnatullah1](https://github.com/Minnatullah1) |

## License

[MIT](LICENSE)
