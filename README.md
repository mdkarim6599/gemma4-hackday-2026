# StudyMate AI

**Turn a photo of your notes into a study sheet — powered by Gemma 4.**

Built for **React Hyderabad × MLH Hack Day 2026 (Hacktoberfest)** — **Track 2: Best Use of Gemma 4**.

---

## Problem

Students revise from **photos of handwritten notes, textbook pages and diagrams**. Turning that raw
material into something they can actually *study from* — an explanation, practice questions, a
summary, or a plan for the days they have left — is slow manual work. Generic chatbots give a wall
of text and no structure you can practise against.

## Solution

StudyMate AI takes **one input** — a photo of notes/diagram, or text you paste, or a question — and
produces a structured **study sheet** in one of five modes:

| Mode | Input | What the student gets |
|---|---|---|
| **Explain a topic** | photo or typed topic | explanation, real-life analogy, key points, exam keywords |
| **Generate a quiz** | photo or typed topic | 3–8 MCQs with four options each, instant scoring and answer explanations |
| **Summarise notes** | pasted notes or a photo of the page | a tight summary, key points to revise, exam keywords |
| **Solve a doubt** | your question (+ optional photo) | concept explanation, analogy, a worked example, code, a step-by-step dry run |
| **Build a study plan** | topics + days left | a day-by-day plan with tasks and a self-test for each day |

The core loop is: **student shares material → Gemma 4 understands it → the app responds → the student acts (reads, attempts, revises)**.

## How It Works

```
Photo of notes / pasted notes / typed topic / a question
                    |
                    v
    Gemma 4 (gemma-4-26b-a4b-it)  -- multimodal: reads the image directly
                    |
                    v
    Structured JSON  ->  validated  ->  StudyPack object
                    |
                    v
    explanation - analogy - key points - keywords - quiz - plan
```

1. **Input** — the student picks a mode, then uploads an image, pastes notes, types a topic or asks
   a question. They also pick a difficulty (Beginner / Intermediate / Exam-ready) and a language
   (English / Hinglish).
2. **Gemma 4** — the model reads the image (multimodal) or the text and returns **JSON only**, shaped
   for the chosen mode.
3. **Validation** — the app parses and validates the JSON into a `StudyPack` (MCQ count, four options
   each, answer index clamped to range; plan entries and keywords cleaned). Fields a mode does not
   use are dropped. Invalid output is retried once; if the API fails, a cached example sheet is shown
   so a walkthrough never breaks.
4. **Result** — the sheet renders in the mode's own layout; the quiz scores the student live.

## Architecture

```
app.py                     Streamlit UI: mode picker, inputs, sheet rendering, quiz scoring
gemma_client.py            Gemma 4 client: per-mode prompt + JSON schema, multimodal call, validation
fallback_study_packs.json  Cached example sheet per mode, used only if the live API call fails
check.py                   CLI end-to-end check: --mode <explain|quiz|summarise|doubt|plan>, --all
samples/sample-notes.png   Sample input image (notes on Binary Search)
```

## AI / Partner Technology Used

**Google Gemma (Track 2 — Best Use of Gemma 4).**

- **Model:** `gemma-4-26b-a4b-it` (Gemma 4, open-weights, served through the **Gemini API**)
- **API:** Gemini API — `generativelanguage.googleapis.com` via the official `google-genai` SDK
- **Where the AI lives in the code:** `gemma_client.generate_study_pack()` — this is the AI
  component, and it is **central to the workflow**: without Gemma 4 there is no study sheet at all.
- **Why Gemma 4 matters here / where multimodality adds value:** the student's real input is a
  **photo of notes**, not typed text. Gemma 4 reads the image directly, so the app needs no separate
  OCR step and can understand handwriting, layout and small diagrams. The same model then turns that
  understanding into structured, validated study material.

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

Then open <http://localhost:8501>, pick a mode, add your material, and generate. To try the photo
flow, upload `samples/sample-notes.png` in **Explain a topic** or **Generate a quiz**.

Quick CLI check without the UI:

```bash
python check.py                              # explain mode (typed topic)
python check.py --mode quiz                  # quiz mode
python check.py --image samples/sample-notes.png   # explain a photo
python check.py --all                        # one live call per mode
```

## Try the app online

The repository page shows the source code; the Python app itself runs on Streamlit Community Cloud.
Deploy it in one click from the **Deploy** button below, or open the deployment page directly:

[![Deploy to Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repo=mdkarim6599/gemma4-hackday-2026&branch=main&mainModule=app.py)

On the deployment page choose:

```text
Repository: mdkarim6599/gemma4-hackday-2026
Branch:     main
Main file:  app.py
```

Before launching, add these values under **Advanced settings → Secrets**. Never put the API key in
the repository or in a public README:

```toml
GEMINI_API_KEY = "your_google_ai_studio_key"
GEMMA_MODEL = "gemma-4-26b-a4b-it"
```

After deployment, Streamlit provides a public `*.streamlit.app` URL that can be shared alongside
this GitHub repository. The app has a cached example sheet per mode, so the UI remains demoable even
when the API quota is temporarily unavailable.

## Current Status

**Works:** all five modes end to end (explain, quiz, summarise, doubt, plan); photo and text input;
difficulty and language controls; MCQ generation with live scoring; JSON validation with retry;
per-mode cached fallback sheets.

**Limitations / next steps:** no user accounts or saved history yet; the quiz is self-scored on one
page; the study plan is generated on request rather than tracked over time; rate limits are whatever
the Gemini API free tier allows.

## Team Members

| Name | GitHub |
|---|---|
| Md karim | [@mdkarim6599](https://github.com/mdkarim6599) |
| Abbani Vaishnavi | [@abbanivaishnavi](https://github.com/abbanivaishnavi) |
| minnatullah | [@Minnatullah1](https://github.com/Minnatullah1) |

## License

[MIT](LICENSE)
