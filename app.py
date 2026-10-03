"""StudyMate AI - Streamlit app.

Photo of your notes (or a topic)  ->  Gemma 4  ->  explanation + quiz + revision keywords.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import gemma_client as gc

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .pill {display:inline-block; padding:5px 13px; margin:4px 6px 4px 0; border-radius:999px;
           background:#eef2ff; color:#3730a3; font-size:0.85rem; font-weight:600;}
    .hint {color:#64748b; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Study settings")
    difficulty = st.radio("Difficulty", gc.DIFFICULTIES, index=1)
    language = st.radio("Language", gc.LANGUAGES, index=0)
    num_mcqs = st.slider("Number of MCQs", 3, 8, 5)
    st.divider()
    if gc.get_api_key():
        st.success("Gemma 4 key loaded")
    else:
        st.error("GEMINI_API_KEY missing — copy .env.example to .env")
    st.caption(f"Model: `{gc.get_model()}` · Gemini API (multimodal)")

# ---------------------------------------------------------------------- header
st.title("📚 StudyMate AI")
st.caption(
    "Snap your handwritten notes or a diagram → get a simple explanation, a quiz "
    "and revision keywords. Powered by **Gemma 4** via the Gemini API."
)

col_input, col_how = st.columns([1.15, 1], gap="large")

with col_input:
    mode = st.radio(
        "Input type",
        ["📷 Photo of notes / diagram", "⌨️ Type a topic"],
        horizontal=True,
    )

    topic: str | None = None
    image_bytes: bytes | None = None
    image_mime: str | None = None

    if mode.startswith("📷"):
        uploaded = st.file_uploader(
            "Upload a photo of your notes, a textbook page or a diagram",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            image_mime = uploaded.type
            st.image(image_bytes, caption="Your input", use_container_width=True)
    else:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. K-Means clustering, Binary Search, Recursion…",
        )

    generate = st.button("✨ Generate study pack", type="primary", use_container_width=True)

with col_how:
    st.markdown("#### How it works — 3 steps")
    st.markdown(
        "1. **You share** a photo of your notes/diagram (or type a topic).\n"
        "2. **Gemma 4 understands** it — multimodal, it reads the image directly.\n"
        "3. **You act** — read the explanation, attempt the quiz, revise the keywords."
    )
    sample = Path("samples/sample-notes.png")
    if sample.exists():
        st.info("No photo handy? Use the bundled sample: `samples/sample-notes.png`")
    st.markdown(
        '<span class="hint">Gemma 4 output is validated into structured data '
        "(JSON → StudyPack) before it is shown, so the UI never renders raw text.</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------- generate
if generate:
    if not topic and not image_bytes:
        st.warning("Pehle ek topic likho ya notes ki photo upload karo.")
    else:
        with st.spinner("Gemma 4 padh raha hai …"):
            try:
                pack = gc.generate_study_pack(
                    topic=topic,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime,
                    difficulty=difficulty,
                    language=language,
                    num_mcqs=num_mcqs,
                )
                st.session_state["pack"] = pack
                st.session_state["source"] = "live"
                st.session_state["snapshot"] = {
                    "difficulty": difficulty,
                    "language": language,
                    "num_mcqs": num_mcqs,
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"Gemma 4 call fail hui: {exc}")
                st.info("Fallback demo pack dikhaya ja raha hai taaki demo ruke na.")
                st.session_state["pack"] = gc.load_fallback()
                st.session_state["source"] = "fallback"

# ----------------------------------------------------------------------- render
pack: gc.StudyPack | None = st.session_state.get("pack")
if pack is not None:
    st.divider()
    st.subheader(f"📖 {pack.topic}")
    snapshot = st.session_state.get("snapshot", {})
    if snapshot:
        st.markdown(
            '<span class="hint">Generated with '
            f"difficulty=<b>{snapshot.get('difficulty')}</b>, "
            f"language=<b>{snapshot.get('language')}</b>, "
            f"MCQs=<b>{snapshot.get('num_mcqs')}</b>, "
            f"model=<b>{pack.model}</b></span>",
            unsafe_allow_html=True,
        )
    if st.session_state.get("source") == "fallback":
        st.warning("Live Gemma 4 call failed — showing the cached demo pack.")
    if pack.detected_text:
        st.caption(f"Read from your image: {pack.detected_text}")

    tab_explain, tab_quiz, tab_revise = st.tabs(["🧠 Explanation", "❓ Quiz", "🔑 Revision"])

    with tab_explain:
        st.markdown(pack.explanation)
        if pack.analogy:
            st.markdown(f"**Real-life analogy:** {pack.analogy}")
        if pack.key_points:
            st.markdown("**Key points**")
            for point in pack.key_points:
                st.markdown(f"- {point}")

    with tab_quiz:
        if not pack.mcqs:
            st.info("Koi MCQ generate nahi hua — dobara try karo.")
        else:
            reveal = st.toggle("Show correct answers & explanations", value=False)
            for i, q in enumerate(pack.mcqs, start=1):
                st.markdown(f"**Q{i}. {q.question}**")
                st.radio(
                    f"Q{i}",
                    q.options,
                    index=None,
                    key=f"mcq_{i}",
                    label_visibility="collapsed",
                )
                if reveal:
                    st.success(f"Answer: {q.options[q.answer_index]} — {q.explanation}")
                st.write("")

            score = sum(
                1
                for i, q in enumerate(pack.mcqs, start=1)
                if st.session_state.get(f"mcq_{i}") == q.options[q.answer_index]
            )
            answered = sum(
                1 for i in range(1, len(pack.mcqs) + 1) if st.session_state.get(f"mcq_{i}")
            )
            st.metric("Score", f"{score} / {len(pack.mcqs)}", f"{answered} answered")

    with tab_revise:
        if pack.keywords:
            st.markdown("**Exam keywords**")
            st.markdown(
                "".join(f'<span class="pill">{k}</span>' for k in pack.keywords),
                unsafe_allow_html=True,
            )
        if pack.key_points:
            st.markdown("**Quick revision**")
            for point in pack.key_points:
                st.markdown(f"- {point}")

st.divider()
st.caption(
    "Built for React Hyderabad × MLH Hack Day 2026 · Track 2 — Best Use of Gemma 4 · "
    f"Model: `{gc.get_model()}`"
)
