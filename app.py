"""StudyMate AI - Streamlit app.

Five study modes over one Gemma 4 call: explain, quiz, summarise, doubt, plan.
Gemma 4 reads the page; this file only presents what the model returns.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import gemma_client as gc

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Karla:wght@400;500;600;700&display=swap');

:root{
  --paper:#F5F6F8; --panel:#EEF1F5; --grid:#EAEEF3; --rule:#DDE3EB;
  --ink:#16233A; --muted:#5F6E85; --mark:#FFE066;
}

html, body, .stApp {background-color:var(--paper);}
body, [data-testid="stAppViewContainer"]{
  font-family:'Karla', system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--ink);
}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"]{display:none;}

/* ---------------------------------------------------- faint graph paper */
[data-testid="stAppViewContainer"]{
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:28px 28px; background-position:-1px -1px;
}

/* --------------------------------------- page column + notebook margin */
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"]{
  max-width:1080px !important;
  padding:2.2rem 2.4rem 5rem 5rem !important;
  position:relative;
}
[data-testid="stMainBlockContainer"]::before,
[data-testid="stAppViewBlockContainer"]::before{
  content:""; position:absolute; left:2.9rem; top:0; bottom:0; width:1px; background:var(--rule);
}

/* ------------------------------------------------------------- type */
[data-testid="stMarkdownContainer"]{
  font-family:'Karla', system-ui, -apple-system, "Segoe UI", sans-serif;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{
  font-family:'Fraunces', Georgia, serif !important;
  color:var(--ink) !important; font-weight:600 !important;
  letter-spacing:-.015em !important; line-height:1.14 !important;
}
[data-testid="stMarkdownContainer"] h1{font-size:clamp(1.95rem, 3.4vw, 2.8rem) !important; margin:.1rem 0 .5rem !important;}
[data-testid="stMarkdownContainer"] h2{font-size:1.5rem !important; margin:.1rem 0 .45rem !important;}
[data-testid="stMarkdownContainer"] h3{font-size:1.16rem !important; margin:.1rem 0 .4rem !important;}
[data-testid="stMarkdownContainer"] h4{font-size:1.02rem !important; margin:.1rem 0 .35rem !important;}
[data-testid="stMarkdownContainer"] p{margin:0 0 .9rem; line-height:1.66; font-size:1.01rem; color:#25334C;}
[data-testid="stMarkdownContainer"] a{color:var(--ink); text-decoration-color:#B9C3D2; text-underline-offset:3px;}

/* ------------------------------------------------------------ hero */
[data-testid="stMarkdownContainer"] .sm-hero{margin:.1rem 0 1.1rem;}
[data-testid="stMarkdownContainer"] .sm-sub{max-width:62ch; font-size:1.08rem; line-height:1.6; color:#33415A;}
[data-testid="stMarkdownContainer"] .sm-note{
  font-family:'Fraunces', Georgia, serif; font-style:italic; color:var(--muted);
  font-size:1rem; line-height:1.6; max-width:70ch;
}
[data-testid="stMarkdownContainer"] .sm-hr{height:1px; background:var(--rule); margin:1.6rem 0;}

/* ----------------------------------------------------- highlighter */
[data-testid="stMarkdownContainer"] .mark{
  background:linear-gradient(180deg, transparent 56%, var(--mark) 56%); padding:0 3px;
}
[data-testid="stMarkdownContainer"] .kw{
  display:inline-block; margin:5px 10px 5px 0; padding:3px 10px; font-weight:600; font-size:.95rem;
  background:linear-gradient(180deg, transparent 54%, var(--mark) 54%);
}

/* ------------------------------------------------------------ copy */
[data-testid="stMarkdownContainer"] .sm-p{max-width:72ch; margin:0 0 .85rem; font-size:1.03rem; line-height:1.68; color:#25334C;}
[data-testid="stMarkdownContainer"] .sm-analogy{
  font-family:'Fraunces', Georgia, serif; font-size:1.1rem; line-height:1.58; max-width:64ch;
  border-left:3px solid var(--mark); padding:.15rem 0 .15rem 1rem; margin:1.4rem 0; color:#22304A;
}
[data-testid="stMarkdownContainer"] .sm-example{
  border:1px solid var(--rule); background:#fff; padding:.95rem 1.1rem; margin:1.2rem 0;
  max-width:72ch; font-size:1.01rem; line-height:1.62; color:#25334C;
}
[data-testid="stMarkdownContainer"] ul.sm-points{list-style:none; padding-left:0; margin:.6rem 0 0; max-width:72ch;}
[data-testid="stMarkdownContainer"] ul.sm-points li{
  position:relative; padding-left:1.5rem; margin:.5rem 0; line-height:1.6; color:#25334C;
}
[data-testid="stMarkdownContainer"] ul.sm-points li::before{
  content:""; position:absolute; left:0; top:.52em; width:9px; height:9px; background:var(--mark);
}
[data-testid="stMarkdownContainer"] .sm-label{
  font-family:'Fraunces', Georgia, serif; font-weight:600; font-size:1.02rem; color:var(--ink);
  margin:.9rem 0 .3rem;
}

/* --------------------------------------------------------- sheet head */
[data-testid="stMarkdownContainer"] .sm-sheet-head{margin:0 0 .9rem;}
[data-testid="stMarkdownContainer"] .sm-sheet-meta{color:var(--muted); font-size:.96rem; line-height:1.55; max-width:70ch;}
[data-testid="stMarkdownContainer"] .sm-flag{
  margin:.9rem 0 0; padding:.65rem .95rem; border-left:3px solid #D9A21B; background:#FFFBEA;
  font-size:.96rem; line-height:1.55; color:#4A3A12; max-width:72ch;
}

/* ----------------------------------------------------------- steps */
[data-testid="stMarkdownContainer"] .sm-step{
  display:flex; gap:.85rem; padding:.72rem 0; border-bottom:1px solid var(--rule); max-width:48ch;
}
[data-testid="stMarkdownContainer"] .sm-step:last-child{border-bottom:none; padding-bottom:.2rem;}
[data-testid="stMarkdownContainer"] .sm-stepnum{
  font-family:'Fraunces', Georgia, serif; font-size:1.05rem; color:var(--muted); min-width:1.1rem;
}
[data-testid="stMarkdownContainer"] .sm-step div{line-height:1.56; color:#33415A; font-size:.98rem;}

/* -------------------------------------------------- dry run sequence */
[data-testid="stMarkdownContainer"] ol.sm-trace{list-style:none; counter-reset:step; padding-left:0; margin:.6rem 0 0; max-width:72ch;}
[data-testid="stMarkdownContainer"] ol.sm-trace li{
  counter-increment:step; position:relative; padding-left:2.1rem; margin:.5rem 0;
  line-height:1.6; color:#25334C;
}
[data-testid="stMarkdownContainer"] ol.sm-trace li::before{
  content:counter(step); position:absolute; left:0; top:.05em; width:1.45rem; height:1.45rem;
  display:flex; align-items:center; justify-content:center; font-size:.82rem; font-weight:700;
  font-family:'Fraunces', Georgia, serif; color:var(--ink); background:var(--mark);
}

/* ------------------------------------------------------ study plan */
[data-testid="stMarkdownContainer"] .sm-day{padding:.95rem 0; border-top:1px solid var(--rule); max-width:72ch;}
[data-testid="stMarkdownContainer"] .sm-day:first-of-type{border-top:none; padding-top:.3rem;}
[data-testid="stMarkdownContainer"] .sm-day-label{
  font-family:'Fraunces', Georgia, serif; font-size:.92rem; color:var(--muted);
}
[data-testid="stMarkdownContainer"] .sm-day-focus{
  font-family:'Fraunces', Georgia, serif; font-size:1.18rem; font-weight:600; margin:.1rem 0 .45rem;
}
[data-testid="stMarkdownContainer"] .sm-day-check{
  margin-top:.55rem; font-size:.96rem; color:#33415A; border-left:3px solid var(--mark); padding-left:.7rem;
}

/* ------------------------------------------------------------ quiz */
[data-testid="stMarkdownContainer"] .sm-q{
  display:flex; gap:.75rem; margin:.9rem 0 .45rem; font-size:1.05rem; font-weight:600;
  line-height:1.5; max-width:70ch; color:var(--ink);
}
[data-testid="stMarkdownContainer"] .sm-qnum{font-family:'Fraunces', Georgia, serif; color:var(--muted); min-width:1.4rem;}
[data-testid="stMarkdownContainer"] .sm-answer{
  margin:.35rem 0 1.4rem; font-size:.98rem; line-height:1.56; max-width:70ch; color:#33415A;
}
[data-testid="stMarkdownContainer"] .sm-answer .sm-label{display:inline; margin-right:.4rem;}
[data-testid="stMarkdownContainer"] .sm-score{margin:.5rem 0 0; font-size:1.06rem; color:#25334C; max-width:70ch;}
[data-testid="stMarkdownContainer"] .sm-score .mark{font-weight:700; padding:0 5px;}
[data-testid="stMarkdownContainer"] .sm-score-sub{color:var(--muted); font-size:.96rem;}

/* ----------------------------------------------------- empty state */
[data-testid="stMarkdownContainer"] .sm-empty{
  font-family:'Fraunces', Georgia, serif; font-style:italic; color:var(--muted);
  font-size:1.04rem; max-width:58ch; line-height:1.6; margin:1.6rem 0 .2rem;
}

/* -------------------------------------------------------- widgets */
[data-testid="stButton"] > button{
  border-radius:3px; font-family:'Karla', sans-serif; font-weight:700; letter-spacing:.01em;
  padding:.58rem 1.15rem;
}
[data-testid="stButton"] > button:focus-visible{outline:2px solid var(--mark); outline-offset:2px;}
[data-testid="stFileUploaderDropzone"]{background:#fff; border:1.5px dashed var(--rule); border-radius:4px;}
div[role="radiogroup"] label{padding:.32rem .1rem; font-size:.99rem;}

[data-testid="stCodeBlock"] pre{border-radius:3px; border:1px solid var(--rule);}

[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--rule);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:.95rem; color:var(--muted);}
[data-testid="stCaptionContainer"]{color:var(--muted);}
[data-testid="stAlert"]{border-radius:3px;}

@media (prefers-reduced-motion: reduce){*{transition:none !important; animation:none !important;}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------ hero
st.markdown(
    """
    <div class="sm-hero">
      <h1>Turn a photo of your notes into a study sheet.</h1>
      <p class="sm-sub">Handwriting, textbook pages, diagrams, or a topic you type — Gemma 4 reads
      the material and writes back whatever you need: an explanation, a quiz, a summary, a walk
      through a doubt, or a plan for the days you have left.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ mode picker
mode_label = st.radio(
    "What do you need?",
    [gc.MODE_LABELS[m] for m in gc.MODES],
    horizontal=True,
    index=0,
    key="mode",
)
mode = gc.MODES[[gc.MODE_LABELS[m] for m in gc.MODES].index(mode_label)]

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Set up the sheet")
    difficulty = st.radio("Level", gc.DIFFICULTIES, index=1, key="difficulty")
    language = st.radio("Language", gc.LANGUAGES, index=0, key="language")
    num_mcqs = 5
    if mode == "quiz":
        num_mcqs = st.slider("Questions", 3, 8, 5, key="num_mcqs")
    st.divider()
    if gc.get_api_key():
        st.markdown("Gemma 4 key loaded.")
    else:
        st.error("No key found. Copy `.env.example` to `.env` and paste your GEMINI_API_KEY.")
    st.caption(f"Reading with `{gc.get_model()}` through the Gemini API.")

# ------------------------------------------------------------------ input area
topic = notes = question = plan_topics = None
image_bytes: bytes | None = None
image_mime: str | None = None
days = 5


def photo_input(label: str, key: str) -> tuple[bytes | None, str | None, bool]:
    uploaded = st.file_uploader(label, type=["png", "jpg", "jpeg", "webp"], key=key)
    if uploaded is not None:
        data = uploaded.getvalue()
        st.image(data, caption="What Gemma 4 will read", use_container_width=True)
        return data, uploaded.type, True
    return None, None, True  # the uploader exists, so an image is possible


if mode in ("explain", "quiz"):
    source = st.radio("Studying from", ["A photo of my notes", "A topic I type"],
                      horizontal=True, key=f"source_{mode}")
    if source == "A photo of my notes":
        image_bytes, image_mime, _ = photo_input(
            "Upload a photo of your notes, a textbook page or a diagram", f"up_{mode}"
        )
    else:
        topic = st.text_input("Topic", placeholder="K-Means clustering, Binary Search, Recursion…", key="topic")

elif mode == "summarise":
    source = st.radio("Summarising", ["Pasted notes", "A photo of the page"],
                      horizontal=True, key="source_summarise")
    if source == "Pasted notes":
        notes = st.text_area(
            "Paste the notes you want condensed",
            height=210,
            placeholder="Paste a long passage, a chapter or your own notes here…",
            key="notes",
        )
    else:
        image_bytes, image_mime, _ = photo_input(
            "Upload a photo of the page to summarise", "up_summarise"
        )

elif mode == "doubt":
    question = st.text_input(
        "What are you stuck on?",
        placeholder="I don't understand recursion — walk me through it",
        key="question",
    )
    with st.expander("Add a photo of the page you're stuck on (optional)"):
        image_bytes, image_mime, _ = photo_input("Upload a photo", "up_doubt")

else:  # study plan
    plan_topics = st.text_area(
        "Topics to cover",
        height=160,
        placeholder="DBMS\nOperating Systems\nComputer Networks\nData Structures",
        key="plan_topics",
    )
    days = st.number_input("Days until the exam", min_value=1, max_value=14, value=5, key="days")

BUTTON = {
    "explain": "Explain this",
    "quiz": "Generate the quiz",
    "summarise": "Summarise these notes",
    "doubt": "Walk me through it",
    "plan": "Build my plan",
}
generate = st.button(BUTTON[mode], type="primary", use_container_width=True)

# ---------------------------------------------------------------------- generate
if generate:
    missing = (
        (mode in ("explain", "quiz") and not topic and not image_bytes)
        or (mode == "summarise" and not notes and not image_bytes)
        or (mode == "doubt" and not question)
        or (mode == "plan" and not plan_topics)
    )
    if missing:
        st.warning("Add the material first, then generate.")
    else:
        with st.spinner("Reading the material…"):
            try:
                pack = gc.generate_study_pack(
                    mode=mode,
                    topic=topic,
                    notes=notes,
                    question=question,
                    plan_topics=plan_topics,
                    days=int(days),
                    image_bytes=image_bytes,
                    image_mime_type=image_mime,
                    difficulty=difficulty,
                    language=language,
                    num_mcqs=num_mcqs,
                )
                st.session_state["pack"] = pack
                st.session_state["source"] = "live"
                st.session_state["settings"] = (difficulty, language, num_mcqs, int(days))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Gemma 4 did not answer: {exc}")
                st.info("Showing the saved example sheet so the walkthrough can continue.")
                st.session_state["pack"] = gc.load_fallback(mode)
                st.session_state["source"] = "fallback"

# ------------------------------------------------------------------------ sheet
pack: gc.StudyPack | None = st.session_state.get("pack")
LETTERS = "ABCDEFGH"


def points(items: list[str]) -> None:
    html = "".join(f"<li>{x}</li>" for x in items)
    st.markdown(f'<ul class="sm-points">{html}</ul>', unsafe_allow_html=True)


def keywords(items: list[str]) -> None:
    st.markdown('<p class="sm-label">Worth memorising</p>', unsafe_allow_html=True)
    st.markdown("".join(f'<span class="kw">{k}</span>' for k in items), unsafe_allow_html=True)


if pack is None:
    st.markdown(
        '<p class="sm-empty">Nothing here yet. Choose what you need above, add your material, and '
        "generate a study sheet.</p>",
        unsafe_allow_html=True,
    )
else:
    used = st.session_state.get("settings", ("Intermediate", "English", 5, 5))
    st.markdown('<div class="sm-hr"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sm-sheet-head">
          <h2>{pack.topic}</h2>
          <div class="sm-sheet-meta">{gc.MODE_LABELS[pack.mode]} at <b>{used[0]}</b> level in
          <b>{used[1]}</b>, by <b>{pack.model}</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pack.detected_text:
        st.markdown(f'<p class="sm-note">Read from your page: {pack.detected_text}</p>',
                    unsafe_allow_html=True)
    if st.session_state.get("source") == "fallback":
        st.markdown(
            '<div class="sm-flag">This is the saved example sheet, not a live Gemma 4 answer.</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------- explain
    if pack.mode in ("explain", "summarise"):
        if pack.summary:
            st.markdown(f'<p class="sm-p">{pack.summary}</p>', unsafe_allow_html=True)
        if pack.explanation and pack.mode == "explain":
            lines = [ln.strip() for ln in pack.explanation.splitlines() if ln.strip()]
            st.markdown("".join(f'<p class="sm-p">{ln}</p>' for ln in lines), unsafe_allow_html=True)
        elif pack.explanation:
            st.markdown(f'<p class="sm-p">{pack.explanation}</p>', unsafe_allow_html=True)
        if pack.analogy:
            st.markdown(
                f'<p class="sm-analogy"><span class="sm-label">Think of it this way. </span>'
                f"{pack.analogy}</p>",
                unsafe_allow_html=True,
            )
        if pack.key_points:
            st.markdown('<p class="sm-label">In short</p>', unsafe_allow_html=True)
            points(pack.key_points)
        if pack.keywords:
            keywords(pack.keywords)

    # ---------------------------------------------------------------- doubt
    elif pack.mode == "doubt":
        if pack.explanation:
            lines = [ln.strip() for ln in pack.explanation.splitlines() if ln.strip()]
            st.markdown("".join(f'<p class="sm-p">{ln}</p>' for ln in lines), unsafe_allow_html=True)
        if pack.analogy:
            st.markdown(
                f'<p class="sm-analogy"><span class="sm-label">Think of it this way. </span>'
                f"{pack.analogy}</p>",
                unsafe_allow_html=True,
            )
        if pack.example:
            st.markdown('<p class="sm-label">A worked example</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="sm-example">{pack.example}</div>', unsafe_allow_html=True)
        if pack.code:
            st.markdown('<p class="sm-label">In code</p>', unsafe_allow_html=True)
            st.code(pack.code)
        if pack.dry_run:
            st.markdown('<p class="sm-label">Dry run</p>', unsafe_allow_html=True)
            html = "".join(f"<li>{step}</li>" for step in pack.dry_run)
            st.markdown(f'<ol class="sm-trace">{html}</ol>', unsafe_allow_html=True)
        if pack.key_points:
            st.markdown('<p class="sm-label">Remember this</p>', unsafe_allow_html=True)
            points(pack.key_points)
        if pack.keywords:
            keywords(pack.keywords)

    # ------------------------------------------------------------------ plan
    elif pack.mode == "plan":
        if pack.plan:
            html = "".join(
                f'<div class="sm-day"><div class="sm-day-label">{d.day}</div>'
                f'<div class="sm-day-focus">{d.focus}</div>'
                + (f'<ul class="sm-points">{"".join(f"<li>{t}</li>" for t in d.tasks)}</ul>' if d.tasks else "")
                + (f'<div class="sm-day-check">Self-test: {d.check}</div>' if d.check else "")
                + "</div>"
                for d in pack.plan
            )
            st.markdown(html, unsafe_allow_html=True)
        if pack.key_points:
            st.markdown('<p class="sm-label">How to use it</p>', unsafe_allow_html=True)
            points(pack.key_points)

    # ------------------------------------------------------------------ quiz
    else:
        if pack.explanation:
            st.markdown(f'<p class="sm-p">{pack.explanation}</p>', unsafe_allow_html=True)
        if not pack.mcqs:
            st.markdown(
                '<p class="sm-empty">No questions came back this time. Generate the sheet again.</p>',
                unsafe_allow_html=True,
            )
        else:
            reveal = st.toggle("Show the answers")
            for i, q in enumerate(pack.mcqs, start=1):
                st.markdown(
                    f'<div class="sm-q"><span class="sm-qnum">{i}.</span>{q.question}</div>',
                    unsafe_allow_html=True,
                )
                display = [f"{LETTERS[j]}. {opt}" for j, opt in enumerate(q.options)]
                st.radio(f"Question {i}", display, index=None, key=f"mcq_{i}",
                         label_visibility="collapsed")
                if reveal:
                    st.markdown(
                        f'<div class="sm-answer"><span class="sm-label">Answer</span>'
                        f'<span class="mark">{q.options[q.answer_index]}</span>'
                        f" {q.explanation}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("")

            score = answered = 0
            for i, q in enumerate(pack.mcqs, start=1):
                value = st.session_state.get(f"mcq_{i}")
                if value:
                    answered += 1
                    if LETTERS.index(value[0]) == q.answer_index:
                        score += 1
            st.markdown(
                f'<div class="sm-score">Marks <span class="mark">{score}</span> of {len(pack.mcqs)}'
                f'<span class="sm-score-sub"> ({answered} answered)</span></div>',
                unsafe_allow_html=True,
            )
        if pack.keywords:
            keywords(pack.keywords)

st.markdown('<div class="sm-hr"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sm-note">StudyMate AI. Built for React Hyderabad × MLH Hack Day 2026, '
    f"Track 2: Best Use of Gemma 4. Reading with {gc.get_model()} through the Gemini API.</p>",
    unsafe_allow_html=True,
)
