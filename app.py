import html  # Escape special characters so LLM text cannot break the page layout
import re  # Strip terminal colour codes out of error messages
import time

import markdown as md  # Convert markdown (lists, bold) into real HTML
import streamlit as st

from utils.audio_processor import prepared_audio  # Prepares audio chunks, deletes them after
from core.transcriber import transcribe_all, segments_to_text
from core.summarizer import summarize, generate_title
from core.extractor import extract_takeaways, extract_topics, extract_quotes
from core.rag_engine import build_rag_chain, ask_question
from core.timestamps import extract_video_id, youtube_link  # Build links into the video

# No load_dotenv() here - core/config.py already loads .env with an absolute path
# before any setting is read.

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Markdown rendered inside a card ──
   The LLM returns markdown lists and bold text. Once converted to real HTML
   these rules give them sensible spacing inside the card. */
.card-content p { margin: 0 0 0.6rem 0; }
.card-content p:last-child { margin-bottom: 0; }

.card-content ol,
.card-content ul { margin: 0 0 0.6rem 0; padding-left: 1.4rem; }
.card-content ol:last-child,
.card-content ul:last-child { margin-bottom: 0; }

.card-content li { margin-bottom: 0.45rem; }
.card-content li:last-child { margin-bottom: 0; }
.card-content li::marker { color: var(--accent-glow); }

.card-content strong { color: var(--accent-glow); font-weight: 600; }
.card-content em { color: var(--accent-2); font-style: normal; }

.card-content code {
    background: var(--surface-2);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.8rem;
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* Markdown rendered inside a chat bubble */
.chat-bubble p { margin: 0 0 0.5rem 0; }
.chat-bubble p:last-child { margin-bottom: 0; }
.chat-bubble ol,
.chat-bubble ul { margin: 0.35rem 0; padding-left: 1.3rem; }
.chat-bubble li { margin-bottom: 0.3rem; }
.chat-bubble li:last-child { margin-bottom: 0; }
.chat-bubble strong { color: var(--accent-glow); font-weight: 600; }

/* ── Timestamp citations ── */
.ts-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.45rem;
    padding-left: 0.1rem;
}

.ts-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-right: 0.15rem;
}

.ts-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    text-decoration: none !important;
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.35);
    color: var(--accent-2) !important;
    transition: all 0.15s;
}

.ts-chip:hover {
    background: rgba(6,182,212,0.25);
    border-color: var(--accent-2);
    transform: translateY(-1px);
}

.ts-chip-plain {
    background: var(--surface-2);
    border-color: var(--border);
    color: var(--text-muted) !important;
    cursor: default;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # Terminal colour codes, e.g. \x1b[0;31m


def clean_error(err) -> str:
    """Turn a raw exception into something worth showing a user."""
    msg = ANSI_RE.sub("", str(err)).strip()  # Drop terminal colour codes

    if msg.upper().startswith("ERROR:"):  # yt-dlp prefixes its own messages
        msg = msg[6:].strip()

    if "403" in msg or "Forbidden" in msg:  # Add a hint for the most common failure
        msg += (
            "  \u2014  YouTube refused the download. Try 'pip install -U yt-dlp', "
            "then install Deno if it still fails."
        )

    return msg or "Something went wrong."


def esc(text) -> str:
    """Make plain text safe to place inside raw HTML."""
    return html.escape(str(text or ""))


def render_md(text) -> str:
    """
    Turn LLM markdown into real HTML so lists and bold actually render.

    A browser treats newlines as ordinary spaces and knows nothing about
    markdown, so dropping raw LLM text into a <div> flattens a numbered list
    into one long line. We escape the text first, so nothing in the transcript
    can break the page, then convert the markdown to HTML.
    """
    if not text:
        return ""

    return md.markdown(
        html.escape(str(text)),
        extensions=["nl2br", "sane_lists"],  # nl2br keeps single line breaks
    )


def render_sources(sources, video_id) -> str:
    """
    Build the row of timestamp chips shown under an answer.

    With a YouTube video ID each chip is a real link that opens the video at
    that second. For a local file there is nothing to link to, so the chips are
    shown as plain labels instead.
    """
    if not sources:
        return ""

    chips = []

    for s in sources:
        label = esc(s["label"])  # e.g. 3:42

        if video_id:
            url = youtube_link(video_id, s["seconds"])
            chips.append(
                f'<a class="ts-chip" href="{esc(url)}" target="_blank" '
                f'rel="noopener">▶ {label}</a>'
            )
        else:
            chips.append(f'<span class="ts-chip ts-chip-plain">🕐 {label}</span>')

    return (
        '<div class="ts-row">'
        '<span class="ts-label">Sources in video</span>'
        + "".join(chips)
        + "</div>"
    )

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Video Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with any video</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            with prepared_audio(source) as chunks:  # Temp WAV files deleted after this block
                update_step("audio", "done")

                update_step("transcript", "active")
                segments = transcribe_all(chunks, language)  # Timed segments
            transcript = segments_to_text(segments)  # Plain text for summary/extraction
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            takeaways = extract_takeaways(transcript)
            topics    = extract_topics(transcript)
            quotes    = extract_quotes(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(segments)  # Segments, so chunks keep their times
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "takeaways": takeaways,
                "topics": topics,
                "quotes": quotes,
                "rag_chain": rag_chain,
                "source": source.strip(),
                "video_id": extract_video_id(source.strip()),  # None for local files
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {clean_error(e)}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {esc(r['title'])}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{render_md(r['summary'])}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{esc(r["transcript"])}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💡 Key Takeaways</div>
            <div class="card-content">{render_md(r['takeaways'])}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🗂️ Topics Covered</div>
            <div class="card-content">{render_md(r['topics'])}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🗣️ Notable Quotes</div>
            <div class="card-content">{render_md(r['quotes'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with this Video</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{esc(msg['content'])}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{render_md(msg['content'])}</div>
                    {render_sources(msg.get('sources'), r.get('video_id'))}
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about this video</div>
            <div style="color:var(--text-muted);font-size:0.72rem;margin-top:0.4rem">
                Every answer comes with the timestamps it was taken from
            </div>
        </div>""", unsafe_allow_html=True)

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What is this video about?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer, sources = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Notes ──────────────────────────────────────────────────────────────────────
# "app.py" is the Streamlit user interface. It collects the YouTube URL or file
# path, runs the full pipeline, and displays the title, summary, transcript,
# key takeaways, topics covered, notable quotes and the grounded RAG chat.
#
# All LLM output passes through render_md() before it is placed inside a styled
# card, so markdown lists and bold text render properly instead of collapsing
# into one line, and no text from the transcript can break the page layout.
#
# Every chat answer also shows the moments in the video it was taken from. For a
# YouTube URL those chips are links that open the video at that second; for a
# local file they are plain labels, because there is nothing to link to.
#
# The row is labelled "Sources in video" rather than "the answer is here",
# because these are the passages the search retrieved - the model may have used
# some of them more than others.
#
# Sidebar input -> pipeline -> result cards + chat with timestamps
#
# NOTE: these notes are # comments, not a ''' string like in the other files.
# Streamlit's "magic" auto-displays any bare expression at the top level of the
# script, so a triple-quoted string here would be printed onto the page.