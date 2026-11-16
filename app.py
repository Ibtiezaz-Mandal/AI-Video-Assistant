"""
AI Video Agent — Modern Streamlit SaaS Dashboard
================================================
A production-grade Streamlit web interface that integrates directly with the
existing AI Video Agent backend (`main.py`, `core/*`, and `utils/*`) without
duplicating or altering the core transcription, summarization, extraction, or
RAG logic.

Run from the project root:
    streamlit run app.py
"""

import io
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env) before importing backend modules
load_dotenv()

# Import existing backend modules and functions
from main import run_pipeline
from utils.audio_processor import chunk_audio, convert_to_wav, download_youtube_audio
from core.transcriber import transcribe_chunk
from core.summarize import generate_title, summarize
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain

# Optional ReportLab imports for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ==============================================================================
# 1. PAGE CONFIGURATION & HIGH-CONTRAST MODERN DARK SAAS STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Video Agent | Video Intelligence & RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_TEMP_DIR = "uploaded_media"
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)


def inject_custom_css() -> None:
    """
    Inject custom CSS for a sleek, modern AI SaaS dark dashboard aesthetic.
    Explicitly styles all Streamlit / BaseWeb inputs, textareas, buttons,
    download buttons, file uploaders, and chat boxes so text is always crisp
    and high-contrast regardless of whether the browser/OS is in Light or Dark mode.
    """
    st.markdown(
        """
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main App & Header Background */
        .stApp {
            background: radial-gradient(circle at top left, #131828 0%, #0B0F19 55%, #080B12 100%) !important;
            color: #F1F5F9 !important;
        }

        header[data-testid="stHeader"] {
            background: rgba(11, 15, 25, 0.85) !important;
        }

        /* Top Hero Header Card */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 16px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .hero-title {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            background: linear-gradient(90deg, #FFFFFF 0%, #A5B4FC 55%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: #CBD5E1 !important;
            font-size: 0.98rem;
            margin-top: 0.3rem;
            margin-bottom: 0;
        }

        .hero-badge-group {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .saas-badge {
            background: rgba(99, 102, 241, 0.18);
            color: #E0E7FF !important;
            border: 1px solid rgba(99, 102, 241, 0.4);
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* Glassmorphic Dashboard Cards */
        .saas-card {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }

        .saas-card-header {
            font-size: 1.05rem;
            font-weight: 600;
            color: #F8FAFC !important;
            margin-bottom: 0.65rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Metadata Info Pills */
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
            margin-bottom: 1.25rem;
        }

        .meta-box {
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
        }

        .meta-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8 !important;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }

        .meta-value {
            font-size: 0.98rem;
            font-weight: 600;
            color: #F8FAFC !important;
            word-break: break-word;
        }

        /* ==================================================================
           HIGH-CONTRAST TEXT INPUTS & TEXT AREAS (Transcript, YouTube URL)
           ================================================================== */
        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        div[data-baseweb="textarea"] {
            background-color: #1E293B !important;
            border: 1px solid rgba(129, 140, 248, 0.45) !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {
            border-color: #818CF8 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.28) !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            -webkit-text-fill-color: #F8FAFC !important;
            caret-color: #818CF8 !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
        }

        /* Disabled or Read-Only Inputs/Textareas still stay crisp & readable */
        .stTextInput input:disabled,
        .stTextArea textarea:disabled {
            background-color: #172033 !important;
            color: #E2E8F0 !important;
            -webkit-text-fill-color: #E2E8F0 !important;
            opacity: 1 !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
            -webkit-text-fill-color: #94A3B8 !important;
            opacity: 1 !important;
        }

        /* ==================================================================
           HIGH-CONTRAST BUTTONS & DOWNLOAD BUTTONS (Primary & Secondary)
           ================================================================== */
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(129, 140, 248, 0.45) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.15rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25) !important;
        }

        /* Ensure nested text/paragraphs inside all buttons are bright white */
        .stButton > button *,
        .stDownloadButton > button * {
            color: #F8FAFC !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #312E81 0%, #1E1B4B 100%) !important;
            border-color: #818CF8 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35) !important;
        }

        /* Primary CTA Buttons (Process Video / Process File) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
            border: 1px solid rgba(165, 180, 252, 0.55) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.55) !important;
        }

        /* Disabled Buttons */
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            background: rgba(30, 41, 59, 0.55) !important;
            color: #94A3B8 !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            opacity: 0.75 !important;
        }

        .stButton > button:disabled *,
        .stDownloadButton > button:disabled * {
            color: #94A3B8 !important;
        }

        /* ==================================================================
           SELECTBOX, RADIO, WIDGET LABELS & FILE UPLOADER
           ================================================================== */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        .stRadio label,
        .stRadio div[role="radiogroup"] label p,
        .stSelectbox label {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(129, 140, 248, 0.45) !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] * {
            color: #F8FAFC !important;
        }

        /* Dropdown menu list */
        div[data-baseweb="popover"] ul,
        div[data-baseweb="menu"] {
            background-color: #1E293B !important;
            border: 1px solid rgba(129, 140, 248, 0.4) !important;
        }

        div[data-baseweb="popover"] li,
        li[role="option"] {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
        }

        li[role="option"]:hover,
        li[aria-selected="true"] {
            background-color: #312E81 !important;
            color: #FFFFFF !important;
        }

        /* File Uploader Dropzone */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #1E293B !important;
            border: 1px dashed rgba(129, 140, 248, 0.55) !important;
            border-radius: 12px !important;
            color: #E2E8F0 !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #E2E8F0 !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #312E81 !important;
            color: #FFFFFF !important;
            border: 1px solid #818CF8 !important;
            border-radius: 8px !important;
        }

        /* ==================================================================
           CHAT INPUT & CHAT MESSAGES (RAG INTERFACE)
           ================================================================== */
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        [data-testid="stChatInput"] {
            background-color: #1E293B !important;
            border: 1px solid rgba(129, 140, 248, 0.5) !important;
            border-radius: 12px !important;
        }

        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #F8FAFC !important;
            -webkit-text-fill-color: #F8FAFC !important;
            caret-color: #818CF8 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #94A3B8 !important;
            -webkit-text-fill-color: #94A3B8 !important;
        }

        [data-testid="stChatMessage"] {
            background: rgba(30, 41, 59, 0.65) !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 12px !important;
            padding: 0.85rem 1.1rem !important;
            margin-bottom: 0.65rem !important;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            color: #F1F5F9 !important;
        }

        /* ==================================================================
           EXPANDERS, TABS & SIDEBAR
           ================================================================== */
        [data-testid="stExpander"] {
            background-color: rgba(17, 24, 39, 0.85) !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.85) !important;
            padding: 0.4rem;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            color: #CBD5E1 !important;
            background: transparent !important;
        }

        .stTabs [data-baseweb="tab"] * {
            color: #CBD5E1 !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(99, 102, 241, 0.28) !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid #818CF8 !important;
        }

        .stTabs [aria-selected="true"] * {
            color: #FFFFFF !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0D1322 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.18) !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #F1F5F9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. SESSION STATE INITIALIZATION & MANAGEMENT
# ==============================================================================
def init_session_state() -> None:
    """Initialize required Streamlit session state keys safely."""
    defaults = {
        "video_processed": False,
        "is_processing": False,
        "results": None,           # Dict returned by backend pipeline
        "source_display": "",      # Original URL or uploaded filename
        "source_type": "",         # "YouTube URL" or "Uploaded File"
        "selected_language": "English",
        "processed_at": "",
        "chat_history": [],        # List of {"role": "user"|"assistant", "content": str}
        "active_nav": "🏠 Workspace Dashboard",
        "error_message": None,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def reset_session_for_new_video() -> None:
    """Clear previous video results and RAG chat history before or on new run."""
    st.session_state.video_processed = False
    st.session_state.results = None
    st.session_state.chat_history = []
    st.session_state.error_message = None


# ==============================================================================
# 3. BACKEND PIPELINE INTEGRATION WITH REAL-TIME STAGE REPORTING
# ==============================================================================
def map_ui_language_to_backend(ui_language: str) -> str:
    """
    Map the user-facing language option ('English' or 'Hindi') to the exact
    parameter expected by `core/transcriber.py`.

    Note: `core/transcriber.py` checks `if language.lower() == "hinglish":`
    to route audio to Sarvam AI (`saaras:v4`), and defaults to OpenAI Whisper
    for `"english"`. Mapping `"Hindi"` -> `"hinglish"` ensures Hindi videos
    automatically use the Sarvam AI pipeline without modifying `transcriber.py`.
    """
    if ui_language.strip().lower() in ("hindi", "hinglish"):
        return "hinglish"
    return "english"


def execute_backend_with_live_status(
    source_path_or_url: str,
    ui_language: str,
    status_container: Any,
) -> Dict[str, Any]:
    """
    Execute the exact backend workflow defined in `main.py` while updating the
    Streamlit `st.status` container at each real processing stage.

    All underlying functions are imported directly from `utils.audio_processor`,
    `core.transcriber`, `core.summarize`, `core.extractor`, and `core.rag_engine`.
    """
    backend_lang = map_ui_language_to_backend(ui_language)
    is_url = source_path_or_url.startswith(("http://", "https://"))

    # Stage 1: Audio Extraction
    if is_url:
        status_container.update(
            label="Stage 1/7 — Extracting audio from YouTube video (yt-dlp)...",
            state="running",
        )
        st.write("📥 **Stage 1:** Extracting audio track from YouTube URL...")
        wav_path = download_youtube_audio(source_path_or_url)
    else:
        status_container.update(
            label="Stage 1/7 — Extracting and normalizing audio from file (Pydub)...",
            state="running",
        )
        st.write("🎵 **Stage 1:** Converting local media file to 16kHz mono WAV...")
        wav_path = convert_to_wav(source_path_or_url)

    # Stage 2: Audio Chunking
    status_container.update(
        label="Stage 2/7 — Splitting audio into manageable chunks...",
        state="running",
    )
    st.write("✂️ **Stage 2:** Splitting audio into chunks for transcription...")
    chunks = chunk_audio(wav_path)
    st.write(f"   ↳ Created **{len(chunks)}** audio chunk(s).")

    # Stage 3: Transcription (Whisper for English, Sarvam AI for Hindi)
    engine_label = "Sarvam AI" if backend_lang == "hinglish" else "OpenAI Whisper"
    status_container.update(
        label=f"Stage 3/7 — Transcribing audio using {engine_label}...",
        state="running",
    )
    st.write(f"🎙️ **Stage 3:** Transcribing audio with **{engine_label}** ({ui_language})...")

    transcript_parts = []
    for idx, chunk_file in enumerate(chunks, start=1):
        status_container.update(
            label=f"Stage 3/7 — Transcribing chunk {idx} of {len(chunks)} ({engine_label})...",
            state="running",
        )
        chunk_text = transcribe_chunk(chunk_file, language=backend_lang)
        transcript_parts.append(chunk_text)

    transcript = " ".join(transcript_parts).strip()
    if not transcript:
        raise ValueError(
            "Transcription returned empty text. Please verify the video has clear speech audio."
        )

    # Stage 4: Generating Video Title & Summary
    status_container.update(
        label="Stage 4/7 — Generating AI video title and executive summary (Gemini)...",
        state="running",
    )
    st.write("📝 **Stage 4:** Generating AI video title and structured summary...")
    title = generate_title(transcript)
    summary = summarize(transcript)

    # Stage 5: Extracting Action Items and Key Decisions
    status_container.update(
        label="Stage 5/7 — Extracting action items and key decisions...",
        state="running",
    )
    st.write("✅ **Stage 5:** Extracting action items and key decisions...")
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)

    # Stage 6: Generating Important Questions
    status_container.update(
        label="Stage 6/7 — Extracting important and unresolved questions...",
        state="running",
    )
    st.write("❓ **Stage 6:** Identifying important questions and follow-up topics...")
    open_questions = extract_questions(transcript)

    # Stage 7: Creating Vector Database for RAG
    status_container.update(
        label="Stage 7/7 — Creating ChromaDB vector store & LangChain RAG pipeline...",
        state="running",
    )
    st.write("🧠 **Stage 7:** Building embeddings and ChromaDB vector store for RAG...")
    rag_chain = build_rag_chain(transcript)

    status_container.update(
        label="Processing complete! Your video knowledge base is ready.",
        state="complete",
        expanded=False,
    )

    # Return the exact dictionary structure produced by main.run_pipeline()
    return {
        "title": title.strip() if title else "Untitled Video Session",
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


# ==============================================================================
# 4. EXPORT & DOWNLOAD HELPERS (TXT & REPORTLAB PDF)
# ==============================================================================
def build_txt_summary_report(results: Dict[str, Any], meta: Dict[str, str]) -> str:
    """Format a clean plain-text summary document."""
    return (
        f"====================================================================\n"
        f"AI VIDEO AGENT — VIDEO SUMMARY\n"
        f"====================================================================\n"
        f"Title      : {results.get('title', 'N/A')}\n"
        f"Source     : {meta.get('source', 'N/A')}\n"
        f"Language   : {meta.get('language', 'N/A')}\n"
        f"Generated  : {meta.get('processed_at', 'N/A')}\n"
        f"====================================================================\n\n"
        f"EXECUTIVE SUMMARY\n"
        f"-----------------\n"
        f"{results.get('summary', '')}\n"
    )


def build_txt_action_decisions_report(
    results: Dict[str, Any], meta: Dict[str, str]
) -> str:
    """Format Action Items, Key Decisions, and Important Questions as a TXT report."""
    return (
        f"====================================================================\n"
        f"AI VIDEO AGENT — ACTION ITEMS, KEY DECISIONS & QUESTIONS REPORT\n"
        f"====================================================================\n"
        f"Title      : {results.get('title', 'N/A')}\n"
        f"Source     : {meta.get('source', 'N/A')}\n"
        f"Language   : {meta.get('language', 'N/A')}\n"
        f"Generated  : {meta.get('processed_at', 'N/A')}\n"
        f"====================================================================\n\n"
        f"1. ACTION ITEMS\n"
        f"---------------\n"
        f"{results.get('action_items', 'No action items found.')}\n\n"
        f"2. KEY DECISIONS\n"
        f"----------------\n"
        f"{results.get('key_decisions', 'No key decisions found.')}\n\n"
        f"3. IMPORTANT / UNRESOLVED QUESTIONS\n"
        f"-----------------------------------\n"
        f"{results.get('open_questions', 'No open questions found.')}\n"
    )


def _sanitize_for_reportlab(text: str) -> str:
    """Escape XML special characters and convert basic Markdown bold to ReportLab tags."""
    if not text:
        return ""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # Convert **bold** to <b>bold</b>
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return escaped


def build_pdf_report(
    results: Dict[str, Any],
    meta: Dict[str, str],
    include_extractions: bool = True,
) -> Optional[bytes]:
    """
    Generate a polished PDF report using ReportLab.
    Returns PDF bytes or None if ReportLab is unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=results.get("title", "AI Video Agent Report"),
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E1B4B"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#4338CA"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )

    story = []

    # Title & Metadata Header
    story.append(
        Paragraph(
            _sanitize_for_reportlab(results.get("title", "AI Video Summary")),
            title_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Source:</b> {_sanitize_for_reportlab(meta.get('source', 'N/A'))} &nbsp;|&nbsp; "
            f"<b>Language:</b> {_sanitize_for_reportlab(meta.get('language', 'N/A'))} &nbsp;|&nbsp; "
            f"<b>Processed:</b> {_sanitize_for_reportlab(meta.get('processed_at', 'N/A'))}",
            meta_style,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=4,
            spaceAfter=12,
        )
    )

    def add_multiline_section(heading: str, content: str) -> None:
        story.append(Paragraph(heading, section_style))
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 4))
                continue
            story.append(Paragraph(_sanitize_for_reportlab(stripped), body_style))

    # Summary Section
    add_multiline_section("Executive Summary", results.get("summary", ""))

    # Optional Action Items, Key Decisions & Questions
    if include_extractions:
        add_multiline_section(
            "Action Items", results.get("action_items", "No action items found.")
        )
        add_multiline_section(
            "Key Decisions", results.get("key_decisions", "No key decisions found.")
        )
        add_multiline_section(
            "Important Questions",
            results.get("open_questions", "No open questions found."),
        )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 5. UI COMPONENTS — HEADER, SIDEBAR, INPUT, RESULTS, AND RAG CHAT
# ==============================================================================
def render_top_header() -> None:
    """Render the top AI Video Agent brand banner and quick navigation pills."""
    st.markdown(
        """
        <div class="hero-banner">
            <div>
                <h1 class="hero-title">🎬 AI Video Agent</h1>
                <p class="hero-subtitle">Transform videos into searchable knowledge with AI.</p>
            </div>
            <div class="hero-badge-group">
                <span class="saas-badge">🎙️ Whisper &amp; Sarvam AI</span>
                <span class="saas-badge">✨ Google Gemini</span>
                <span class="saas-badge">🧠 LangChain + ChromaDB RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render sidebar navigation, session status card, and quick reset controls."""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        nav_selection = st.radio(
            "Go to section",
            options=[
                "🏠 Workspace Dashboard",
                "💬 AI Chat with Video (RAG)",
                "📥 Downloads & Exports",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Language Configuration")
        selected_lang = st.selectbox(
            "Video Audio Language",
            options=["English", "Hindi"],
            index=0 if st.session_state.selected_language == "English" else 1,
            help=(
                "Select 'English' for local OpenAI Whisper transcription, or "
                "'Hindi' for Sarvam AI speech-to-text translation."
            ),
            disabled=st.session_state.is_processing,
        )
        st.session_state.selected_language = selected_lang

        engine_info = (
            "OpenAI Whisper (Local)"
            if selected_lang == "English"
            else "Sarvam AI (Hindi → English STT)"
        )
        st.caption(f"**Active STT Engine:** `{engine_info}`")

        st.markdown("---")
        st.markdown("### 📊 Session Status")
        if st.session_state.video_processed and st.session_state.results:
            st.success("Video Processed & Indexed")
            st.caption(f"**Title:** {st.session_state.results.get('title', 'N/A')}")
            st.caption(f"**Source:** {st.session_state.source_display}")
            st.caption(f"**Language:** {st.session_state.selected_language}")
            st.caption(f"**Chat Messages:** {len(st.session_state.chat_history)}")

            if st.button(
                "🔄 Process Another Video",
                use_container_width=True,
                disabled=st.session_state.is_processing,
            ):
                reset_session_for_new_video()
                st.rerun()
        else:
            st.info("No video processed yet. Submit a YouTube URL or upload a file to begin.")

        return nav_selection


def render_input_section() -> None:
    """Render the dual-mode Video Input Section (YouTube URL or Local File Upload)."""
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">📥 Video / Audio Input &amp; Language Selection</div>
            <p style="color:#CBD5E1; font-size:0.9rem; margin-bottom:0.2rem;">
                Choose your input source below and select the spoken language before starting the AI pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-level Language Selector inside main dashboard alongside sidebar sync
    col_input_mode, col_lang_mode = st.columns([3, 1])
    with col_lang_mode:
        main_lang = st.radio(
            "Spoken Language",
            options=["English", "Hindi"],
            index=0 if st.session_state.selected_language == "English" else 1,
            horizontal=True,
            disabled=st.session_state.is_processing,
            help="Passed directly to the backend transcription pipeline.",
        )
        st.session_state.selected_language = main_lang

    with col_input_mode:
        input_tab_yt, input_tab_upload = st.tabs(
            ["🔗 Option 1: YouTube URL", "📂 Option 2: Upload Video / Audio"]
        )

    # Option 1: YouTube URL Input
    with input_tab_yt:
        yt_col1, yt_col2 = st.columns([4, 1])
        with yt_col1:
            youtube_url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
                disabled=st.session_state.is_processing,
            )
        with yt_col2:
            process_yt_clicked = st.button(
                "🚀 Process Video",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_processing,
                key="btn_process_youtube",
            )

        if process_yt_clicked:
            cleaned_url = youtube_url.strip()
            if not cleaned_url:
                st.warning("⚠️ Please enter a valid YouTube video URL before clicking **Process Video**.")
            elif not cleaned_url.startswith(("http://", "https://")):
                st.error("❌ Invalid URL format. Please include `https://` (e.g., `https://www.youtube.com/watch?v=...`).")
            else:
                _trigger_processing(
                    source_input=cleaned_url,
                    display_name=cleaned_url,
                    source_type="YouTube URL",
                )

    # Option 2: Upload Video / Audio File
    with input_tab_upload:
        uploaded_file = st.file_uploader(
            "Upload MP4, MOV, MP3, WAV, or M4A file",
            type=["mp4", "mov", "mp3", "wav", "m4a"],
            disabled=st.session_state.is_processing,
            help="Supported formats: MP4, MOV, MP3, WAV, M4A",
        )
        process_file_clicked = st.button(
            "⚙️ Process File",
            type="primary",
            disabled=st.session_state.is_processing or (uploaded_file is None),
            key="btn_process_file",
        )

        if process_file_clicked and uploaded_file is not None:
            # Save uploaded file to local disk so pydub/ffmpeg can process it
            safe_filename = os.path.basename(uploaded_file.name)
            saved_path = os.path.join(UPLOAD_TEMP_DIR, safe_filename)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            _trigger_processing(
                source_input=saved_path,
                display_name=safe_filename,
                source_type="Uploaded File",
            )


def _trigger_processing(
    source_input: str,
    display_name: str,
    source_type: str,
) -> None:
    """
    Coordinate session state reset, duplicate click protection, real-time
    status updates, and graceful error handling during pipeline execution.
    """
    if st.session_state.is_processing:
        st.info("⏳ A video is currently being processed. Please wait...")
        return

    st.session_state.is_processing = True
    reset_session_for_new_video()

    try:
        with st.status(
            "Initializing AI Video Agent pipeline...",
            expanded=True,
        ) as status_box:
            results = execute_backend_with_live_status(
                source_path_or_url=source_input,
                ui_language=st.session_state.selected_language,
                status_container=status_box,
            )

        # Store results in session_state so tab switches & RAG chat never re-run transcription
        st.session_state.results = results
        st.session_state.video_processed = True
        st.session_state.source_display = display_name
        st.session_state.source_type = source_type
        st.session_state.processed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history = []
        st.toast("Video processed! Scroll down to explore insights or chat.", icon="🎉")

    except Exception as exc:
        st.session_state.video_processed = False
        st.session_state.error_message = str(exc)
        st.error(
            f"❌ **Pipeline Error:** Could not complete processing.\n\n"
            f"**Details:** `{exc}`\n\n"
            f"Please check that FFmpeg is installed, your `.env` API keys (`GOOGLE_API_KEY` / `SARVAM_API_KEY`) "
            f"are configured, and the video source is accessible."
        )
    finally:
        st.session_state.is_processing = False


def render_results_dashboard() -> None:
    """Render Section 5: Video Information, Summary, Action Items, Decisions, Questions, and Transcript."""
    results = st.session_state.results
    if not st.session_state.video_processed or not results:
        return

    st.markdown("### 📊 Video Intelligence Dashboard")

    # A. Video Information Metadata Cards
    word_count = len(results.get("transcript", "").split())
    source_val = st.session_state.source_display
    source_html = (
        f'<a href="{source_val}" target="_blank" style="color:#A5B4FC; text-decoration:none;">{source_val}</a>'
        if st.session_state.source_type == "YouTube URL"
        else source_val
    )

    st.markdown(
        f"""
        <div class="meta-grid">
            <div class="meta-box">
                <div class="meta-label">🏷️ Generated Video Title</div>
                <div class="meta-value">{results.get("title", "Untitled")}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">🔗 Source ({st.session_state.source_type})</div>
                <div class="meta-value">{source_html}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">🌐 Selected Language</div>
                <div class="meta-value">{st.session_state.selected_language}</div>
            </div>
            <div class="meta-box">
                <div class="meta-label">📄 Transcript Volume</div>
                <div class="meta-value">{word_count:,} words</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Organized Tabs for Results Sections (B, C, D, E, F)
    tab_overview, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
        [
            "📝 Video Summary",
            "✅ Action Items",
            "🎯 Key Decisions",
            "❓ Important Questions",
            "📜 Full Transcript",
        ]
    )

    with tab_overview:
        st.markdown(
            '<div class="saas-card"><div class="saas-card-header">📝 AI-Generated Executive Summary</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results.get("summary", "_No summary available._"))
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick side-by-side preview of Action Items & Decisions on the main summary view
        col_a, col_b = st.columns(2)
        with col_a:
            with st.expander("✅ Quick View: Action Items", expanded=True):
                st.markdown(results.get("action_items", "_No action items found._"))
        with col_b:
            with st.expander("🎯 Quick View: Key Decisions", expanded=True):
                st.markdown(results.get("key_decisions", "_No key decisions found._"))

    with tab_actions:
        st.markdown(
            '<div class="saas-card"><div class="saas-card-header">✅ Extracted Action Items</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results.get("action_items", "_No action items found._"))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(
            '<div class="saas-card"><div class="saas-card-header">🎯 Key Decisions Identified</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results.get("key_decisions", "_No key decisions found._"))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_questions:
        st.markdown(
            '<div class="saas-card"><div class="saas-card-header">❓ Important &amp; Unresolved Questions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results.get("open_questions", "_No open questions found._"))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(
            '<div class="saas-card"><div class="saas-card-header">📜 Complete Video Transcript</div>',
            unsafe_allow_html=True,
        )
        transcript_text = results.get("transcript", "")
        st.text_area(
            "Full Transcript (Read-Only)",
            value=transcript_text,
            height=360,
            label_visibility="collapsed",
        )
        st.caption(
            f"Total length: {len(transcript_text):,} characters ({word_count:,} words)."
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_rag_chat_section() -> None:
    """
    Render Section 6: Interactive ChatGPT-style RAG Question-Answering interface
    powered by the existing `ask_question(rag_chain, question)` function.
    """
    st.markdown("### 💬 AI Chat with Video (RAG)")

    if not st.session_state.video_processed or not st.session_state.results:
        st.info(
            "👋 **Welcome to AI Video Chat!** Please process a YouTube URL or upload a video/audio file "
            "above first. Once your video's ChromaDB vector database is built, this interactive chat will unlock."
        )
        # Disabled chat input with friendly prompt
        st.chat_input(
            "Process a video first to start asking questions...",
            disabled=True,
            key="disabled_rag_chat_input",
        )
        return

    results = st.session_state.results
    rag_chain = results.get("rag_chain")

    # Header bar with example prompts & Clear Chat History button
    hdr_col1, hdr_col2 = st.columns([4, 1])
    with hdr_col1:
        st.caption(
            f"Ask anything about **{results.get('title', 'this video')}**. Answers are grounded strictly in the "
            f"video's transcript via LangChain + ChromaDB retrieval."
        )
    with hdr_col2:
        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True,
            disabled=len(st.session_state.chat_history) == 0,
        ):
            st.session_state.chat_history = []
            st.rerun()

    # Quick-ask suggestion pills when chat is empty
    if not st.session_state.chat_history:
        st.markdown("**💡 Suggested questions to try:**")
        q_cols = st.columns(3)
        sample_questions = [
            "What is the main topic discussed in this video?",
            "What action items were mentioned?",
            "What are the important conclusions?",
        ]
        for idx, sample_q in enumerate(sample_questions):
            if q_cols[idx].button(sample_q, key=f"sample_q_{idx}", use_container_width=True):
                _handle_user_rag_question(rag_chain, sample_q)
                st.rerun()

    # Render conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat Input Box
    user_query = st.chat_input(
        "Ask a question about the processed video (e.g., 'Explain the second point in more detail')..."
    )
    if user_query:
        _handle_user_rag_question(rag_chain, user_query)
        st.rerun()


def _handle_user_rag_question(rag_chain: Any, question: str) -> None:
    """Append user question, invoke existing `ask_question(rag_chain, question)`, and store response."""
    cleaned_q = question.strip()
    if not cleaned_q:
        return

    st.session_state.chat_history.append({"role": "user", "content": cleaned_q})

    with st.spinner("Searching video transcript & generating grounded response..."):
        try:
            answer = ask_question(rag_chain, cleaned_q)
        except Exception as exc:
            answer = (
                f"⚠️ An error occurred while querying the RAG chain: `{exc}`. "
                f"Please verify your `GOOGLE_API_KEY` and try again."
            )

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def render_downloads_section() -> None:
    """Render Section 7: Download & Export buttons for TXT and PDF deliverables."""
    st.markdown("### 📥 Download & Export Reports")

    if not st.session_state.video_processed or not st.session_state.results:
        st.info("Process a video first to unlock TXT and PDF report downloads.")
        return

    results = st.session_state.results
    meta = {
        "source": st.session_state.source_display,
        "language": st.session_state.selected_language,
        "processed_at": st.session_state.processed_at,
    }

    # Create safe filename slug from generated video title
    raw_title = results.get("title", "video_report")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_title.strip()).strip("_").lower() or "video_report"

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    # 1. Full Transcript as TXT
    with dl_col1:
        st.download_button(
            label="📄 Full Transcript (.TXT)",
            data=results.get("transcript", "").encode("utf-8"),
            file_name=f"{slug}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 2. Video Summary as TXT
    with dl_col2:
        summary_txt = build_txt_summary_report(results, meta)
        st.download_button(
            label="📝 Video Summary (.TXT)",
            data=summary_txt.encode("utf-8"),
            file_name=f"{slug}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 3. Video Summary as PDF (ReportLab)
    with dl_col3:
        pdf_bytes = build_pdf_report(results, meta, include_extractions=True)
        if pdf_bytes is not None:
            st.download_button(
                label="📕 Summary & Insights (.PDF)",
                data=pdf_bytes,
                file_name=f"{slug}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button(
                "📕 Summary (.PDF - Install reportlab)",
                disabled=True,
                use_container_width=True,
            )

    # 4. Action Items & Key Decisions Report (.TXT)
    with dl_col4:
        actions_txt = build_txt_action_decisions_report(results, meta)
        st.download_button(
            label="✅ Actions & Decisions (.TXT)",
            data=actions_txt.encode("utf-8"),
            file_name=f"{slug}_actions_decisions.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ==============================================================================
# 6. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    """Main Streamlit entrypoint."""
    inject_custom_css()
    init_session_state()

    render_top_header()
    active_nav = render_sidebar()

    if active_nav == "🏠 Workspace Dashboard":
        render_input_section()
        if st.session_state.video_processed:
            st.markdown("---")
            render_results_dashboard()
            st.markdown("---")
            render_downloads_section()
            st.markdown("---")
            render_rag_chat_section()

    elif active_nav == "💬 AI Chat with Video (RAG)":
        render_rag_chat_section()

    elif active_nav == "📥 Downloads & Exports":
        render_downloads_section()
        if st.session_state.video_processed:
            st.markdown("---")
            render_results_dashboard()


if __name__ == "__main__":
    main()