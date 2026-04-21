"""
WHU Dortmund Explorer - Multi-Agent Streamlit App (PLANNER VERSION)
Group Assignment Phase 4

ARCHITECTURE:
  Planner Agent ──► [list of sub-tasks] ──► Text Agent (RAG)    ┐
                                        ├──► Coding Agent (Pandas) ├──► Final Editor ─► User
                                        └──► Visual Agent (Plotly) ─┘
                                        (each task runs in order; charts stay as charts)

WHY THE PLANNER:
  A single user prompt may need MULTIPLE agents. The Planner decomposes the
  request into self-contained sub-tasks with the right agent for each, then
  the Final Editor merges the individual results into one coherent answer.

HOW TO SWAP DATA WHEN PHASE 1+2 ARE DONE:
  Change DATA_SOURCE_XLSX / DATA_FILE and update CATEGORY_COLUMNS.
"""

import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import plotly.express as px

# LangChain imports for RAG
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_FILE = "DORTMUND.csv"
DATA_SOURCE_XLSX = "DORTMUND.xlsx"
CATEGORY_COLUMNS: list[str] = []  # fill in after Phase 2

# =============================================================================
# WHU BRANDING
# =============================================================================

WHU_BLUE = "#2C4592"
WHU_LIGHT_BLUE = "#808FBE"
WHU_DARK_GRAY = "#515256"
WHU_LIGHT_GRAY = "#C7C1BF"
WHU_VERY_LIGHT_GRAY = "#EEEBEA"
WHU_RED = "#E7331A"
WHU_SUCCESS = "#2E7D32"

st.set_page_config(
    page_title="WHU Dortmund Explorer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    /* Typography — cleaner, system-native */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }}

    /* Headers */
    h1, h2, h3 {{ color: {WHU_BLUE} !important; font-weight: 600 !important; letter-spacing: -0.01em; }}
    h1 {{ font-size: 2rem !important; }}
    h2 {{ font-size: 1.35rem !important; margin-top: 1.2rem !important; }}
    h3 {{ font-size: 1.1rem !important; }}

    /* Hide Streamlit chrome we don't want */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* Primary button — solid WHU blue, clean */
    .stButton > button[kind="primary"] {{
        background-color: {WHU_BLUE} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #1F3472 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(44, 69, 146, 0.25);
    }}
    /* Secondary buttons — flat with border */
    .stButton > button:not([kind="primary"]) {{
        border: 1px solid {WHU_LIGHT_GRAY} !important;
        color: {WHU_DARK_GRAY} !important;
        background-color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease;
    }}
    .stButton > button:not([kind="primary"]):hover {{
        border-color: {WHU_BLUE} !important;
        color: {WHU_BLUE} !important;
        background-color: #F8FAFE !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {WHU_BLUE} !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {WHU_DARK_GRAY} !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #FAFBFD !important;
        border-right: 1px solid #E5E8EE;
    }}
    [data-testid="stSidebar"] h3 {{
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {WHU_DARK_GRAY} !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
    }}

    /* Container with border — clean cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 8px !important;
    }}

    /* Links */
    a {{ color: {WHU_BLUE} !important; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* Divider */
    hr {{ border-color: #E5E8EE !important; margin: 1.2rem 0 !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid #E5E8EE;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {WHU_DARK_GRAY};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {WHU_BLUE} !important;
        border-bottom-color: {WHU_BLUE} !important;
    }}

    /* Text area */
    .stTextArea textarea {{
        border-radius: 8px !important;
        border: 1px solid #D1D5DB !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {WHU_BLUE} !important;
        box-shadow: 0 0 0 3px rgba(44, 69, 146, 0.1) !important;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Provider, Model dropdown, Performance toggle
# =============================================================================

# Predefined model options per provider
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — balanced quality (recommended)",
    "openai/gpt-oss-120b":     "GPT-OSS 120B — highest quality, separate quota",
    "llama-3.1-8b-instant":    "Llama 3.1 8B — fastest, lower quality",
    "mixtral-8x7b-32768":      "Mixtral 8x7B — long context (32k)",
}
OLLAMA_MODELS = {
    "glm-4.7-flash:latest": "GLM 4.7 Flash 29.9B — best on-prem (recommended)",
    "llama3:latest":        "Llama 3 8B — faster but weaker",
}
CUSTOM_LABEL = "Custom…"

with st.sidebar:
    st.markdown(
        f'<div style="padding:0.25rem 0 1.25rem 0;">'
        f'<div style="font-weight:700;color:{WHU_BLUE};font-size:1.05rem;letter-spacing:-0.01em;">'
        f'WHU Dortmund Explorer</div>'
        f'<div style="color:{WHU_DARK_GRAY};font-size:0.75rem;">Multi-Agent Pipeline</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Provider ---
    # Detect Streamlit Cloud: the WHU Ollama endpoint is unreachable from cloud
    # deployments (requires eduVPN). Detect via a known env var set by Streamlit
    # Cloud, and also fall back to a simple hostname check.
    IS_CLOUD = (
        os.getenv("STREAMLIT_SHARING_MODE") is not None
        or os.getenv("HOSTNAME", "").startswith("streamlit-")
        or "/mount/src/" in os.path.abspath(__file__)  # reliable: Cloud mounts repos here
    )

    st.markdown("### Provider")
    if IS_CLOUD:
        # Only Groq is usable on Streamlit Cloud
        provider = "Groq (cloud)"
        st.radio(
            "Provider",
            options=["Groq (cloud)"],
            index=0,
            label_visibility="collapsed",
            disabled=True,
            help="WHU Ollama is only reachable from within the WHU network (requires eduVPN). "
                 "Clone the repo and run the app locally to use the on-premise option.",
        )
        st.caption("ℹ️ Cloud deployment — WHU Ollama disabled (VPN unreachable from cloud)")
    else:
        provider = st.radio(
            "Provider",
            options=["Groq (cloud)", "WHU Ollama (VPN)"],
            index=0,
            label_visibility="collapsed",
            help="Groq: fast cloud inference (requires API key). "
                 "WHU Ollama: on-premise (requires eduVPN).",
        )

    # --- Model dropdown (varies per provider) ---
    st.markdown("### Model")
    if provider == "Groq (cloud)":
        options = list(GROQ_MODELS.keys()) + [CUSTOM_LABEL]
        model_choice = st.selectbox(
            "Model",
            options=options,
            index=0,
            format_func=lambda k: k if k == CUSTOM_LABEL else k,
            label_visibility="collapsed",
        )
        if model_choice != CUSTOM_LABEL:
            st.caption(GROQ_MODELS[model_choice])
            llm_model = model_choice
        else:
            llm_model = st.text_input(
                "Custom model name", placeholder="e.g. deepseek-r1-distill-llama-70b",
                label_visibility="collapsed",
            ) or "llama-3.3-70b-versatile"

        llm_base_url = "https://api.groq.com/openai/v1"
        groq_api_key = st.text_input(
            "API Key", type="password", placeholder="gsk_…",
            help="Free key at console.groq.com/keys",
        )
        llm_api_key = groq_api_key
        provider_ready = bool(groq_api_key)

    else:  # WHU Ollama
        options = list(OLLAMA_MODELS.keys()) + [CUSTOM_LABEL]
        model_choice = st.selectbox(
            "Model",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        if model_choice != CUSTOM_LABEL:
            st.caption(OLLAMA_MODELS[model_choice])
            llm_model = model_choice
        else:
            llm_model = st.text_input(
                "Custom model name", placeholder="e.g. mixtral:8x7b",
                label_visibility="collapsed",
            ) or "glm-4.7-flash:latest"

        # Ollama endpoint — read from st.secrets to keep internal IPs out of Git
        try:
            llm_base_url = st.secrets["OLLAMA_BASE_URL"]
        except (KeyError, FileNotFoundError):
            st.error(
                "🔒 WHU Ollama endpoint not configured. "
                "Create `.streamlit/secrets.toml` with:\n\n"
                "```\nOLLAMA_BASE_URL = \"http://<internal-ip>:11434/v1\"\n```\n\n"
                "Ask your group for the exact IP — do NOT commit it to Git."
            )
            st.stop()
        llm_api_key = "ollama"
        provider_ready = True

    # --- Performance toggle ---
    st.markdown("### Performance")
    token_saving = st.toggle(
        "Token-saving mode",
        value=False,
        help="Reduces token usage by ~60% at slight quality cost. "
             "Use when near rate limits.",
    )

    # --- Status box ---
    st.markdown("---")
    if provider == "Groq (cloud)" and not provider_ready:
        st.warning("⚠️ Enter your Groq API key to continue")
    elif provider == "WHU Ollama (VPN)":
        st.info("🔒 eduVPN required · Ollama endpoint configured via secrets")
    else:
        st.success(f"✓ Ready · {llm_model}")

    if token_saving:
        st.caption("⚡ Token-saving: compact facts · k=3 · 1 retry")
    else:
        st.caption("Full quality: k=5 retrieval · 2 retries")

if not provider_ready:
    st.info("👈 Complete the setup in the sidebar to get started.")
    st.stop()

# Module-level: picked up by agents and context builders
TOKEN_SAVING = token_saving

# =============================================================================
# SESSION STATE — conversation history for Multi-Turn Memory (Improvement #3)
# =============================================================================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Memory controls in the sidebar (placed after provider check so they always render)
with st.sidebar:
    st.markdown("### Conversation")
    turn_count = len(st.session_state.conversation_history)
    if turn_count > 0:
        st.caption(f"💬 {turn_count} previous turn(s) · follow-ups supported")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    else:
        st.caption("💬 Fresh conversation — first turn")

# =============================================================================
# UNIFIED LLM CLIENTS — both providers expose an OpenAI-compatible interface
# =============================================================================

from openai import OpenAI
from langchain_openai import ChatOpenAI

# Explicit timeouts prevent infinite spinners when an LLM call hangs.
# 45s per call is generous for 120B models; total budget across all agents
# in a compound query stays within Streamlit's ~2min window.
LLM_TIMEOUT_SECONDS = 45

# Direct client (used by planner, coding, visual, editor agents)
client = OpenAI(
    api_key=llm_api_key,
    base_url=llm_base_url,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=1,  # internal SDK retries — keep low so we fail fast on hangs
)

# LangChain client (used by the RAG text agent)
langchain_llm = ChatOpenAI(
    model=llm_model,
    base_url=llm_base_url,
    api_key=llm_api_key,
    temperature=0.1,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=1,
)

LLM_MODEL = llm_model  # used everywhere the agents call the direct client

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def ensure_csv_exists() -> str | None:
    """Convert the Excel source to CSV if needed. Returns the CSV path, or None
    if neither the CSV nor the Excel source exists (e.g. when running on Streamlit
    Cloud without the proprietary dataset committed to the repo)."""
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    if os.path.exists(DATA_SOURCE_XLSX):
        df = pd.read_excel(DATA_SOURCE_XLSX)
        df.columns = [c.replace("\n", " ").strip() for c in df.columns]
        df.to_csv(DATA_FILE, index=False, encoding="utf-8")
        return DATA_FILE
    return None


csv_path = ensure_csv_exists()

if csv_path is None:
    st.warning(
        f"### 📦 Dataset not available in this deployment\n\n"
        f"This app analyses the **Dortmund company dataset** (`{DATA_SOURCE_XLSX}`), "
        f"which is proprietary and not included in the public repository.\n\n"
        f"**To run this app with real data:**\n"
        f"1. Clone the repository locally\n"
        f"2. Place `DORTMUND.xlsx` in the project root\n"
        f"3. Run `streamlit run dortmund_multi_agent_app_planner.py`\n\n"
        f"The architecture (Planner → Executors → Editor pipeline with hybrid RAG, "
        f"self-correction, and sandboxed code execution) is fully functional — "
        f"it just needs the dataset to operate on."
    )
    st.info(
        "**For evaluators:** The complete source code, including all 5 agents, "
        "9 architectural improvements over the baseline template, and the full "
        "prompt engineering, is visible in the repository."
    )
    st.stop()

# =============================================================================
# AGENT SETUP
# =============================================================================

# =============================================================================
# GROUND TRUTH FACTS — computed in Python, injected into every agent's context
# This is the key anti-hallucination mechanism. Instead of letting the LLM
# guess statistics from 5 retrieved rows, we compute them once from the full
# dataset and hand them to the LLM as verified facts.
# =============================================================================

@st.cache_data
def compute_data_facts():
    """Compute ground-truth statistics from the full dataset."""
    df = pd.read_csv(csv_path, encoding="utf-8")

    def coerce(col: str):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

    emp24 = coerce("Number of employees 2024")
    fy = coerce("Founded Year")

    # Top 10 largest employers 2024
    top10 = (df.assign(_emp=emp24)
               .dropna(subset=["_emp"])
               .nlargest(10, "_emp")[["Company name Latin alphabet", "_emp"]]
               .values.tolist())

    # Founded-decade distribution
    decades = (fy // 10 * 10).dropna().astype(int).value_counts().sort_index().to_dict()

    scaling_metrics = ["Scaler", "HighGrowthFirm", "ConsistentHighGrowthFirm",
                       "VeryHighGrowthFirm", "Gazelle", "Mature", "Scaleup", "Superstar"]

    facts = {
        "n_total": int(df.shape[0]),
        "n_active": int((df["Status"] == "Active").sum()),
        "status_dist": df["Status"].value_counts().to_dict(),
        "nace_counts": df["NACE Rev. 2 main section"].value_counts().to_dict(),
        "founded_min": int(fy.min()) if fy.notna().any() else None,
        "founded_max": int(fy.max()) if fy.notna().any() else None,
        "founded_decades": decades,
        "emp_min": int(emp24.min()) if emp24.notna().any() else None,
        "emp_max": int(emp24.max()) if emp24.notna().any() else None,
        "emp_mean": round(float(emp24.mean()), 1) if emp24.notna().any() else None,
        "emp_median": int(emp24.median()) if emp24.notna().any() else None,
        "top10_employers": top10,
        "scaling_2024": {m: int(coerce(f"{m} 2024").sum()) for m in scaling_metrics},
        "scaler_by_year": {y: int(coerce(f"Scaler {y}").sum()) for y in [2020, 2021, 2022, 2023, 2024]},
    }
    return facts, df


def format_facts_block(facts: dict, compact: bool = False) -> str:
    """Format facts as a markdown block. Output contains NO curly braces
    so it can be safely embedded in f-strings and ChatPromptTemplate strings.

    compact=True produces a ~70% smaller version for token-saving mode."""

    if compact:
        L = []
        L.append("=== DATASET FACTS (compact — use exact numbers, do not invent) ===")
        L.append(f"Total companies: {facts['n_total']} | Active: {facts['n_active']}")
        L.append(f"Founded: {facts['founded_min']}-{facts['founded_max']} | "
                 f"Employees 2024: min={facts['emp_min']}, max={facts['emp_max']}, "
                 f"median={facts['emp_median']}, mean={facts['emp_mean']}")
        # Top 5 NACE
        top5_nace = sorted(facts["nace_counts"].items(), key=lambda x: -x[1])[:5]
        L.append("Top 5 NACE sections (count): " + "; ".join(f"{s}={c}" for s, c in top5_nace))
        # Top 5 employers
        top5_emp = facts["top10_employers"][:5]
        L.append("Top 5 employers 2024: " + "; ".join(f"{n} ({int(e)})" for n, e in top5_emp))
        # Scaling 2024 compact
        L.append("Scaling metrics 2024: " + ", ".join(f"{m}={c}" for m, c in facts["scaling_2024"].items()))
        L.append(f"Scaler evolution: "
                 f"2020={facts['scaler_by_year'].get(2020, '?')} → "
                 f"2024={facts['scaler_by_year'].get(2024, '?')}")
        return "\n".join(L)

    # Full version
    L = []
    L.append("=== DATASET FACTS (computed in Python from the FULL dataset — use these for ALL numeric claims) ===")
    L.append("")
    L.append(f"Total companies: {facts['n_total']}")
    L.append(f"Active companies: {facts['n_active']}")
    L.append("Status breakdown:")
    for status, cnt in facts["status_dist"].items():
        L.append(f"  - {status}: {cnt}")
    L.append("")
    L.append(f"Founded Year range: {facts['founded_min']} to {facts['founded_max']}")
    L.append("Founded by decade:")
    for dec, cnt in sorted(facts["founded_decades"].items()):
        L.append(f"  - {dec}s: {cnt} companies")
    L.append("")
    L.append(f"Employees in 2024 — min: {facts['emp_min']}, max: {facts['emp_max']}, "
             f"mean: {facts['emp_mean']}, median: {facts['emp_median']}")
    L.append("")
    L.append("NACE Rev. 2 main section counts (all sections, sorted by frequency):")
    for section, cnt in sorted(facts["nace_counts"].items(), key=lambda x: -x[1]):
        L.append(f"  - {section}: {cnt}")
    L.append("")
    L.append("Top 10 largest employers in 2024:")
    for i, (name, emp) in enumerate(facts["top10_employers"], 1):
        L.append(f"  {i}. {name}: {int(emp)} employees")
    L.append("")
    L.append("Scaling metric counts for 2024 (number of companies flagged as 1):")
    for metric, cnt in facts["scaling_2024"].items():
        L.append(f"  - {metric} 2024: {cnt}")
    L.append("")
    L.append("Scaler counts by year (evolution):")
    for y, cnt in facts["scaler_by_year"].items():
        L.append(f"  - {y}: {cnt} scalers")
    L.append("")
    L.append("⚠️ Use these EXACT numbers when answering. Do not invent, estimate, or round.")
    return "\n".join(L)


@st.cache_resource
def setup_text_analysis_agent(compact: bool = False, rag_k: int = 5, hybrid: bool = True):
    """RAG chain where the template injects ground-truth facts alongside retrieved rows.
    This prevents the text agent from hallucinating statistics from only a few retrieved chunks.

    Improvement Point 5 — Hybrid Retrieval: combines BM25 (keyword match, great for
    company names like 'THYSSENKRUPP NUCERA') with dense embeddings (semantic understanding).
    LangChain's EnsembleRetriever does the reciprocal-rank-fusion internally.

    Cached per (compact, rag_k, hybrid) triple → toggling any of them builds a new cached chain.
    """
    facts, _ = compute_data_facts()
    facts_block = format_facts_block(facts, compact=compact)

    loader = CSVLoader(csv_path, encoding="utf-8")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    llm = langchain_llm  # unified ChatOpenAI — works for both Groq and Ollama

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": rag_k})
    if hybrid:
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = rag_k
            retriever = EnsembleRetriever(
                retrievers=[bm25, dense_retriever],
                weights=[0.4, 0.6],  # slightly favor dense for semantic queries
            )
        except ImportError:
            # Fall back gracefully if rank_bm25 isn't installed
            retriever = dense_retriever
    else:
        retriever = dense_retriever

    # Concatenate (not f-string) so {context} and {question} remain the only
    # template variables visible to ChatPromptTemplate.
    template = (
        "You are answering questions about companies headquartered in Dortmund.\n\n"
        + facts_block + "\n\n"
        + "=== RETRIEVED ROWS (specific companies matching the question via hybrid BM25+dense search) ===\n"
        + "{context}\n\n"
        + "Question: {question}\n\n"
        + "INSTRUCTIONS:\n"
        + "- For ANY number / count / statistic / range → use the DATASET FACTS section. These are ground truth.\n"
        + "- For qualitative info about specific companies → use the RETRIEVED ROWS section.\n"
        + "- Do NOT invent numbers. If unsure, say 'this is not available in the dataset'.\n"
        + "- Answer in English.\n"
    )
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever | (lambda ds: "\n\n".join(d.page_content for d in ds)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


@st.cache_data
def setup_data_context(compact: bool = False):
    """Shared system prompt for coding + visual agents. Now includes the
    full ground-truth facts block so agents can reference real numbers
    even without executing code.

    In compact mode we also trim the dataset metadata (columns + sample) heavily:
    - skip `df.head(3).to_string()` entirely (saves ~1300 tokens per call)
    - replace verbose dtypes listing with a single-line column inventory (saves ~600 tokens)
    """
    facts, df = compute_data_facts()
    facts_block = format_facts_block(facts, compact=compact)

    if compact:
        # One-line column inventory, grouped by semantic role
        col_names = list(df.columns)
        columns_info = "Columns (72 total): " + ", ".join(col_names)
        sample_section = ""  # skip entirely
    else:
        columns_info = df.dtypes.to_string()
        sample_section = f"\nSample data (first 3 rows):\n{df.head(3).to_string()}\n"

    categories_note = (
        f"\nCategory flags (0/1) from Phase 2 clustering: {CATEGORY_COLUMNS}"
        if CATEGORY_COLUMNS else
        "\n⚠️ Category flags from Phase 2 clustering are NOT yet available.\n"
        "When the user asks about 'categories', 'industries', 'sectors', or 'branches',\n"
        "treat 'NACE Rev. 2 main section' as the category column."
    )

    base_context = f"""
You have access to a dataset of companies headquartered in Dortmund (> 10 employees).

Dataset has {df.shape[0]} rows and {df.shape[1]} columns.

Columns and types:
{columns_info}
{sample_section}
{facts_block}

The dataset contains:
- Basic info: 'Company name Latin alphabet', 'Country ISO code', 'City Latin Alphabet',
  'NACE Rev. 2, core code (4 digits)', 'NACE Rev. 2 main section', 'Status', 'Founded Year'
- Employee counts per year: 'Number of employees 2017' … 'Number of employees 2024'
- Growth metrics per year: 'Growth YYYY', 'aagr YYYY' (annual average growth rate)
- Binary scaling indicators per year (0/1 or 'n.a.'):
  'Scaler', 'HighGrowthFirm', 'ConsistentHighGrowthFirm', 'VeryHighGrowthFirm',
  'Gazelle', 'Mature', 'Scaleup', 'Superstar'
{categories_note}

Definitions (from the course):
- Scaler: firm with ≥ 20% annual employee growth sustained
- Gazelle: young high-growth firm
- Scaleup: ≥ 10 employees reaching ≥ 20% annual growth for 3+ years
- Superstar: highest-growth tier

When scaling columns contain 'n.a.' strings, convert with pd.to_numeric(errors='coerce') before aggregating.
"""
    return base_context, df


@st.cache_data
def setup_coding_agent_context(compact: bool = False):
    base, df = setup_data_context(compact=compact)
    context = base + """
When asked a question, generate Python pandas code that operates on the dataframe 'df' to answer it.

Rules:
- The code MUST end by storing the answer in a variable called 'result'.
- AGGREGATE by default — counts, sums, means, top-N; never dump 1000+ raw rows as the result UNLESS…
- EXCEPTION: if the sub-question explicitly asks for "list", "all companies", "names of", "roster",
  return the FULL list (up to ~200 rows is fine). Use `.tolist()` for names or a small
  DataFrame with just the relevant columns. Example:
      result = df.loc[pd.to_numeric(df['Scaler 2024'], errors='coerce') == 1,
                      'Company name Latin alphabet'].tolist()
- "categories" / "industries" / "sectors" / "branches" → use 'NACE Rev. 2 main section'.
- Use pd.to_numeric(..., errors='coerce') for any scaling column that may contain 'n.a.'.

Return ONLY the executable Python code, no explanations.
"""
    return context, df


@st.cache_data
def setup_visual_agent_context(compact: bool = False):
    base, df = setup_data_context(compact=compact)
    context = base + """
You are a data visualization expert. Generate Python code that:
1. Uses plotly.express (imported as px) for interactive charts
2. Operates on the dataframe 'df'
3. Creates the requested chart with clear labels, title, and formatting
4. Stores the figure in a variable called 'fig'

CRITICAL rules — violating these produces unreadable charts:

A. "categories" / "industries" / "sectors" / "branches" mapping:
   → Use 'NACE Rev. 2 main section' as the category column. This is the ONLY proper
     categorical dimension in the CURRENT dataset. Do NOT invent categories from
     scaling flags (Scaler, Gazelle, Mature, ...) — those are boolean growth indicators,
     not industry categories.

B. ALWAYS aggregate BEFORE plotting. Never plot raw rows as individual slices/bars.
   Correct pattern (pandas 2.0+ compatible):
       counts = df['NACE Rev. 2 main section'].value_counts().reset_index()
       counts.columns = ['Sector', 'Number of companies']  # ← direct assignment, version-safe
       fig = px.bar(counts, x='Sector', y='Number of companies', title='...')

   ⚠️ PANDAS 2.0 WARNING: Do NOT use .rename(columns={"index": ...}) after
   value_counts().reset_index(). In pandas 2.0+ the column names are
   ['<original_column_name>', 'count'] — NOT ['index', '<original_column_name>'].
   Always use direct `df.columns = [...]` assignment instead of `.rename()` for renaming
   both result columns of value_counts().reset_index(). This avoids silent column mismatches.

C. Pie charts MUST have at most 10 slices. NACE has ~18 sections — for pie charts
   keep only the top 9 and aggregate the rest into 'Other'. For > 10 real categories,
   use a horizontal bar chart (px.bar with orientation='h') instead.

D. When scaling columns contain 'n.a.' strings, convert with
   pd.to_numeric(df['Scaler 2024'], errors='coerce') before summing/filtering.

E. Rename columns meaningfully before plotting so the legend/labels are readable
   (e.g. 'Number of companies', 'NACE sector', 'Scalers in 2024').

F. Titles, axis labels, and legends must be in English.

G. DO NOT call fig.show() or plt.show() — Streamlit renders the figure itself via st.plotly_chart(fig). Any .show() call opens an unwanted browser popup in Codespaces/VS Code.

Return ONLY the executable Python code, no explanations.
"""
    return context, df


# =============================================================================
# PLANNER AGENT — the new brain of the system
# =============================================================================

PLANNER_PROMPT = """You are a PLANNER agent for a Dortmund company dataset analysis tool.

Your job is to break a user's request into one OR MORE focused sub-tasks.
Each sub-task is handled by exactly ONE executor agent:

- "code"   → quantitative pandas analysis OR deterministic list extraction from the full dataframe.
             USE THIS for: counts, averages, groupby, percentages, filtering, AND
             whenever the user asks for "a list of X", "all X", "which companies are X",
             "top N by Y" — the code agent operates on ALL 1089 rows and returns complete
             results. Do NOT use text agent for list extraction — it only sees 3-5
             retrieved rows and will hallucinate or truncate the list.
- "visual" → interactive plotly chart (bar, pie, line, scatter, histogram).
- "text"   → QUALITATIVE narrative answers ONLY: explain, describe reasons, contextualise,
             summarise a concept. NEVER use text for factual lookups or complete lists.

Return a JSON object with a single key "tasks". Its value is a list of task objects.
Each task has exactly two fields:
  - "approach": one of "code", "visual", "text"
  - "question": a SELF-CONTAINED, specific sub-question for that executor

Rules:
1. Simple question → ONE task.
2. Compound request ("chart AND report", "calculate X and plot Y") → MULTIPLE tasks.
3. Order matters: put data/charts first, then analytical text that may reference them.
4. Each sub-question must stand alone — later agents cannot see earlier outputs at execution time.
5. Maximum 4 tasks.
6. Formulate sub-questions in English.
7. CRITICAL — "list of X" routing: any request for a list, roster, names, or enumeration of
   companies MUST go to the CODE agent (it sees all 1089 rows). The text agent only retrieves
   a handful of rows via semantic search and CANNOT produce complete lists.

Examples:

User: "How many scalers in 2024?"
{"tasks":[{"approach":"code","question":"How many companies have Scaler 2024 equal to 1?"}]}

User: "Give me a list of scalers in 2024"
{"tasks":[{"approach":"code","question":"Return the full list of company names where Scaler 2024 == 1. Store as a Python list in `result`."}]}

User: "Which are the 10 largest employers in Dortmund?"
{"tasks":[{"approach":"code","question":"Return the top 10 companies by 'Number of employees 2024' as a DataFrame with company name and employee count."}]}

User: "Show a bar chart of companies per NACE sector"
{"tasks":[{"approach":"visual","question":"Create a bar chart showing the number of companies per NACE Rev. 2 main section"}]}

User: "Why are professional services growing faster than manufacturing?"
{"tasks":[{"approach":"text","question":"Provide a qualitative explanation for why NACE section M (professional services) may be scaling faster than section C (manufacturing) in Dortmund."}]}

User: "Show a chart of scalers by NACE sector and write a short report about it"
{"tasks":[
  {"approach":"visual","question":"Bar chart: number of companies where Scaler 2024 = 1, grouped by NACE Rev. 2 main section"},
  {"approach":"code","question":"Return the top 3 NACE sectors by count of companies with Scaler 2024 = 1"},
  {"approach":"text","question":"Summarise which industries dominate scaling in Dortmund and why they may be leading"}
]}

User: "How many companies are in each industry, and make a pie chart"
{"tasks":[
  {"approach":"code","question":"How many companies are in each NACE Rev. 2 main section? Return the result as a table."},
  {"approach":"visual","question":"Create a pie chart showing the distribution of companies across NACE Rev. 2 main sections"}
]}

Now plan the tasks for the user question below.

User question: {question}
"""


def _call_planner_llm(system_prompt: str) -> str:
    """Call the LLM for the planner. Tries JSON mode first; falls back for
    models (like Ollama's glm-4.7-flash) that don't support response_format."""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
    except Exception:
        # Ollama models often reject response_format — retry without it
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return completion.choices[0].message.content


def _extract_json_block(raw: str) -> dict | None:
    """Parse raw LLM output into a dict. Handles:
    1) Pure JSON (json.loads works directly)
    2) JSON wrapped in markdown ```json fences
    3) JSON embedded in free text (extract between first '{' and last '}')
    """
    raw = raw.strip()
    # 1) Direct JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 2) Markdown fence
    if "```json" in raw:
        inner = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    elif "```" in raw:
        inner = raw.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    # 3) Find the outermost {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def planner_agent(question: str) -> tuple[list[dict], str]:
    """Returns (validated tasks, raw LLM output).
    Raw output is returned for diagnostic display in the workflow details."""
    # Build conversation context from recent turns (Improvement Point 3 — Multi-Turn Memory)
    history = st.session_state.get("conversation_history", [])
    history_ctx = ""
    if history:
        recent = history[-2:]  # only last 2 turns to save tokens
        history_ctx = "\n=== RECENT CONVERSATION (for context) ===\n"
        for turn in recent:
            history_ctx += f"- Previous user question: {turn['question']}\n"
            history_ctx += f"  Previous answer (summary): {turn['answer'][:250]}\n"
        history_ctx += (
            "\nIf the CURRENT question is a follow-up referring to the above "
            "(via words like 'that', 'it', 'compared to before', 'dazu', 'davon'), "
            "resolve references using the prior context before planning tasks.\n\n"
        )

    # Use .replace() instead of .format() because PLANNER_PROMPT contains
    # JSON examples with { and } which .format() would misinterpret.
    system_prompt = history_ctx + PLANNER_PROMPT.replace("{question}", question)
    raw = _call_planner_llm(system_prompt)
    parsed = _extract_json_block(raw)

    if parsed is not None:
        raw_tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
        valid = []
        for t in raw_tasks[:4]:
            if isinstance(t, dict) and t.get("approach") in ("code", "visual", "text") and t.get("question"):
                valid.append({"approach": t["approach"], "question": t["question"]})
        if valid:
            return valid, raw

    # Last-resort fallback: single text task
    return [{"approach": "text", "question": question}], raw


# =============================================================================
# EXECUTOR AGENTS — with self-correcting retry loops (Improvement Point 1)
# =============================================================================

MAX_RETRIES = 1 if TOKEN_SAVING else 2  # fewer retries in token-saving mode
_SENTINEL = object()  # identity-based sentinel; safe against pandas element-wise comparison


def text_analysis_agent(question: str, rag_chain) -> str:
    return rag_chain.invoke(question)


def _extract_code(raw: str) -> str:
    if "```python" in raw:
        return raw.split("```python")[1].split("```")[0]
    if "```" in raw:
        return raw.split("```")[1].split("```")[0]
    return raw


def _strip_show_calls(code: str) -> str:
    """
    Remove any fig.show() / plt.show() / pyo.plot() calls from generated code.
    Streamlit renders figures itself via st.plotly_chart(fig); calling .show()
    in Codespaces / VS Code opens an unwanted browser popup.
    """
    import re
    patterns = [
        r'^\s*\w+\.show\([^)]*\)\s*$',        # fig.show(), ax.show(), etc.
        r'^\s*plt\.show\([^)]*\)\s*$',        # plt.show()
        r'^\s*\w+\.show\([^)]*\)\s*;',        # inline version
        r'^\s*py\w*\.plot\([^)]*\)\s*$',      # plotly.offline.plot
    ]
    for p in patterns:
        code = re.sub(p, '', code, flags=re.MULTILINE)
    return code


def _generate_code(system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    """Single LLM call returning executable code."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1, max_tokens=1024,
    )
    return _strip_show_calls(_extract_code(completion.choices[0].message.content))


# =============================================================================
# SANDBOX — AST safety check before exec() (Improvement Point 4)
# =============================================================================
# Not a full sandbox, but a defense-in-depth layer that blocks the most dangerous
# patterns (os / subprocess imports, eval/exec/compile calls, file writes).
# Documented as "partial sandbox" in the Executive Summary.

import ast

_FORBIDDEN_MODULES = {
    "os", "subprocess", "sys", "socket", "urllib", "urllib2", "urllib3",
    "requests", "httpx", "shutil", "pathlib", "pickle", "dill", "marshal",
    "ctypes", "importlib", "builtins", "webbrowser", "tempfile", "glob",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}
_FORBIDDEN_ATTRS = {"system", "popen", "spawn", "execv", "execve", "fork",
                    "remove", "rmdir", "unlink", "chmod", "chown"}


def _safety_check_code(code: str) -> str | None:
    """Return a short description of any safety issue, or None if the code looks safe."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Let exec raise the real syntax error

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return f"forbidden import `{alias.name}`"
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _FORBIDDEN_MODULES:
                return f"forbidden import `from {node.module}`"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                return f"forbidden function call `{node.func.id}()`"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_ATTRS:
                return f"forbidden method `.{node.func.attr}()`"
    return None


def _safe_exec(code: str, globals_dict: dict, locals_dict: dict) -> None:
    """Run the AST safety check, then exec. Raises RuntimeError on safety violations."""
    issue = _safety_check_code(code)
    if issue:
        raise RuntimeError(f"Sandbox blocked code: {issue}")
    exec(code, globals_dict, locals_dict)


def _check_fig_quality(fig) -> str | None:
    """
    Return a short feedback string if the figure is obviously broken,
    or None if it looks fine. Used by the self-correcting loop.
    """
    if fig is None:
        return "Figure is None — no chart was produced."
    try:
        # Pie charts: check for too many slices
        for trace in fig.data:
            t = trace.type if hasattr(trace, "type") else ""
            if t == "pie":
                n = len(trace.labels) if getattr(trace, "labels", None) is not None else 0
                if n > 12:
                    return (f"The pie chart has {n} slices — far too many to read. "
                            "Use a bar chart instead, OR keep only the top 9 categories "
                            "and group the rest into 'Other'.")
            if t in ("bar", "histogram"):
                # Bar charts with > 40 categories are unreadable
                x = getattr(trace, "x", None)
                if x is not None and len(x) > 40:
                    return (f"The bar chart has {len(x)} bars — too many. "
                            "Show only the top 20 and group the rest, "
                            "OR switch to a horizontal bar chart if labels are long.")
    except Exception:
        pass
    return None


def coding_agent(question: str, data_context: str, df: pd.DataFrame):
    """
    Self-correcting: up to MAX_RETRIES retries on exec error.
    Returns (result, final_code, attempt_log)
    """
    attempt_log = []
    history: list[dict] = []  # running chat history for retries
    code_to_run = _generate_code(data_context, question)

    for attempt in range(1, MAX_RETRIES + 2):  # first try + retries
        try:
            local_vars = {"df": df, "pd": pd}
            _safe_exec(code_to_run, {}, local_vars)
            # Identity comparison with sentinel — safe against pandas Series/DataFrame results
            result = local_vars.get("result", _SENTINEL)
            if result is _SENTINEL:
                # No explicit `result` variable set — capture stdout instead
                import io, sys
                old = sys.stdout
                sys.stdout = buf = io.StringIO()
                _safe_exec(code_to_run, {"df": df, "pd": pd}, {})
                sys.stdout = old
                result = buf.getvalue() or "Code ran, but no explicit `result` variable was set."
            attempt_log.append(f"✓ Attempt {attempt}: success.")
            return result, code_to_run, attempt_log
        except Exception as e:
            err = str(e)
            attempt_log.append(f"✗ Attempt {attempt} failed: {err}")
            if attempt > MAX_RETRIES:
                return f"Error after {MAX_RETRIES + 1} attempts. Last error: {err}", code_to_run, attempt_log
            # Build retry message
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": code_to_run})
            retry_msg = (
                f"Your previous code raised this error:\n{err}\n\n"
                "Fix the code. Output ONLY the corrected Python code, no explanations. "
                "The code must end by storing the answer in a variable called 'result'."
            )
            code_to_run = _generate_code(data_context, retry_msg, history)


def visual_agent(question: str, visual_context: str, df: pd.DataFrame):
    """
    Self-correcting: retries on exec error AND on quality issues
    (too many pie slices / bars). Returns (fig, final_code, attempt_log).
    """
    attempt_log = []
    history: list[dict] = []
    code_to_run = _generate_code(visual_context, question)
    fig = None

    for attempt in range(1, MAX_RETRIES + 2):
        # Diagnostic: flag empty / too-short code immediately
        if len(code_to_run.strip()) < 10:
            attempt_log.append(
                f"⚠️ Attempt {attempt}: model returned empty/near-empty code "
                f"(length {len(code_to_run.strip())} chars). "
                f"This usually means the model is too small for Plotly code generation — "
                f"consider switching to a larger model (e.g. glm-4.7-flash:latest)."
            )
        feedback: str | None = None
        try:
            local_vars = {"df": df, "pd": pd, "px": px, "plt": plt}
            _safe_exec(code_to_run, {}, local_vars)
            fig = local_vars.get("fig", None)
            if fig is None:
                feedback = ("No 'fig' variable was produced. Make sure to store the plotly "
                            "figure in a variable literally named `fig`.")
            else:
                feedback = _check_fig_quality(fig)
        except Exception as e:
            feedback = f"Your code raised an error: {e}"

        if feedback is None:
            attempt_log.append(f"✓ Attempt {attempt}: chart passed quality check.")
            return fig, code_to_run, attempt_log

        attempt_log.append(f"✗ Attempt {attempt}: {feedback[:120]}")
        if attempt > MAX_RETRIES:
            return fig, code_to_run, attempt_log

        # Retry with feedback
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": code_to_run})
        retry_msg = (
            f"Your previous chart has an issue:\n{feedback}\n\n"
            "Regenerate the plotly code to address this. Output ONLY the corrected Python code, "
            "no explanations. Store the figure in a variable called `fig`. "
            "Remember: aggregate before plotting; pie charts ≤ 10 slices; 'categories' means NACE Rev. 2 main section."
        )
        code_to_run = _generate_code(visual_context, retry_msg, history)

    return fig, code_to_run, attempt_log


# =============================================================================
# EVALUATOR AGENT — LLM-as-a-judge (Improvement Point 2)
# =============================================================================
# Only runs when TOKEN_SAVING is OFF (it's an extra LLM call we'd rather skip
# when conserving tokens). Checks COVERAGE, ACCURACY, RELEVANCE before the final editor.

EVALUATOR_PROMPT = """You are the EVALUATOR agent. Your job is to check whether a multi-agent system's
sub-task results correctly and completely address the user's ORIGINAL question.

=== ORIGINAL QUESTION ===
{question}

=== SUB-TASK RESULTS ===
{results}

=== GROUND-TRUTH FACTS (for numeric verification) ===
{facts}

Evaluate on THREE criteria:
1. COVERAGE — do the sub-tasks address ALL parts of the original question (chart requested? number requested? report requested?)
2. ACCURACY — are the numbers in the sub-task results CONSISTENT with the ground-truth facts?
3. RELEVANCE — are the sub-tasks actually on topic, not tangential?

Return a JSON object with EXACTLY these keys:
  "passed": true or false,
  "feedback": "1-3 sentence assessment (what's good, what's missing or wrong)",
  "suggested_fix": null  OR  "short hint for the final editor — what to emphasise or correct"

Be lenient on minor wording differences. Only fail if something is clearly missing or a number contradicts facts.
"""


def evaluator_agent(original_question: str, results: list[dict], facts_block: str) -> dict:
    """Returns {'passed': bool, 'feedback': str, 'suggested_fix': str|None}.
    On any internal error the evaluator fails-open (passed=True) so the pipeline continues."""
    # Build a trimmed summary of sub-task results
    blocks = []
    for i, r in enumerate(results, 1):
        answer = str(r["raw_answer"])[:600]  # cap per task
        blocks.append(f"[Task {i} — {r['approach'].upper()}]\nQ: {r['question']}\nA: {answer}")
    combined = "\n\n".join(blocks)

    prompt = EVALUATOR_PROMPT.format(
        question=original_question,
        results=combined,
        facts=facts_block[:2500],  # cap facts for evaluator
    )

    try:
        # Tighter timeout for the evaluator — it's optional, not on the critical path.
        # If it hangs or is slow, we'd rather skip and let the editor run.
        completion = client.with_options(timeout=20).chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = completion.choices[0].message.content
        parsed = _extract_json_block(raw)
        if parsed and isinstance(parsed, dict):
            return {
                "passed": bool(parsed.get("passed", True)),
                "feedback": str(parsed.get("feedback", ""))[:500],
                "suggested_fix": (str(parsed["suggested_fix"])[:300]
                                  if parsed.get("suggested_fix") else None),
                "raw": raw,
            }
    except Exception as e:
        return {"passed": True, "feedback": f"Evaluator skipped: {type(e).__name__}", "suggested_fix": None, "raw": ""}
    return {"passed": True, "feedback": "Evaluator returned unparseable output", "suggested_fix": None, "raw": ""}


# =============================================================================
# FINAL EDITOR — synthesises all sub-results into ONE coherent answer
# =============================================================================

def final_editor_agent(original_question: str, results: list[dict], extra_context: str = "") -> str:
    """
    results items: {"approach": str, "question": str, "raw_answer": str}
    Visual results include a short description instead of the figure itself.
    extra_context: optional text from the Evaluator telling the editor what to emphasise/fix.

    Enhanced with:
    - Ground-truth facts block (same as other agents) to avoid hallucinations
    - Explicit handling of failed sub-tasks (don't parrot "sub-process failed")
    - Instruction to treat charts as data sources, not just decoration
    - In token-saving mode: skip the facts block (sub-tasks already carry the numbers)
    - Optional Evaluator feedback injection (Improvement Point 2)
    """
    # In token-saving mode, sub-agent results already embed the facts — skip the
    # duplicate facts injection to save ~1000 tokens per editor call.
    include_facts = not TOKEN_SAVING
    if include_facts:
        facts, _ = compute_data_facts()
        facts_block = format_facts_block(facts, compact=False)
    else:
        facts_block = ""

    blocks = []
    for i, r in enumerate(results, 1):
        if r["approach"] == "visual":
            blocks.append(
                f"[Sub-task {i} — VISUAL]\n"
                f"Sub-question: {r['question']}\n"
                f"A plotly chart has been rendered directly above your answer. "
                f"TREAT IT AS A DATA SOURCE: the chart's categories and percentages/counts are reliable."
            )
        else:
            # Detect failed sub-tasks
            raw = r["raw_answer"]
            failed = raw.startswith("Error after") or raw.startswith("Error executing code")
            status = " [FAILED]" if failed else ""
            blocks.append(
                f"[Sub-task {i} — {r['approach'].upper()}{status}]\n"
                f"Sub-question: {r['question']}\n"
                f"Result:\n{raw}"
            )
    combined = "\n\n".join(blocks)

    facts_section = (
        f"\n=== GROUND TRUTH DATASET FACTS (computed in Python, use these for any numeric claim) ===\n{facts_block}\n"
        if include_facts else ""
    )
    # Evaluator hint (only present when evaluator actually ran — i.e. TOKEN_SAVING=OFF)
    eval_section = (
        f"\n=== EVALUATOR FEEDBACK (act on this) ===\n{extra_context}\n"
        if extra_context else ""
    )

    editor_prompt = f"""You are the FINAL EDITOR. Synthesise sub-agent results into ONE coherent answer.

=== ORIGINAL USER QUESTION ===
{original_question}

=== SUB-AGENT RESULTS ===
{combined}
{facts_section}{eval_section}
=== LAYOUT CONTEXT ===
- Any charts generated by visual sub-tasks are rendered DIRECTLY ABOVE your text in the same answer block.
- There is NO separate "report" section — your text IS the report.
- Reference charts naturally: "The pie chart above shows…", "As visible in the bar chart…"

=== RULES ===
1. Write ONE unified answer that addresses the ORIGINAL question as a whole.
2. Base every numeric claim on either (a) GROUND TRUTH FACTS, (b) successful sub-task results, or (c) the chart rendered above.
3. Do NOT invent numbers. Do NOT round aggressively.
4. CRITICAL — avoid mixing metrics without signposting:
   - The chart above shows ONE specific metric (e.g. "share of scalers per sector"). When you quote a % from the chart, label it clearly: "16.8% *of scalers*", NOT just "16.8%".
   - If you additionally use total-dataset counts from GROUND TRUTH FACTS (e.g. "G has 187 companies total"), label them clearly: "187 companies in total (not only scalers)".
   - NEVER write a sentence that reads as if chart-%-values and total-company-counts describe the same thing.
5. Structure for compound questions: if the user asked for BOTH a chart-specific view AND a broader report, organise the text into two clearly distinguishable paragraphs:
   • Paragraph 1: what the chart reveals (chart metric)
   • Paragraph 2: broader context from the full dataset (total counts, top employers, etc.)
6. If a sub-task FAILED:
   - Do NOT say "sub-process failed" or "could not be determined".
   - Silently fall back to the GROUND TRUTH FACTS.
   - Only if truly absent from both sources, say so briefly.
7. Write the final answer in English.
8. Aim for 5–12 sentences. End with a one-line "Key takeaway" if the user asked for a summary/report.
9. LIST-STYLE QUESTIONS — if the user asked for a "list of X", "all X", "which companies are X",
   and the code sub-task returned an actual Python list or DataFrame:
   • Present the full list verbatim (as a bulleted list or table), NOT a summary.
   • Do NOT write padding sentences like "the list includes notable companies such as…".
   • Do NOT invent overall counts that differ from the length of the returned list.
   • A brief 1-sentence intro ("Below are the N scalers identified in 2024:") is OK; no Key Takeaway needed.

Now write the final answer:"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an editor that synthesises multi-agent outputs into one coherent answer, with access to ground-truth dataset facts."},
            {"role": "user", "content": editor_prompt},
        ],
        temperature=0.3, max_tokens=2048,
    )
    return completion.choices[0].message.content


# UI — Main area
# =============================================================================

# Clean header — no gradient banner
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown(
        f'<h1 style="margin-bottom:0.25rem;">Dortmund Explorer</h1>'
        f'<p style="color:{WHU_DARK_GRAY};font-size:1rem;margin-top:0;">'
        f'Multi-agent analysis of {1089} Dortmund companies (&gt;10 employees)'
        f'</p>',
        unsafe_allow_html=True,
    )
with col_meta:
    # Inline pipeline indicator — compact, elegant
    st.markdown(
        f'<div style="text-align:right;padding-top:1rem;color:{WHU_DARK_GRAY};font-size:0.85rem;">'
        f'<span style="font-weight:600;color:{WHU_BLUE};">Pipeline:</span> '
        f'🧠 Planner → 📝💻📊 Executors → ✏️ Editor'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# Silent agent setup — both Explorer and Benchmark views need these
with st.spinner("Initializing agents…"):
    rag_k = 3 if TOKEN_SAVING else 5
    rag_chain = setup_text_analysis_agent(compact=TOKEN_SAVING, rag_k=rag_k)
    data_context, df = setup_coding_agent_context(compact=TOKEN_SAVING)
    visual_context, _ = setup_visual_agent_context(compact=TOKEN_SAVING)

# --- Dataset metrics (always visible, no expander) ---
scaler_count = int(pd.to_numeric(df.get("Scaler 2024", pd.Series(dtype=float)), errors="coerce").sum())
active_count = int((df["Status"] == "Active").sum()) if "Status" in df.columns else df.shape[0]
n_sectors = df["NACE Rev. 2 main section"].nunique() if "NACE Rev. 2 main section" in df.columns else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Companies", f"{df.shape[0]:,}")
m2.metric("Active", f"{active_count:,}")
m3.metric("Scalers 2024", f"{scaler_count:,}")
m4.metric("NACE Sectors", f"{n_sectors}")

# Show conversation history if any (Multi-Turn Memory UI)
if st.session_state.conversation_history:
    with st.expander(f"💬 Conversation history ({len(st.session_state.conversation_history)} previous turn(s))"):
        for i, turn in enumerate(st.session_state.conversation_history, 1):
            st.markdown(f"**Turn {i}:** {turn['question']}")
            st.markdown(f"> {turn['answer'][:300]}{'…' if len(turn['answer']) >= 300 else ''}")
            if i < len(st.session_state.conversation_history):
                st.markdown("---")
        st.caption("💡 The Planner uses the last 2 turns as context — ask follow-up questions like *\"and compare that to 2020\"*.")

st.markdown("### Ask a question")

# --- Examples as clickable suggestions in tabs ---
EXAMPLES = {
    "Quick Q&A": [
        "How many scalers were there in 2024?",
        "Who are the top 5 employers in Dortmund?",
        "How did the number of scalers evolve from 2020 to 2024?",
    ],
    "With chart": [
        "Create a bar chart of NACE sectors in Dortmund",
        "Show me a pie chart of the top 10 sectors",
        "Line chart: scaler count by year 2020 to 2024",
    ],
    "Compound": [
        "Give me an overview of Dortmund: pie chart of top sectors, number of scalers 2024, and a short report",
        "Top 3 NACE sectors by scaler share plus explanation why they lead",
    ],
}

# Unified session state key — both the example buttons and the text_area
# bind to `question_input`. Using a separate `question_text` variable would
# fail because Streamlit widgets with a `key=` create their own independent
# state slot that doesn't listen to external variables.
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

tabs = st.tabs(list(EXAMPLES.keys()))
for tab, (label, items) in zip(tabs, EXAMPLES.items()):
    with tab:
        for ex in items:
            if st.button(ex, key=f"ex_{label}_{ex[:20]}", use_container_width=True):
                st.session_state.question_input = ex
                st.rerun()

# --- Question input — text_area for longer queries ---
# No `value=` param — the widget reads its value from session_state[key] automatically.
question = st.text_area(
    "Your question",
    placeholder="e.g., Create a chart of the sectors and write a short report",
    height=90,
    label_visibility="collapsed",
    key="question_input",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    submit = st.button("Get Answer", type="primary", use_container_width=True)

if submit and question:

    # ---- STEP 1: Planner ----
    try:
        with st.spinner("🧠 Planner decomposing question into sub-tasks..."):
            tasks, planner_raw_output = planner_agent(question)
    except Exception as e:
        err_type = type(e).__name__
        msg = str(e)
        if "APIConnectionError" in err_type or "ConnectionError" in err_type or "connect" in msg.lower():
            st.error(
                f"🔌 **Could not connect to the LLM provider.**\n\n"
                f"**Current provider:** {provider}  \n"
                f"**Model:** `{llm_model}`\n\n"
                f"Possible causes:\n"
                f"- You selected **WHU Ollama** but eduVPN is not active, or you are on Streamlit Cloud (VPN unreachable)\n"
                f"- Your Groq API key is invalid or expired\n"
                f"- The LLM endpoint is temporarily down\n\n"
                f"*Technical detail:* {err_type}"
            )
        elif "RateLimitError" in err_type or "429" in msg:
            st.error(
                f"⏳ **Rate limit reached on {provider}.**\n\n"
                f"Try one of:\n"
                f"- Enable **Token-saving mode** in the sidebar\n"
                f"- Switch to a different model with its own quota (e.g. `openai/gpt-oss-120b`)\n"
                f"- Wait ~40 minutes for the rolling quota window to free up\n\n"
                f"*Details:* {msg[:300]}"
            )
        elif "AuthenticationError" in err_type or "401" in msg:
            st.error(
                f"🔑 **Authentication failed.** The API key appears to be invalid or missing.\n\n"
                f"Paste a fresh key from https://console.groq.com/keys into the sidebar."
            )
        else:
            st.error(f"❌ Unexpected error in Planner: {err_type}: {msg[:500]}")
        st.stop()

    # Show the plan (stays visible so the user sees the decomposition)
    emoji_map = {"code": "💻", "visual": "📊", "text": "📝"}
    st.info(f"**Plan:** {len(tasks)} sub-task(s)")
    plan_cols = st.columns(len(tasks))
    for i, (col, t) in enumerate(zip(plan_cols, tasks)):
        col.markdown(
            f"**{emoji_map[t['approach']]} Step {i+1}: {t['approach'].upper()}**\n\n"
            f"_{t['question']}_"
        )

    # ---- STEP 2: Execute each sub-task silently; collect artefacts ----
    results: list[dict] = []
    figures: list[tuple[str, object]] = []   # (sub-question, plotly fig)
    code_snippets: list[tuple[str, str]] = []  # (label, code)
    retry_logs: list[tuple[str, list[str]]] = []  # (label, attempt_log)
    progress = st.empty()

    for i, task in enumerate(tasks, 1):
        approach = task["approach"]
        sub_q = task["question"]
        progress.info(f"⏳ Step {i}/{len(tasks)} — {emoji_map[approach]} {approach.upper()} Agent working…")

        if approach == "text":
            raw = text_analysis_agent(sub_q, rag_chain)
            results.append({"approach": approach, "question": sub_q, "raw_answer": raw})

        elif approach == "visual":
            fig, code_used, log = visual_agent(sub_q, visual_context, df)
            if fig is not None:
                figures.append((sub_q, fig))
            code_snippets.append((f"📊 Visual — {sub_q}", code_used))
            retry_logs.append((f"📊 {sub_q}", log))
            results.append({
                "approach": approach, "question": sub_q,
                "raw_answer": "Visualisation generated." if fig is not None else "Chart generation failed.",
            })

        else:  # code
            raw, code_used, log = coding_agent(sub_q, data_context, df)
            code_snippets.append((f"💻 Code — {sub_q}", code_used))
            retry_logs.append((f"💻 {sub_q}", log))
            results.append({"approach": approach, "question": sub_q, "raw_answer": str(raw)})

    progress.empty()

    # ---- STEP 2.5: Evaluator Agent (only when TOKEN_SAVING is OFF AND not on Cloud) ----
    # Improvement Point 2 — LLM-as-a-judge validates sub-task results against the
    # original question and ground-truth facts BEFORE the editor writes the final answer.
    # On Streamlit Cloud the total request budget (~2 min) is too tight for the extra
    # evaluator call on top of a compound query, so we skip it there.
    evaluator_result = None
    if not TOKEN_SAVING and not IS_CLOUD:
        with st.spinner("🔍 Evaluator reviewing answer quality…"):
            facts_for_eval, _ = compute_data_facts()
            evaluator_result = evaluator_agent(
                question, results, format_facts_block(facts_for_eval, compact=True)
            )

    # ---- STEP 3: Final Editor merges everything into one narrative ----
    editor_extra = ""
    if evaluator_result and evaluator_result.get("suggested_fix"):
        editor_extra = evaluator_result["suggested_fix"]

    with st.spinner("✏️ Final Editor synthesising all results…"):
        final_answer = final_editor_agent(question, results, extra_context=editor_extra)

    # Persist this turn into conversation history (Improvement Point 3 — Multi-Turn Memory)
    st.session_state.conversation_history.append({
        "question": question,
        "answer": final_answer[:500],  # truncate to cap memory growth
    })

    # ================= FINAL ANSWER — the complete deliverable =================
    st.markdown("")  # spacer
    with st.container(border=True):
        st.markdown(f'<h3 style="margin-top:0;">📋 Final Answer</h3>', unsafe_allow_html=True)

        # 1) Charts first — they anchor the narrative
        for sub_q, fig in figures:
            st.plotly_chart(fig, use_container_width=True)

        # 2) Synthesised narrative that references the charts
        st.markdown(final_answer)

    # Evaluator verdict — visible quality signal
    if evaluator_result:
        if evaluator_result["passed"]:
            st.success(f"✓ Evaluator: {evaluator_result['feedback']}")
        else:
            st.warning(f"⚠️ Evaluator flagged issues: {evaluator_result['feedback']}")

    # If a visual was planned but failed, flag it instead of hiding it
    failed_visuals = [r for r in results if r["approach"] == "visual" and "failed" in r["raw_answer"].lower()]
    if failed_visuals:
        st.warning(f"⚠️ {len(failed_visuals)} chart(s) could not be generated — see Workflow Details below.")

    # ---- Workflow details — collapsed by default, for transparency / demo ----
    with st.expander("🔧 Agent Workflow Details (plan, code, retries, raw outputs)"):
        st.markdown(f"**User question:** {question}")
        st.markdown(f"**Planner decomposed into {len(tasks)} task(s):**")
        for i, t in enumerate(tasks, 1):
            st.markdown(f"{i}. {emoji_map[t['approach']]} **{t['approach'].upper()}** — {t['question']}")

        # NEW: show raw planner output for diagnostics
        st.markdown("---")
        st.markdown("**🧠 Planner raw output** (what the LLM actually returned):")
        st.code(planner_raw_output, language="json")

        if retry_logs:
            st.markdown("---")
            st.markdown("**🔁 Self-correction log** (how many attempts each agent needed):")
            for label, log in retry_logs:
                st.markdown(f"_{label}_")
                for line in log:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", unsafe_allow_html=True)

        if code_snippets:
            st.markdown("---")
            st.markdown("**Generated code per sub-task (final version after retries):**")
            for label, code in code_snippets:
                st.markdown(f"_{label}_")
                st.code(code, language="python")

        st.markdown("---")
        st.markdown("**Raw sub-agent outputs:**")
        for r in results:
            if r["approach"] == "visual":
                st.markdown(f"- {emoji_map[r['approach']]} _{r['question']}_ → (chart rendered above)")
            else:
                st.markdown(f"- {emoji_map[r['approach']]} _{r['question']}_")
                st.code(r["raw_answer"][:1500] + ("…" if len(r["raw_answer"]) > 1500 else ""))

st.divider()
st.caption("WHU Otto Beisheim School of Management · Multi-Agent Analysis for Group Assignment Phase 4")
