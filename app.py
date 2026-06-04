import streamlit as st
import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Load API key securely from st.secrets (never from user input or .env) ───
# Locally:  .streamlit/secrets.toml  →  MISTRAL_API_KEY = "sk-..."
# Deployed: Streamlit Cloud dashboard → Secrets tab
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --card:      #181c26;
    --border:    #252a38;
    --accent:    #5b8fff;
    --accent2:   #a78bfa;
    --green:     #34d399;
    --amber:     #fbbf24;
    --red:       #f87171;
    --text:      #e8eaf0;
    --muted:     #7880a0;
    --radius:    14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

/* ── Brand header ── */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
}
.brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.brand-text { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; line-height: 1.2; }
.brand-sub  { font-size: 0.72rem; color: var(--muted); font-weight: 400; }

/* ── Nav pills ── */
.nav-section { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
               color: var(--muted); font-weight: 600; margin: 1.4rem 0 0.5rem; }
.nav-pill {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px; cursor: pointer;
    font-size: 0.88rem; font-weight: 500; color: var(--muted);
    transition: all .2s ease; margin-bottom: 4px;
    border: 1px solid transparent;
}
.nav-pill:hover  { background: var(--card); color: var(--text); border-color: var(--border); }
.nav-pill.active { background: linear-gradient(135deg,rgba(91,143,255,.18),rgba(167,139,250,.12));
                   color: var(--accent); border-color: rgba(91,143,255,.4); }
.nav-pill .icon  { font-size: 1.1rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg,rgba(91,143,255,.08) 0%,rgba(167,139,250,.06) 50%,rgba(52,211,153,.04) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(91,143,255,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow { font-size: 0.72rem; letter-spacing: .12em; text-transform: uppercase;
                color: var(--accent); font-weight: 600; margin-bottom: .5rem; }
.hero-title   { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
                line-height: 1.15; margin-bottom: .6rem; }
.hero-title span { background: linear-gradient(90deg, var(--accent), var(--accent2));
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-desc    { color: var(--muted); font-size: .9rem; max-width: 520px; line-height: 1.6; }

/* ── Upload zone ── */
.upload-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 2rem; }
.upload-card {
    background: var(--card); border: 1.5px dashed var(--border);
    border-radius: var(--radius); padding: 1.4rem 1.2rem;
    text-align: center; transition: border-color .2s;
}
.upload-card:hover { border-color: var(--accent); }
.upload-card .uc-icon { font-size: 1.8rem; margin-bottom: .5rem; }
.upload-card .uc-label { font-family:'Syne',sans-serif; font-size:.82rem; font-weight:700;
                         color:var(--text); margin-bottom:.25rem; }
.upload-card .uc-hint  { font-size:.74rem; color:var(--muted); }
.upload-card.ready { border-color: var(--green); border-style: solid;
                     background: rgba(52,211,153,.05); }

/* ── Status badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 99px; font-size: .74rem; font-weight: 600;
}
.badge-green  { background:rgba(52,211,153,.12);  color:var(--green);  border:1px solid rgba(52,211,153,.25); }
.badge-amber  { background:rgba(251,191,36,.12);  color:var(--amber);  border:1px solid rgba(251,191,36,.25); }
.badge-blue   { background:rgba(91,143,255,.12);  color:var(--accent); border:1px solid rgba(91,143,255,.25); }
.badge-purple { background:rgba(167,139,250,.12); color:var(--accent2);border:1px solid rgba(167,139,250,.25); }

/* ── Mode selector cards ── */
.mode-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: .9rem; margin-bottom: 1.5rem; }
.mode-card {
    background: var(--card); border: 1.5px solid var(--border);
    border-radius: var(--radius); padding: 1.1rem 1.2rem; cursor: pointer;
    transition: all .2s;
}
.mode-card:hover  { border-color: var(--accent); }
.mode-card.sel    { border-color: var(--accent); background:rgba(91,143,255,.08); }
.mode-card .mc-icon { font-size:1.4rem; margin-bottom:.4rem; }
.mode-card .mc-title{ font-family:'Syne',sans-serif; font-size:.88rem; font-weight:700; margin-bottom:.2rem; }
.mode-card .mc-desc { font-size:.74rem; color:var(--muted); line-height:1.5; }

/* ── Chat bubbles ── */
.chat-wrap { display:flex; flex-direction:column; gap:1rem; margin-bottom:1.5rem; }
.msg { display:flex; gap:10px; }
.msg.user  { flex-direction: row-reverse; }
.msg .avatar {
    width:32px; height:32px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:14px;
}
.msg.ai   .avatar { background:linear-gradient(135deg,var(--accent),var(--accent2)); }
.msg.user .avatar { background:linear-gradient(135deg,var(--green),#059669); }
.bubble {
    max-width: 80%; padding: 12px 16px; border-radius: 14px;
    font-size: .88rem; line-height: 1.65;
}
.msg.ai   .bubble { background:var(--card); border:1px solid var(--border); border-top-left-radius:4px; }
.msg.user .bubble { background:linear-gradient(135deg,rgba(91,143,255,.22),rgba(167,139,250,.18));
                    border:1px solid rgba(91,143,255,.3); border-top-right-radius:4px; }

/* ── Source chips ── */
.source-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.source-chip {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(255,255,255,.04); border:1px solid var(--border);
    border-radius:6px; padding:3px 9px; font-size:.7rem; color:var(--muted);
}

/* ── Section heading ── */
.sec-heading {
    font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
    margin-bottom:1rem; display:flex; align-items:center; gap:8px;
}
.sec-heading .line { flex:1; height:1px; background:var(--border); }

/* ── Divider ── */
hr.fancy { border:none; border-top:1px solid var(--border); margin: 1.5rem 0; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity .2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: .88 !important; }

.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(91,143,255,.12) !important;
}

.stFileUploader > div {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

.stAlert { border-radius: 10px !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history":    [],
        "display_history": [],
        "vectorstore":     None,
        "retriever":       None,
        "llm":             None,
        "mode":            "chat",
        "page":            "home",
        "uploaded_files":  {},
        "system_ready":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Prompts (exact from user code) ──────────────────────────────────────────
ats_prompt = ChatPromptTemplate.from_template("""
You are an ATS evaluator.

Analyze the resume and provide:

1. ATS Score
2. Strengths
3. Weaknesses
4. Missing Skills
5. Improvements

Resume:

{context}
""")

main_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI Career Coach, Senior Recruiter, ATS Specialist, and Technical Interview Mentor.

Your responsibility is to help users improve their resumes, understand job descriptions, identify skill gaps, prepare for interviews, and generate professional career-related documents.

You must answer only using the provided context retrieved from the knowledge base. The context may include:

* Candidate Resume
* Job Description (JD)
* Company Information
* Interview Notes
* Career Resources

Rules:

1. Use the retrieved context as the primary source of truth.
2. If the answer is not present in the context, clearly say:
   "I could not find sufficient information in the provided documents."
3. Do not hallucinate facts.
4. Be professional, concise, and actionable.
5. When comparing Resume and JD, explicitly mention:

   * Matching Skills
   * Missing Skills
   * Recommended Improvements
6. When generating interview questions:

   * Include HR Questions
   * Technical Questions
   * Scenario-Based Questions
7. When generating ATS analysis:

   * Provide ATS Score (estimated)
   * Strengths
   * Weaknesses
   * Action Items
8. Use bullet points whenever possible.
9. Format output clearly using markdown.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

ASSISTANT RESPONSE:
Think step-by-step using the provided context and generate a helpful response."""),
    ("user", "{question}")
])

interview_prompt = ChatPromptTemplate.from_template("""
You are a Senior Technical Interviewer and Hiring Manager.

Using the provided context, generate interview questions that are highly relevant to the candidate's profile.

CONTEXT:
{context}

Instructions:

1. Generate 10 HR Questions.
2. Generate 10 Technical Questions.
3. Generate 5 Scenario-Based Questions.
4. Generate 5 Project-Based Questions.
5. Questions should be tailored to the candidate's skills, projects, and experience.
6. Avoid generic questions.
7. Format the response in markdown.

Output Format:

# HR Questions
- Question 1
- Question 2

# Technical Questions
- Question 1
- Question 2

# Scenario-Based Questions
- Question 1
- Question 2

# Project-Based Questions
- Question 1
- Question 2
""")

roadmap_prompt = ChatPromptTemplate.from_template("""
You are an Expert Career Mentor and Industry Coach.

Analyze the candidate's profile and identify missing skills, technologies, and knowledge areas.

CONTEXT:
{context}

Instructions:

1. Identify important missing skills.
2. Explain why each skill matters.
3. Create a practical learning roadmap.
4. Divide roadmap into weekly milestones.
5. Prioritize skills based on industry demand.
6. Suggest projects for each phase.
7. Keep roadmap realistic and actionable.

Output Format:

# Missing Skills
- Skill 1
- Skill 2

# Why These Skills Matter
- Explanation

# Learning Roadmap

## Week 1
Topics:
Projects:

## Week 2
Topics:
Projects:

## Week 3
Topics:
Projects:

## Week 4
Topics:
Projects:

# Final Portfolio Projects
- Project 1
- Project 2
- Project 3
""")


# ─── Backend Helpers ──────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file, filename):
    path = f"/tmp/{filename}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def pdf_system(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def build_rag(file_paths: dict):
    """file_paths: {'resume': path, 'jd': path, 'company': path}"""
    all_docs = []
    label_map = {"resume": "resume.pdf", "jd": "jd.pdf", "company": "company.pdf"}
    for key, path in file_paths.items():
        chunks = pdf_system(path)
        for chunk in chunks:
            chunk.metadata["source"] = label_map.get(key, key)
        all_docs.extend(chunks)

    embedding_model = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    return vectorstore, retriever


def get_llm():
    return ChatMistralAI(api_key=MISTRAL_API_KEY, temperature=0)


def run_query(query: str):
    retriever = st.session_state.retriever
    llm       = st.session_state.llm
    docs      = retriever.invoke(query)
    context   = "\n\n".join([d.page_content for d in docs])
    sources   = list({d.metadata.get("source", "unknown") for d in docs})

    q_lower = query.lower()
    if "analyze" in q_lower:
        chain    = ats_prompt | llm
        response = chain.invoke({"context": context})
    elif "interview" in q_lower:
        chain    = interview_prompt | llm
        response = chain.invoke({"context": context})
    elif "roadmap" in q_lower:
        chain    = roadmap_prompt | llm
        response = chain.invoke({"context": context})
    else:
        final_prompt = main_prompt.invoke({
            "context":      context,
            "question":     query,
            "chat_history": st.session_state.chat_history,
        })
        response = llm.invoke(final_prompt)

    return response.content, sources


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-text">CareerAI</div>
            <div class="brand-sub">Powered by Mistral + RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System status ──
    if st.session_state.system_ready:
        st.markdown('<span class="badge badge-green">● System Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-amber">○ Awaiting Setup</span>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("home",      "🏠", "Home"),
        ("setup",     "⚙️", "Setup & Upload"),
        ("chat",      "💬", "Career Chat"),
    ]
    for pid, icon, label in pages:
        active = "active" if st.session_state.page == pid else ""
        if st.button(f"{icon}  {label}", key=f"nav_{pid}",
                     use_container_width=True):
            st.session_state.page = pid
            st.rerun()

    st.markdown('<hr class="fancy">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.72rem;color:var(--muted)">Chat turns: <b>{len(st.session_state.display_history)}</b></div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history    = []
        st.session_state.display_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI-Powered Career Intelligence</div>
        <div class="hero-title">Your Personal <span>Career Coach</span><br>& ATS Expert</div>
        <div class="hero-desc">
            Upload your resume, job description, and company info — then get instant ATS analysis,
            tailored interview prep, skill roadmaps, and career guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("🎯", "ATS Analysis",     "badge-green",  "Score your resume against any JD instantly"),
        ("🎤", "Interview Prep",   "badge-blue",   "HR, Technical, Scenario & Project questions"),
        ("🗺️", "Skill Roadmap",    "badge-purple", "Weekly learning plan with project milestones"),
        ("💬", "Career Chat",      "badge-amber",  "Ask anything about your career journey"),
    ]
    for col, (icon, title, badge_cls, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;">
                <div style="font-size:1.8rem;margin-bottom:.5rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.3rem">{title}</div>
                <div style="font-size:.74rem;color:var(--muted);line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-heading">
        How It Works
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Upload Documents",   "Add your Resume, Job Description, and Company Info PDFs in the Setup page."),
        ("02", "Initialize System",  "Click 'Build Knowledge Base' — the RAG pipeline indexes your documents."),
        ("03", "Choose Your Mode",   "Head to Career Chat and pick a mode: Chat, ATS Analyze, Interview, or Roadmap."),
        ("04", "Get Actionable Help","Ask questions or trigger specialized analysis — all grounded in your documents."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:1rem;
                    background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.3rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                        color:rgba(91,143,255,.25);line-height:1;flex-shrink:0;min-width:40px">{num}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.25rem">{title}</div>
                <div style="font-size:.82rem;color:var(--muted);line-height:1.55">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→  Get Started — Go to Setup", use_container_width=False):
        st.session_state.page = "setup"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "setup":
    st.markdown("""
    <div class="sec-heading">⚙️ Setup &amp; Document Upload <div class="line"></div></div>
    <p style="color:var(--muted);font-size:.88rem;margin-bottom:1.5rem">
        Upload all three PDFs to build your personal knowledge base. All analysis is grounded in these documents.
    </p>
    """, unsafe_allow_html=True)

    col_r, col_j, col_c = st.columns(3)

    with col_r:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;margin-bottom:.5rem">
            📄 Resume <span style="color:var(--red);font-size:.7rem">*required</span>
        </div>""", unsafe_allow_html=True)
        resume_file = st.file_uploader("Resume PDF", type=["pdf"], key="upload_resume", label_visibility="collapsed")
        if resume_file:
            st.markdown('<span class="badge badge-green">✓ Uploaded</span>', unsafe_allow_html=True)
            st.session_state.uploaded_files["resume"] = save_uploaded_file(resume_file, "resume.pdf")

    with col_j:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;margin-bottom:.5rem">
            📋 Job Description <span style="color:var(--red);font-size:.7rem">*required</span>
        </div>""", unsafe_allow_html=True)
        jd_file = st.file_uploader("JD PDF", type=["pdf"], key="upload_jd", label_visibility="collapsed")
        if jd_file:
            st.markdown('<span class="badge badge-green">✓ Uploaded</span>', unsafe_allow_html=True)
            st.session_state.uploaded_files["jd"] = save_uploaded_file(jd_file, "jd.pdf")

    with col_c:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;margin-bottom:.5rem">
            🏢 Company Info <span style="color:var(--muted);font-size:.7rem">optional</span>
        </div>""", unsafe_allow_html=True)
        company_file = st.file_uploader("Company PDF", type=["pdf"], key="upload_company", label_visibility="collapsed")
        if company_file:
            st.markdown('<span class="badge badge-green">✓ Uploaded</span>', unsafe_allow_html=True)
            st.session_state.uploaded_files["company"] = save_uploaded_file(company_file, "company.pdf")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Readiness check ──
    has_resume = "resume" in st.session_state.uploaded_files
    has_jd     = "jd"     in st.session_state.uploaded_files
    n_uploaded = len(st.session_state.uploaded_files)

    if has_resume and has_jd:
        st.markdown(f'<span class="badge badge-blue">📂 {n_uploaded}/3 documents ready</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("🚀 Build Knowledge Base", use_container_width=True):
                with st.spinner("Embedding documents and building vector store…"):
                    try:
                        vs, ret = build_rag(st.session_state.uploaded_files)
                        st.session_state.vectorstore  = vs
                        st.session_state.retriever    = ret
                        st.session_state.llm          = get_llm()
                        st.session_state.system_ready = True
                        st.success("✅ Knowledge base ready! Head to Career Chat.")
                        time.sleep(1)
                        st.session_state.page = "chat"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error building RAG system: {e}")
    else:
        missing = []
        if not has_resume: missing.append("Resume")
        if not has_jd:     missing.append("Job Description")
        st.markdown(f'<span class="badge badge-amber">⚠️ Still needed: {", ".join(missing)}</span>', unsafe_allow_html=True)

    # ── Tip box ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(91,143,255,.06);border:1px solid rgba(91,143,255,.2);
                border-radius:var(--radius);padding:1.1rem 1.3rem;font-size:.82rem;color:var(--muted);">
        <b style="color:var(--accent)">💡 Tips for best results</b><br><br>
        • Make sure your resume PDF contains <b>selectable text</b> (not a scanned image).<br>
        • The JD should include the full job description with required skills.<br>
        • Company info PDF can be an About page, culture doc, or any background material.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "chat":
    st.markdown("""
    <div class="sec-heading">💬 Career Chat <div class="line"></div></div>
    """, unsafe_allow_html=True)

    if not st.session_state.system_ready:
        st.markdown("""
        <div style="background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.25);
                    border-radius:var(--radius);padding:1.4rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:.5rem">⚠️</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:.3rem">System Not Initialized</div>
            <div style="font-size:.84rem;color:var(--muted)">
                Please go to <b>Setup & Upload</b> and build the knowledge base first.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("→  Go to Setup"):
            st.session_state.page = "setup"
            st.rerun()
    else:
        # ── Mode selector (matching the 3 chains + general) ──
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;
                    margin-bottom:.7rem;color:var(--muted)">SELECT MODE</div>
        """, unsafe_allow_html=True)

        mode_options = [
            ("chat",      "💬", "Career Chat",    "General career Q&A grounded in your documents"),
            ("analyze",   "🎯", "ATS Analysis",   "Triggered when your query contains 'analyze'"),
            ("interview", "🎤", "Interview Prep", "Triggered when your query contains 'interview'"),
            ("roadmap",   "🗺️", "Skill Roadmap",  "Triggered when your query contains 'roadmap'"),
        ]

        cols = st.columns(4)
        for col, (mid, icon, label, hint) in zip(cols, mode_options):
            with col:
                is_sel = st.session_state.mode == mid
                border = "rgba(91,143,255,.6)" if is_sel else "var(--border)"
                bg     = "rgba(91,143,255,.08)" if is_sel else "var(--card)"
                if st.button(f"{icon} {label}", key=f"mode_{mid}", use_container_width=True):
                    st.session_state.mode = mid
                    st.rerun()
                st.markdown(f'<div style="font-size:.7rem;color:var(--muted);text-align:center;margin-top:-6px;line-height:1.3">{hint}</div>', unsafe_allow_html=True)

        # ── Auto-fill hint ──
        mode_hints = {
            "chat":      "Ask me anything about your resume, JD match, career advice…",
            "analyze":   "Type 'analyze my resume' to trigger full ATS evaluation…",
            "interview": "Type 'generate interview questions' to get tailored questions…",
            "roadmap":   "Type 'create a roadmap for me' to get a weekly learning plan…",
        }
        st.markdown(f"""
        <div style="background:rgba(91,143,255,.05);border:1px solid rgba(91,143,255,.15);
                    border-radius:8px;padding:.7rem 1rem;font-size:.78rem;color:var(--muted);
                    margin:.8rem 0 1.2rem">
            💡 <b style="color:var(--accent)">Current mode:</b> {mode_hints[st.session_state.mode]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="fancy">', unsafe_allow_html=True)

        # ── Chat history ──
        if st.session_state.display_history:
            for turn in st.session_state.display_history:
                role, content, sources = turn["role"], turn["content"], turn.get("sources", [])

                if role == "user":
                    st.markdown(f"""
                    <div class="msg user">
                        <div class="avatar">👤</div>
                        <div class="bubble">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    source_chips = "".join(
                        f'<span class="source-chip">📎 {s}</span>' for s in sources
                    )
                    source_row = f'<div class="source-row">{source_chips}</div>' if sources else ""
                    st.markdown(f"""
                    <div class="msg ai">
                        <div class="avatar">🧠</div>
                        <div class="bubble">
                    """, unsafe_allow_html=True)
                    st.markdown(content)
                    if source_row:
                        st.markdown(source_row, unsafe_allow_html=True)
                    st.markdown("</div></div>", unsafe_allow_html=True)

        # ── Input ──
        with st.container():
            col_input, col_send = st.columns([5, 1])
            with col_input:
                # Pre-fill based on mode
                prefills = {
                    "analyze":   "Analyze my resume against the job description",
                    "interview": "Generate interview questions for my profile",
                    "roadmap":   "Create a skill roadmap for me based on the JD",
                    "chat":      "",
                }
                placeholder = mode_hints[st.session_state.mode]
                user_query = st.text_input(
                    "Your question",
                    label_visibility="collapsed",
                    placeholder=placeholder,
                    key="user_input",
                    value=prefills.get(st.session_state.mode, "") if not st.session_state.display_history else ""
                )
            with col_send:
                send = st.button("Send →", use_container_width=True)

        if send and user_query.strip():
            q = user_query.strip()
            st.session_state.display_history.append({"role": "user", "content": q, "sources": []})
            st.session_state.chat_history.append(("human", q))

            with st.spinner("Thinking…"):
                try:
                    answer, sources = run_query(q)
                    st.session_state.display_history.append({"role": "ai", "content": answer, "sources": sources})
                    st.session_state.chat_history.append(("ai", answer))
                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.session_state.display_history.append({"role": "ai", "content": err, "sources": []})
            st.rerun()

        # ── Quick-action buttons ──
        if not st.session_state.display_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.8rem;
                        color:var(--muted);margin-bottom:.6rem">QUICK ACTIONS</div>
            """, unsafe_allow_html=True)
            q_cols = st.columns(3)
            quick = [
                "Analyze my resume against the job description",
                "Generate interview questions for my profile",
                "Create a skill roadmap for me based on the JD",
            ]
            for qcol, qlabel in zip(q_cols, quick):
                with qcol:
                    if st.button(qlabel, use_container_width=True, key=f"quick_{qlabel[:20]}"):
                        st.session_state.display_history.append({"role": "user", "content": qlabel, "sources": []})
                        st.session_state.chat_history.append(("human", qlabel))
                        with st.spinner("Thinking…"):
                            try:
                                answer, sources = run_query(qlabel)
                                st.session_state.display_history.append({"role": "ai", "content": answer, "sources": sources})
                                st.session_state.chat_history.append(("ai", answer))
                            except Exception as e:
                                st.session_state.display_history.append({"role": "ai", "content": f"⚠️ {e}", "sources": []})
                        st.rerun()
