"""
SkillSync — AI Resume Screener
Match your skills to any job description instantly.
Built with TF-IDF + Cosine Similarity
"""

import re
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="SkillSync",
    page_icon="🎯",
    layout="wide"
)

# ── LINKEDIN THEME ────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #f3f2ef;
    }

    /* Main header */
    .skillsync-header {
        background: linear-gradient(135deg, #0077b5, #00a0dc);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,119,181,0.3);
    }
    .skillsync-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    .skillsync-header p {
        color: #cce8f5;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }

    /* Score circle */
    .score-container {
        text-align: center;
        padding: 2rem;
    }
    .score-circle {
        display: inline-block;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        line-height: 160px;
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 1rem auto;
    }

    /* Skill tags */
    .skill-tag-match {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-tag-missing {
        display: inline-block;
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #0077b5, #00a0dc);
        color: white;
        border: none;
        padding: 0.75rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 30px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #005f8e, #0077b5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,119,181,0.4);
    }

    /* Section titles */
    .section-title {
        color: #0077b5;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0077b5;
    }

    /* Text areas */
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #0077b5;
        box-shadow: 0 0 0 3px rgba(0,119,181,0.15);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── HELPER FUNCTIONS ──────────────────────────────────────

STOPWORDS = {
    "i", "me", "my", "we", "you", "he", "she", "it", "they", "what",
    "this", "that", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "a", "an", "the", "and", "but", "if",
    "or", "as", "at", "by", "for", "with", "about", "to", "from",
    "in", "out", "on", "is", "will", "can", "may", "should", "would",
    "could", "not", "no", "so", "than", "too", "very", "just", "also",
    "of", "up", "its", "our", "your", "their", "who", "which", "how",
    "all", "more", "other", "any", "each", "both", "such", "into",
    "through", "during", "including", "while", "although", "strong",
    "experience", "work", "working", "ability", "skills", "skill",
    "knowledge", "understanding", "looking", "seeking", "candidate",
    "must", "required", "preferred", "plus", "well", "good", "great",
    "excellent", "minimum", "years", "year", "role", "position", "job",
    "team", "company", "join", "help", "using", "use", "used",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def extract_keywords(text: str, top_n: int = 30):
    clean = clean_text(text)
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([clean])
        return set(vectorizer.get_feature_names_out())
    except Exception:
        return set(clean.split())


def compute_match(jd_text: str, resume_text: str):
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except Exception:
        score = 0.0

    jd_keywords = extract_keywords(jd_text, top_n=40)
    resume_keywords = extract_keywords(resume_text, top_n=60)

    matched = jd_keywords & resume_keywords
    missing = jd_keywords - resume_keywords

    return round(score * 100, 1), matched, missing


def get_score_color(score: float) -> str:
    if score >= 75:
        return "#27ae60"
    elif score >= 50:
        return "#f39c12"
    else:
        return "#e74c3c"


def get_score_label(score: float) -> str:
    if score >= 75:
        return "Strong Match 🎉"
    elif score >= 50:
        return "Moderate Match ⚡"
    elif score >= 25:
        return "Weak Match 📈"
    else:
        return "Poor Match ❌"


# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="skillsync-header">
    <h1>🎯 SkillSync</h1>
    <p>AI-powered resume screener — match your skills to any job, instantly</p>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">📋 Job Description</div>', unsafe_allow_html=True)
    jd_input = st.text_area(
        label="job_description",
        placeholder="Paste the job description here...\n\nExample:\nWe are looking for a Python developer with experience in machine learning, SQL, and cloud platforms like AWS...",
        height=300,
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="section-title">📄 Your Resume</div>', unsafe_allow_html=True)
    resume_input = st.text_area(
        label="resume",
        placeholder="Paste your resume text here...\n\nExample:\nExperienced in Python, machine learning, data analysis, scikit-learn, SQL, and NLP...",
        height=300,
        label_visibility="collapsed"
    )

# ── ANALYZE BUTTON ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    analyze = st.button("🔍 Analyze My Resume")

# ── RESULTS ───────────────────────────────────────────────
if analyze:
    if not jd_input.strip() or not resume_input.strip():
        st.warning("⚠️ Please paste both a job description and your resume!")
    else:
        with st.spinner("Analyzing your resume..."):
            score, matched, missing = compute_match(jd_input, resume_input)

        st.markdown("---")
        st.markdown("<h2 style='color:#0077b5; font-weight:900; font-size:2rem;'>📊 Results</h2>", unsafe_allow_html=True)

        # Score + Label
        color = get_score_color(score)
        label = get_score_label(score)

        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="card score-container">
                <div style="color:#1a1a1a; font-size:1rem; font-weight:700;">Match Score</div>
                <div class="score-circle" style="background:{color};">{score}%</div>
                <div style="color:{color}; font-size:1.1rem; font-weight:700;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="color:#1a1a1a; font-size:1rem; font-weight:700;">Matched Keywords</div>
                <div style="font-size:2.5rem; font-weight:800; color:#27ae60;">{len(matched)}</div>
                <div style="color:#333; font-weight:600;">skills found in your resume</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="color:#1a1a1a; font-size:1rem; font-weight:700;">Missing Keywords</div>
                <div style="font-size:2.5rem; font-weight:800; color:#e74c3c;">{len(missing)}</div>
                <div style="color:#333; font-weight:600;">skills to add to your resume</div>
            </div>
            """, unsafe_allow_html=True)

        # Matched Skills
        if matched:
            st.markdown("<h3 style='color:#1a6b1a; font-weight:800;'>✅ Matched Skills</h3>", unsafe_allow_html=True)
            tags = "".join([f'<span class="skill-tag-match">{s}</span>' for s in sorted(matched)])
            st.markdown(f'<div class="card">{tags}</div>', unsafe_allow_html=True)

        # Missing Skills
        if missing:
            st.markdown("<h3 style='color:#a00000; font-weight:800;'>❌ Missing Skills</h3>", unsafe_allow_html=True)
            tags = "".join([f'<span class="skill-tag-missing">{s}</span>' for s in sorted(missing)])
            st.markdown(f'<div class="card">{tags}</div>', unsafe_allow_html=True)

        # Suggestions
        st.markdown("<h3 style='color:#0077b5; font-weight:800;'>💡 How to Improve</h3>", unsafe_allow_html=True)
        if score >= 75:
            st.markdown("<div style='background:#d4edda; padding:1rem; border-radius:8px; color:#155724; font-weight:700; font-size:1rem; border:1px solid #c3e6cb;'>🎉 Your resume is a strong match! You are ready to apply.</div>", unsafe_allow_html=True)
        elif score >= 50:
            st.markdown(f"<div style='background:#cce5ff; padding:1rem; border-radius:8px; color:#004085; font-weight:700; font-size:1rem; border:1px solid #b8daff;'>⚡ Good start! Add these missing skills to boost your score: <strong>{', '.join(list(missing)[:5])}</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#fff3cd; padding:1rem; border-radius:8px; color:#856404; font-weight:700; font-size:1rem; border:1px solid #ffeeba;'>📈 Your resume needs work. Focus on adding: <strong>{', '.join(list(missing)[:5])}</strong>. Tailor your resume specifically for this role.</div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    Built with ❤️ using TF-IDF + Cosine Similarity | SkillSync by Smarika Singh
</div>
""", unsafe_allow_html=True)