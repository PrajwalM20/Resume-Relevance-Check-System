import os
import sqlite3
import streamlit as st
import fitz  # PyMuPDF

# ==============================
# DATABASE SETUP
# ==============================
DB_FILE = "submissions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jd_file TEXT,
            resume_file TEXT,
            score REAL,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==============================
# HELPER FUNCTIONS
# ==============================
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF safely"""
    try:
        uploaded_file.seek(0)  # reset stream
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        st.error(f"❌ Failed to extract text from {uploaded_file.name}: {e}")
        return ""

def calculate_score(jd_text, resume_text):
    """Simple word overlap score"""
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    if not jd_words:
        return 0.0
    matched = jd_words.intersection(resume_words)
    return round(len(matched) / len(jd_words) * 100, 2)

def rule_based_feedback(jd_text, resume_text):
    """Simple rule-based feedback"""
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    missing = jd_words - resume_words
    top_missing = list(missing)[:20]  # top 20 missing words
    return "Missing important skills:⚠️ " + ", ".join(top_missing)

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="Resume Relevance Checker", page_icon="📄", layout="wide")
st.title("📄 Resume Relevance Check System")

jd_files = st.file_uploader("📌 Upload Job Description(s) (PDF) [Multiple]", type=["pdf"], accept_multiple_files=True)
resume_files = st.file_uploader("📌 Upload Resumes (PDF) [Multiple]", type=["pdf"], accept_multiple_files=True)

if st.button("Evaluate "):
    if jd_files and resume_files:
        st.subheader("Evaluation Results📊")

        for jd_file in jd_files:
            jd_text = extract_text_from_pdf(jd_file)
            if not jd_text:
                st.warning(f"⚠️ Skipping JD {jd_file.name} due to extraction error.")
                continue

            st.markdown(f"📌 **Job Description:** {jd_file.name}")

            for resume_file in resume_files:
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text:
                    st.warning(f"⚠️ Skipping resume {resume_file.name} due to extraction error.")
                    continue

                score = calculate_score(jd_text, resume_text)
                feedback = rule_based_feedback(jd_text, resume_text)

                st.write(f"👤 **{resume_file.name}** | Score: {score}%")
                with st.expander("Feedback"):
                    st.write(feedback)

                # Save results in DB
                try:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO submissions (jd_file, resume_file, score, feedback) VALUES (?, ?, ?, ?)",
                        (jd_file.name, resume_file.name, score, feedback)
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"❌ Database error: {e}")
    else:
        st.warning("Please upload JD(s) and resume(s) to evaluate.")

if st.button("📂 View All Submissions"):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT jd_file, resume_file, score, feedback FROM submissions")
        rows = c.fetchall()
        conn.close()

        if rows:
            st.subheader("📂 All Submissions")
            for row in rows:
                st.write(f"JD: {row[0]} | Resume: {row[1]} | Score: {row[2]}%")
                with st.expander("Feedback"):
                    st.write(row[3])
        else:
            st.info("No submissions yet.")
    except Exception as e:
        st.error(f"❌ Could not fetch submissions: {e}")

st.markdown("---")
st.markdown("© 2025 Hackathon Demo | Automated Resume Relevance Check System")








