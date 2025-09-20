# Streamlit MVP - Automated Resume Relevance Check System
import os, io, re, tempfile, glob
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import docx
from rapidfuzz import fuzz
import openai

# Optional LangChain embeddings
try:
    from langchain.embeddings import OpenAIEmbeddings
except:
    OpenAIEmbeddings = None

# -----------------------
# Configuration
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-3-small"
HARD_WEIGHT = 0.6
SOFT_WEIGHT = 0.4
HIGH_THRESH = 75
MEDIUM_THRESH = 50

# -----------------------
# Helper functions
# -----------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = docx.Document(tmp.name)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_file(uploaded_file) -> str:
    content = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"): return extract_text_from_pdf_bytes(content)
    if fname.endswith(".docx"): return extract_text_from_docx_bytes(content)
    if fname.endswith(".txt"):
        try: return content.decode("utf-8")
        except: return content.decode("latin-1")
    return ""

def simple_skill_extractor(text: str, top_k=50) -> List[str]:
    text = text.lower()
    tokens = re.split(r"[^a-z+#0-9\.]+", text)
    tokens = [t for t in tokens if 2 <= len(t) <= 30 and re.search(r"[a-z0-9]", t)]
    freq = {}
    for t in tokens: freq[t] = freq.get(t,0)+1
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:top_k]]

def parse_jd(text: str) -> Dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else ""
    skills = simple_skill_extractor(text, top_k=80)
    qualifications = [kw for kw in ["bachelor","masters","phd","btech","mtech","degree"] if kw in text.lower()]
    return {"title": title, "skills": skills, "qualifications": qualifications, "raw": text}

def hard_match_score(jd_skills: List[str], resume_text: str) -> Tuple[float,List[str]]:
    resume_text_lower = resume_text.lower()
    matched = 0
    matched_skills = []
    for skill in jd_skills:
        if not skill or len(skill)<2: continue
        if skill in resume_text_lower:
            matched +=1; matched_skills.append(skill)
        else:
            if fuzz.partial_ratio(skill,resume_text_lower) >= 85:
                matched +=1; matched_skills.append(skill)
    total = max(1,len(jd_skills))
    missing = [s for s in jd_skills if s not in matched_skills]
    return (matched/total)*100, missing

def get_embedding(text: str) -> np.ndarray:
    if OpenAIEmbeddings is not None:
        emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        vec = emb.embed_documents([text])[0]
        return np.array(vec)
    else:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=[text])
        return np.array(resp['data'][0]['embedding'])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom==0: return 0.0
    return float(np.dot(a,b)/denom)

def compute_final_score(hard_pct: float, soft_cosine: float) -> float:
    soft_pct = max(0.0,(soft_cosine+1)/2)*100
    return round(HARD_WEIGHT*hard_pct + SOFT_WEIGHT*soft_pct,2)

def verdict_from_score(score: float) -> str:
    if score>=HIGH_THRESH: return "High"
    elif score>=MEDIUM_THRESH: return "Medium"
    return "Low"

def generate_feedback_openai(resume_text: str, jd_text: str, missing_skills: List[str]) -> str:
    prompt = f"Job Description:\n{jd_text[:2000]}\nResume:\n{resume_text[:2000]}\nMissing skills: {', '.join(missing_skills[:20])}\nSuggestions:"
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=250)
        return resp['choices'][0]['message']['content'].strip()
    except: return "Feedback could not be generated."

# -----------------------
# Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="Automated Resume Relevance Check", layout="wide")
    st.title("Automated Resume Relevance Check System — MVP")

    st.sidebar.header("Configuration")
    demo_mode = st.sidebar.checkbox("Load sample data for demo", value=True)

    # Upload JD
    st.header("1) Upload Job Description")
    jd_file = None
    jd_text = ""
    jd_meta = None

    if demo_mode:
        sample_jd_path = "sample_data/JD1.pdf"
        with open(sample_jd_path,"rb") as f:
            jd_file = f
            jd_text = extract_text_from_file(jd_file)
            jd_meta = parse_jd(jd_text)
            st.success("Sample JD loaded")
            st.subheader("Parsed JD Title & Skills")
            st.write(jd_meta['title'])
            st.write(pd.DataFrame({'Skills': jd_meta['skills'][:30]}))
    else:
        jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=['pdf','docx','txt'])
        if jd_file is not None:
            jd_text = extract_text_from_file(jd_file)
            jd_meta = parse_jd(jd_text)
            st.success("JD parsed")
            st.subheader("Parsed JD Title & Skills")
            st.write(jd_meta['title'])
            st.write(pd.DataFrame({'Skills': jd_meta['skills'][:30]}))

    # Upload Resumes
    st.header("2) Upload Resumes")
    resumes = []
    if demo_mode:
        for file_path in glob.glob("sample_data/resumes/*"):
            with open(file_path,"rb") as f:
                class DummyFile:
                    def __init__(self,f,name): self.file=f; self.name=name
                    def read(self): self.file.seek(0); return self.file.read()
                resumes.append(DummyFile(f, os.path.basename(file_path)))
    else:
        resumes = st.file_uploader("Upload Resumes", type=['pdf','docx'], accept_multiple_files=True)

    # Process resumes
    results=[]
    if jd_meta and resumes:
        jd_embedding = get_embedding(jd_text)
        for up in resumes:
            rtext = extract_text_from_file(up)
            hard_pct, missing = hard_match_score(jd_meta['skills'][:40], rtext)
            r_emb = get_embedding(rtext)
            cosine = cosine_similarity(jd_embedding,r_emb)
            final_score = compute_final_score(hard_pct, cosine)
            verdict = verdict_from_score(final_score)
            feedback=""
            if st.checkbox(f"Generate feedback for {up.name}",value=False,key=f"fb_{up.name}"):
                feedback = generate_feedback_openai(rtext,jd_text,missing)
            results.append({'filename':up.name,'hard_pct':round(hard_pct,2),'soft_cosine':round(cosine,4),'final_score':final_score,'verdict':verdict,'missing_skills':', '.join(missing[:10]),'feedback':feedback})
        df=pd.DataFrame(results).sort_values('final_score',ascending=False)
        st.header("Results")
        st.dataframe(df)
        st.download_button("Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name='resume_scores.csv')
        st.subheader("Top Candidates")
        st.table(df.head(5))

if __name__=="__main__":
    main()




