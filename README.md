# Automated Resume Relevance Check System 📄 

## Problem Statement
At many companies and educational institutions, resume evaluation is manual, time-consuming, and inconsistent. Recruiters often receive hundreds or thousands of applications per job description (JD), making it challenging to quickly shortlist the most relevant candidates.  

This project automates **resume relevance checking** by comparing resumes against job descriptions and providing a **rule-based relevance score** along with **feedback on missing skills**.

---

## Approach

1. **PDF Parsing:**  
   - Extract text from Job Descriptions and resumes using `PyMuPDF (fitz)`.  

2. **Relevance Scoring:**  
   - Simple word overlap score between JD and resume.  
   - Rule-based feedback to highlight missing skills.  

3. **Batch Evaluation:**  
   - Multiple JDs × Multiple resumes in one go.  
   - Saves results to a SQLite database (`submissions.db`).  

4. **LLM Feedback (Optional):**  
   - AI-based feedback can be added using small LLMs (like tiny-T5) if memory permits.  
   - If unavailable, fallback to rule-based feedback.  

---

## Installation

1. Clone the repository:
      ```bash
      git clone https://github.com/yourusername/resume-relevance-checker.git
      cd resume-relevance-checker

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
# Activate venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Streamlit app:

```bash
streamlit run hackathon_resume_checker_v7.py
```

2. **Interface:**

* Select a Job Description (JD) PDF from `sample_data/`.
* All resumes in `sample_data/resumes/` are processed automatically.
* View for each resume:

  * Final Score (%)
  * Verdict (High / Medium / Low)
  * Missing Skills
  * LLM-generated Feedback

3. **Summary:**

* Average score across resumes.
* Counts of High/Medium/Low verdicts.

4. **Database:**

* Evaluations stored in `resume_scores.db`.
* Can be queried for further analysis or exported.

---

## 🧰 Demo Data

```
sample_data/
├── resumes/
│   ├── resume-1.pdf
│   ├── RESUME-2.pdf
│   └── resume-3.pdf
├── sample_jd_1.pdf
└── sample_jd_2.pdf
```

> Demo files allow instant testing without manual uploads.

---

## ⚡ Optional Enhancements

* Dashboard tab to filter by score, verdict, or missing skills.
* CSV export for placement teams.
* Integration with FAISS/Chroma for semantic search over large resume datasets.
* Support for dynamic resume uploads by students.

---

🌐 Live Demo
Streamlit Cloud Link: https://resume-relevance-check-system-nxudheqnyhnkpmxnvsjpwk.streamlit.app

- © 2025 Hackathon Demo | Automated Resume Relevance Check System
- Built by Prajwal M and Shreeja Hebbar for Code4EdTech'25 💻
