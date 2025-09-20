# Automated Resume Relevance Check System (v5)

## 📝 Problem Statement

At Innomatics Research Labs, resume evaluation is currently **manual, inconsistent, and time-consuming**. Placement teams across multiple locations receive dozens of job requirements weekly, each attracting thousands of applications.  

**Challenges:**

- Manual shortlisting causes delays.  
- Evaluation is inconsistent across reviewers.  
- High workload reduces time for interview preparation and student guidance.  

**Objective:** Build an **AI-powered automated system** that:  

- Evaluates resumes against job descriptions.  
- Generates a **Relevance Score (0–100)** for each resume.  
- Highlights missing skills or projects.  
- Provides a **High/Medium/Low suitability verdict**.  
- Gives **personalized improvement feedback** to students.  
- Stores evaluations in a **dashboard/database** for easy access.

---

## 🛠 Approach / Features

1. **Resume Parsing:**  
   - Extract text from PDF/DOCX resumes using `PyMuPDF` and `docx2txt`.  

2. **Job Description Parsing:**  
   - Extract required skills and qualifications from JD PDF.  

3. **Scoring:**  
   - **Hard Match:** Keyword matching via TF-IDF and cosine similarity.  
   - **Semantic Match:** Embedding similarity using `sentence-transformers`.  
   - **Weighted Final Score:** Hard match 40%, semantic match 60%.  
   - **Verdict:** High / Medium / Low suitability.  

4. **Missing Skills Detection & Feedback:**  
   - Identify skills present in JD but missing in resume.  
   - Generate **personalized feedback** using `transformers` text-generation pipeline.  

5. **Database Storage:**  
   - Store results in **SQLite**.  
   - Columns: Resume name, JD file, hard/semantic score, final score, verdict, missing skills, feedback.  

6. **Dashboard (Streamlit):**  
   - Select JD and process all resumes in `sample_data/resumes/`.  
   - Display scores, verdicts, missing skills, feedback, and summary statistics.  

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Resume-Relevance-Check-System.git
cd Resume-Relevance-Check-System
````

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

> `requirements.txt` includes:
> `streamlit`, `PyMuPDF`, `docx2txt`, `pandas`, `scikit-learn`, `sentence-transformers`, `transformers`, `torch`

---

## 🚀 Usage

1. Run the Streamlit app:

```bash
streamlit run hackathon_resume_checker_v5.py
```

2. **Interface:**

* Select a Job Description (JD) PDF from `sample_data/`.
* All resumes in `sample_data/resumes/` are processed automatically.
* View for each resume:

  * Hard Match (%)
  * Semantic Match (%)
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

© 2025 Hackathon Demo | Automated Resume Relevance Check System

```

---

