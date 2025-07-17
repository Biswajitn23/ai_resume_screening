# 🤖 AI-Powered Resume Screening System

> Automate resume shortlisting using AI and NLP — smarter than keyword-based ATS systems.

---

## 🔍 Overview

This project addresses the challenge of filtering thousands of resumes for every job application. Traditional ATS (Applicant Tracking Systems) rely heavily on keyword matching and often fail to interpret the true context of a candidate's profile.

The **AI Resume Screening System** uses **Natural Language Processing (NLP)** to understand the resume content, identify key skills and experiences, and match them intelligently with the job description — even if the terms used are different.

---

## 🎯 Features

- Upload one or more resumes (PDF, DOCX, TXT)
- Paste a job description
- Get AI-generated **match scores** for each resume
- Shortlist resumes based on scores
- Extract and highlight **top skills/keywords**
- Dashboard built with Flask

---

## 🧠 Technologies Used

| Library        | Purpose                              |
|----------------|--------------------------------------|
| `pandas`       | Data manipulation                    |
| `spaCy`        | NLP and entity extraction            |
| `scikit-learn` | Machine Learning classification      |
| `Flask`        | Web framework for dashboard          |
| `joblib`       | Model serialization                  |

---

## 📁 Dataset (Example)

To train the model, a labeled dataset of resumes is used where:

- `1` = Hired  
- `0` = Not Hired  

```csv
resume_text,label
"Experienced Python Developer skilled in Django and Flask",1
"Worked in sales, support, and customer relations",0
"Strong in Machine Learning, NLP, Data Analysis",1
"Experience in cold calling, lead generation, CRM tools",0
