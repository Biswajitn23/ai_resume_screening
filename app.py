from flask import Flask, render_template, request, send_file, redirect, url_for, session, jsonify, send_from_directory
import joblib
import spacy
import os
from sklearn.metrics.pairwise import cosine_similarity
import io
import csv
import re
from rapidfuzz import fuzz
from io import BytesIO
from collections import Counter

class FileLikeWithAttrs(BytesIO):
    def __init__(self, content, filename, mimetype):
        super().__init__(content)
        self.filename = filename
        self.mimetype = mimetype

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# Load trained model
model = joblib.load("model/model.pkl")

# Smaller, focused stopword list
CUSTOM_STOPWORDS = set([
    'the', 'and', 'or', 'a', 'an', 'in', 'on', 'for', 'to', 'of', 'as', 'by', 'at', 'from',
    'with', 'is', 'are', 'was', 'were', 'be', 'has', 'have', 'had', 'will', 'can', 'may', 'should', 'would', 'could',
    'that', 'this', 'these', 'those', 'their', 'his', 'her', 'its', 'our', 'your', 'my', 'i', 'we', 'you', 'he', 'she', 'they', 'it',
    'experience', 'knowledge', 'working', 'responsible', 'proficient', 'familiar', 'ability', 'skills',
    'work', 'good', 'excellent', 'strong', 'background', 'understanding', 'capable', 'competent',
    'expertise', 'expert', 'well', 'team', 'member', 'years', 'including', 'etc', 'various', 'using', 'used',
    'customer'  # Added to ignore simple word 'customer'
])

# Expanded skills and education lists
SKILLS = set([
    'python', 'java', 'c++', 'c#', 'sql', 'javascript', 'html', 'css', 'aws', 'azure', 'docker', 'kubernetes',
    'machine learning', 'deep learning', 'nlp', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'react', 'angular', 'node', 'django', 'flask', 'excel', 'powerbi', 'tableau', 'salesforce', 'git', 'linux',
    'spark', 'hadoop', 'mongodb', 'postgresql', 'mysql', 'rest', 'graphql', 'cloud', 'devops', 'etl', 'data analysis',
    'data science', 'ai', 'ml', 'frontend', 'backend', 'fullstack', 'data engineer', 'data scientist', 'business analyst',
    'project manager', 'scrum', 'agile', 'jira', 'kanban', 'crm', 'erp', 'marketing', 'sales', 'seo', 'sem', 'content',
    'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'presentation', 'negotiation',
    'typescript', 'swift', 'go', 'ruby', 'php', 'objective-c', 'scala', 'perl', 'sas', 'r', 'matlab', 'stata', 'spss',
    'cloud computing', 'big data', 'data visualization', 'data mining', 'data warehousing', 'business intelligence',
    'blockchain', 'cybersecurity', 'networking', 'system administration', 'qa', 'testing', 'automation', 'robotics',
    'mobile development', 'android', 'ios', 'ui', 'ux', 'design', 'illustrator', 'photoshop', 'indesign', 'after effects',
    'public speaking', 'coaching', 'mentoring', 'training', 'customer service', 'account management', 'supply chain',
    'logistics', 'procurement', 'finance', 'accounting', 'investment', 'banking', 'risk management', 'compliance',
    'legal', 'regulatory', 'medical', 'healthcare', 'clinical', 'pharmaceutical', 'biotech', 'chemistry', 'biology',
    'physics', 'mathematics', 'statistics', 'economics', 'psychology', 'sociology', 'anthropology', 'history', 'geography',
    'political science', 'international relations', 'linguistics', 'journalism', 'media', 'writing', 'editing', 'publishing',
    'translation', 'interpretation', 'teaching', 'education', 'research', 'grant writing', 'fundraising', 'event planning',
    'hospitality', 'tourism', 'real estate', 'construction', 'architecture', 'urban planning', 'environmental science',
    'sustainability', 'energy', 'renewable energy', 'manufacturing', 'production', 'quality assurance', 'safety', 'maintenance',
    'operations', 'strategy', 'consulting', 'entrepreneurship', 'startups', 'innovation', 'product management', 'scrum master',
    'business development', 'partnerships', 'vendor management', 'negotiation', 'public relations', 'advertising', 'branding',
    'market research', 'digital marketing', 'social media', 'content marketing', 'email marketing', 'e-commerce', 'retail',
    'merchandising', 'fashion', 'luxury', 'sports', 'fitness', 'wellness', 'nutrition', 'culinary', 'food safety', 'agriculture',
    'horticulture', 'forestry', 'mining', 'oil and gas', 'transportation', 'aviation', 'aerospace', 'marine', 'naval', 'defense',
    'security', 'law enforcement', 'firefighting', 'emergency management', 'disaster recovery', 'human resources', 'recruiting',
    'talent acquisition', 'compensation', 'benefits', 'employee relations', 'labor relations', 'diversity', 'inclusion', 'training and development',
    'organizational development', 'change management', 'performance management', 'succession planning', 'workforce planning', 'analytics',
    'data governance', 'data stewardship', 'data quality', 'master data management', 'metadata management', 'information management',
    'records management', 'document management', 'knowledge management', 'library science', 'museum studies', 'archival science', 'curation',
    'collection management', 'exhibit design', 'conservation', 'restoration', 'preservation', 'archaeology', 'paleontology', 'geology', 'meteorology',
    'oceanography', 'marine biology', 'zoology', 'botany', 'ecology', 'environmental policy', 'environmental law', 'environmental health', 'public health',
    'epidemiology', 'biostatistics', 'bioinformatics', 'genomics', 'proteomics', 'metabolomics', 'systems biology', 'synthetic biology', 'biomaterials',
    'biomechanics', 'biophysics', 'biochemistry', 'molecular biology', 'cell biology', 'developmental biology', 'neuroscience', 'cognitive science',
    'behavioral science', 'social science', 'demography', 'criminology', 'forensic science', 'criminal justice', 'law', 'legal research', 'legal writing',
    'litigation', 'trial practice', 'appellate practice', 'corporate law', 'intellectual property', 'patent law', 'trademark law', 'copyright law',
    'real estate law', 'tax law', 'international law', 'immigration law', 'family law', 'estate planning', 'elder law', 'health law', 'employment law',
    'labor law', 'securities law', 'antitrust law', 'bankruptcy law', 'environmental law', 'energy law', 'transportation law', 'aviation law', 'maritime law',
    'military law', 'sports law', 'entertainment law', 'media law', 'communications law', 'cyber law', 'space law', 'animal law', 'agricultural law',
    'food law', 'gaming law', 'hospitality law', 'insurance law', 'product liability', 'toxic torts', 'mass torts', 'class actions', 'complex litigation',
    'alternative dispute resolution', 'mediation', 'arbitration', 'negotiation', 'settlement', 'trial advocacy', 'jury selection', 'jury instructions',
    'jury verdicts', 'appeals', 'post-conviction', 'clemency', 'pardons', 'expungement', 'record sealing', 'probation', 'parole', 'sentencing', 'incarceration',
    'reentry', 'victim advocacy', 'witness protection', 'child advocacy', 'elder advocacy', 'disability advocacy', 'civil rights', 'human rights', 'animal rights',
    'consumer rights', 'patient rights', 'tenant rights', 'landlord rights', 'property rights', 'intellectual property rights', 'privacy rights', 'freedom of speech',
    'freedom of religion', 'freedom of assembly', 'freedom of the press', 'freedom of association', 'freedom of movement', 'freedom from discrimination',
    'freedom from harassment', 'freedom from retaliation', 'freedom from violence', 'freedom from abuse', 'freedom from neglect', 'freedom from exploitation',
    'freedom from oppression', 'freedom from persecution', 'freedom from torture', 'freedom from slavery', 'freedom from trafficking', 'freedom from forced labor',
    'freedom from child labor', 'freedom from hazardous work', 'freedom from unfair labor practices', 'freedom from wage theft', 'freedom from unsafe working conditions',
    'freedom from discrimination in employment', 'freedom from harassment in employment', 'freedom from retaliation in employment', 'freedom from wrongful termination',
    'freedom from constructive discharge', 'freedom from demotion', 'freedom from discipline', 'freedom from suspension', 'freedom from layoff', 'freedom from furlough',
    'freedom from reduction in force', 'freedom from downsizing', 'freedom from outsourcing', 'freedom from offshoring', 'freedom from automation', 'freedom from technological change',
    'freedom from economic change', 'freedom from globalization', 'freedom from trade agreements', 'freedom from tariffs', 'freedom from quotas', 'freedom from embargoes',
    'freedom from sanctions', 'freedom from boycotts', 'freedom from blockades', 'freedom from sieges', 'freedom from occupation', 'freedom from annexation', 'freedom from colonization',
    'freedom from imperialism', 'freedom from neocolonialism', 'freedom from dependency', 'freedom from underdevelopment', 'freedom from poverty', 'freedom from hunger', 'freedom from malnutrition',
    'freedom from disease', 'freedom from disability', 'freedom from injury', 'freedom from illness', 'freedom from death', 'freedom from suffering', 'freedom from pain', 'freedom from distress',
    'freedom from anxiety', 'freedom from depression', 'freedom from mental illness', 'freedom from addiction', 'freedom from substance abuse', 'freedom from alcoholism', 'freedom from drug abuse',
    'freedom from gambling', 'freedom from smoking', 'freedom from obesity', 'freedom from eating disorders', 'freedom from malnutrition', 'freedom from dehydration', 'freedom from starvation',
    'freedom from exposure', 'freedom from homelessness', 'freedom from displacement', 'freedom from migration', 'freedom from exile', 'freedom from statelessness', 'freedom from detention',
    'freedom from imprisonment', 'freedom from arrest', 'freedom from prosecution', 'freedom from investigation', 'freedom from surveillance', 'freedom from censorship', 'freedom from propaganda',
    'freedom from indoctrination', 'freedom from brainwashing', 'freedom from mind control', 'freedom from coercion', 'freedom from manipulation', 'freedom from intimidation', 'freedom from threats',
    'freedom from violence', 'freedom from abuse', 'freedom from neglect', 'freedom from exploitation', 'freedom from oppression', 'freedom from persecution', 'freedom from torture', 'freedom from slavery',
    'freedom from trafficking', 'freedom from forced labor', 'freedom from child labor', 'freedom from hazardous work', 'freedom from unfair labor practices', 'freedom from wage theft', 'freedom from unsafe working conditions',
    'freedom from discrimination in employment', 'freedom from harassment in employment', 'freedom from retaliation in employment', 'freedom from wrongful termination', 'freedom from constructive discharge',
    'freedom from demotion', 'freedom from discipline', 'freedom from suspension', 'freedom from layoff', 'freedom from furlough', 'freedom from reduction in force', 'freedom from downsizing',
    'freedom from outsourcing', 'freedom from offshoring', 'freedom from automation', 'freedom from technological change', 'freedom from economic change', 'freedom from globalization', 'freedom from trade agreements',
    'freedom from tariffs', 'freedom from quotas', 'freedom from embargoes', 'freedom from sanctions', 'freedom from boycotts', 'freedom from blockades', 'freedom from sieges', 'freedom from occupation', 'freedom from annexation',
    'freedom from colonization', 'freedom from imperialism', 'freedom from neocolonialism', 'freedom from dependency', 'freedom from underdevelopment', 'freedom from poverty', 'freedom from hunger', 'freedom from malnutrition',
    'freedom from disease', 'freedom from disability', 'freedom from injury', 'freedom from illness', 'freedom from death', 'freedom from suffering', 'freedom from pain', 'freedom from distress', 'freedom from anxiety', 'freedom from depression',
    'freedom from mental illness', 'freedom from addiction', 'freedom from substance abuse', 'freedom from alcoholism', 'freedom from drug abuse', 'freedom from gambling', 'freedom from smoking', 'freedom from obesity', 'freedom from eating disorders',
    'freedom from malnutrition', 'freedom from dehydration', 'freedom from starvation', 'freedom from exposure', 'freedom from homelessness', 'freedom from displacement', 'freedom from migration', 'freedom from exile', 'freedom from statelessness',
    'freedom from detention', 'freedom from imprisonment', 'freedom from arrest', 'freedom from prosecution', 'freedom from investigation', 'freedom from surveillance', 'freedom from censorship', 'freedom from propaganda', 'freedom from indoctrination',
    'freedom from brainwashing', 'freedom from mind control', 'freedom from coercion', 'freedom from manipulation', 'freedom from intimidation', 'freedom from threats'
])

# Replace EDU_KEYWORDS with a more comprehensive list
EDU_KEYWORDS = [
    "ba", "bachelor", "bachelor of science", "bachelor of arts", "b.e.", "be", "bsc", "b.sc", "btech", "b.tech",
    "ma", "master", "master of science", "master of arts", "m.e.", "me", "msc", "m.sc", "mtech", "m.tech",
    "phd", "doctor", "doctorate", "mba", "ug", "pg", "ssc", "hsc", "certificate", "diploma", "associate",
    "degree", "graduate", "postgraduate", "undergraduate", "honours", "hons", "education"
]

TAG_KEYWORDS = {
    'Backend': ['python', 'java', 'c++', 'node', 'django', 'flask', 'backend', 'api', 'sql', 'mongodb', 'postgresql', 'mysql'],
    'Frontend': ['html', 'css', 'javascript', 'react', 'angular', 'frontend', 'ui', 'ux'],
    'AI': ['machine learning', 'deep learning', 'nlp', 'ai', 'ml', 'tensorflow', 'pytorch', 'data science'],
    'Sales': ['sales', 'crm', 'marketing', 'negotiation', 'lead generation', 'customer', 'business development']
}

# --- Skill Synonyms and Degree Abbreviations ---
SKILL_SYNONYMS = {
    "rest api": "api",
    "rest": "api",
    "python (basic)": "python",
    "python (advanced)": "python",
    # Add more as needed
}
DEGREE_ABBREVIATIONS = [
    "btech", "be", "bsc", "msc", "mba", "bba", "mtech", "phd", "ba", "ma", "bs", "ms"
]

# --- Resume Download Endpoint ---
@app.route('/download_resume/<filename>')
def download_resume(filename):
    return send_from_directory('uploads', filename, as_attachment=True)

# --- Improved Education Extraction ---
def extract_education(text):
    from rapidfuzz import fuzz
    text_lower = text.lower()
    lines = text_lower.splitlines()
    found_edu = set()
    # Lowercase all education keywords for matching
    edu_keywords_lower = [edu.lower() for edu in EDU_KEYWORDS]
    # Whitelist of valid short degree abbreviations
    SHORT_DEGREES = {'be', 'me', 'ms', 'ma', 'ba', 'bs', 'phd', 'mba', 'bca', 'mca', 'bsc', 'msc', 'btech', 'mtech', 'ug', 'pg', 'ssc', 'hsc'}
    # Fuzzy match education keywords in the whole text and extract full lines
    for line in lines:
        for edu in edu_keywords_lower:
            if len(edu) < 3 and edu not in SHORT_DEGREES:
                continue
            threshold = 95 if len(edu) < 4 else 85
            if fuzz.partial_ratio(edu, line) >= threshold:
                found_edu.add(line.strip())
    # Also look for education section headers and extract lines below
    section_headers = ['education', 'academic background', 'qualifications', 'educational background', 'academic qualifications']
    for i, line in enumerate(lines):
        if any(header in line for header in section_headers):
            for next_line in lines[i+1:i+6]:
                for edu in edu_keywords_lower:
                    if len(edu) < 3 and edu not in SHORT_DEGREES:
                        continue
                    threshold = 95 if len(edu) < 4 else 85
                    if fuzz.partial_ratio(edu, next_line) >= threshold:
                        found_edu.add(next_line.strip())
    # Remove likely Roman numerals and unrelated short matches
    roman_numerals = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}
    found_edu = {e for e in found_edu if e not in roman_numerals and (len(e) > 2 or e in SHORT_DEGREES)}
    return sorted(found_edu)

# --- Improved Skills Extraction ---
def normalize_skill(skill):
    return SKILL_SYNONYMS.get(skill.lower(), skill.title())

def extract_skills(text):
    text_lower = text.lower()
    found_skills = []
    for skill in SKILLS:
        if skill in text_lower:
            found_skills.append(normalize_skill(skill))
    # Sort by frequency (most common first)
    sorted_skills = [skill for skill, _ in Counter(found_skills).most_common()]
    return ', '.join(sorted_skills)

# --- Improved Match Scoring ---
def compute_match_score(skills_score, experience_years, has_degree, has_cert):
    exp_score = min(experience_years / 10, 1)  # Cap at 10 years
    edu_score = 1 if has_degree else 0
    cert_score = 1 if has_cert else 0
    return (
        0.5 * skills_score +
        0.2 * exp_score +
        0.2 * edu_score +
        0.1 * cert_score
    ) * 100

def filter_relevant_terms(terms):
    return [t for t in terms if t.lower() not in CUSTOM_STOPWORDS and len(t) > 2]

# Clean text using SpaCy
def clean_resume(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_text_from_file(file):
    filename = file.filename
    if filename.endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif filename.endswith('.docx'):
        import docx
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return ""

def extract_years_experience(text):
    # Improved regex for more patterns
    years = re.findall(r'(\d+)\s*(?:\+|plus)?\s*years?', text.lower())
    ranges = re.findall(r'(\d{4})\s*[-–]\s*(\d{4})', text)
    since = re.findall(r'since\s*(\d{4})', text.lower())
    max_years = int(max(years, default=0))
    for start, end in ranges:
        try:
            diff = int(end) - int(start)
            if diff > max_years:
                max_years = diff
        except:
            pass
    for s in since:
        try:
            diff = 2024 - int(s)
            if diff > max_years:
                max_years = diff
        except:
            pass
    return max_years

def extract_tags(text):
    text_lower = text.lower()
    tags = set()
    for tag, keywords in TAG_KEYWORDS.items():
        if any(word in text_lower for word in keywords):
            tags.add(tag)
    return sorted(tags)

def extract_keywords_from_job_desc(job_desc):
    job_desc_lower = job_desc.lower()
    # Sort skills by length (longest first) to match multi-word skills first
    sorted_skills = sorted(SKILLS, key=lambda x: -len(x))
    found = set()
    for skill in sorted_skills:
        if skill in job_desc_lower:
            found.add(skill.title())
    return sorted(found)

@app.route('/', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        job_desc = request.form.get('job_desc', '').strip()
        resume_text = request.form.get('resume_text', '').strip()
        uploaded_files = []
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        for file in request.files.getlist('resumes'):
            if file and file.filename:
                save_path = os.path.join(upload_dir, file.filename)
                file.save(save_path)
                uploaded_files.append({
                    'filename': file.filename,
                    'filepath': save_path,
                    'mimetype': file.mimetype
                })
        print('DEBUG: Received job_desc:', repr(job_desc))
        print('DEBUG: Received resume_text:', repr(resume_text))
        print('DEBUG: Received uploaded_files:', [f['filename'] for f in uploaded_files])
        session['job_desc'] = job_desc
        session['resume_text'] = resume_text
        # Only store filenames, not file contents, in session
        session['uploaded_files'] = [{'filename': f['filename'], 'filepath': f['filepath'], 'mimetype': f['mimetype']} for f in uploaded_files]
        print('DEBUG: Session set for job_desc:', repr(session.get('job_desc')))
        print('DEBUG: Session set for resume_text:', repr(session.get('resume_text')))
        print('DEBUG: Session set for uploaded_files:', [f['filename'] for f in session.get('uploaded_files', [])])
        return redirect(url_for('dashboard'))
    return render_template('submit.html')

# --- Dashboard Route Update ---
@app.route('/dashboard', methods=['GET'])
def dashboard():
    results = []
    job_description = session.get('job_desc', '')
    resume_text = session.get('resume_text', '')
    uploaded_files = session.get('uploaded_files', [])
    print('DEBUG: job_description:', repr(job_description))
    print('DEBUG: resume_text:', repr(resume_text))
    print('DEBUG: uploaded_files:', [f['filename'] for f in uploaded_files])
    # Extract technical keywords from job description for keyword cloud
    keywords = extract_keywords_from_job_desc(job_description)
    shortlisted_count = 0
    rejected_count = 0
    tfidf = model.named_steps['tfidf']
    cleaned_job_desc = clean_resume(job_description) if job_description else ''
    job_vec = tfidf.transform([cleaned_job_desc]) if cleaned_job_desc else None

    def get_top_keywords(resume_vec, job_vec, top_n=5):
        feature_names = tfidf.get_feature_names_out()
        import numpy as np
        scores = (resume_vec.multiply(job_vec)).toarray().flatten()
        top_indices = np.argsort(scores)[::-1]
        filtered = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0 and feature_names[i] not in CUSTOM_STOPWORDS]
        if not filtered:
            resume_scores = resume_vec.toarray().flatten()
            tfidf_indices = np.argsort(resume_scores)[::-1]
            filtered = [(feature_names[i], resume_scores[i]) for i in tfidf_indices if resume_scores[i] > 0 and feature_names[i] not in CUSTOM_STOPWORDS]
        if not filtered:
            resume_scores = resume_vec.toarray().flatten()
            tfidf_indices = np.argsort(resume_scores)[::-1]
            filtered = [(feature_names[i], resume_scores[i]) for i in tfidf_indices if resume_scores[i] > 0]
        return [term for term, _ in filtered[:top_n]]

    def enrich_resume_analysis(text):
        # Extract skills, years, education, certifications
        skills = extract_skills(text)
        years_experience = extract_years_experience(text)
        education = extract_education(text)
        # Simple cert detection
        has_cert = 1 if "certified" in text.lower() or "certificate" in text.lower() else 0
        return {
            'skills': skills,
            'years_experience': years_experience,
            'education': education,
            'has_cert': has_cert
        }

    # Process uploaded files
    for file_info in uploaded_files:
        filename = file_info['filename']
        filepath = file_info['filepath']
        mimetype = file_info['mimetype']
        with open(filepath, 'rb') as f:
            file_obj = FileLikeWithAttrs(f.read(), filename, mimetype)
        resume_text_content = extract_text_from_file(file_obj)
        cleaned = clean_resume(resume_text_content)
        skills_score = 0.0
        top_keywords = []
        if job_vec is not None:
            resume_vec = tfidf.transform([cleaned])
            skills_score = cosine_similarity(resume_vec, job_vec)[0][0]
            top_keywords = get_top_keywords(resume_vec, job_vec)
        analysis = enrich_resume_analysis(resume_text_content)
        try:
            exp_years = int(analysis['years_experience']) if analysis['years_experience'] else 0
        except:
            exp_years = 0
        has_degree = bool(analysis['education'])
        has_cert = analysis['has_cert']
        # Weighted score components
        exp_score = min(exp_years / 10, 1)
        edu_score = 1 if has_degree else 0
        cert_score = 1 if has_cert else 0
        score = (
            0.5 * skills_score +
            0.2 * exp_score +
            0.2 * edu_score +
            0.1 * cert_score
        ) * 100
        # Feedback with all percentages
        feedback = [
            f"Skills: {skills_score*100:.1f}% (50%)",
            f"Experience: {exp_score*100:.1f}% (20%)",
            f"Education: {edu_score*100:.1f}% (20%)",
            f"Certifications: {cert_score*100:.1f}% (10%)"
        ]
        if not has_degree:
            feedback.append("Missing relevant degree")
        if exp_years < 2:
            feedback.append("Low experience")
        if skills_score < 0.3:
            feedback.append("Low skill match")
        # Shortlist if 'python' is in skills
        skills_list = [s.strip().lower() for s in analysis['skills'].split(',')]
        if 'python' in skills_list:
            status = "Shortlisted"
        else:
            status = "Shortlisted" if score >= 60 else "Rejected"
        results.append({
            "Candidate": filename,
            "Education": analysis['education'],
            "Experience": f"{exp_years} years" if exp_years else "",
            "Score": int(score),
            "Status": status,
            "Top Skills": ', '.join(top_keywords),
            "Feedback": '; '.join(feedback),
            "Download": f"/download_resume/{filename}"
        })

    # Process pasted resume text
    if resume_text:
        cleaned = clean_resume(resume_text)
        skills_score = 0.0
        top_keywords = []
        if job_vec is not None:
            resume_vec = tfidf.transform([cleaned])
            skills_score = cosine_similarity(resume_vec, job_vec)[0][0]
            top_keywords = get_top_keywords(resume_vec, job_vec)
        analysis = enrich_resume_analysis(resume_text)
        try:
            exp_years = int(analysis['years_experience']) if analysis['years_experience'] else 0
        except:
            exp_years = 0
        has_degree = bool(analysis['education'])
        has_cert = analysis['has_cert']
        exp_score = min(exp_years / 10, 1)
        edu_score = 1 if has_degree else 0
        cert_score = 1 if has_cert else 0
        score = (
            0.5 * skills_score +
            0.2 * exp_score +
            0.2 * edu_score +
            0.1 * cert_score
        ) * 100
        feedback = [
            f"Skills: {skills_score*100:.1f}% (50%)",
            f"Experience: {exp_score*100:.1f}% (20%)",
            f"Education: {edu_score*100:.1f}% (20%)",
            f"Certifications: {cert_score*100:.1f}% (10%)"
        ]
        if not has_degree:
            feedback.append("Missing relevant degree")
        if exp_years < 2:
            feedback.append("Low experience")
        if skills_score < 0.3:
            feedback.append("Low skill match")
        # Shortlist if 'python' is in skills
        skills_list = [s.strip().lower() for s in analysis['skills'].split(',')]
        if 'python' in skills_list:
            status = "Shortlisted"
        else:
            status = "Shortlisted" if score >= 60 else "Rejected"
        results.append({
            "Candidate": "Pasted Resume",
            "Education": analysis['education'],
            "Experience": f"{exp_years} years" if exp_years else "",
            "Score": int(score),
            "Status": status,
            "Top Skills": ', '.join(top_keywords),
            "Feedback": '; '.join(feedback),
            "Download": None
        })

    results.sort(key=lambda x: x["Score"], reverse=True)
    shortlisted_count = sum(1 for r in results if r["Status"] == "Shortlisted")
    rejected_count = sum(1 for r in results if r["Status"] == "Rejected")
    # Store shortlisted results for export
    session['shortlisted_results'] = [
        [r["Candidate"], "Shortlisted ✅", r["Score"], r["Top Skills"].split(", ")]
        for r in results if r["Status"] == "Shortlisted"
    ]
    print('DEBUG: results:', results)
    return render_template(
        "dashboard.html",
        results=results,
        job_description=job_description,
        keywords=keywords,
        shortlisted_count=shortlisted_count,
        rejected_count=rejected_count
    )

@app.route('/export', methods=['POST'])
def export():
    results = session.get('shortlisted_results', [])
    # Only export resumes that are Shortlisted
    shortlisted = [r for r in results if r[1] == 'Shortlisted ✅']
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Resume', 'Status', 'Score', 'Top Keywords'])
    for row in shortlisted:
        writer.writerow([row[0], row[1], f"{row[2]:.2f}", ', '.join(row[3])])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='shortlisted_resumes.csv')

@app.route('/api/results', methods=['GET'])
def api_results():
    results = []
    job_description = session.get('job_desc', '')
    resume_text = session.get('resume_text', '')
    uploaded_files = session.get('uploaded_files', [])
    tfidf = model.named_steps['tfidf']
    cleaned_job_desc = clean_resume(job_description) if job_description else ''
    job_vec = tfidf.transform([cleaned_job_desc]) if cleaned_job_desc else None

    def get_top_keywords(resume_vec, job_vec, top_n=5):
        feature_names = tfidf.get_feature_names_out()
        import numpy as np
        scores = (resume_vec.multiply(job_vec)).toarray().flatten()
        top_indices = np.argsort(scores)[::-1]
        filtered = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0 and feature_names[i] not in CUSTOM_STOPWORDS]
        if not filtered:
            resume_scores = resume_vec.toarray().flatten()
            tfidf_indices = np.argsort(resume_scores)[::-1]
            filtered = [(feature_names[i], resume_scores[i]) for i in tfidf_indices if resume_scores[i] > 0 and feature_names[i] not in CUSTOM_STOPWORDS]
        if not filtered:
            resume_scores = resume_vec.toarray().flatten()
            tfidf_indices = np.argsort(resume_scores)[::-1]
            filtered = [(feature_names[i], resume_scores[i]) for i in tfidf_indices if resume_scores[i] > 0]
        return [term for term, _ in filtered[:top_n]]

    # Process uploaded files
    for file_info in uploaded_files:
        filename = file_info['filename']
        filepath = file_info['filepath']
        mimetype = file_info['mimetype']
        with open(filepath, 'rb') as f:
            file_obj = FileLikeWithAttrs(f.read(), filename, mimetype)
        resume_text_content = extract_text_from_file(file_obj)
        cleaned = clean_resume(resume_text_content)
        score = 0.0
        top_keywords = []
        if job_vec is not None:
            resume_vec = tfidf.transform([cleaned])
            score = cosine_similarity(resume_vec, job_vec)[0][0]
            top_keywords = get_top_keywords(resume_vec, job_vec)
        status = "Shortlisted" if score >= 0.6 else "Rejected"
        results.append({
            "candidate": filename,
            "score": int(score * 100),
            "status": status,
            "top_keywords": top_keywords
        })

    # Process pasted resume text
    if resume_text:
        cleaned = clean_resume(resume_text)
        score = 0.0
        top_keywords = []
        if job_vec is not None:
            resume_vec = tfidf.transform([cleaned])
            score = cosine_similarity(resume_vec, job_vec)[0][0]
            top_keywords = get_top_keywords(resume_vec, job_vec)
        status = "Shortlisted" if score >= 0.6 else "Rejected"
        results.append({
            "candidate": "Pasted Resume",
            "score": int(score * 100),
            "status": status,
            "top_keywords": top_keywords
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False)
