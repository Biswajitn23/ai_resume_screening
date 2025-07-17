import pandas as pd
import os

try:
    df = pd.read_csv("dataset/resumes.csv")
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import re
from sklearn.ensemble import RandomForestClassifier

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to clean text using SpaCy
def spacy_clean(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Function to extract NER features and concatenate with cleaned text
ENTITY_LABELS = {'ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'SKILL', 'DEGREE'}
def enrich_with_ner(text):
    doc = nlp(text)
    # Cleaned text
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    cleaned = " ".join(tokens)
    # Extract NER entities (skills, degrees, orgs, etc.)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ENTITY_LABELS]
    enriched = cleaned + " " + " ".join(entities)
    return enriched

# Feature extraction functions
SKILL_LIST = [
    'python', 'java', 'c++', 'aws', 'kubernetes', 'docker', 'django', 'rest', 'api', 'tensorflow', 'pytorch',
    'sql', 'pandas', 'scikit-learn', 'react', 'flutter', 'firebase', 'nlp', 'deep learning', 'machine learning',
    'cloud', 'google cloud', 'gcp', 'ci/cd', 'devops', 'security', 'penetration testing', 'incident response',
    'hr', 'payroll', 'crm', 'seo', 'social media', 'campaign management', 'data entry', 'data visualization',
    'scikit-learn', 'database', 'mobile apps', 'react native', 'office coordination', 'inventory management',
    'sales', 'customer service', 'call center', 'lead generation', 'billing', 'conflict resolution', 'pos systems',
    'onboarding', 'performance management', 'hris systems', 'employee engagement', 'market research', 'client acquisition',
    'cloud infrastructure', 'google research', 'published papers', 'ai', 'computer vision', 'reinforcement learning',
    'data science', 'data scientist', 'data analyst', 'software engineer', 'software developer', 'backend developer',
    'security engineer', 'marketing manager', 'recruitment specialist', 'retail store manager', 'office administrator',
    'hr coordinator', 'customer service representative', 'sales executive', 'machine learning engineer', 'ai researcher',
    'devops engineer', 'data scientist', 'security engineer', 'call center agent', 'hr coordinator', 'office administrator'
]
EDU_LIST = [
    'bachelor', 'master', 'phd', 'msc', 'bsc', 'mba', 'ba', 'ma', 'be', 'me', 'b.tech', 'm.tech', 'degree', 'internship', 'published papers'
]

def extract_skills(text):
    text = text.lower()
    return ', '.join(sorted(set([skill for skill in SKILL_LIST if skill in text])))

def extract_education(text):
    text = text.lower()
    return ', '.join(sorted(set([edu for edu in EDU_LIST if edu in text])))

def extract_experience(text):
    # Look for years of experience
    match = re.search(r'(\d+)\+?\s*years?', text.lower())
    if match:
        return match.group(1) + ' years'
    return ''

# Load dataset
print("File exists:", os.path.exists("dataset/resumes.csv"))
print("File size:", os.path.getsize("dataset/resumes.csv"))
df = pd.read_csv("dataset/resumes.csv")

# Extract features
print('Extracting features...')
df['skills'] = df['resume_text'].apply(extract_skills)
df['education'] = df['resume_text'].apply(extract_education)
df['experience'] = df['resume_text'].apply(extract_experience)
df['features'] = df['skills'] + ' ' + df['education'] + ' ' + df['experience']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['features'], df['label'], test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
print('Training model...')
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model/model.pkl')
print('Model saved to model/model.pkl')
