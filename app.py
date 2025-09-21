import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, session, url_for, flash
import fitz  # PyMuPDF
import spacy
import language_tool_python
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------- FLASK SETUP ---------------- #
app = Flask(__name__)
app.secret_key = 'super_secret_key'

# ---------------- NLP SETUP ---------------- #
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

# ---------------- TRANSFORMERS SETUP ---------------- #
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------- KEYWORDS & SECTIONS ---------------- #
REQUIRED_KEYWORDS = {
    "python", "java", "c", "c++", "html", "css", "javascript", "php", "sql", "mysql",
    "mongodb", "react", "angular", "flutter", "firebase", "node.js", "django", "flask",
    "data structures", "algorithms", "nlp", "data science", "excel", "github", "linux",
    "teamwork", "communication", "problem solving", "leadership"
}
EXPECTED_SECTIONS = ["objective", "education", "projects", "skills", "experience", "certifications", "internship", "declaration"]

# ---------------- DEMO USERS ---------------- #
users = {
    "student@gmail.com": {"password": "123", "role": "student"},
    "recruiter@gmail.com": {"password": "123", "role": "recruiter"}
}

# ---------------- HELPER FUNCTIONS ---------------- #
def clean_resume_text(text, max_length=1500):
    lines = text.splitlines()
    seen = set()
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower and line_lower not in seen:
            seen.add(line_lower)
            cleaned_lines.append(line)
        if len(" ".join(cleaned_lines)) > max_length:
            break
    return "\n".join(cleaned_lines)

def extract_text_from_pdf(file, max_chars=5000):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
        if len(text) > max_chars:
            text = text[:max_chars]
            break
    return text

def score_resume(text):
    text_lower = text.lower()
    doc = nlp(text_lower)
    found_keywords = set()
    for keyword in REQUIRED_KEYWORDS:
        if keyword in text_lower:
            found_keywords.add(keyword)
        else:
            for chunk in doc.noun_chunks:
                if keyword in chunk.text.lower():
                    found_keywords.add(keyword)
                    break
    missing_keywords = REQUIRED_KEYWORDS - found_keywords
    match_count = len(found_keywords)
    total_keywords = len(REQUIRED_KEYWORDS)
    score = (match_count / total_keywords) * 100
    return round(score, 2), found_keywords, missing_keywords

def check_sections(text):
    text_lower = text.lower()
    found = [s for s in EXPECTED_SECTIONS if s in text_lower]
    missing = [s for s in EXPECTED_SECTIONS if s not in found]
    return found, missing

def check_grammar(text):
    matches = tool.check(text)
    filtered = []
    for match in matches:
        ctx = match.context.lower()
        if re.search(r"(https?://|\.com|@|github|linkedin|\.edu|\d{4})", ctx): continue
        if any(section in ctx for section in EXPECTED_SECTIONS): continue
        if match.context.strip().split()[0].istitle() and match.ruleId == "MORFOLOGIK_RULE_EN_US": continue
        filtered.append(match)
    suggestions = [{"message": m.message, "error": m.context} for m in filtered[:10]]
    return len(filtered), suggestions

# ---------------- GENERATIVE AI ---------------- #
def rewrite_resume(text):
    prompt = f"Rewrite this resume professionally and ATS-friendly:\n{text}\n"
    inputs = tokenizer(prompt[:1500], return_tensors="pt").to(device)
    outputs = model.generate(
    **inputs,
    max_new_tokens=180,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.5   # increase penalty
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_cover_letter(resume_text, job_description="", role="Software Developer Intern", company=""):
    prompt = f"""
Using the resume below, write a professional and ATS-friendly cover letter:

Resume:
{resume_text}

Role: {role}
Company: {company if company else '[Company Name]'}
Job Description:
{job_description}

Cover letter requirements:
- 3–5 paragraphs
- Introduce yourself and the role
- Highlight relevant skills, education, projects
- Explain why you want to join the company
- Conclude politely with a call to action
"""
    inputs = tokenizer(prompt[:1500], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    return render_template("landing.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)
        if user and user["password"] == password:
            session['user'] = email
            session['role'] = user["role"]
            return redirect(url_for(user["role"]))
        flash("Invalid credentials!")
    return render_template("login.html")

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        if email in users:
            flash("User already exists!")
        else:
            users[email] = {"password": password, "role": role}
            flash("Signup successful. Please login.")
            return redirect("/login")
    return render_template("signup.html")

@app.route('/student', methods=["GET", "POST"])
def student():
    if session.get("role") != "student":
        return redirect("/login")
    if request.method == "POST":
        file = request.files['resume']
        filename = file.filename.lower()
        if filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        else:
            return render_template("student.html", error="Only .txt and .pdf allowed")
        text = clean_resume_text(text)
        keyword_score, matched, missing = score_resume(text)
        grammar_errors, grammar_suggestions = check_grammar(text)
        sections_found, sections_missing = check_sections(text)
        grammar_score = 100 if grammar_errors == 0 else 90 if grammar_errors <= 3 else 75 if grammar_errors <= 7 else 60 if grammar_errors <= 10 else 50
        section_score = (len(sections_found) / len(EXPECTED_SECTIONS)) * 100
        final_score = round((keyword_score * 0.4) + (grammar_score * 0.25) + (section_score * 0.35), 2)
        role = request.form.get("role", "Software Developer Intern")
        company = request.form.get("company", "")
        job_description = request.form.get("job_description", "")
        try:
            rewritten_resume = rewrite_resume(text)
        except:
            rewritten_resume = "Error generating rewritten resume"
        try:
            ai_cover_letter = generate_cover_letter(text, job_description, role, company)
        except:
            ai_cover_letter = "Error generating cover letter"
        return render_template(
            "student.html",
            score=final_score,
            matched=matched,
            missing=missing,
            sections_found=sections_found,
            sections_missing=sections_missing,
            grammar_errors=grammar_errors,
            suggestions=grammar_suggestions,
            rewritten_resume=rewritten_resume,
            ai_cover_letter=ai_cover_letter,
            role=role,
            company=company,
            job_description=job_description
        )
    return render_template("student.html")
def process_resume(file, job_description, jd_keywords):
    filename = file.filename.lower()
    if filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    else:
        return None

    text = clean_resume_text(text)
    score, matched_keywords, missing_keywords = score_resume(text)
    # You can also compare with jd_keywords to see relevance
    jd_matches = jd_keywords.intersection(set(text.lower().split()))
    return {
        "filename": file.filename,
        "score": score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "jd_matches": jd_matches
    }

@app.route('/recruiter', methods=["GET", "POST"])
def recruiter():
    if session.get("role") != "recruiter":
        return redirect("/login")
    candidates = []
    if request.method == "POST":
        files = request.files.getlist("resumes")
        job_description = request.form.get("job_description", "").lower()
        jd_doc = nlp(job_description)
        jd_keywords = set([token.lemma_ for token in jd_doc if token.is_alpha and not token.is_stop])
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda f: process_resume(f, job_description, jd_keywords), files))
        candidates = [res for res in results if res]
        candidates.sort(key=lambda c: c['score'], reverse=True)
    return render_template("recruiter.html", candidates=candidates)


@app.route('/logout')
def logout():
    session.clear()
    return redirect("/")

# ---------------- RUN APP ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
