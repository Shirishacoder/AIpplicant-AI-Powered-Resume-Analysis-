# AIpplicant - AI-Powered Resume Analysis System
# Main application file (app.py)

from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import spacy
import pymongo
from pymongo import MongoClient
import PyPDF2
import docx
import re
from datetime import datetime
from bson import ObjectId
import json
from typing import Dict, List, Tuple
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    logger.error("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['aipplicant']
    resumes_collection = db['resumes']
    jobs_collection = db['jobs']
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    client = None
    db = None

class ResumeParser:
    def _init_(self, nlp_model):
        self.nlp = nlp_model
        
        # Predefined skill categories and keywords
        self.skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 
                          'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express', 
                         'laravel', 'rails', 'asp.net', 'bootstrap', 'jquery', 'tensorflow', 'pytorch'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis', 'cassandra', 
                        'elasticsearch', 'neo4j', 'dynamodb'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ansible'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'photoshop', 'illustrator', 'figma'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 
                          'creative', 'adaptable', 'organized', 'detail-oriented']
        }
        
        # Education patterns
        self.education_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|associate|diploma|certificate)\b.*?(?:in|of)\s+([^\n.]+)',
            r'\b(computer science|engineering|mathematics|physics|chemistry|biology|business|marketing|finance)\b',
            r'\b(university|college|institute|school)\s+of\s+([^\n.]+)',
            r'\b([a-z\s]+)\s+(?:university|college|institute)\b'
        ]
        
        # Experience patterns
        self.experience_patterns = [
            r'(\d{1,2}[\+]?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:from|since)\s*(\d{4})\s*(?:to|[-–—])\s*(?:(\d{4})|present|current)',
            r'(\d{4})\s*[-–—]\s*(?:(\d{4})|present|current)'
        ]

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            return ""

    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Phone extraction
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phone'] = ''.join(phones[0]) if phones else None
        
        # LinkedIn extraction
        linkedin_pattern = r'linkedin\.com/in/([A-Za-z0-9-]+)'
        linkedin = re.search(linkedin_pattern, text)
        contact_info['linkedin'] = linkedin.group(0) if linkedin else None
        
        # GitHub extraction
        github_pattern = r'github\.com/([A-Za-z0-9-]+)'
        github = re.search(github_pattern, text)
        contact_info['github'] = github.group(0) if github else None
        
        return contact_info

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        extracted_skills = {}
        
        for category, skills in self.skill_keywords.items():
            found_skills = []
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
            extracted_skills[category] = found_skills
        
        # Extract additional skills using NLP
        if self.nlp:
            doc = self.nlp(text)
            additional_skills = []
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'TECH']:
                    additional_skills.append(ent.text)
            extracted_skills['additional'] = list(set(additional_skills))
        
        return extracted_skills

    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume text"""
        education = []
        
        for pattern in self.education_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education.append({
                    'text': match.group(0),
                    'field': match.group(1) if match.lastindex and match.lastindex >= 1 else None
                })
        
        return education

    def extract_experience(self, text: str) -> Dict:
        """Extract experience information from resume text"""
        experience = {'years': 0, 'positions': []}
        
        # Extract years of experience
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'years' in pattern or 'yrs' in pattern:
                    try:
                        years = int(match.group(1).replace('+', ''))
                        experience['years'] = max(experience['years'], years)
                    except (ValueError, AttributeError):
                        continue
                else:
                    # Calculate years from date ranges
                    try:
                        start_year = int(match.group(1))
                        end_year = int(match.group(2)) if match.group(2) else datetime.now().year
                        years = end_year - start_year
                        experience['years'] = max(experience['years'], years)
                    except (ValueError, AttributeError, TypeError):
                        continue
        
        # Extract job positions (simplified)
        position_patterns = [
            r'\b(?:senior|junior|lead|principal|chief)?\s*(?:software|web|mobile|data|machine learning|ai|devops|full stack)?\s*(?:developer|engineer|analyst|scientist|manager|director|architect)\b'
        ]
        
        for pattern in position_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                experience['positions'].append(match.group(0).strip())
        
        return experience

    def parse_resume(self, file_path: str) -> Dict:
        """Main method to parse resume and extract all information"""
        # Determine file type and extract text
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Extract all information
        parsed_data = {
            'raw_text': text,
            'contact_info': self.extract_contact_info(text),
            'skills': self.extract_skills(text),
            'education': self.extract_education(text),
            'experience': self.extract_experience(text),
            'extracted_at': datetime.now(),
            'file_name': os.path.basename(file_path)
        }
        
        return parsed_data

class CandidateRanker:
    def _init_(self):
        pass
    
    def calculate_skill_match(self, candidate_skills: Dict, required_skills: List[str]) -> float:
        """Calculate skill match percentage"""
        if not required_skills:
            return 0.0
        
        # Flatten candidate skills
        all_candidate_skills = []
        for skill_list in candidate_skills.values():
            all_candidate_skills.extend([skill.lower() for skill in skill_list])
        
        # Calculate matches
        matches = 0
        for required_skill in required_skills:
            if required_skill.lower() in all_candidate_skills:
                matches += 1
        
        return (matches / len(required_skills)) * 100

    def calculate_experience_score(self, candidate_experience: int, required_experience: int) -> float:
        """Calculate experience match score"""
        if required_experience == 0:
            return 100.0
        
        if candidate_experience >= required_experience:
            return 100.0
        else:
            return (candidate_experience / required_experience) * 100 * 0.8  # Penalty for insufficient experience

    def rank_candidates(self, candidates: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Rank candidates based on job requirements"""
        required_skills = job_requirements.get('skills', [])
        required_experience = job_requirements.get('experience_years', 0)
        
        for candidate in candidates:
            skill_score = self.calculate_skill_match(candidate['skills'], required_skills)
            experience_score = self.calculate_experience_score(
                candidate['experience']['years'], 
                required_experience
            )
            
            # Weighted overall score
            overall_score = (skill_score * 0.7) + (experience_score * 0.3)
            
            candidate['scores'] = {
                'skill_match': skill_score,
                'experience_match': experience_score,
                'overall_score': overall_score
            }
        
        # Sort by overall score (descending)
        return sorted(candidates, key=lambda x: x['scores']['overall_score'], reverse=True)

# Initialize components
resume_parser = ResumeParser() if nlp else None
candidate_ranker = CandidateRanker()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Upload and parse resume"""
    if not resume_parser:
        return jsonify({'error': 'NLP model not loaded'}), 500
    
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file uploaded'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Parse resume
            parsed_data = resume_parser.parse_resume(file_path)
            
            # Save to database
            if db:
                result = resumes_collection.insert_one(parsed_data)
                parsed_data['_id'] = str(result.inserted_id)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            # Remove raw text from response for brevity
            response_data = parsed_data.copy()
            response_data.pop('raw_text', None)
            
            return jsonify({
                'success': True,
                'data': response_data,
                'message': 'Resume parsed successfully'
            })
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Error parsing resume: {str(e)}'}), 500

@app.route('/create_job', methods=['POST'])
def create_job():
    """Create a new job posting with requirements"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    job_data = {
        'title': data.get('title'),
        'description': data.get('description'),
        'skills': data.get('skills', []),
        'experience_years': data.get('experience_years', 0),
        'education_requirements': data.get('education_requirements', []),
        'created_at': datetime.now()
    }
    
    if db:
        result = jobs_collection.insert_one(job_data)
        job_data['_id'] = str(result.inserted_id)
    
    return jsonify({
        'success': True,
        'data': job_data,
        'message': 'Job created successfully'
    })

@app.route('/rank_candidates/<job_id>')
def rank_candidates_for_job(job_id):
    """Rank all candidates for a specific job"""
    if not db:
        return jsonify({'error': 'Database not connected'}), 500
    
    try:
        # Get job requirements
        job = jobs_collection.find_one({'_id': ObjectId(job_id)})
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Get all candidates
        candidates = list(resumes_collection.find())
        
        if not candidates:
            return jsonify({'error': 'No candidates found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        for candidate in candidates:
            candidate['_id'] = str(candidate['_id'])
            candidate.pop('raw_text', None)  # Remove raw text for response
        
        # Rank candidates
        ranked_candidates = candidate_ranker.rank_candidates(candidates, job)
        
        return jsonify({
            'success': True,
            'job': {
                '_id': str(job['_id']),
                'title': job['title'],
                'skills': job['skills'],
                'experience_years': job['experience_years']
            },
            'candidates': ranked_candidates,
            'total_candidates': len(ranked_candidates)
        })
        
    except Exception as e:
        logger.error(f"Error ranking candidates: {e}")
        return jsonify({'error': f'Error ranking candidates: {str(e)}'}), 500

@app.route('/candidates')
def get_all_candidates():
    """Get all parsed candidates"""
    if not db:
        return jsonify({'error': 'Database not connected'}), 500
    
    try:
        candidates = list(resumes_collection.find())
        
        # Convert ObjectId to string and remove raw text
        for candidate in candidates:
            candidate['_id'] = str(candidate['_id'])
            candidate.pop('raw_text', None)
        
        return jsonify({
            'success': True,
            'candidates': candidates,
            'total': len(candidates)
        })
        
    except Exception as e:
        logger.error(f"Error fetching candidates: {e}")
        return jsonify({'error': f'Error fetching candidates: {str(e)}'}), 500

@app.route('/jobs')
def get_all_jobs():
    """Get all job postings"""
    if not db:
        return jsonify({'error': 'Database not connected'}), 500
    
    try:
        jobs = list(jobs_collection.find())
        
        # Convert ObjectId to string
        for job in jobs:
            job['_id'] = str(job['_id'])
        
        return jsonify({
            'success': True,
            'jobs': jobs,
            'total': len(jobs)
        })
        
    except Exception as e:
        logger.error(f"Error fetching jobs: {e}")
        return jsonify({'error': f'Error fetching jobs: {str(e)}'}), 500

@app.route('/candidate/<candidate_id>')
def get_candidate_details(candidate_id):
    """Get detailed information about a specific candidate"""
    if not db:
        return jsonify({'error': 'Database not connected'}), 500
    
    try:
        candidate = resumes_collection.find_one({'_id': ObjectId(candidate_id)})
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        candidate['_id'] = str(candidate['_id'])
        
        return jsonify({
            'success': True,
            'candidate': candidate
        })
        
    except Exception as e:
        logger.error(f"Error fetching candidate details: {e}")
        return jsonify({'error': f'Error fetching candidate details: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)