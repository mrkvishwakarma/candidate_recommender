from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import numpy as np
import pdfplumber
import os
import io
import zipfile

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_sbert_model():
    """Load the Sentence-Transformer model for generating embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_groq_client(api_key):
    """Initialize and return a Groq client."""
    if not api_key:
        raise ValueError("Groq API key is not set. Please add it to your .env file.")
    return Groq(api_key=api_key)



def read_pdf_file(file):
    """Reads the content of a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"An unexpected error occurred while processing PDF with pdfplumber: {e}"


def read_text_file(file):
    """Reads the content of an uploaded file, handling both PDF and TXT."""
    file_ext = file.name.split('.')[-1].lower()

    if file_ext == 'txt':
        return file.getvalue().decode("utf-8")
    elif file_ext == 'pdf':
        return read_pdf_file(file)
    else:
        return f"Unsupported file type: {file_ext}"


import re


def extract_contact_info(resume_text):
    """
    Extracts name and email address from the resume.
    """
    email_match = re.search('\S+@\S+', resume_text)

    return email_match.group(0) if email_match else ''


def get_resume_summary_prompts(resume_text):
    """
    Generates and returns a dictionary of prompts for extracting structured
    information from a resume.
    """

    resume_prompts = {
        "Qualifications and Education": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
            Given the following resume text:
            --------------------
            {resume_text}
            --------------------
            Directly extract all the sections which indicates qualifications or education of the candidate from the above given resume.
            """
        ),

        "Skills and Certifications": (
            f"""
               You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
               Given the following resume text:
               --------------------
               {resume_text}
               --------------------
               Directly extract all the sections which indicates Skills and certifications of the candidate from the above given resume.
               """
        ),

        "Projects and Work Experience": (
            f"""
               You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
               Given the following resume text:
               --------------------
               {resume_text}
               --------------------
               Directly extract all the sections which indicates Projects and Work Experience of the candidate from the above given resume.
               """
        ),
    }

    return resume_prompts


def get_jd_summary_prompts(jd_text):
    """
    Generates and returns a dictionary of prompts for extracting structured
    information from a job description.
    """

    jd_prompts = {
        "About Company": (
            f"""
                You are an ATS (Applicant Tracking System) that extracts structured information from job descriptions.
                Given the following job description text:
                --------------------
                {jd_text}
                --------------------
                Directly state the company information given in the above job description.
                """
        ),
        "Role Overview": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts structured information from job descriptions.
            Given the following job description text:
            --------------------
            {jd_text}
            --------------------
            Directly state the role overview  given in the above job description.
            """
        ),

        "Required Skills and Technologies": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts structured information from job descriptions.
            Given the following job description text:
            --------------------
            {jd_text}
            --------------------
            Directly list all the required Skills and technologies given in the above job description.
            """
        ),

        "Qualifications and Education": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts structured information from job descriptions.
            Given the following job description text:
            --------------------
            {jd_text}
            --------------------
            Directly list all the required qualifications or education given in the above job description.
            """
        ),

        "Responsibilities and Duties": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts structured information from job descriptions.
            Given the following job description text:
            --------------------
            {jd_text}
            --------------------
            Directly list all the responsibilities and duties given in the above job description.
            """
        ),
    }

    return jd_prompts


def generate_summary(prompts):
    """
    Extracts structured sections from a job description using guided prompts.
    Returns a dictionary with extracted information.
    """

    section_summaries = {}
    client = get_groq_client(GROQ_API_KEY)

    for section, prompt in prompts.items():
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            section_summaries[section] = response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    return section_summaries


def compute_section_similarity(resume_sections, jd_sections):
    """
    Computes cosine similarity between relevant resume and job description sections.

    Args:
        resume_sections: dict with resume section names and text
        job_sections: dict with job section names and text

    Returns:
        dict of section-wise similarity and overall average similarity
    """
    model = load_sbert_model()

    # Define section mappings (resume → job description)
    section_pairs = {
        "Qualifications and Education": "Qualifications and Education",
        "Skills and Certifications": "Required Skills and Technologies",
        "Projects and Work Experience": "Responsibilities and Duties",
    }

    similarities = {}

    for resume_key, job_key in section_pairs.items():
        resume_text = resume_sections.get(resume_key, "")
        job_text = jd_sections.get(job_key, "")

        if not resume_text or not job_text:
            similarities[f"{resume_key} ↔ {job_key}"] = 0.0
            continue

        # Generate embeddings
        embeddings = model.encode([resume_text, job_text])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        similarities[f"{job_key}"] = sim_score

    # Calculate average similarity
    average_similarity = np.mean(list(similarities.values()))
    similarities["Overall Score"] = average_similarity

    return similarities


def get_summary_prompt(jd_text, resume_text):
    """
    Generate an AI-powered summary for why a candidate is a good fit
    using a Groq LLM.
    """
    summary_prompt = {
            "Summary": (
                f"""
                You are an expert recruiter. 
                Analyze the following job description and candidate resume to explain in 3 sentences why this person is a great fit for the role. Focus on matching keywords, skills, experience and be precise.
            
                Job Description:
                {jd_text}
            
                Resume:
                {resume_text}
            
                Summary:
                """
            ),
    }

    return summary_prompt


def create_zip_file_for_resumes(resumes):
    """
    Creates a zip file of top candidates resumes
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, resume_data in enumerate(resumes):
            score = f"{resume_data['section_scores']['Overall Score']:.2f}"
            email = resume_data['email']
            name = resume_data['name']

            filename = f"{score}_{email}_{name}.pdf"

            zip_file.writestr(filename, resume_data.get('text', ''))

    return zip_buffer.getvalue()