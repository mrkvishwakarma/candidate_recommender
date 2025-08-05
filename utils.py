from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import numpy as np
import pdfplumber
import os
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_sbert_model():
    """Load the Sentence-Transformer model for generating embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts, model):
    """Generate embeddings for a list of texts."""
    return model.encode(texts)


@st.cache_resource
def get_groq_client(api_key):
    """Initialize and return a Groq client."""
    if not api_key:
        raise ValueError("Groq API key is not set. Please add it to your .env file.")
    return Groq(api_key=api_key)


def calculate_similarity(job_embedding, resume_embeddings):
    """
    Compute cosine similarity between the job description embedding
    and each resume embedding.
    """
    job_embedding_reshaped = job_embedding.reshape(1, -1)
    return cosine_similarity(job_embedding_reshaped, resume_embeddings)[0]


def generate_summary(jd_text, resume_text):
    """
    Generate an AI-powered summary for why a candidate is a good fit
    using a Groq LLM.
    """
    prompt = f"""
    You are an expert recruiter. 
    Analyze the following job description and candidate resume to explain in 3 sentences why this person is a great fit for the role. Focus on matching keywords, skills, experience and be precise.

    Job Description:
    {jd_text}

    Resume:
    {resume_text}

    Summary:
    """

    client = get_groq_client(GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

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
    file_extension = file.name.split('.')[-1].lower()

    if file_extension == 'txt':
        return file.getvalue().decode("utf-8")
    elif file_extension == 'pdf':
        return read_pdf_file(file)
    else:
        return f"Unsupported file type: {file_extension}"


def extract_key_sections_from_resume( resume_text):
    """
    Summarizes the full resume into key sections using guided prompts.
    Returns a dictionary with 3 summarized parts:
        - Qualifications and Education
        - Skills and Certifications
        - Projects and Work Experience
    """

    prompts = {
        "Qualifications and Education": (
            f"""
            You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
            Given the following resume text:
            --------------------
            {resume_text}
            --------------------
            Directly list all the qualifications or education of the candidate from the above given resume.
            """
        ),

        "Skills and Certifications": (
            f"""
               You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
               Given the following resume text:
               --------------------
               {resume_text}
               --------------------
               Directly list all the Skills and certifications of the candidate from the above given resume.
               """
        ),

        "Projects and Work Experience": (
            f"""
               You are an ATS (Applicant Tracking System) that extracts relevant information from resumes
               Given the following resume text:
               --------------------
               {resume_text}
               --------------------
               Directly list all the Projects and Work Experience of the candidate from the above given resume.
               """
        ),
    }

    section_summaries = {}

    client = get_groq_client(GROQ_API_KEY)

    for section, prompt in prompts.items():
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=250,
            )
            section_summaries[section] = response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    return section_summaries

def extract_key_sections_from_jd(jd_text):
    """
    Extracts structured sections from a job description using guided prompts.
    Returns a dictionary with:
        - Role Overview
        - Required Skills and Technologies
        - Qualifications and Education
        - Responsibilities and Duties
    """

    prompts = {
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

    section_summaries = {}

    client = get_groq_client(GROQ_API_KEY)

    for section, prompt in prompts.items():
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=250,
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