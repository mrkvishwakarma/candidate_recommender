from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import streamlit as st
import pdfplumber
import re


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


def generate_summary(client, job_description, resume_text):
    """
    Generate an AI-powered summary for why a candidate is a good fit
    using a Groq LLM.
    """
    prompt = f"""
    You are an expert recruiter. 
    Analyze the following job description and candidate resume to explain in 3 sentences why this person is a great fit for the role. Focus on matching keywords, skills, experience and be precise.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Summary:
    """
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

def read_pdf_text(file):
    """Reads the content of a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"An unexpected error occurred while processing PDF with pdfplumber: {e}"


def read_file_text(file):
    """Reads the content of an uploaded file, handling both PDF and TXT."""
    file_extension = file.name.split('.')[-1].lower()

    if file_extension == 'txt':
        return file.getvalue().decode("utf-8")
    elif file_extension == 'pdf':
        return read_pdf_text(file)
    else:
        return f"Unsupported file type: {file_extension}"


def extract_key_sections(text):
    """
    Extracts key sections (Summary, Experience, Skills, Projects, Education)
    from a resume using regex.
    """
    pattern = re.compile(
        r'(Summary|Professional Summary|Experience|Work Experience|Skills|Technical Skills|Projects|Education|Responsibilities|Qualifications|Requirements|Certifications)\s*(.*?)'
        r'(?=\b(Summary|Professional Summary|Experience|Work Experience|Skills|Technical Skills|Projects|Education|Responsibilities|Qualifications|Requirements|Certifications)\b|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    sections = {
        'content': ''
    }

    matches = pattern.finditer(text)

    for match in matches:
        sections['content'] += match.group(2).strip() + ' '

    if not sections['content'].strip():
        return text

    return sections['content'].strip()