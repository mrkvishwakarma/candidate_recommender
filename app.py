import os
from dotenv import load_dotenv
from utils import *

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Candidate Recommendation Agent", layout="wide")

st.title("Candidate Recommendation Agent 🤖")
st.markdown("Enter a job description and upload a list of resumes to find the best candidates.")

embedding_method = st.radio(
    "Select Recommendation Method:",
    ('Over Complete Resume', 'Over SkillSets & Requirements'),
    horizontal=True,
    index=0
)

# Main input area
job_description = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (.txt or .pdf files)",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Find Top Candidates"):
    if not job_description:
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Processing resumes and finding top candidates..."):
            sbert_model = load_sbert_model()

            job_description_text = job_description

            if embedding_method == 'Over SkillSets & Requirements':
                job_description_text = extract_key_sections(job_description)

            job_embedding = get_embeddings([job_description_text], sbert_model)[0]

            resumes = []
            resume_texts = []
            for uploaded_file in uploaded_files:
                text = read_file_text(uploaded_file)
                if "unsupported" in text or "Failed to read" in text:
                    st.warning(f"Skipping {uploaded_file.name}: {text}")
                    continue
                resumes.append({"name": uploaded_file.name, "text": text})
                resume_texts.append(text)

            resumes = []
            resume_texts = []
            for uploaded_file in uploaded_files:
                full_text = read_file_text(uploaded_file)
                if "unsupported" in full_text or "Failed to read" in full_text:
                    st.warning(f"Skipping {uploaded_file.name}: {full_text}")
                    continue

                if embedding_method == 'Over SkillSets & Requirements':
                    processed_text = extract_key_sections(full_text)
                else:
                    processed_text = full_text

                resumes.append({"name": uploaded_file.name, "full_text": full_text, "processed_text": processed_text})
                resume_texts.append(processed_text)

            if not resume_texts:
                st.error("No valid resumes were processed.")
                st.session_state.results = None
            else:
                resume_embeddings = get_embeddings(resume_texts, sbert_model)
                similarity_scores = calculate_similarity(job_embedding, resume_embeddings)

                candidate_list = []
                for i, resume in enumerate(resumes):
                    candidate_list.append({
                        "name": resume["name"],
                        "score": similarity_scores[i],
                        "text": resume["full_text"]
                    })

                candidate_list.sort(key=lambda x: x["score"], reverse=True)

                st.session_state.results = candidate_list
                st.success("Analysis complete!")

if st.session_state.results:
    st.header("Top Candidate Recommendation")

    top_candidate = st.session_state.results[0]

    st.markdown(f"**1. {top_candidate['name']}**")
    st.markdown(f"**Similarity Score:** `{top_candidate['score']:.4f}`")

    try:
        groq_client = get_groq_client(GROQ_API_KEY)
        summary = generate_summary(groq_client, job_description, top_candidate['text'])
        st.write(summary)
    except Exception as e:
        st.error(f"Failed to connect to Groq API: {e}")

    with st.expander("View Full Resume"):
        st.text(top_candidate['text'])