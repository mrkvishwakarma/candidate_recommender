from utils import *

st.set_page_config(page_title="Candidate Recommendation Agent", layout="wide")

st.title("Candidate Recommendation Agent 🤖")
st.markdown("Enter a job description and upload a list of resumes to find the best candidates.")

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

            resumes = []
            for uploaded_file in uploaded_files:
                full_text = read_text_file(uploaded_file)  # Uses the combined reader
                if "unsupported" in full_text or "Failed to read" in full_text:
                    st.warning(f"Skipping {uploaded_file.name}: {full_text}")
                    continue
                resumes.append({"name": uploaded_file.name, "full_text": full_text})

            if not resumes:
                st.error("No valid resumes were processed.")
                st.session_state.results = None
            else:
                candidate_list = []

                jd_sections = extract_key_sections_from_jd(job_description)

                for resume in resumes:
                    resume_sections = extract_key_sections_from_resume(resume['full_text'])

                    if isinstance(resume_sections, str) and resume_sections.startswith("Error"):
                        st.warning(f"Skipping {resume['name']} due to extraction error: {resume_sections}")
                        continue

                    similarities = compute_section_similarity(resume_sections, jd_sections)

                    candidate_list.append({
                        "name": resume["name"],
                        "score": similarities.get("Overall Score", 0.0),
                        "text": resume["full_text"],
                        "section_scores": similarities
                    })

                if not candidate_list:
                    st.error("No candidates could be processed for scoring.")
                    st.session_state.results = None
                else:
                    candidate_list.sort(key=lambda x: x["score"], reverse=True)
                    st.session_state.results = candidate_list
                    st.success("Analysis complete!")

if st.session_state.results:
    st.header("Top Candidate Recommendation")

    if st.session_state.results:
        top_candidate = st.session_state.results[0]
        st.markdown(f"**1. {top_candidate['name']}** ")

        for section, score in top_candidate['section_scores'].items():
            if section != "Overall Score":
                st.markdown(f"- **{section}:** `{score:.4f}`")

        st.markdown(f"**Overall Similarity Score:** `{top_candidate['score']:.4f}`")

        try:
            summary = generate_summary(job_description, top_candidate['text'])
            st.success("Summary generated!")
            st.write(summary)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to generate summary: {e}")

        with st.expander("View Full Resume"):
            st.text(top_candidate['text'])
    else:
        st.warning("No candidates found after processing.")