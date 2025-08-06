from utils import *

st.set_page_config(page_title="Candidate Recommendation Agent", layout="wide")

st.title("Candidate Recommendation Agent 🤖")
st.markdown("Enter a job description and upload a resumes to find the best candidates.")

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
                full_text = read_text_file(uploaded_file)
                if "unsupported" in full_text or "Failed to read" in full_text:
                    st.warning(f"Skipping {uploaded_file.name}: {full_text}")
                    continue

                email = extract_contact_info(full_text)

                resumes.append({
                    "name": uploaded_file.name,
                    "email": email,
                    "full_text": full_text
                })

            if not resumes:
                st.error("No valid resumes were processed.")
                st.session_state.results = None
            else:
                candidate_list = []

                jd_prompts = get_jd_summary_prompts(job_description)
                jd_sections = generate_summary(jd_prompts)

                for resume in resumes:
                    resume_prompts = get_resume_summary_prompts(resume['full_text'])
                    resume_sections = generate_summary(resume_prompts)

                    similarities = compute_section_similarity(resume_sections, jd_sections)

                    candidate_list.append({
                        "name": resume["name"],
                        "email": resume["email"],
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
        top_candidates = st.session_state.results[:10]

        zip_data = create_zip_file_for_resumes(top_candidates)
        st.download_button(
            label="Download All Top Resumes",
            data=zip_data,
            file_name="top_candidates.zip",
            mime="application/zip",
        )

        for i, top_candidate in enumerate(top_candidates):
            st.markdown(f"**{i+1}. {top_candidate['name']}, {top_candidate['email']}** ")

            for section, score in top_candidate['section_scores'].items():
                st.markdown(f"- **{section}:** `{score:.2%}`")

            try:
                summary_prompt = get_summary_prompt(job_description, top_candidate['text'])
                summary = generate_summary(summary_prompt)['Summary']
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