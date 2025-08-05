import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
def summarize_resume_into_sections(resume_text):
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

    for section, prompt in prompts.items():
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        section_summaries[section] = response.choices[0].message.content.strip()

    return section_summaries

def summarize_job_description_into_sections(jd_text):
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

    for section, prompt in prompts.items():
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        section_summaries[section] = response.choices[0].message.content.strip()

    return section_summaries

def compute_section_similarity(resume_sections: dict, job_sections: dict):
    """
    Computes cosine similarity between relevant resume and job description sections.

    Args:
        resume_sections: dict with resume section names and text
        job_sections: dict with job section names and text

    Returns:
        dict of section-wise similarity and overall average similarity
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define section mappings (resume → job description)
    section_pairs = {
        "Qualifications and Education": "Qualifications and Education",
        "Skills and Certifications": "Required Skills and Technologies",
        "Projects and Work Experience": "Responsibilities and Duties",
    }

    similarities = {}

    for resume_key, job_key in section_pairs.items():
        resume_text = resume_sections.get(resume_key, "")
        job_text = job_sections.get(job_key, "")

        if not resume_text or not job_text:
            similarities[f"{resume_key} ↔ {job_key}"] = 0.0
            continue

        # Generate embeddings
        embeddings = model.encode([resume_text, job_text])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        similarities[f"{resume_key} ↔ {job_key}"] = sim_score

    # Calculate average similarity
    average_similarity = np.mean(list(similarities.values()))
    similarities["Average Similarity"] = average_similarity

    return similarities


resume_text = """
KRISHNA VISHWAKARMA
Newark, NJ,USA | +1 (585) 434-8178 | mrkvishwakarma@gmail.com| Website: mrkvishwakarma.github.io
GitHub: hJps://github.com/mrkvishwakarma | LinkedIn: hJps://www.linkedin.com/in/mrkvishwakarma
Dec 2025 (Expected)
GPA: 3.6/4.0
July 2017 – May 2021
CGPA: 9.15/10
Education
Master of Science in Computer Science Rochester Ins6tute of Technology, Rochester, United States Bachelor of Engineering in Informa8on Technology University of Mumbai, Mumbai, India Work Experience
Data Analy8cs Intern (Hackensack Meridian Health, New Jersey, USA) June 2025 – Aug 2025
• Conducted in-depth data analysis for the U.S. News Best Hospitals 2025–2026 report, covering 22 procedures and condiYons and
15 adult specialYes. Analyzed CMS RaYngs data to support internal benchmarking, compliance and strategic planning.
• Automated data workflows using Google Collab, BigQuery and Hospital Data Insights, reducing manual analysis Yme by 90%
enabling faster and streamline delivery of hospital performance metrics across 18 hospital sites.
• Built and maintained site-specific Looker dashboards for 22 procedures and condiYons across a network of 18 hospitals,
improving visibility into key performance indicators such as readmission rates, mortality, and paYent volume.
• Presented key findings and data-driven insights to CEO, directors, business line stakeholders, and senior leadership, influencing
strategic decisions for the U.S. News Best Hospitals report and earning formal recogniYon for clarity, depth, and impact.
• Collaborated with cross-funcYonal teams, clinical and data team, to ensure accurate data alignment with US News methodology.
Data Engineer (Capgemini Technology India Pvt Ltd, Mumbai, India) June 2021 – Jan 2024
• Designed and built scalable ETL pipelines using PySpark and Azure Databricks, enabling processing of millions of records weekly
and improving analyYcs eﬃciency across mulYple teams while integraYng data from mulYple sources, improving data accuracy
and reliability on built ETL robust pipelines by 25% and streamlining reporYng for 15+ business stakeholders
• Developed automated workflows for pre-deployment, post-deployment, and unit tesYng in Databricks notebooks, reducing
processing Yme by 30% and Integrated data quality checks with 95% code coverage using SonarQube to ensure accuracy.
• Led file processing and data ingesYon pipelines using InformaYca, SQL Server, Azure Storage Explorer and Databricks, cujng
processing Yme by 15% and implemented CI/CD pipelines via Azure DevOps, achieving a 99% deployment success rate.
• Collaborated with cross-funcYonal teams, including analyst and stakeholders, to translate business requirements into technical
deliverables and mentored new team members to ensure smooth onboarding and team producYvity.
Business Intelligence Intern (Bhak6vedanta Research Hospital, Mumbai, India) Nov 2020 – Feb 2021
• Collaborated with clinicians to design a real-Yme paYent occupancy dashboard, enabling faster bed allocaYon during peak hours.
• Managed SQL Server databases for 10K+ paYent records weekly, ensuring HIPAA compliance and data integrity across
inpaYent/outpaYent workflows and automated monthly financial reconciliaYon processes using SQL.
• Led a 15-member cross-funcYonal team to design and deploy Tableau dashboards for inpaYent/outpaYent department analyYcs,
improving data-driven decision-making for hospital administrators.
Technical Skills
• Programming Languages: Python, Java, JavaScript, SQL, MySQL, PostgreSQL, MongoDB.
• Cloud Technologies: Google Cloud, Azure Cloud, Azure Databricks, Looker, Power BI, Tableau.
• Project Management & Methodologies: Azure DevOps, Jenkins, Jira, Asana, Git, SharePoint, Confluence, Agile, Scrum & SDLC.
• Packages & Frameworks: Pandas, NumPy, PyTorch, PySpark, dbt, HuggingFace, LangChain, CrewAI, OpenAI, LLMs, RAGs.
• Tech Stack: Data Pipeline, Data Modelling, ETL Pipeline, CI/CD pipelines, Data IntegraYon, Big Data Technologies.
• SoF Skills: Team lead, Cross-FuncYonal CollaboraYon, AnalyYcal Problem Solving, Root-Cause Analysis.
• Relevant Coursework: Big Data & AnalyYcs, AI, ML, OOPs, Data Structures and Algorithms, Database Systems.
Projects
F1 Data Engineering— PySpark, Databricks, Azure • Designed and implemented data pipelines using Azure Databricks and Spark to process and analyze Formula 1 data.
• Created and managed Azure Data Factory pipelines to automate ETL process with trigger funcYonality.
• Leveraged PySpark and Spark SQL for complex data processing and manipulaYon, resulYng in eﬃcient and scalable soluYons.
Big Star Collec8bles— Airflow, Airbyte, Dagster, DBT , GCP GitHub Link
• Employed SonarQube to achieve a code coverage of 95% and conducted DQ checks, ensuring data reliability and accuracy.
• Seamlessly extracted data from various sources using Airbyte and transformed it into an analyYcs-ready state with dbt.
• Managed data pipelines using Dagster, hosted on Google Cloud Planorm (GCP), and uYlized Google BigQuery.
• Leveraged Docker for containerizaYon and used PostgreSQL as the primary database for storing and processing data.
Certifications
• Google Cloud Cer8fied: Data Analy6cs & Advanced Analy6cs Engineer
• Databricks Cer8fied: Data Engineer Associate
• MicrosoF Cer8fied: Azure Fundamentals & Administrator
"""

job_desc = """
About the job
About Byline Bank

Headquartered in Chicago, Byline Bank is a full-service commercial bank serving small- and medium-sized businesses, financial sponsors and consumers. Byline Bank has approximately $9.6 billion in assets and operates 46 branch locations throughout the Chicago and Milwaukee metropolitan areas. Byline Bank offers a broad range of commercial and community banking products and services, including small-ticket equipment leasing solutions, and is one of the top Small Business Administration lenders in the United States according to the national SBA ranking by the U.S. Small Business Administration by volume FY2023. Byline Bank is a member of FDIC and an Equal Housing Lender.

At Byline Bank, we take pride in being an award-winning workplace. Some of our recent recognitions include:

U.S. News & World Report named Byline Bank as one of the Best Companies to Work for in the Midwest, Finance & Overall in 2024-2025, 2025-2026.
Chicago Sun Times Chicago’s Best Workplaces 2024
Best Workplaces in Illinois in 2024 by Best Companies Group and Illinois SHRM (Society for Human Resource Management)
Forbes America’s Best Small Employers 2023

The Credit Internship at Byline Bank will give an in-depth exposure to the world of commercial banking, specifically from a leader in the area of Small Business lending. As an intern you will be responsible for entering information into our online loan tracking system, assist with the underwriting reviews of our appraisals and environmental reports ordered from third party vendors, and analyze and verify tax return information used in our underwriting process. You will also have the opportunity to sit in on weekly credit meetings where loans are presented and approved and be involved in departmental meetings and training. This is a great opportunity for Finance or Accounting majors to learn firsthand about future career opportunities in their chosen field.

Key Responsibilities

Extracting and organizing financial data from a company's financial statements (like balance sheets, income statements, and cash flow statements) into a standardized format for analysis and comparison.
Review third party reports including equipment appraisals, environmental reports, and business valuations and write a short memo of pertinent information reviewed.
Participate in Credit Committee as an observer to learn the credit procedures.
Work with management on data review projects.
Prepare post-close credit change memos for closed loans – reallocating funds for final disbursements, clearing exceptions or clarifying information from original approval. 

Qualifications

Currently enrolled as a Junior or Senior majoring in Finance or Accounting at an accredited college/university program.
Familiarity with financial documents (tax returns, balance sheets, profit & loss statements) preferred.
Strong written communication skills required.
Ability to work independently.
Strong organization skills.
Detail-oriented and accuracy a must.
Confident in a professional office environment.

Physical Demands/Work Environment

Usual office environment with frequent sitting, walking, and standing, and occasional climbing, stooping, kneeling, crouching, crawling, and balancing. Frequent use of eye, hand, and finger coordination enabling the use of office equipment. Oral and auditory capacity enabling interpersonal communication as well as communication through automated devices.

At Byline Bank, we value work-life flexibility and support a hybrid work environment for this position. This role allows for a combination of remote and in-office work, with occasional visits to the office based on business needs. Specific in-office days may vary and will be discussed during the interview process.

Compensation & Benefits

Byline Bank offers competitive industry rate salary bands with the goal of retaining and growing talented individuals. We regularly review and adjust our compensation structure as needed to ensure equity across teams and levels.

The hourly rate for this position is $18.00.

In addition, Byline provides benefits including medical coverage, dental, vision, disability, 401k, paid time off and much more! Depending on the specific role, compensation may also include discretionary bonuses and other benefit programs. The actual compensation package may vary based on factors such as skill set, experience level, and candidate's location.

Additional Information

Byline Bank is an Equal Opportunity Employment / Affirmative Action employer dedicated to providing an inclusive workplace where the unique differences of our employees are welcomed, respected, and valued. We are committed to the principle of equal employment opportunity for all employees and to providing employees with a work environment free of discrimination and harassment.

All employment decisions at Byline Bank are based on business needs, job requirements and individual qualifications, without regard to race, color, religion or belief, national, social or ethnic origin, sex (including pregnancy), age, physical, mental or sensory disability, HIV Status, sexual orientation, gender identity and/or expression, marital, civil union or domestic partnership status, past or present military service, protected veteran status, family medical history or genetic information, family or parental status, or any other status protected by the laws or regulations in the locations where we operate. Byline Bank will not tolerate discrimination or harassment based on any of these characteristics.

Byline Bank is committed to providing reasonable accommodations for candidates with disabilities in our recruiting process. If you need any assistance or accommodations due to a disability, please contact us directly at 773-475-2900, Option #2.

If applying within the US, this role is not eligible for visa sponsorship now or in the future.

Note To Recruitment Agencies And Third-Party Recruiters

Byline Bank kindly requests that third-party recruiters, staffing agencies, and recruitment firms refrain from submitting resumes or candidate profiles without a prior agreement in place. Any unsolicited submissions will be considered property of Byline Bank, and no fees will be paid for placements resulting from such submissions. We appreciate your understanding and cooperation.
"""




if __name__ == "__main__":
    print("Resume Key Sections:")
    resume_sections = summarize_resume_into_sections(resume_text)
    print(resume_sections)

    print("Job Key Sections:")
    job_sections = summarize_job_description_into_sections(job_desc)
    print(job_sections)

    print("Similarity Score:")
    similarity_scores = compute_section_similarity(resume_sections, job_sections)
    for section, score in similarity_scores.items():
        print(f"{section}: {score:.4f}")
