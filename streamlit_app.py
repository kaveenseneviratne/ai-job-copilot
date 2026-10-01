"""
Phase 4: Streamlit frontend.

Calls the FastAPI backend over HTTP rather than importing src/ directly
-- keeping the UI decoupled from the underlying pipeline, same as any
real client of this API would be.

Run (with the API already running separately):
    streamlit run streamlit_app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

VERDICT_ICON = {"match": "✅", "partial": "⚠️", "gap": "❌"}

st.set_page_config(page_title="AI Job-Search Copilot", page_icon="🧭", layout="centered")
st.title("🧭 AI Job-Search Copilot")
st.caption("Retrieval-augmented Q&A and automated gap analysis over your profile.")

tab_ask, tab_analyze = st.tabs(["💬 Ask a question", "📋 Analyze a job posting"])


def _render_assessment_group(title: str, assessments: list[dict]) -> None:
    if not assessments:
        return
    must_haves = title.startswith("Must")
    if must_haves:
        matched = sum(1 for a in assessments if a["verdict"] == "match")
        st.subheader(f"{title}: {matched}/{len(assessments)} matched ({matched / len(assessments):.0%})")
    else:
        st.subheader(title)

    for a in assessments:
        icon = VERDICT_ICON.get(a["verdict"], "•")
        with st.expander(f"{icon}  {a['requirement']}"):
            st.write(a["evidence"])
            if a.get("sources"):
                st.caption("Sources: " + ", ".join(a["sources"]))


with tab_ask:
    st.write("Ask anything about your profile or an ingested job posting.")
    question = st.text_input(
        "Question", placeholder="Does Kaveen have Docker experience?", key="ask_question"
    )
    if st.button("Ask", key="ask_btn") and question.strip():
        with st.spinner("Retrieving and thinking..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(f"**Answer:** {data['answer']}")
                if data.get("sources"):
                    st.caption("Sources: " + ", ".join(data["sources"]))
            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the API. Is it running? "
                    "Start it with: `uvicorn src.api:app --reload --port 8000`"
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")

with tab_analyze:
    st.write("Paste a job posting to get a structured gap analysis against your profile.")
    company = st.text_input(
        "Company / role label (optional)", placeholder="e.g. DKSR - Data Engineer", key="analyze_label"
    )
    job_text = st.text_area("Job posting text", height=280, key="analyze_text")

    if st.button("Analyze", key="analyze_btn") and job_text.strip():
        with st.spinner("Extracting requirements and checking each against your profile... this can take a minute."):
            try:
                resp = requests.post(
                    f"{API_URL}/analyze",
                    json={"job_text": job_text, "company_or_role": company or "this role"},
                    timeout=300,
                )
                resp.raise_for_status()
                report = resp.json()

                assessments = report["assessments"]
                must_haves = [a for a in assessments if a["importance"] == "must_have"]
                nice_to_haves = [a for a in assessments if a["importance"] == "nice_to_have"]

                _render_assessment_group("Must-haves", must_haves)
                _render_assessment_group("Nice-to-haves", nice_to_haves)

                gaps = [a["requirement"] for a in must_haves if a["verdict"] == "gap"]
                if gaps:
                    st.subheader("🎯 Priority gaps to address")
                    for g in gaps:
                        st.write(f"- {g}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the API. Is it running? "
                    "Start it with: `uvicorn src.api:app --reload --port 8000`"
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")

st.divider()
st.caption("AI Job-Search Copilot — portfolio project. Answers are grounded in ingested profile and job-posting documents only.")
