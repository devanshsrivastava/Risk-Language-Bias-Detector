import streamlit as st
import PyPDF2
import ollama
import json
import plotly.graph_objects as go
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(page_title="Risk Language Bias Detector", page_icon="🚩", layout="wide")

# ---------------- Header ----------------
st.markdown(
    """
    <div style="text-align:center; padding:15px; background-color:#f5f5f5; border-radius:12px;">
        <h1 style="margin-bottom:0;">🚩 Risk Language Bias Detector</h1>
        <p style="font-size:18px; color:#555;">AI-powered assistant to identify biased or vague language in vendor risk documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------- Sidebar ----------------
st.sidebar.header("📂 Upload & Controls")
uploaded_file = st.sidebar.file_uploader("Upload Vendor Document", type=["pdf", "txt"])

# ---------------- Helpers ----------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    segments = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        seg = " ".join(words[start:end])
        segments.append(seg)
        start += chunk_size - overlap
    return segments

def normalize_keys(item):
    mapping = {
        "phrase": "phrase",
        "category": "category",
        "explanation": "explanation",
        "bias_score": "bias_score",
        "recommendation": "recommendation"
    }
    return {mapping.get(k.lower(), k): v for k, v in item.items()}

# --- New: Risk Relevance Check ---
def check_relevance(text, model="llama3"):
    prompt = f"""
    You are an expert in third-party risk management.
    Determine if the following text is a vendor risk/compliance/security document 
    (like risk questionnaire responses, security policies, or audit reports).  

    Text: "{text[:1000]}"  # only first 1000 chars for quick check

    Answer ONLY with a single word:
    - "YES" if the text is related to vendor risk/compliance/security
    - "NO" if it is unrelated (story, essay, random text, etc.)
    """
    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content": prompt}]
    )
    return response["message"]["content"].strip().upper()

# --- Bias Analysis ---
def analyze_bias(segment, model="llama3"):
    prompt = f"""
    You are an expert in third-party risk management.
    Analyze the following vendor response for biased, vague, or evasive language.

    Text: "{segment}"

    Respond ONLY with a valid JSON array of issues.
    Each issue must include:
    - phrase
    - category: Overconfidence | Ambiguity | Deflection | Downplaying
    - explanation
    - bias_score: 0-100
    - recommendation
    """
    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content": prompt}]
    )
    raw_output = response["message"]["content"].strip()

    try:
        parsed = json.loads(raw_output)
    except:
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw_output[start:end])
            except:
                return [{"error": "Parse error", "raw_output": raw_output}]
        else:
            return [{"error": "No JSON detected", "raw_output": raw_output}]

    return [normalize_keys(item) for item in parsed] if isinstance(parsed, list) else parsed

# --- Visualizations ---
def show_gauge(score, title="Overall Bias Score"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {"text": title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_category_chart(categories):
    df = pd.DataFrame(categories, columns=["Category"])
    fig = go.Figure(data=[go.Pie(labels=df["Category"].value_counts().index,
                                 values=df["Category"].value_counts().values,
                                 hole=0.4)])
    fig.update_layout(title="Risk Category Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Main App ----------------
if not uploaded_file:
    # Landing Page
    st.markdown(
        """
        ## 👋 Welcome to the Risk Language Bias Detector  

        This tool helps **risk management & compliance teams** analyze vendor responses for vague, overconfident, or biased language.  
        Upload a vendor risk document (📄 PDF or TXT) and let AI highlight potential red flags.  

        ### ✨ Key Features:
        - 🚩 Identify biased or evasive phrases in vendor responses  
        - 📊 Dashboard showing distribution of risk categories  
        - ⏱ Overall Bias Score for the entire document  
        - 📋 Auto-generated follow-up questionnaire for vendors  
        - 🎯 Focus on selected document *segments* (instead of whole doc)  

        ---
        👉 To get started, upload a document from the sidebar.  
        """
    )

else:
    raw_text = extract_text(uploaded_file)
    segments = chunk_text(raw_text)

    # --- Run Risk Relevance Check ---
    relevance = check_relevance(raw_text)
    if relevance == "NO":
        st.error("⚠️ This document doesn’t appear to be a vendor risk/compliance document. Bias analysis may not be meaningful.")
    else:
        st.sidebar.success(f"📄 Document Loaded — {len(segments)} Segments")

        seg_options = [f"Segment {i+1}: {segments[i][:80]}..." for i in range(len(segments))]
        selected = st.sidebar.multiselect("Select Segments to Analyze", seg_options)

        if st.sidebar.button("🚀 Run Bias Analysis"):
            if not selected:
                st.warning("⚠️ Please select at least one segment")
            else:
                all_scores, all_categories, followups = [], [], []

                for choice in selected:
                    idx = int(choice.split()[1].replace(":", "")) - 1
                    st.markdown(f"### 📌 {choice}")

                    results = analyze_bias(segments[idx])

                    if isinstance(results, list):
                        for item in results:
                            if "error" in item:
                                st.error("⚠️ Parse error")
                                st.text(item.get("raw_output", ""))
                            else:
                                phrase = item.get('phrase', 'N/A')
                                category = item.get('category', 'N/A')
                                explanation = item.get('explanation', 'N/A')
                                recommendation = item.get('recommendation', 'N/A')
                                score = item.get('bias_score', 0)

                                if category != "N/A":
                                    all_categories.append(category)
                                if recommendation != "N/A":
                                    followups.append(f"- {recommendation}")
                                try:
                                    all_scores.append(int(score))
                                except:
                                    pass

                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div style="padding:12px; border:1px solid #ddd; border-radius:10px; margin-bottom:8px; background:#fafafa;">
                                            <b>🚩 Phrase:</b> {phrase}<br>
                                            <b>🏷 Category:</b> {category}<br>
                                            <b>📖 Why risky:</b> {explanation}<br>
                                            <b>📝 Recommendation:</b> {recommendation}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                # ---- Overall Results ----
                st.markdown("---")
                st.header("📊 Summary")

                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    show_gauge(avg_score, title="Overall Document Bias Score")
                    st.write(f"**Overall Avg Bias Score:** {avg_score:.1f}/100")

                if all_categories:
                    st.subheader("📊 Risk Category Dashboard")
                    show_category_chart(all_categories)

                if followups:
                    st.subheader("📋 Follow-up Questionnaire")
                    questionnaire = "\n".join(set(followups))
                    st.text_area("Vendor Follow-up Questions", questionnaire, height=200)
                    st.download_button("⬇️ Download Questionnaire", questionnaire, file_name="follow_up_questions.txt")
