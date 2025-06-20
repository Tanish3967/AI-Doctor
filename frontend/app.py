import streamlit as st
import requests
import io
from fpdf import FPDF
import re
from io import BytesIO
import pandas as pd
import qrcode
from PIL import Image as PILImage
import time

# --- Session Expiry ---
SESSION_TIMEOUT = 600  # 10 minutes
if 'last_active' not in st.session_state:
    st.session_state['last_active'] = time.time()
if time.time() - st.session_state['last_active'] > SESSION_TIMEOUT:
    for k in list(st.session_state.keys()):
        if k not in ['font_size', 'language', 'allergies']:
            del st.session_state[k]
    st.warning('Session expired. Sensitive data cleared for privacy.')
st.session_state['last_active'] = time.time()

# --- Stub Translation ---
TRANSLATIONS = {
    'English': {
        'Symptoms': 'Symptoms',
        'Medical History (optional)': 'Medical History (optional)',
        'Upload CT scans or reports': 'Upload CT scans or reports',
        'Analyze': 'Analyze',
        'Diagnosis & Recommendations': 'Diagnosis & Recommendations',
        'Medicines Table': 'Medicines Table',
        'Download Report as PDF': 'Download Report as PDF',
        'Important Notice:': 'Important Notice:',
        'Downloaded reports may contain sensitive information. Please handle with care.': 'Downloaded reports may contain sensitive information. Please handle with care.'
    },
    'Hindi (Coming Soon)': {
        'Symptoms': 'लक्षण',
        'Medical History (optional)': 'चिकित्सा इतिहास (वैकल्पिक)',
        'Upload CT scans or reports': 'सीटी स्कैन या रिपोर्ट अपलोड करें',
        'Analyze': 'विश्लेषण करें',
        'Diagnosis & Recommendations': 'निदान और सिफारिशें',
        'Medicines Table': 'दवाओं की तालिका',
        'Download Report as PDF': 'पीडीएफ के रूप में रिपोर्ट डाउनलोड करें',
        'Important Notice:': 'महत्वपूर्ण सूचना:',
        'Downloaded reports may contain sensitive information. Please handle with care.': 'डाउनलोड की गई रिपोर्ट में संवेदनशील जानकारी हो सकती है। कृपया सावधानी से संभालें।'
    }
}
def t(key):
    lang = st.session_state.get('language', 'English')
    return TRANSLATIONS.get(lang, TRANSLATIONS['English']).get(key, key)

# --- PDF Logo (placeholder if not present) ---
PDF_LOGO_PATH = 'logo.png'  # Place your logo.png in the same directory as app.py
import os
logo_exists = os.path.exists(PDF_LOGO_PATH)

# --- Custom Banner/Header ---
st.markdown(
    f"""
    <div style='background-color: #1a73e8; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
        <h1 style='color: white; margin: 0; display: flex; align-items: center; font-size: {st.session_state['font_size']+8}px;'>
            🩺&nbsp; <span style='font-size: 2rem;'>AI Doctor</span>
        </h1>
        <p style='color: #e3e3e3; margin: 0.5rem 0 0 0;'>Your AI-powered medical assistant for diagnosis, test recommendations, and safe prescriptions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Download Notice ---
st.info(t('Downloaded reports may contain sensitive information. Please handle with care.'))

# --- Main Form ---
st.set_page_config(page_title="AI Doctor", layout="wide")

st.title("AI Doctor")

col1, col2 = st.columns(2)

with col1:
    st.header("Patient Input")
    symptoms = st.text_area("Symptoms", key="symptoms")
    history = st.text_area("Medical History (optional)", key="history")
    uploaded_files = st.file_uploader("Upload CT scans or reports", type=["tif", "tiff", "png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    if st.button("Analyze", help="Submit for AI analysis"):
        files = [("files", (f.name, f, f.type)) for f in uploaded_files] if uploaded_files else []
        data = {"symptoms": symptoms, "history": history}
        with st.spinner("AI Doctor is analyzing..."):
            try:
                response = requests.post(f"http://backend:5000/ai_doctor", data=data, files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
                diagnosis = result.get('diagnosis', '')
                disclaimer = result.get('disclaimer', 'This is not a substitute for professional medical advice.')
                st.session_state['ai_result'] = (diagnosis, disclaimer)
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend service is taking too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend service. Please check if the service is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with col2:
    st.header("AI Diagnosis & Recommendations")
    if 'ai_result' in st.session_state:
        diagnosis, disclaimer = st.session_state['ai_result']
        st.markdown(diagnosis)
        st.info(disclaimer)

# --- Diagnosis & Medicines Display ---
def parse_sections(diagnosis):
    # Simple parser for sections (Diagnosis, Tests, Medicines, Explanation, References)
    sections = {}
    current = None
    for line in diagnosis.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('diagnosis:'):
            current = 'Diagnosis'
            sections[current] = [line[len('diagnosis:'):].strip()]
        elif line.lower().startswith('recommended tests:'):
            current = 'Recommended Tests'
            sections[current] = [line[len('recommended tests:'):].strip()]
        elif line.lower().startswith('medicines:'):
            current = 'Medicines'
            sections[current] = [line[len('medicines:'):].strip()]
        elif line.lower().startswith('explanation'):
            current = 'Explanation'
            sections[current] = [line[len('explanation:'):].strip()]
        elif line.lower().startswith('references'):
            current = 'References'
            sections[current] = [line[len('references:'):].strip()]
        elif current:
            sections[current].append(line)
    return sections

def display_sections(sections):
    for sec, content in sections.items():
        st.markdown(f"### {sec}", unsafe_allow_html=True)
        for item in content:
            if item:
                st.markdown(f"- {item}", unsafe_allow_html=True)

# --- Medicines Table ---
def parse_medicines_section(sections):
    # Dummy parser: expects medicines as bullet points or lines in 'Medicines' section
    med_rows = []
    meds = sections.get('Medicines', [])
    for med in meds:
        # Try to split by ':' or '-' or '.'
        parts = re.split(r'[:\-\.]', med, maxsplit=1)
        if len(parts) == 2:
            name, details = parts
        else:
            name, details = med, ''
        med_rows.append({
            'Medicine': name.strip(),
            'Details': details.strip(),
            'Type': 'Indian' if 'indian' in details.lower() else ('Global' if 'global' in details.lower() else ''),
            'Dosage': '',
            'Duration': '',
            'Side Effects': ''
        })
    return med_rows

def display_medicine_info(med_rows):
    if not med_rows:
        return
    df = pd.DataFrame(med_rows)
    st.dataframe(df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': '#e1e4e8',
        'font-size': f'{st.session_state["font_size"]}px'
    }), use_container_width=True)

# --- PDF Generation ---
def generate_pdf_report(diagnosis, disclaimer, med_rows):
    pdf = FPDF()
    pdf.add_page()
    # Branding/Header
    if logo_exists:
        pdf.image(PDF_LOGO_PATH, x=10, y=8, w=30)
        pdf.set_xy(45, 10)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "AI Doctor Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, "Generated by AI Doctor App", ln=True, align='C')
    pdf.ln(5)
    # QR Code
    qr = qrcode.make("https://github.com/Tanish3967/AI-Doctor")
    qr_bytes = BytesIO()
    qr.save(qr_bytes, format='PNG')
    qr_bytes.seek(0)
    pdf.image(qr_bytes, x=170, y=10, w=25)
    pdf.ln(20)
    # Table of Contents
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Table of Contents", ln=True)
    toc = [t("Diagnosis & Recommendations"), "Treatment Regimen", t("Important Notice:")]
    for idx, item in enumerate(toc, 1):
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"{idx}. {item}", ln=True)
    pdf.ln(5)
    # Diagnosis Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, t("Diagnosis & Recommendations"), ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, diagnosis)
    pdf.ln(5)
    # Medicines Table
    if med_rows:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Treatment Regimen", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(40, 8, "Medicine", 1, 0, 'L', True)
        pdf.cell(30, 8, "Type", 1, 0, 'L', True)
        pdf.cell(30, 8, "Brand Name", 1, 0, 'L', True)
        pdf.cell(30, 8, "Generic Name", 1, 0, 'L', True)
        pdf.cell(30, 8, "Side Effects", 1, 0, 'L', True)
        pdf.cell(0, 8, "Details", 1, 1, 'L', True)
        pdf.set_font("Arial", '', 10)
        for row in med_rows:
            pdf.cell(40, 8, row.get("Medicine", ""), 1)
            pdf.cell(30, 8, row.get("Type", ""), 1)
            pdf.cell(30, 8, row.get("Brand Name", ""), 1)
            pdf.cell(30, 8, row.get("Generic Name", ""), 1)
            pdf.cell(30, 8, row.get("Side Effects", ""), 1)
            pdf.cell(0, 8, row.get("Details", ""), 1, 1)
        pdf.ln(3)
    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, t("Important Notice:"), ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 8, disclaimer)
    return pdf.output(dest='S').encode('latin1')

# --- Main Display ---
if 'ai_result' in st.session_state:
    diagnosis, disclaimer = st.session_state['ai_result']
    sections = parse_sections(diagnosis)
    display_sections(sections)
    med_rows = parse_medicines_section(sections)
    if med_rows:
        st.markdown("#### 💊 Medicines Table")
        display_medicine_info(med_rows)
    pdf_bytes = generate_pdf_report(diagnosis, disclaimer, med_rows)
    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="ai_doctor_report.pdf",
        mime="application/pdf",
        key="download_report"
    )
    st.markdown(f"---\n*_{disclaimer}_*")

# --- Knowledge Base UI ---
st.header("Medical Knowledge Base")
kb_query = st.text_input("Search KB:")
if kb_query:
    response = requests.get(f"http://backend:5000/kb_search", params={"q": kb_query})
    if response.status_code == 200:
        results = response.json().get('results', [])
        for r in results:
            st.markdown(f"**Q:** {r['question']}\n\n**A:** {r['answer']}")
    else:
        st.error("KB search failed.")
with st.expander("Add to Knowledge Base"):
    kb_q = st.text_input("New KB Question:", key="kbq")
    kb_a = st.text_area("New KB Answer:", key="kba")
    if st.button("Add KB Entry"):
        response = requests.post(f"http://backend:5000/kb", json={"question": kb_q, "answer": kb_a})
        if response.status_code == 200:
            st.success("KB entry added!")
        else:
            st.error("Failed to add KB entry.")

# --- Drug Interaction Checker ---
with st.expander("💊 Drug Interaction Checker"):
    st.write("Enter the names of medicines to check for interactions and see cost info (Indian dataset).")
    med_input = st.text_area("Medicines (comma-separated)", key="interaction_meds")
    # Dosage input
    st.write("(Optional) Enter dosages for each medicine (mg):")
    dosage_inputs = {}
    med_list = [m.strip() for m in med_input.split(",") if m.strip()]
    for med in med_list:
        dosage_inputs[med] = st.text_input(f"Dosage for {med}", key=f"dosage_{med}")
    # Allergies from sidebar
    allergies = [a.strip() for a in st.session_state.get('allergies', '').split(',') if a.strip()]
    if st.button("Check Interactions"):
        if len(med_list) < 2:
            st.warning("Please enter at least two medicines.")
        else:
            payload = {
                "medicines": med_list,
                "allergies": allergies,
                "dosages": {k: v for k, v in dosage_inputs.items() if v}
            }
            response = requests.post(f"http://backend:5000/drug_interactions", json=payload)
            if response.status_code == 200:
                result = response.json()
                if result["interactions"]:
                    st.error("Interactions found:")
                    for inter in result["interactions"]:
                        st.write(f"- {inter['warning']}")
                else:
                    st.success("No known interactions found.")
                if result["allergy_warnings"]:
                    st.warning("Allergy Warnings:")
                    for warn in result["allergy_warnings"]:
                        st.write(f"- {warn['warning']}")
                if result["dosage_warnings"]:
                    st.warning("Dosage Warnings:")
                    for warn in result["dosage_warnings"]:
                        st.write(f"- {warn['warning']}")
                if result["cost_info"]:
                    st.markdown("#### Medicine Cost Info (INR)")
                    st.table(result["cost_info"])
                # --- Detailed Medicine Info ---
                if result.get("medicine_info"):
                    st.markdown("#### Medicine Details")
                    for med, info in result["medicine_info"].items():
                        with st.expander(f"{info.get('brand_name', med.title())} ({info.get('generic_name', '')})"):
                            st.write(f"**Brand Name:** {info.get('brand_name', '')}")
                            st.write(f"**Generic Name:** {info.get('generic_name', '')}")
                            st.write(f"**Manufacturer:** {info.get('manufacturer', '')}")
                            st.write(f"**Type:** {info.get('type', '')}")
                            st.write(f"**Form:** {info.get('form', '')}")
                            st.write(f"**Strength:** {info.get('strength', '')}")
                            st.write(f"**Price:** ₹{info.get('price', '')}")
                            st.write(f"**Packaging:** {info.get('packaging', '')}")
                            st.write(f"**Schedule:** {info.get('schedule', '')}")
                            if info.get('dosage'):
                                st.write("**Dosage Guidelines:**")
                                for k, v in info['dosage'].items():
                                    st.write(f"- {k.title()}: {v}")
                            if info.get('side_effects'):
                                st.write(f"**Side Effects / Warnings:** {info['side_effects']}")
                            if info.get('contraindications'):
                                st.write("**Contraindications:**")
                                for c in info['contraindications']:
                                    st.write(f"- {c}")
                            if info.get('interactions'):
                                st.write("**Known Interactions:**")
                                for i in info['interactions']:
                                    st.write(f"- {i}")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# --- Footer ---
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: #888;'>
        <small>
            Powered by <a href='https://github.com/Tanish3967/AI-Doctor' target='_blank'>AI-Doctor</a> |
            <i>This tool does not replace professional medical advice.</i>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Theme suggestion (add to .streamlit/config.toml) ---
# [theme]
# primaryColor="#1a73e8"
# backgroundColor="#f5f6fa"
# secondaryBackgroundColor="#e3e3e3"
# textColor="#222"
# font="sans serif"

# Remove the old mobile toggle button from the bottom
if st.session_state.get('mobile_view', False):
    st.markdown("""
        <div style='position: fixed; bottom: 20px; right: 20px; z-index: 1000;
             background-color: #1a73e8; color: white; padding: 10px;
             border-radius: 50%; width: 50px; height: 50px;
             display: flex; align-items: center; justify-content: center;
             box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
            📱
        </div>
    """, unsafe_allow_html=True)
