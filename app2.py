import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import pdfplumber


# Define which columns are label encoded
Label_Encoder_columns = ["Insurance Company", "Physician Name", "Denial Reason"]
Numerical_columns = ["CPT Code", "Payment Amount", "Balance"]

# Load your original data for dropdowns
df = pd.read_csv('C:/Users/Admin/Desktop/Tensaw AR project/AF_DATA_cleaned.csv')

# Load the saved model, scaler, and encoders
def load_objects():
    with open('pickle\\GradientBoostingClassifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('pickle\\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    label_encoders = {}
    for col in Label_Encoder_columns:
        with open(f'pickle\\label_encoder_{col}.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)
    
    return model, scaler, label_encoders

# Load objects
model, scaler, label_encoders = load_objects()

st.set_page_config(page_title="RCM Denial Insights", layout="wide")

st.title("📊 RCM Denial Insights Dashboard")
st.markdown("Upload your `.xlsx` file to explore CPT denials, payer issues, and revenue impact.")

# File upload
uploaded_file = st.file_uploader("Upload Excel, CSV, or PDF File", type=['xlsx', 'csv', 'pdf'])


if uploaded_file:
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'xlsx':
        df = pd.read_excel(uploaded_file)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'pdf':
        
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
        st.subheader("📄 PDF Content Preview")
        st.text_area("Extracted Text", text[:2000], height=300)
        st.warning("PDF parsing shows raw text only. Structured analysis not supported yet.")
        st.stop()
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.success("✅ File uploaded successfully!")
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    required_cols = ['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance', 'Denial Reason']

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Missing required columns. Please ensure the file includes:\n" + ", ".join(required_cols))
    else:
        # Clean column names
        df.columns = df.columns.str.strip()

        st.markdown("---")
        st.header("1️⃣ Top Denied CPT Codes")
        denied_df = df[df['Payment Amount'] == 0]

        top_denied = denied_df['CPT Code'].value_counts().head(10)
        st.plotly_chart(px.bar(top_denied, title="Top 10 Denied CPT Codes", labels={'value': 'Denial Count', 'index': 'CPT Code'}))

        st.markdown("---")
        st.header("2️⃣ Denial Reasons by CPT Code")

        selected_cpt = st.selectbox("Select CPT Code", df['CPT Code'].unique())
        denial_reasons = df[df['CPT Code'] == selected_cpt]['Denial Reason'].value_counts()

        if not denial_reasons.empty:
            st.plotly_chart(px.bar(denial_reasons, title=f"Denial Reasons for CPT {selected_cpt}", labels={'value': 'Count', 'index': 'Reason'}))
        else:
            st.warning("No denial reasons found for selected CPT.")

        # st.markdown("---")
        # st.header("3️⃣ Denials by Insurance Company")

        # payer_ctp = st.selectbox("Select Insurance Company", df['Insurance Company'].unique())
        # payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['CPT Code'].value_counts().head(10)
        # st.plotly_chart(px.bar(payer_denials, title=f"Top Denied CPTs by {payer_ctp}", labels={'value': 'Denial Count', 'index': 'CPT Code'}))

        # payer_ctp = st.selectbox("Select Insurance Company", df['Insurance Company'].unique())
        # payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['CPT Code'].value_counts().head(10)
        # st.plotly_chart(px.bar(payer_denials, x='CPT Code', y='Denial Count', title=f"Top Denied CPTs by {payer_ctp}"))




        # payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['CPT Code'].value_counts().head(10).reset_index()
        # payer_denials.columns = ['CPT Code', 'Denial Count']  # Rename for clarity
        # st.plotly_chart(px.bar(payer_denials, x='CPT Code', y='Denial Count', title=f"Top Denied CPTs by {payer_ctp}"))

        # payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['CPT Code'].value_counts().head(10).reset_index()
        # payer_denials.columns = ['CPT Code', 'count']  # Rename the column to 'count' to match expected input

        # st.plotly_chart(px.bar(payer_denials, x='CPT Code', y='count', title=f"Top Denied CPTs by {payer_ctp}"))


        st.markdown("---")
        st.header("3️⃣ Denials by Insurance Company")

        # Select Insurance Company
        payer_ctp = st.selectbox("Select Insurance Company", df['Insurance Company'].unique())

        # Debugging: Display selected company
        st.write(f"Selected Insurance Company: {payer_ctp}")

        # Filter denied claims by selected company and count CPT code occurrences
        payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['Insurance Company'].value_counts().head(10).reset_index()
        payer_denials.columns = ['Insurance Company', 'count']

        # Create visualization
        st.plotly_chart(px.bar(payer_denials, x='Insurance Company', y='count', title=f"Top Denied CPTs by {payer_ctp}"))
