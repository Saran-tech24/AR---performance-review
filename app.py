import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import pdfplumber

# Define label and numerical columns
Label_Encoder_columns = ["Insurance Company", "Physician Name", "Denial Reason"]
Numerical_columns = ["CPT Code", "Payment Amount", "Balance"]
Model_Features = ["CPT Code", "Insurance Company", "Physician Name", "Payment Amount", "Balance", "Denial Reason"]

# Load model, scaler, and encoders
def load_objects():
    with open('pickle/GradientBoostingClassifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('pickle/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    label_encoders = {}
    for col in Label_Encoder_columns:
        with open(f'pickle/label_encoder_{col}.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_objects()

# Streamlit UI
st.set_page_config(page_title="RCM Denial Insights", layout="wide")
st.title("📊 RCM Denial Insights Dashboard")
st.markdown("Upload your `.xlsx`, `.csv`, or `.pdf` file to explore CPT denials, payer issues, and predict denial status.")

# File upload
uploaded_file = st.file_uploader("Upload Excel, CSV, or PDF File", type=['xlsx', 'csv', 'pdf'])

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

    # Clean column names
    df.columns = df.columns.str.strip()
    st.success("✅ File uploaded successfully!")
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Check required columns
    if not all(col in df.columns for col in Model_Features):
        st.error("❌ Missing required columns. Please ensure the file includes:\n" + ", ".join(Model_Features))
        st.stop()

    ### 🧠 PREDICTION LOGIC ###
    try:
        # Fill and clean Denial Reason in both df and predict_df
        df['Denial Reason'] = df['Denial Reason'].fillna('No Denial')
        df['Denial Reason'] = df['Denial Reason'].replace([' ', None, np.nan], 'No Denial')

        predict_df = df[Model_Features].copy()

        # Label Encoding
        for col in Label_Encoder_columns:
            encoder = label_encoders[col]
            predict_df[col] = predict_df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            predict_df[col] = encoder.transform(predict_df[col])

        # Scale input
        input_scaled = scaler.transform(predict_df)

        # Make predictions
        model_predictions = model.predict(input_scaled)

        # Interpret predictions
        if hasattr(model, 'predict_proba') or np.array_equal(np.unique(model_predictions), [0, 1]):
            result_labels = ["Paid" if pred == 1 else "Denied" for pred in model_predictions]
        else:
            result_labels = ["Paid" if pred >= 0.5 else "Denied" for pred in model_predictions]

        # Add predictions to original df
        df['Prediction'] = result_labels

        st.markdown("---")
        st.header("🧾 Predictions")
        st.dataframe(df.head(20))

        # Download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    ### 📊 VISUALIZATIONS ###
    denied_df = df[df['Prediction'] == "Denied"]

    st.markdown("---")
    st.header("1️⃣ Top Denied CPT Codes")
    top_denied = denied_df['CPT Code'].value_counts().head(10)
    st.plotly_chart(px.bar(top_denied, title="Top 10 Denied CPT Codes", labels={'value': 'Denial Count', 'index': 'CPT Code'}))

    st.markdown("---")
    st.header("2️⃣ Denial Reasons by CPT Code")
    selected_cpt = st.selectbox("Select CPT Code", df['CPT Code'].unique())
    denial_reasons = denied_df[denied_df['CPT Code'] == selected_cpt]['Denial Reason'].value_counts()
    if not denial_reasons.empty:
        st.plotly_chart(px.bar(denial_reasons, title=f"Denial Reasons for CPT {selected_cpt}", labels={'value': 'Count', 'index': 'Reason'}))
    else:
        st.warning("No denial reasons found for selected CPT.")

    st.markdown("---")
    st.header("3️⃣ Denials by Insurance Company")
    payer_ctp = st.selectbox("Select Insurance Company", df['Insurance Company'].unique())
    payer_denials = denied_df[denied_df['Insurance Company'] == payer_ctp]['CPT Code'].value_counts().head(10).reset_index()
    payer_denials.columns = ['CPT Code', 'Denial Count']
    st.plotly_chart(px.bar(payer_denials, x='CPT Code', y='Denial Count', title=f"Top Denied CPTs by {payer_ctp}"))

    st.markdown("---")
    st.header("🔎 Explore Predictions")
    if st.checkbox("Show full prediction table"):
        st.dataframe(df)
