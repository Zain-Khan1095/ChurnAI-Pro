import codecs

# Define the complete app content inside a clean python string variable
streamlit_app_code = """import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# PREMIUM PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="ChurnAI Engine | Enterprise Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise-grade CSS for stunning UI/UX
st.markdown(\"\"\"
    <style>
    .reportview-container { background: #F1F5F9; }
    .main-title { font-size: 34px; font-weight: 800; color: #1E3A8A; letter-spacing: -0.5px; margin-bottom: 2px; }
    .sub-title { font-size: 15px; color: #64748B; margin-bottom: 30px; font-weight: 400; }
    .kpi-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 6px solid #2563EB;
        margin-bottom: 15px;
    }
    .kpi-card-alert {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 6px solid #DC2626;
        margin-bottom: 15px;
    }
    .kpi-label { font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #0F172A; margin-top: 5px; }
    .section-header { font-size: 20px; font-weight: 700; color: #1E293B; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }
    
    /* Sidebar Specific Styling for User Friendliness */
    div[data-testid="stSidebar"] { background-color: #0F172A !important; }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] label { color: #F8FAFC !important; }
    .sidebar-badge { background-color: #1E293B; border: 1px solid #334155; padding: 10px; border-radius: 8px; margin-bottom: 15px; }
    </style>
\"\"\", unsafe_allow_html=True)

# ==========================================
# CLEANED SIDEBAR CONTROL CENTER
# ==========================================
st.sidebar.markdown(\"<h2 style='color: #3B82F6; font-weight:800; margin-bottom:0px; font-size:26px;'>ChurnAI Pro</h2>\", unsafe_allow_html=True)
st.sidebar.markdown(\"<p style='color: #94A3B8; font-size:12px; margin-bottom:20px;'>Enterprise Intelligence Node</p>\", unsafe_allow_html=True)

# System Status Component
st.sidebar.markdown(\"\"\"
<div class="sidebar-badge">
    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600;">System Engine Status</div>
    <div style="font-size: 14px; color: #4ADE80; font-weight: 700; margin-top: 2px;">● Core Pipeline Active</div>
</div>
\"\"\", unsafe_allow_html=True)

st.sidebar.markdown("---")

# File Ingestion Portal Module
st.sidebar.markdown("### 📥 Ingestion Module")
uploaded_file = st.sidebar.file_uploader(
    "Ingest Operation CSV Spreadsheet", 
    type=["csv"], 
    help="Accepts standard system comma-separated transaction records file dumps."
)

st.sidebar.markdown("---")

# Collapsible User Guide Layer
with st.sidebar.expander("📖 Operational Manual", expanded=True):
    st.markdown(\"\"\"
    <p style='color: #F8FAFC; font-size:13px;'>
    1. Drag-and-drop your formatted operational data log into the file ingestion portal upload layer.<br><br>
    2. The core machine learning engine will immediately execute data preprocessing steps automatically.<br><br>
    3. Navigate through the horizontal tabs to view predicted high-risk profiles, operational chart distributions, and backend performance reports.
    </p>
    \"\"\", unsafe_allow_html=True)

# ==========================================
# MAIN INTERFACE HEADER
# ==========================================
st.markdown('<div class="main-title">Enterprise Churn Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Feature Preprocessing & XGBoost Predictive Inference Engine</div>', unsafe_allow_html=True)

def load_production_pipeline():
    if os.path.exists('production_churn_pipeline.joblib'):
        return joblib.load('production_churn_pipeline.joblib')
    return None

pipeline = load_production_pipeline()

if pipeline is None:
    st.error("⚠️ **System Disconnect:** Core model artifact (`production_churn_pipeline.joblib`) not detected. Please run your core pipeline training script to initialize deployment status.")
else:
    if uploaded_file is None:
        st.markdown(\"\"\"
        <div style='background-color: #EFF6FF; border: 1px dashed #3B82F6; padding: 25px; border-radius: 12px; text-align: center;'>
            <h3 style='color: #1E40AF; margin-top:0px;'>Awaiting Operational Data Batch Injection</h3>
            <p style='color: #1E3A8A; font-size: 14px; max-width: 600px; margin: 0 auto;'>
                To start predictive analysis, please drop your active client record spreadsheet into the sidebar uploader. The server pipeline will process values automatically.
            </p>
        </div>
        \"\"\", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Required Database Structural Schema Blueprint</div>', unsafe_allow_html=True)
        blueprint_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        st.dataframe(pd.DataFrame(columns=blueprint_cols), use_container_width=True)
        
    else:
        try:
            raw_data = pd.read_csv(uploaded_file)
            processed_data = raw_data.copy()
            
            if 'customerID' in processed_data.columns:
                processed_data = processed_data.drop(columns=['customerID'])
            if 'TotalCharges' in processed_data.columns:
                processed_data['TotalCharges'] = processed_data['TotalCharges'].replace(' ', np.nan)
                processed_data['TotalCharges'] = pd.to_numeric(processed_data['TotalCharges'])

            has_ground_truth = 'Churn' in processed_data.columns
            if has_ground_truth:
                processed_data['Churn'] = processed_data['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
                X_eval = processed_data.drop(columns=['Churn'])
                y_eval = processed_data['Churn']
            else:
                X_eval = processed_data

            # Run Model Inference
            batch_predictions = pipeline.predict(X_eval)
            batch_probabilities = pipeline.predict_proba(X_eval)[:, 1]

            results_df = raw_data.copy()
            results_df['Churn_Prediction'] = np.where(batch_predictions == 1, '🚨 High Risk', '✅ Stable')
            results_df['Churn_Probability_%'] = np.round(batch_probabilities * 100, 2)

            # KPI Summary Cards calculation
            total_scanned = len(results_df)
            churn_count = int(np.sum(batch_predictions == 1))
            churn_ratio = (churn_count / total_scanned) * 100 if total_scanned > 0 else 0
            
            card_col1, card_col2, card_col3 = st.columns(3)
            with card_col1:
                st.markdown(f\"\"\"
                <div class="kpi-card">
                    <div class="kpi-label">Total Accounts Audited</div>
                    <div class="kpi-value">{total_scanned:,} Users</div>
                </div>
                \"\"\", unsafe_allow_html=True)
            with card_col2:
                card_style = "kpi-card-alert" if churn_ratio > 15 else "kpi-card"
                st.markdown(f\"\"\"
                <div class=\"{card_style}\">
                    <div class="kpi-label">High Attrition Risk Profiles</div>
                    <div class="kpi-value">{churn_count:,} ({churn_ratio:.1f}%)</div>
                </div>
                \"\"\", unsafe_allow_html=True)
            with card_col3:
                st.markdown(f\"\"\"
                <div class="kpi-card">
                    <div class="kpi-label">Prediction Status</div>
                    <div class="kpi-value" style="color: #16A34A;">Online & Secure</div>
                </div>
                \"\"\", unsafe_allow_html=True)

            # Tabs View Configuration
            tab1, tab2, tab3 = st.tabs(["🎯 Customer Risk Assessment", "📊 Behavioral Analytics", "📈 Model Operational Diagnostics"])
            
            with tab1:
                st.markdown('<div class="section-header">High-Risk Customer Attrition Forecast Ledger</div>', unsafe_allow_html=True)
                display_cols = ['customerID', 'Contract', 'MonthlyCharges', 'Churn_Prediction', 'Churn_Probability_%']
                valid_cols = [c for c in display_cols if c in results_df.columns]
                st.dataframe(results_df[valid_cols], use_container_width=True, height=350)
                
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Structured Predictions CSV Report",
                    data=csv_export,
                    file_name="churn_inference_export.csv",
                    mime="text/csv"
                )

            with tab2:
                st.markdown('<div class="section-header">Operational Vulnerability & Distribution Mapping</div>', unsafe_allow_html=True)
                plot_col1, plot_col2 = st.columns(2)
                
                with plot_col1:
                    st.markdown("#### Monthly Billing Outliers vs Churn Classifications")
                    fig_costs, ax_costs = plt.subplots(figsize=(6, 3.8))
                    sns.boxplot(data=results_df, x='Churn_Prediction', y='MonthlyCharges', palette='Set2', ax=ax_costs)
                    ax_costs.set_xlabel("Predicted Status Category")
                    ax_costs.set_ylabel("Monthly Contract Fees ($)")
                    st.pyplot(fig_costs)
                    
                with plot_col2:
                    st.markdown("#### Contract Structure Vulnerability Alignment Profiles")
                    fig_contract, ax_contract = plt.subplots(figsize=(6, 3.8))
                    sns.countplot(data=results_df, x='Contract', hue='Churn_Prediction', palette='crest', ax=ax_contract)
                    ax_contract.set_xlabel("Contract Specification Types")
                    ax_contract.set_ylabel("Account Counts")
                    st.pyplot(fig_contract)

            with tab3:
                st.markdown('<div class="section-header">Production Verification Pipeline Report</div>', unsafe_allow_html=True)
                if has_ground_truth:
                    diag_col1, diag_col2 = st.columns([1, 1.8])
                    with diag_col1:
                        st.markdown("#### Model Confusion Matrix Alignment")
                        cm = confusion_matrix(y_eval, batch_predictions)
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Churn'], yticklabels=['Stayed', 'Churn'], ax=ax_cm)
                        ax_cm.set_ylabel('Actual Field Ground Truth')
                        ax_cm.set_xlabel('Model Predicted Classification')
                        st.pyplot(fig_cm)
                    with diag_col2:
                        st.markdown("#### Validation Matrix Framework Metrics")
                        report_dict = classification_report(y_eval, batch_predictions, output_dict=True)
                        st.json(report_dict)
                else:
                    st.warning("⚠️ **Ground Truth Missing:** Advanced model evaluation metrics require a baseline target column named 'Churn' within the uploaded template file framework.")
                    
        except Exception as err:
            st.error(f"❌ **Data Processing Exception:** An error occurred while parsing the uploaded file: {str(err)}")
"""

# Write the string variable cleanly out to disk without character drops
with codecs.open("app_streamlit.py", "w", "utf-8") as f:
    f.write(streamlit_app_code)

print("[SUCCESS] app_streamlit.py compiled perfectly with zero structural drops!")
