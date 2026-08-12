# ============================================================
# Heart Disease Classification — Streamlit App
# ML Assignment 2 | M.Tech AIML/DSE | BITS Pilani WILP
# ============================================================

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler

# ---- Page Configuration ----
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for Premium Look ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="main-header">🫀 Heart Disease Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Assignment 2 — M.Tech (AIML/DSE) | BITS Pilani WILP</p>', unsafe_allow_html=True)
st.markdown('---')

# ---- Model Loading ----
MODEL_DIR = 'model'

model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}

@st.cache_resource
def load_models():
    """Load all saved models and scaler."""
    loaded_models = {}
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    return loaded_models, scaler, feature_names

try:
    loaded_models, scaler, feature_names = load_models()
    st.sidebar.success(f"✅ {len(loaded_models)} models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ---- Sidebar ----
st.sidebar.header('⚙️ Configuration')

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    '🤖 Select ML Model',
    options=list(loaded_models.keys()),
    index=0,
    help='Choose a classification model to evaluate on the uploaded test data.'
)

compare_all = st.sidebar.checkbox('📊 Compare All Models', value=True,
                                   help='Show comparison table of all models.')

st.sidebar.markdown('---')
st.sidebar.markdown('### 📋 About')
st.sidebar.info(
    '**Dataset:** Heart Disease (UCI/Kaggle)\n\n'
    '**Features:** 13 clinical features\n\n'
    '**Target:** Heart Disease (Yes/No)\n\n'
    '**Models:** 5 ML classifiers'
)

# ---- File Upload ----
st.header('📂 Upload Test Data')
uploaded_file = st.file_uploader(
    'Upload your test data CSV file (must include a `target` column)',
    type=['csv'],
    help='Upload the test_data.csv file. It should contain the same features as the training data plus a target column.'
)

if uploaded_file is not None:
    # Read uploaded file
    test_data = pd.read_csv(uploaded_file)
    st.success(f'✅ File uploaded: {test_data.shape[0]} rows × {test_data.shape[1]} columns')

    # Show preview
    with st.expander('🔍 Preview Uploaded Data', expanded=False):
        st.dataframe(test_data.head(10), use_container_width=True)

    # Validate columns
    if 'target' not in test_data.columns:
        st.error('❌ The uploaded CSV must contain a `target` column!')
        st.stop()

    missing_features = [f for f in feature_names if f not in test_data.columns]
    if missing_features:
        st.error(f'❌ Missing features in uploaded data: {missing_features}')
        st.stop()

    # Separate features and target
    X_uploaded = test_data[feature_names]
    y_uploaded = test_data['target']

    # Scale features
    X_uploaded_scaled = scaler.transform(X_uploaded)

    st.markdown('---')

    # ---- Evaluate Selected Model ----
    st.header(f'🎯 Results: {selected_model_name}')

    selected_model = loaded_models[selected_model_name]
    y_pred = selected_model.predict(X_uploaded_scaled)

    if hasattr(selected_model, 'predict_proba'):
        y_prob = selected_model.predict_proba(X_uploaded_scaled)[:, 1]
        auc_val = roc_auc_score(y_uploaded, y_prob)
    else:
        auc_val = roc_auc_score(y_uploaded, y_pred)

    acc = accuracy_score(y_uploaded, y_pred)
    prec = precision_score(y_uploaded, y_pred, zero_division=0)
    rec = recall_score(y_uploaded, y_pred, zero_division=0)
    f1 = f1_score(y_uploaded, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_uploaded, y_pred)

    # Display metrics in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric('Accuracy', f'{acc:.4f}')
    col2.metric('AUC Score', f'{auc_val:.4f}')
    col3.metric('Precision', f'{prec:.4f}')
    col4.metric('Recall', f'{rec:.4f}')
    col5.metric('F1 Score', f'{f1:.4f}')
    col6.metric('MCC', f'{mcc:.4f}')

    st.markdown('---')

    # ---- Confusion Matrix & Classification Report ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('📉 Confusion Matrix')
        cm = confusion_matrix(y_uploaded, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    cbar=True, linewidths=1, linecolor='black',
                    annot_kws={'size': 16})
        ax_cm.set_xlabel('Predicted', fontsize=12)
        ax_cm.set_ylabel('Actual', fontsize=12)
        ax_cm.set_title(f'Confusion Matrix — {selected_model_name}', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_right:
        st.subheader('📝 Classification Report')
        report = classification_report(
            y_uploaded, y_pred,
            target_names=['No Disease', 'Disease'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).T
        # Format floats to 4 decimals; leave 'support' as integer
        fmt_dict = {col: '{:.4f}' for col in report_df.columns if col != 'support'}
        if 'support' in report_df.columns:
            fmt_dict['support'] = '{:.0f}'
        st.dataframe(report_df.style.format(fmt_dict), use_container_width=True)

    st.markdown('---')

    # ---- ROC Curve (pure matplotlib — no sklearn display API) ----
    if hasattr(selected_model, 'predict_proba'):
        st.subheader('📈 ROC Curve')
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_uploaded, y_prob)
        ax_roc.plot(fpr, tpr, color='#3498db', linewidth=2,
                    label=f'{selected_model_name} (AUC = {auc_val:.4f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'ROC Curve — {selected_model_name}', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown('---')

    # ---- Compare All Models ----
    if compare_all:
        st.header('📊 All Models — Comparison Table')

        all_results = {}
        for name, model in loaded_models.items():
            y_p = model.predict(X_uploaded_scaled)
            if hasattr(model, 'predict_proba'):
                y_pr = model.predict_proba(X_uploaded_scaled)[:, 1]
                auc_v = roc_auc_score(y_uploaded, y_pr)
            else:
                auc_v = roc_auc_score(y_uploaded, y_p)

            all_results[name] = {
                'Accuracy': accuracy_score(y_uploaded, y_p),
                'AUC': auc_v,
                'Precision': precision_score(y_uploaded, y_p, zero_division=0),
                'Recall': recall_score(y_uploaded, y_p, zero_division=0),
                'F1': f1_score(y_uploaded, y_p, zero_division=0),
                'MCC': matthews_corrcoef(y_uploaded, y_p)
            }

        comp_df = pd.DataFrame(all_results).T.round(4)
        comp_df.index.name = 'ML Model'

        # Highlight best values
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #2ecc71; color: white; font-weight: bold' if v else '' for v in is_max]

        st.dataframe(
            comp_df.style.apply(highlight_max, axis=0).format('{:.4f}'),
            use_container_width=True
        )

        # Winner
        best_model = comp_df['F1'].idxmax()
        st.success(f'🏆 **Best Model (by F1 Score): {best_model}** — F1: {comp_df.loc[best_model, "F1"]:.4f}')

        # Bar chart comparison
        st.subheader('📊 Visual Comparison')
        fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
        comp_df.plot(kind='bar', ax=ax_comp, edgecolor='black', alpha=0.85, width=0.8)
        ax_comp.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax_comp.set_xlabel('Model', fontsize=12)
        ax_comp.set_ylabel('Score', fontsize=12)
        ax_comp.set_ylim(0, 1.1)
        ax_comp.legend(loc='upper right', ncol=3)
        ax_comp.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close(fig_comp)

else:
    st.info('👆 Please upload a test data CSV file to begin evaluation.')
    st.markdown(
        '**Expected CSV format:** The file should contain the same feature columns as the training data '
        '(`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, '
        '`slope`, `ca`, `thal`) plus a `target` column.'
    )

# ---- Footer ----
st.markdown('---')
st.markdown(
    '<div style="text-align:center; color:#888; font-size:0.9rem;">'
    '🫀 Heart Disease Classification App | ML Assignment 2 | BITS Pilani WILP'
    '</div>',
    unsafe_allow_html=True
)
