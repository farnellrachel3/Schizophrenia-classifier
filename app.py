import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set page config and branding elements
st.set_page_config(page_title="Multi-Pathway Psychiatric Classifier", page_icon="🧠", layout="wide")
try:
    st.sidebar.image("https://icons8.com", use_container_width=True)
except Exception:
    st.sidebar.markdown("### 🏥 **CLINICAL AI PORTAL**")
st.sidebar.write("---")

st.title("🧠 Resolving Diagnostic Specificity in Schizophrenia")
st.markdown("### A Multi-Pathway Machine Learning Classifier with Explainable AI (SHAP)")
st.write("---")

# 1. Background Engine: Train the Model on Data Load
@st.cache_data
def load_and_train_model():
    np.random.seed(101)
    n_total_samples = 150
    diagnostic_labels = np.random.choice(
        ['Schizophrenia', 'Bipolar Disorder', 'Major Depression'], 
        size=n_total_samples, 
        p=[0.4, 0.3, 0.3]
    )
    
    sod_m, il6_m, bdnf_m = [], [], []
    for label in diagnostic_labels:
        if label == 'Schizophrenia':
            sod_m.append(np.random.normal(5.2, 1.1))
            il6_m.append(np.random.normal(9.4, 1.8))
            bdnf_m.append(np.random.normal(12.1, 2.5))
        elif label == 'Bipolar Disorder':
            sod_m.append(np.random.normal(8.5, 1.4))
            il6_m.append(np.random.normal(6.1, 1.2))
            bdnf_m.append(np.random.normal(18.4, 3.1))
        else: # Major Depression
            sod_m.append(np.random.normal(11.8, 1.9))
            il6_m.append(np.random.normal(3.8, 0.9))
            bdnf_m.append(np.random.normal(8.2, 1.7))
            
    df_multi = pd.DataFrame({
        'Diagnosis': diagnostic_labels, 
        'SOD1_Level': sod_m, 
        'IL6_Level': il6_m, 
        'BDNF_Level': bdnf_m
    })
    
    X_m = df_multi.drop(columns=['Diagnosis'])
    y_m = df_multi['Diagnosis']
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_m, y_m, test_size=0.30, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_m, y_train_m)
    
    preds = clf.predict(X_test_m)
    acc = accuracy_score(y_test_m, preds) * 100
    
    cm = confusion_matrix(y_test_m, preds, labels=['Schizophrenia', 'Bipolar Disorder', 'Major Depression'])
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual SZ', 'Actual BP', 'Actual MDD'], 
        columns=['Predicted SZ', 'Predicted BP', 'Predicted MDD']
    )
    
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test_m)
    
    return clf, acc, cm_df, shap_vals, X_test_m

# Run pipeline execution
multi_classifier, accuracy, cm_dataframe, shap_values, X_test_data = load_and_train_model()

# 2. Layout Distribution: Sidebar Inputs vs Dashboard Graphics
st.sidebar.header("🔬 Patient Biomarker Entry Panel")
st.sidebar.write("Adjust serum baseline inputs to test specific diagnostic clinical thresholds:")

input_sod = st.sidebar.slider("SOD1 Level (Antioxidant baseline)", 2.0, 16.0, 7.5, step=0.1)
input_il6 = st.sidebar.slider("IL-6 Level (Acute Neuroinflammation)", 1.0, 14.0, 6.5, step=0.1)
input_bdnf = st.sidebar.slider("BDNF Level (Synaptic Plasticity)", 3.0, 25.0, 13.0, step=0.1)

damage_repair_index = (input_sod * input_il6) / (input_bdnf + 0.1)

# Split view into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Live Diagnostic Inference")
    
    patient_vector = pd.DataFrame([{
        'SOD1_Level': input_sod,
        'IL6_Level': input_il6,
        'BDNF_Level': input_bdnf
    }])
    
    prediction = multi_classifier.predict(patient_vector)
    prediction_proba = multi_classifier.predict_proba(patient_vector)
    classes = multi_classifier.classes_
    
    st.metric(label="Predicted Diagnostics Classification Target", value=f"⚠️ {prediction[0]}")
    st.metric(label="Derived Bio-Damage vs Repair Index Score", value=f"{damage_repair_index:.3f}")
    
    st.write("**Algorithm Confidence Distribution Matrix:**")
    
    # FIX: Flatten the multi-dimensional prediction probabilities array to line up matching array rows
    flattened_probabilities = prediction_proba[0]
    
    proba_df = pd.DataFrame({
        "Diagnostic Group": classes, 
        "Confidence Probability": flattened_probabilities
    })
    proba_df["Confidence Probability"] = proba_df["Confidence Probability"].map(lambda x: f"{x*100:.1f}%")
    st.table(proba_df)

with col2:
    st.subheader("📊 System Pipeline Verification")
    st.success(f"Validated Model Accuracy: **{accuracy:.1f}%**")
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
    sns.heatmap(
        cm_dataframe, annot=True, fmt='d', cmap='Purples', 
        linewidths=1.5, linecolor='white', ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title('Cross-Diagnostic Error Margin (Test Set)', fontweight='bold', fontsize=10)
    ax.set_ylabel('True Gold-Standard Diagnosis', fontsize=9)
    ax.set_xlabel('Algorithm Predictions', fontsize=9)
    st.pyplot(fig)

st.write("---")
st.subheader("🕵️ Explainable AI Validation Layer (SHAP)")
st.markdown("This live-rendering panel maps exactly how heavily each biological pathway influences the model's global diagnostic process, ensuring validation doesn't rely on a 'black box' script.")

# Render SHAP Plot directly into Streamlit UI layout
fig_shap, ax_shap = plt.subplots(figsize=(10, 4), dpi=100)
shap.summary_plot(shap_values, X_test_data, class_names=list(multi_classifier.classes_), show=False)
plt.title("SHAP Global Feature Importance Profile", fontweight='bold', fontsize=12, pad=15)
plt.tight_layout()
st.pyplot(fig_shap)

