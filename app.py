import streamlit as st
import pandas as pd
st.set_page_config(page_title="Multi-Pathway Psychiatric Classifier", layout="wide")
st.sidebar.image("hospital_logo.png", use_container_width=True)
st.sidebar.write("---")


import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



@st.cache_resource
def train_pipeline_engine():
    np.random.seed(101)
    n_total_samples = 150
    diagnostic_labels = np.random.choice(['Schizophrenia', 'Bipolar Disorder', 'Major Depression'], size=n_total_samples, p=[0.4, 0.3, 0.3])
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
        else:
            sod_m.append(np.random.normal(11.8, 1.9))
            il6_m.append(np.random.normal(3.8, 0.9))
            bdnf_m.append(np.random.normal(8.2, 1.7))
    df_multi = pd.DataFrame({'Diagnosis': diagnostic_labels, 'SOD1_Level': sod_m, 'IL6_Level': il6_m, 'BDNF_Level': bdnf_m})
    X_m = df_multi.drop(columns=['Diagnosis'])
    y_m = df_multi['Diagnosis']
    multi_classifier = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    multi_classifier.fit(X_m, y_m)
    return multi_classifier

clf_model = train_pipeline_engine()

st.title("🔬 Multi-Pathway Psychiatric Classifier")
st.subheader("Objective Machine Learning Diagnostics")

st.sidebar.header("📋 Patient Biomarker Profile")
sod1_input = st.sidebar.slider("SOD1 Level", 2.0, 16.0, 5.2, 0.1)
il6_input = st.sidebar.slider("IL-6 Level", 1.0, 14.0, 9.4, 0.1)
bdnf_input = st.sidebar.slider("BDNF Level", 4.0, 26.0, 12.1, 0.1)

damage_index = (sod1_input * il6_input) / (bdnf_input + 0.1)
input_vector = pd.DataFrame([{'SOD1_Level': sod1_input, 'IL6_Level': il6_input, 'BDNF_Level': bdnf_input}])

prediction = clf_model.predict(input_vector)[0]
probabilities = clf_model.predict_proba(input_vector)[0]
classes = clf_model.classes_

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Primary Pathology", value=prediction)
    st.metric(label="Calculated Damage-vs-Repair Index", value=f"{damage_index:.3f}")
    for c, prob in zip(classes, probabilities):
        st.write(f"**{c}**: {prob*100:.1f}%")
        st.progress(float(prob))

with col2:
    metrics_data = pd.DataFrame({'Biomarker': ['SOD1', 'IL-6', 'BDNF'], 'Value': [sod1_input, il6_input, bdnf_input]})
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=metrics_data, x='Biomarker', y='Value', palette='Purples_r', ax=ax)
    st.pyplot(fig)
