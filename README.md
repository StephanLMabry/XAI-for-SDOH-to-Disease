# XAI-for-SDOH-to-Disease
# Explainable AI (XAI) for Diagnosis Prediction Using MEPS SDOH Data

This project explores how Social Determinants of Health (SDOH) from the 2021 Medical Expenditure Panel Survey (MEPS) can be used to predict a set of 23 self-reported diagnoses. The primary goal is not only predictive performance, but interpretability—assessing which variables drive model decisions using multiple Explainable AI (XAI) techniques.

---

## 🔧 Files Included

| File Name                           | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `MEPSAnalysis.py`                  | Main script for processing, model training, and running XAI techniques     |
| `model_results.csv`                | Model performance results (F1, accuracy, precision, recall by diagnosis)   |
| `xai_summary_normalized.csv`       | Normalized XAI attribution scores across all models and methods            |
| `Column_Specs_and_Names_trimmed.csv` | Cleaned list of variable names and specifications for SDOH inputs          |

---

## 🧠 Project Highlights

- 23 binary classification models (1 per diagnosis)
- 4 model types: XGBoost, Neural Network, Decision Tree, Ensemble
- 6 XAI methods:
  - SHAP
  - LIME
  - Integrated Gradients
  - Saliency Maps
  - Permutation Importance
  - Feature Ablation
- Attribution results normalized for comparability
- Focused on explanation consistency, insight validity, and methodological artifacts

---

## 📊 Data Sources

- **Dataset**: [MEPS 2021 Full-Year Consolidated (HC-233)](https://www.meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-233)  
- **SDOH Definitions**: [Healthy People 2030 – ODPHP](https://odphp.health.gov/healthypeople/priority-areas/social-determinants-health)

---

## 🚧 Known Limitations

- MEPS is a self-reported survey dataset, with no clinical depth or longitudinal tracking
- Only non-clinical SDOH variables were used for modeling
- Some XAI methods (e.g., permutation importance) identified spurious features such as survey month
- Attribution ≠ causation—interpretations should be treated cautiously

---

## 🔮 Future Work

- **Integrate Clinical Data**: Add diagnoses, labs, and treatment history from EHRs
- **Clinician Validation**: Align explanations with clinical reasoning and workflows
- **Causal Inference**: Apply DAGs, causal SHAP, or twin networks for causal discovery
- **Next-Gen Explainability**: Build contrastive, counterfactual, or interactive explanation tools

---

## 👋 Contact

**Author**: Stephan Mabry  
PhD Student, University of North Dakota  
📫 [LinkedIn](https://www.linkedin.com/in/stephan-mabry-8ab25833/)

---

## 📜 Citation
If you use or build upon this work, please cite:
- Mabry, S. Explainable AI for Diagnosis Prediction from MEPS SDOH Data. University of North Dakota, May 2025.

---

# 🤝 Acknowledgements
- Data provided by the Medical Expenditure Panel Survey (MEPS)
- SDOH categories based on Healthy People 2030 (ODPHP)
