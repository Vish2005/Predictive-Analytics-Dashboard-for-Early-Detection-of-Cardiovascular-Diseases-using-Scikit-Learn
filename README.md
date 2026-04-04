# ❤️ Predictive Analytics Dashboard for Early Detection of Cardiovascular Diseases

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen)

A professional, clinical-grade Explainable AI (XAI) web application designed to predict the risk of cardiovascular disease based on patient health metrics. The dashboard provides a dynamic, user-friendly interface that analyzes medical data against multiple trained ML models to deliver a heavily scrutinized risk assessment.

## ✨ Key Features
- **Ensemble ML Architecture:** Uses a Clinical Pessimistic Ensemble approach drawing from K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest Classifier models.
- **Explainable AI (XAI):** Integrated `SHAP` (SHapley Additive exPlanations) values to mathematically explain exactly which health factors drove the AI's decision and by what percentage.
- **Visual Data Diagnostics:** Interactive Plotly histograms seamlessly compare the current patient's blood pressure, age, and cholesterol explicitly against historical high/low-risk clinical distributions.
- **Complete Clinical Override Hooks:** Ensures hard-coded medical logic (e.g., severe cholesterol thresholds) can override underlying ML abstractions to prioritize absolute patient safety.
- **Modern UI/UX:** Styled comprehensively with custom glassmorphism metrics cards, status animations, clinical emojis, and cleanly formatted side-by-side data expanders.

## 📂 Project Structure
- `app.py`: The main Streamlit dashboard application.
- `train_model.py`: Script to download the UCI Heart Disease dataset, map target severities, build robust preprocessing pipelines, and train the scikit-learn models.
- `predict.py`: Houses the `HeartDiseasePredictor` class handling data preprocessing mapping and generating clinical consensus from the models.
- `explain.py`: Extracts SHAP summary matrices to render quantitative Feature Importance statistics.
- `utils.py`: Auxiliary tools such as dynamically generating PDF reports of the risk assessment.

---

## 🚀 Setup & Installation Instructions

**1. Clone the repository:**
```bash
git clone https://github.com/Vish2005/Predictive-Analytics-Dashboard-for-Early-Detection-of-Cardiovascular-Diseases-using-Scikit-Learn.git
cd Predictive-Analytics-Dashboard-for-Early-Detection-of-Cardiovascular-Diseases-using-Scikit-Learn
```

**2. Install dependencies:**
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

**3. Train the Backend Models:**
Before running the dashboard, generate the machine learning models. *This command will automatically download the UCI dataset into a `.data/` folder and export the models to a `.models/` directory.*
```bash
python train_model.py
```

**4. Run the Streamlit Application:**
```bash
streamlit run app.py
```
*Your browser will automatically open the application at `http://localhost:8501`.*

---

## ⚕️ Disclaimer
This project is an AI demonstration built for educational and research purposes. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions regarding a medical condition.
