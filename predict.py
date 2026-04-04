import os
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "models"

class HeartDiseasePredictor:
    def __init__(self):
        # We now load full preprocessing pipelines
        self.knn = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
        self.lr = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        self.rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        
        # Extracted purely for SHAP
        self.rf_classifier = self.rf.named_steps["classifier"]
        self.rf_preprocessor = self.rf.named_steps["preprocessor"]

    def predict(self, input_data):
        df = pd.DataFrame([input_data])[self.feature_names]

        # The pipelines handle scaling and OneHotEncoding natively
        pred_knn = self.knn.predict(df)[0]
        pred_lr = self.lr.predict(df)[0]
        pred_rf = self.rf.predict(df)[0]
        
        probas = [
            self.knn.predict_proba(df)[0],
            self.lr.predict_proba(df)[0],
            self.rf.predict_proba(df)[0]
        ]
        
        # Format predictions
        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        predictions_raw = [int(pred_knn), int(pred_lr), int(pred_rf)]
        
        # Clinical Pessimistic Ensemble: Prioritize higher risk safely
        final_risk_raw = max(predictions_raw)
        final_risk = risk_map[final_risk_raw]
        
        # Determine confidence score based on the model that gave this max risk
        max_risk_probas = [p[final_risk_raw] for p in probas if len(p) > final_risk_raw and np.argmax(p) == final_risk_raw]
        if max_risk_probas:
            confidence_score = np.mean(max_risk_probas) * 100
        else:
            confidence_score = np.max([p[final_risk_raw] for p in probas if len(p) > final_risk_raw]) * 100

        model_breakdown = {
            "KNN": risk_map[pred_knn],
            "Logistic Regression": risk_map[pred_lr],
            "Random Forest": risk_map[pred_rf]
        }
        
        return final_risk, confidence_score, model_breakdown, df

