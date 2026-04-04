import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

DATA_DIR = "data"
MODELS_DIR = "models"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_PATH = os.path.join(DATA_DIR, "heart_disease.csv")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")

def train_and_save_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(DATA_PATH, names=columns, na_values="?")
    
    # Impute missing values so we don't lose data
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=columns)
    
    # Map target: 0 -> Low, 1 -> Moderate, 2/3/4 -> High
    def map_target(val):
        if val == 0:
            return 0
        elif val == 1:
            return 1
        else:
            return 2

    df_imputed['target'] = df_imputed['target'].apply(map_target)
    
    X = df_imputed.drop("target", axis=1)
    y = df_imputed["target"]
    
    # Categorical and Numerical features
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Save feature names and preprocessing config
    joblib.dump(numeric_features + categorical_features, os.path.join(MODELS_DIR, "feature_names.pkl"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create unified pipelines
    print("Training KNN...")
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    knn_pipeline.fit(X_train, y_train)
    joblib.dump(knn_pipeline, os.path.join(MODELS_DIR, "knn_model.pkl"))
    
    print("Training Logistic Regression...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "lr_model.pkl"))
    
    print("Training Random Forest...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, class_weight='balanced', random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "rf_model.pkl"))
    
    print("All models trained and saved successfully with advanced preprocessing pipelines!")

if __name__ == "__main__":
    download_data()
    train_and_save_models()
