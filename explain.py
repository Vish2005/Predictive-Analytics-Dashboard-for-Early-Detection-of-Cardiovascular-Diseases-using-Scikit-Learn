import shap
import matplotlib.pyplot as plt
import io

def generate_shap_explanation(rf_classifier, rf_preprocessor, df, original_feature_names):
    """
    Generates a SHAP Waterfall plot or bar plot for the prediction natively handling OneHotEncoding pipelines.
    """
    X_transformed = rf_preprocessor.transform(df)
    
    # Extract the true transformed feature names
    try:
        encoded_features = rf_preprocessor.get_feature_names_out()
    except:
        # Fallback if get_feature_names_out unsupported
        encoded_features = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
        
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_transformed)
    
    pred_class = rf_classifier.predict(X_transformed)[0]
    
    if isinstance(shap_values, list):
        target_shap_values = shap_values[int(pred_class)][0]
    else:
        if len(shap_values.shape) == 3:
             target_shap_values = shap_values[0, :, int(pred_class)]
        else:
             target_shap_values = shap_values[0]
             
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 5))
    
    # Clean up feature names for display
    cleaned_features = []
    for f in encoded_features:
        f_clean = f.split('__')[-1] if '__' in f else f
        cleaned_features.append(f_clean)
        
    feature_map = {
        'cp': 'Chest Pain',
        'chol': 'Cholesterol',
        'thalach': 'Heart Rate',
        'oldpeak': 'ST Depression',
        'exang': 'Exercise Chest Pain',
        'ca': 'Blocked Arteries Indicator',
        'thal': 'Thalassemia Test Result',
        'trestbps': 'Blood Pressure',
        'age': 'Age',
        'sex': 'Gender (Male/Female)'
    }
    
    final_features = []
    for f in cleaned_features:
        base = f.split('_')[0]
        final_features.append(feature_map.get(base, base.capitalize()))
    
    importances = list(zip(final_features, target_shap_values))
    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Filter out duplicate display names if any emerged from One Hot Encoding
    seen = set()
    unique_importances = []
    for name, val in importances:
        if name not in seen:
            seen.add(name)
            unique_importances.append((name, val))
        if len(unique_importances) >= 8:
            break
            
    features = [x[0] for x in unique_importances]
    values = [x[1] for x in unique_importances]
    
    # Calculate percentage impacts
    total_abs_impact = sum(abs(v) for v in values)
    top_factors_stats = []
    if total_abs_impact > 0:
        for f, v in unique_importances:
            pct = (abs(v) / total_abs_impact) * 100
            top_factors_stats.append({"feature": f, "impact_pct": pct, "shap_value": v})
    
    if int(pred_class) == 0:
        colors = ['#00cc96' if v > 0 else '#ff4b4b' for v in values]
        title_suffix = "LOW RISK"
    elif int(pred_class) == 1:
        colors = ['#FFA15A' if v > 0 else '#00cc96' for v in values]
        title_suffix = "MODERATE RISK"
    else:
        colors = ['#ff4b4b' if v > 0 else '#00cc96' for v in values]
        title_suffix = "HIGH RISK"
        
    plt.barh(features[::-1], values[::-1], color=colors[::-1])
    plt.xlabel('Impact on Risk Prediction (SHAP value)')
    plt.title(f'FACTORS DRIVING YOUR {title_suffix}')
    
    plt.tight_layout()
    return fig, top_factors_stats
