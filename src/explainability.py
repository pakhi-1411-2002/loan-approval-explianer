# src/explainability.py
import shap
import pandas as pd
import matplotlib.pyplot as plt

def create_shap_explainer(model, background_data):
    return shap.LinearExplainer(model, background_data)

def compute_shap_values(explainer, processed_data):
    return explainer(processed_data)

def get_shap_dataframe(shap_values, processed_data, feature_names):
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(processed_data.shape[1])]
    
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": processed_data[0],
        "SHAP Value": shap_values.values[0]
    })
    
    shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
    return shap_df.sort_values("Abs SHAP Value", ascending=False)

def generate_waterfall_plot(shap_values, max_display=10):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
    return fig

def explain_feature(feature_name, feature_value, shap_value, user_df):
    feature_clean = feature_name.strip()
    direction = "positively" if shap_value > 0 else "negatively"
    impact_strength = abs(shap_value)
    
    if impact_strength > 0.15:
        strength = "significantly"
    elif impact_strength > 0.08:
        strength = "moderately"
    else:
        strength = "slightly"
    
    # Feature-specific explanations
    if "cibil" in feature_clean.lower() or "credit" in feature_clean.lower():
        actual_value = user_df[' cibil_score'].values[0] if ' cibil_score' in user_df.columns else "N/A"
        if shap_value > 0:
            return f"**Credit Score ({actual_value})**: Your credit score is strong and {strength} supports approval (impact: +{shap_value:.3f})"
        else:
            return f"**Credit Score ({actual_value})**: Your credit score raises concerns and {strength} works against approval (impact: {shap_value:.3f})"
    
    elif "income" in feature_clean.lower():
        actual_value = user_df[' income_annum'].values[0] if ' income_annum' in user_df.columns else "N/A"
        if shap_value > 0:
            return f"**Annual Income (${actual_value:,.0f})**: Your income level is sufficient and {strength} supports approval (impact: +{shap_value:.3f})"
        else:
            return f"**Annual Income (${actual_value:,.0f})**: Your income level is a concern and {strength} works against approval (impact: {shap_value:.3f})"
    
    # Add other feature-specific logic...
    
    # Generic fallback
    return f"**{feature_clean}**: This factor {strength} influenced the decision {direction} (impact: {shap_value:+.3f})"

def generate_narrative_explanation(prediction, probability, shap_df_sorted, user_df):
    if prediction[0] == 1:
        decision = "approved"
        prob_text = f"{probability[1]*100:.1f}%"
    else:
        decision = "rejected"
        prob_text = f"{probability[0]*100:.1f}%"
    
    top_positive = shap_df_sorted[shap_df_sorted["SHAP Value"] > 0].head(3)
    top_negative = shap_df_sorted[shap_df_sorted["SHAP Value"] < 0].head(3)
    
    explanation = []
    explanation.append(f"**Decision: Loan {decision.upper()} with {prob_text} confidence**\n")
    
    if prediction[0] == 1:
        explanation.append("Here's why this application was approved:\n")
        if len(top_positive) > 0:
            explanation.append("**Positive Factors:**")
            for idx, row in top_positive.iterrows():
                feature_explanation = explain_feature(row['Feature'], row['Value'], row['SHAP Value'], user_df)
                explanation.append(f"• {feature_explanation}")
    else:
        explanation.append("Here's why this application was rejected:\n")
        if len(top_negative) > 0:
            explanation.append("**Main Concerns:**")
            for idx, row in top_negative.iterrows():
                feature_explanation = explain_feature(row['Feature'], row['Value'], row['SHAP Value'], user_df)
                explanation.append(f"• {feature_explanation}")
    
    return "\n".join(explanation)