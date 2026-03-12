import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Import your modules
from src.model_utils import load_models, get_feature_names, make_prediction
from src.explainability import (
    create_shap_explainer, 
    compute_shap_values, 
    get_shap_dataframe,
    generate_waterfall_plot,
    generate_narrative_explanation
)
from src.ui_components import (
    display_prediction_result,
    display_key_factors,
    display_explanation_panel
)

# Load models once
@st.cache_resource
def initialize_models():
    model, preprocessor = load_models()
    feature_names = get_feature_names(preprocessor)
    
    # Load background data for SHAP
    import test_shap
    X_train = test_shap.df_train
    background = shap.sample(X_train, 100)
    explainer = create_shap_explainer(model, background)
    
    return model, preprocessor, feature_names, explainer

model, preprocessor, feature_names, explainer = initialize_models()

# UI
st.title(":blue[Explainable] Loan Approval Predictor")
st.write("Enter applicant information to predict loan approval and understand why")

# Form
with st.form(key="my_form"):
    st.write("Please enter the following details to predict loan approval:")
    
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education", options=[" Graduate", " Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=[" Yes", " No"])
    anual_income = st.number_input("Annual Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=0)
    loan_term = st.number_input("Loan Term (in months)", min_value=0, value=0)
    cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900, value=0)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0)
    
    submitted = st.form_submit_button(label="Predict Loan Approval")

# Process submission
if submitted:
    # Prepare input
    user_input = {
        " no_of_dependents": no_of_dependents,
        " education": education,
        " self_employed": self_employed,
        " income_annum": anual_income,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil_score,
        " residential_assets_value": residential_assets_value,
        " commercial_assets_value": commercial_assets_value,
        " luxury_assets_value": luxury_assets_value,
        " bank_asset_value": bank_asset_value
    }
    
    user_df = pd.DataFrame([user_input])
    
    # Validate
    if anual_income <= 0:
        st.error("Annual Income must be greater than 0.")
    elif loan_amount <= 0:
        st.error("Loan Amount must be greater than 0.")
    else:
        # Make prediction
        prediction, probability, processed_data = make_prediction(model, preprocessor, user_input)
        
        # Display result
        display_prediction_result(prediction, probability)
        
        # Show input summary
        with st.expander("View Input Summary"):
            st.dataframe(user_df.T.rename(columns={0: "Input Value"}))

        # SHAP explanation
        st.header("Why This Decision?")
        
        # Compute SHAP values
        shap_vals = explainer(processed_data)
        
        # Create Explanation object with feature names
        shap_explanation = shap.Explanation(
            values=shap_vals.values,
            base_values=shap_vals.base_values,
            data=shap_vals.data,
            feature_names=feature_names
        )
        
        # Create DataFrame with feature names
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": processed_data[0],
            "SHAP Value": shap_vals.values[0]
        })
        
        # Sort by absolute SHAP value
        shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
        shap_df_sorted = shap_df.sort_values("Abs SHAP Value", ascending=False)
        
        # Display key factors
        st.subheader("Key Factors")
        
        top_positive = shap_df_sorted[shap_df_sorted["SHAP Value"] > 0].head(3)
        top_negative = shap_df_sorted[shap_df_sorted["SHAP Value"] < 0].head(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Factors** (Helped Approval)")
            for idx, row in top_positive.iterrows():
                st.write(f"✅ {row['Feature']}: +{row['SHAP Value']:.4f}")
        
        with col2:
            st.write("**Top Negative Factors** (Hurt Approval)")
            for idx, row in top_negative.iterrows():
                st.write(f"❌ {row['Feature']}: {row['SHAP Value']:.4f}")
        
        # Waterfall plot with feature names
        st.subheader("Detailed Explanation: Waterfall Chart")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **How to read this chart:**
        - Starting point (bottom): The base prediction (average across all applicants)
        - Each bar shows how one feature changes the prediction
        - Red bars = push toward approval
        - Blue bars = push toward rejection  
        - Final value (top): The actual prediction for this applicant
        """)

        st.subheader("Plain-Language Explanation")
        narrative = generate_narrative_explanation(prediction, probability, shap_df, user_df)
        display_explanation_panel(narrative)