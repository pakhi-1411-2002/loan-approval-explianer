# src/ui_components.py
import streamlit as st

def display_prediction_result(prediction, probability):
    st.header("Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ Congratulations! Your loan application is likely to be approved.")
        st.metric(label="Approval Probability", value=f"{probability[1]*100:.2f}%")
    else:
        st.error("❌ We regret to inform you that your loan application is likely to be rejected.")
        st.metric(label="Rejection Probability", value=f"{probability[0]*100:.2f}%")

def display_key_factors(shap_df_sorted):
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

def display_explanation_panel(narrative):
    st.markdown("""
        <style>
        .explanation-box {
            background-color: #f0f2f6;
            border-left: 5px solid #1f77b4;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown(narrative)
    st.markdown('</div>', unsafe_allow_html=True)