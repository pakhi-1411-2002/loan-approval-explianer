# Loan Approval Explianer 

## Overview

Predicts loan approval (Approve/Reject) with confidence scores and shows exactly which features influenced each decision using SHAP explainability. Built with Logistic Regression for interpretability, not accuracy maximization.

**Key Features:**
- SHAP-based local explanations (per applicant) and global feature importance
- Interactive Streamlit web interface
- Waterfall charts showing feature contributions
- Complete preprocessing pipeline

**Educational project** - not for production use. Does not use real bank data, verify identity, or ensure regulatory compliance

## Demo

```
Input: Income $65k, Credit 720, Loan $25k  →  APPROVED (78% confidence)
Explanation: Credit Score (+), Income (+), Loan Amount (-)
```

## Tech Stack

**scikit-learn** (Logistic Regression, preprocessing) · **SHAP** (explainability) · **Streamlit** (web UI) · **pandas/numpy** (data) · **matplotlib/plotly** (charts)

## Quick Start

```bash
git clone https://github.com/yourusername/loan-approval-explainer.git
cd loan-approval-explainer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model
python preprocess.py
python train.py

# Run app
streamlit run app.py  # Opens at http://localhost:8501
```

## Project Structure

```
loan-approval-explainer/
├── README.md
├── requirements.txt
├── exploration.py          # Data loading and initial preprocessing
├── preprocess.py           # Feature engineering and scaling
├── train.py                # Model training script
├── app.py                  # Streamlit web application
├── data/
│   └── loan_data.csv       # Training dataset
├── models/
│   └── loan_model.pkl      # Saved trained model
```

## Model

**Logistic Regression** with SAGA solver
- Chosen for interpretability over black-box models (Random Forest, XGBoost)
- Linear coefficients make SHAP explanations clearer
- Features: StandardScaler (numeric) + OneHotEncoder (categorical)

**Why not Random Forest?** Higher accuracy (2-5%) isn't worth losing transparency in an explainability showcase.

## Explainability

**Local (per applicant):** Waterfall chart showing how each feature pushes toward approval/rejection  
**Global (model-wide):** Feature importance ranking across all predictions

SHAP breaks predictions into additive feature contributions: baseline + feature effects = final score

## Dataset

**Features:** Income, loan amount/term, credit score, asset values, dependents, education, employment status  
**Target:** Binary loan approval decision (Approved/Rejected)

## Future Ideas

Counterfactual explanations · Fairness metrics · Model comparison dashboard · REST API · Cloud deployment
