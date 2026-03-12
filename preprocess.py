import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df_raw = pd.read_csv('data/loan_approval_dataset.csv')

X = df_raw.drop(columns=['loan_id', ' loan_status'], axis=1)

numerical_features = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']
categorical_features = [' education', ' self_employed']
categorical_features_extended = [' education_ Graduate', ' education_ Not Graduate', ' self_employed_ No', ' self_employed_ Yes']   

# ----- Scale the numerical features and one-hot encode the categorical features using ColumnTransformer and Pipeline -----
preproc = ColumnTransformer(transformers=[(
    'num', StandardScaler(), numerical_features), 
    ('cat', OneHotEncoder(sparse_output=False), categorical_features)], 
    remainder='drop')

preproc.fit(X)

joblib.dump(preproc, 'models/preprocessor.pkl')

test_applicant = pd.DataFrame({
    ' no_of_dependents': [2],
    ' income_annum': [50000],
    ' loan_amount': [200000],
    ' loan_term': [360],
    ' cibil_score': [700],
    ' residential_assets_value': [100000],
    ' commercial_assets_value': [0],
    ' luxury_assets_value': [0],
    ' bank_asset_value': [50000],
    ' education': [' Graduate'],
    ' self_employed': [' No']
})

# print("Test applicant data before preprocessing:")
# print(test_applicant)
test_transformed = preproc.transform(test_applicant)
# print("Transformed shape: ", test_transformed.shape)
# print("Test applicant data after preprocessing:", test_transformed)
