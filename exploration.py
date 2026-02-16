import pandas as pd 
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----- Loading Dataset into a DataFrame -----
df = pd.read_csv('loan_approval_dataset.csv')

# ----- Initial Exploration of the Data -----
print(df.shape)
print(list(df.columns))
print(list(df.dtypes))
print(df.describe())

# ----- Checking for missing values and class imbalance -----
print(df.columns[df.isnull().any()])
print(df[' loan_status'].value_counts()*100/df[' loan_status'].count())

# ----- Data Preprocessing -----
print("Unique count in A:", df[' education'].nunique())
print("Unique count in B:", df[' self_employed'].nunique())
print("Unique count in C:", df[' loan_status'].nunique())



# ----- One-hot encode the categorical features ----- 
categorical_features = [' education', ' self_employed']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
df_encoded = pd.concat([df, encoded_df], axis=1)
df_encoded = df_encoded.drop(columns=categorical_features, axis=1)
df_encoded[" loan_status"] = [1 if " Approved" in status else 0 for status in df_encoded[" loan_status"]]
print(f"Encoded Data: \n{df_encoded}\n")
print(df_encoded.head())

# ----- Scale the numerical features -----
data_to_scale = df_encoded.drop(columns=[' education_ Graduate', ' education_ Not Graduate', ' self_employed_ No', ' self_employed_ Yes', 'loan_id', ' loan_status'], axis=1)
print(data_to_scale.head())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)
print(scaled_data.shape)
print(data_to_scale.columns)
scaled_df = pd.DataFrame(scaled_data, columns= ['no_of_dependents', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value'])
print(scaled_df.head())

# ----- Check the standard deviation of the scaled features -----
print(round(np.std(scaled_df[' income_annum']), 2))

# ----- Combine the scaled numerical features with the one-hot encoded categorical features and the target variable -----
scaled_data_combined = pd.concat([scaled_df, df_encoded[[' education_ Graduate', ' education_ Not Graduate', ' self_employed_ No', ' self_employed_ Yes', 'loan_id', ' loan_status']]], axis=1)
print(scaled_data_combined.head())

# ----- Reorder the columns so that the target variable is the first column and the loan_id is the last column -----
scaled_data_reordered = scaled_data_combined.iloc[:, [13, 0, 9, 10, 12, 11, 1, 2, 3, 4, 5, 6, 7, 8, 14]]
print(scaled_data_reordered.head())