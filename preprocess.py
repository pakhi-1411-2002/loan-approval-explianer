import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import exploration

# ----- Scale the numerical features and one-hot encode the categorical features using ColumnTransformer and Pipeline -----
scaler = ColumnTransformer([('scaler', StandardScaler(), ['no_of_dependents', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value'])], remainder='passthrough', verbose_feature_names_out=False)
encoder = ColumnTransformer([('encoder', OneHotEncoder(sparse_output=False), [' education_ Graduate', ' education_ Not Graduate', ' self_employed_ No', ' self_employed_ Yes'])], remainder='passthrough', verbose_feature_names_out=False)

preproc = Pipeline(steps=[('scaler', scaler), ('encoder', encoder)])

# ----- Apply the transformations to the data -----
new_scaled_data_reordered = exploration.scaled_data_reordered.drop(columns=['loan_id', ' loan_status'], axis=1)

df_std = preproc.fit_transform(new_scaled_data_reordered)
df_std = pd.DataFrame(df_std, columns=preproc.get_feature_names_out())

print(df_std.head(5))
