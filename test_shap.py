import shap 
import joblib as jl
import train_model as tm
import preprocess as pp

# ----- Importing test data, model and preprocessor -----
df_test = tm.x_test
df_train = tm.x_train
model = jl.load('logistic_regression_model.pkl')
preproc = pp.preproc

explainer = shap.LinearExplainer(model, df_train)
shap_values = explainer.shap_values(df_test)
exp = explainer(df_test)
print(shap_values.shape)

# ----- Findind candidates for which we want to explain the model's predictions -----
for index, row in df_test.iterrows():
    prob = model.predict_proba([row])[0][1]
    if prob >= 0.85:
        print("Model's Decision: Approved")
        print(f"index {index} is classified as Approved with a probability of {prob:.4f}\n")
    if prob <= 0.15:
        print("Model's Decision: Rejected")
        print(f"index {index} is classified as Rejected with a probability of {prob:.4f}\n")
    if 0.48 <= prob <= 0.52:
        print("Model's Decision: Uncertain")
        print(f"index {index} is classified as Uncertain with a probability of {prob:.4f}\n")

# ----- Candidates chosen for explanation -----
rejected = df_test.iloc[[87]]
uncertain = df_test.iloc[[413]]
approved = df_test.iloc[[685]]

print(f"Rejected candidate: \n{rejected}\n")
print(f"Uncertain candidate: \n{uncertain}\n")
print(f"Approved candidate: \n{approved}\n")

shapvalue_rejected = explainer.shap_values(rejected)
shapvalue_uncertain = explainer.shap_values(uncertain)
shapvalue_approved = explainer.shap_values(approved)
print(f"SHAP values for rejected candidate: \n{shapvalue_rejected}\n")
print(f"SHAP values for uncertain candidate: \n{shapvalue_uncertain}\n")
print(f"SHAP values for approved candidate: \n{shapvalue_approved}\n")

print("Rejected")
shap.summary_plot(shapvalue_rejected, rejected, plot_type="bar")
print("Uncertain")
shap.summary_plot(shapvalue_uncertain, uncertain, plot_type="bar")
print("Approved")
shap.summary_plot(shapvalue_approved, approved, plot_type="bar")

shap.plots.waterfall(exp[0], max_display=10, show=True)
