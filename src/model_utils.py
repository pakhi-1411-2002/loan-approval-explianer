# src/model_utils.py
import joblib
import numpy as np

def load_models():
    model = joblib.load('models/logistic_regression_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    return model, preprocessor

def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
            
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names)
        else:
            if isinstance(columns, list):
                feature_names.extend(columns)
            else:
                feature_names.append(columns)
    
    return feature_names

def make_prediction(model, preprocessor, user_input_dict):
    import pandas as pd
    
    user_df = pd.DataFrame([user_input_dict])
    processed_data = preprocessor.transform(user_df)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0]
    
    return prediction, probability, processed_data