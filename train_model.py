from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import exploration
import joblib 

# ----- Splitting the data -----
features = exploration.scaled_data_reordered.drop(columns=[' loan_status', 'loan_id'], axis=1)
target = exploration.scaled_data_reordered[[' loan_status']]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.38, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
# print(x_train.shape)

# ----- Training the model -----
logit = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
logit.fit(x_train, y_train)
print(logit.score(x_train, y_train))
logit_pred = logit.predict(x_test)

# print(classification_report(y_test, logit_pred))
dict_report = classification_report(y_test, logit_pred, output_dict=True)

with open('model_metrics.txt', 'w') as f:
    f.write(f"Classification Report: \n{classification_report(y_test, logit_pred)}\n")
    f.write(f"Accuracy: {dict_report['accuracy']}\n")
    f.write(f"Precision (Approved): {dict_report['1']['precision']}\n")
    f.write(f"Recall (Approved): {dict_report['1']['recall']}\n")
    f.write(f"F1-score (Approved): {dict_report['1']['f1-score']}\n")
    f.write(f"Precision (Rejected): {dict_report['0']['precision']}\n")
    f.write(f"Recall (Rejected): {dict_report['0']['recall']}\n")
    f.write(f"F1-score (Rejected): {dict_report['0']['f1-score']}\n")

print("Model metrics saved to model_metrics.txt")

#----- Finding the importance of the features ----- 
print(logit.coef_)

features_importance_logit = pd.DataFrame({'Feature': features.columns, 'Importance': logit.coef_[0]})
print(features_importance_logit.sort_values(by='Importance', ascending=False))

confusion_matrix = confusion_matrix(y_test, logit_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=logit.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

joblib.dump(logit, 'logistic_regression_model.pkl')
print("Model saved as logistic_regression_model.pkl")