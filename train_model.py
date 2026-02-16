from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import exploration

# ----- Splitting the data -----
features = exploration.scaled_data_reordered.drop(columns=[' loan_status', 'loan_id'], axis=1)
target = exploration.scaled_data_reordered[[' loan_status']]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print(x_train.shape)

# ----- Training the model -----
logit = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
logit.fit(x_train, y_train)
print(logit.score(x_train, y_train))
logit_pred = logit.predict(x_test)

print(classification_report(y_test, logit_pred))

#----- Finding the importance of the features ----- 
print(logit.coef_)

features_importance_logit = pd.DataFrame({'Feature': features.columns, 'Importance': logit.coef_[0]})
print(features_importance_logit.sort_values(by='Importance', ascending=False))