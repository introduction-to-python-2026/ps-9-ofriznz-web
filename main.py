from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

input_features = ['spread1', 'PPE']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=15)

model_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(random_state=15))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

joblib.dump(model_pipeline, 'model.joblib') 
print("Model saved successfully!")
