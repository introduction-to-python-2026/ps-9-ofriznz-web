from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lab_setup_do_not_edit
import pandas as pd
import joblib

input_features = ['spread1', 'PPE']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=15)

model = LogisticRegression(random_state=15)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("Accuracy requirement met!")
else:
    print("Accuracy is below 0.8. Consider re-evaluating features or model choices.")
