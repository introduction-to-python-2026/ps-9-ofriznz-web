from sklearn.metrics import accuracy_score

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
