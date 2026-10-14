
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# Load training and testing data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")["Email_Status"]
y_test = pd.read_csv("data/processed/y_test.csv")["Email_Status"]

print("Training model...")

# Create the Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions on unseen test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\n========== MODEL EVALUATION ==========")

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)
