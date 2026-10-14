
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# Load training and testing data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")["Email_Status"]
y_test = pd.read_csv("data/processed/y_test.csv")["Email_Status"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n========== RANDOM FOREST EVALUATION ==========")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    zero_division=0
))

# Show the ten most important features
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10).round(4))
