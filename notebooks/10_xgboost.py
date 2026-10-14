
import pandas as pd

from xgboost import XGBClassifier
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

# Create the XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")

# Train the model
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Evaluate
print("\n========== XGBOOST EVALUATION ==========")

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

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10).round(4))
