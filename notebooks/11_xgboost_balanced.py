
import pandas as pd

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# Load train/test data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")["Email_Status"]
y_test = pd.read_csv("data/processed/y_test.csv")["Email_Status"]

# Give more weight to less frequent classes
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Training class distribution:")
print(y_train.value_counts())

print("\nTraining class weights:")
for class_label in sorted(y_train.unique()):
    print(
        f"Class {class_label}:",
        round(
            sample_weights[y_train.to_numpy() == class_label][0],
            3
        )
    )

# Create XGBoost model
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

print("\nTraining class-weighted XGBoost model...")

# Train with sample weights
model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n========== BALANCED XGBOOST EVALUATION ==========")

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
