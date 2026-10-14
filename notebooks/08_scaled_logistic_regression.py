
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the existing train/test split
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")["Email_Status"]
y_test = pd.read_csv("data/processed/y_test.csv")["Email_Status"]

# Scale numeric columns; keep one-hot encoded columns unchanged
numeric_columns = [
    "Subject_Hotness_Score",
    "Total_Past_Communications",
    "Word_Count",
    "Total_Links",
    "Total_Images"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("scale_numeric", StandardScaler(), numeric_columns)
    ],
    remainder="passthrough"
)

# Build the complete pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training scaled Logistic Regression model...")

# Fit preprocessing and model using training data only
model.fit(X_train, y_train)

# Predict on unseen test data
y_pred = model.predict(X_test)

# Evaluate
print("\n========== SCALED MODEL EVALUATION ==========")

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
