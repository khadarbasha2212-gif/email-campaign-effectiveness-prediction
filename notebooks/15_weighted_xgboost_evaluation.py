
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# 1. Load the original raw data
df = pd.read_csv("data/raw/data_email_campaign.csv")

# Remove rows with a missing target, if any
df = df.dropna(subset=["Email_Status"]).copy()

# Separate features and target
X = df.drop(columns=["Email_ID", "Email_Status"])
y = df["Email_Status"].astype(int)

# Identify feature types
numeric_features = [
    "Subject_Hotness_Score",
    "Total_Past_Communications",
    "Word_Count",
    "Total_Links",
    "Total_Images"
]

categorical_features = [
    "Email_Type",
    "Email_Source_Type",
    "Customer_Location",
    "Email_Campaign_Type",
    "Time_Email_sent_Category"
]

# 2. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Build preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit preprocessing using training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 4. Calculate class weights using training labels only
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# 5. Train weighted XGBoost
model = XGBClassifier(
    n_estimators=200,
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

model.fit(
    X_train_processed,
    y_train,
    sample_weight=sample_weights
)

# 6. Evaluate on the held-out test set
y_pred = model.predict(X_test_processed)

print("\n========== WEIGHTED XGBOOST TEST RESULTS ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(
    "Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred):.4f}"
)
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import os
import joblib

os.makedirs("models", exist_ok=True)

email_campaign_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

joblib.dump(
    email_campaign_pipeline,
    "models/weighted_xgboost_pipeline.joblib"
)

print("\nModel pipeline saved successfully!")