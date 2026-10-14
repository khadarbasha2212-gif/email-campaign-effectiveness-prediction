
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load cleaned data
df = pd.read_csv("data/processed/email_campaign_cleaned.csv")

X = df.drop(columns=["Email_ID", "Email_Status"])
y = df["Email_Status"]

numeric_columns = [
    "Subject_Hotness_Score",
    "Total_Past_Communications",
    "Word_Count",
    "Total_Links",
    "Total_Images"
]

categorical_columns = [
    "Email_Type",
    "Email_Source_Type",
    "Customer_Location",
    "Email_Campaign_Type",
    "Time_Email_sent_Category"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

balanced_scores = []
macro_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Fit preprocessing on this fold's training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Calculate weights using only this fold's training labels
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

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
        n_jobs=1
    )

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights
    )

    predictions = model.predict(X_val_processed)

    bal_acc = balanced_accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    balanced_scores.append(bal_acc)
    macro_f1_scores.append(macro_f1)

    print(
        f"Fold {fold}: "
        f"Balanced Accuracy = {bal_acc:.4f}, "
        f"Macro F1 = {macro_f1:.4f}"
    )

print("\n========== WEIGHTED XGBOOST CV RESULTS ==========")
print("Mean balanced accuracy:", round(np.mean(balanced_scores), 4))
print("Balanced accuracy std:", round(np.std(balanced_scores), 4))
print("Mean macro F1:", round(np.mean(macro_f1_scores), 4))
print("Macro F1 std:", round(np.std(macro_f1_scores), 4))
