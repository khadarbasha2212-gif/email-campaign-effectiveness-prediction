
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Load the original cleaned data
# This lets preprocessing happen inside each training fold.
df = pd.read_csv("data/processed/email_campaign_cleaned.csv")

X = df.drop(columns=["Email_ID", "Email_Status"])
y = df["Email_Status"]

# Convert categorical columns to strings so OneHotEncoder can process them
categorical_columns = [
    "Email_Type",
    "Email_Source_Type",
    "Customer_Location",
    "Email_Campaign_Type",
    "Time_Email_sent_Category"
]

numeric_columns = [
    "Subject_Hotness_Score",
    "Total_Past_Communications",
    "Word_Count",
    "Total_Links",
    "Total_Images"
]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns
        ),
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

# XGBoost classifier
classifier = XGBClassifier(
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

# Complete pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# Stratified 3-fold cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Score each fold on balanced accuracy and macro F1
from sklearn.metrics import f1_score

scoring = {
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro")
}

print("Running 3-fold cross-validation...")
print("This may take a few minutes.")

# Note: preprocessing is fit independently within each fold.
# Sample weights are not passed in this initial CV experiment.
scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=1
)

print("\n========== CROSS-VALIDATION RESULTS ==========")

for metric in scoring:
    values = scores[f"test_{metric}"]
    print(f"\n{metric}:")
    print("Fold scores:", values.round(4))
    print("Mean:", round(values.mean(), 4))
    print("Standard deviation:", round(values.std(), 4))
