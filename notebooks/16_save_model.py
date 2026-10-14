
import os
import joblib

from sklearn.pipeline import Pipeline

# Create the models folder if needed
os.makedirs("models", exist_ok=True)

# Combine preprocessing and model into one reusable pipeline
email_campaign_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Save the complete pipeline
joblib.dump(
    email_campaign_pipeline,
    "models/weighted_xgboost_pipeline.joblib"
)

print("Model pipeline saved successfully!")
print("Path: models/weighted_xgboost_pipeline.joblib")
