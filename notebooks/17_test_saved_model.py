
import joblib
import pandas as pd

# 1. Load the saved pipeline
model = joblib.load(
    "models/weighted_xgboost_pipeline.joblib"
)

# 2. Create sample email data
# Use the same feature columns as the training dataset.
new_email = pd.DataFrame([{
    "Email_Type": 1,
    "Subject_Hotness_Score": 2.5,
    "Email_Source_Type": "Marketing",
    "Customer_Location": "Unknown",
    "Email_Campaign_Type": 2,
    "Total_Past_Communications": 5,
    "Time_Email_sent_Category": 2,
    "Word_Count": 150,
    "Total_Links": 3,
    "Total_Images": 1
}])

# 3. Predict the class
predicted_status = model.predict(new_email)[0]

# 4. Get probabilities for each class
probabilities = model.predict_proba(new_email)[0]

print("\n========== NEW EMAIL PREDICTION ==========")
print("Predicted status:", predicted_status)

print("\nPredicted probability by class:")
for class_label, probability in enumerate(probabilities):
    print(f"Class {class_label}: {probability:.2%}")
