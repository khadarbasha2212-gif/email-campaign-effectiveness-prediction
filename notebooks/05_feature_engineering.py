
import pandas as pd
from pathlib import Path

# Load cleaned dataset
input_path = Path(
    r"F:\AIML\email_campaign_prediction\data\processed\email_campaign_cleaned.csv"
)

df = pd.read_csv(input_path)

# Remove identifier because it is not a meaningful predictive feature
df = df.drop(columns=["Email_ID"])

# Define target and input features
target_column = "Email_Status"

X = df.drop(columns=[target_column])
y = df[target_column]

# Specify categorical columns
categorical_columns = [
    "Email_Type",
    "Email_Source_Type",
    "Customer_Location",
    "Email_Campaign_Type",
    "Time_Email_sent_Category"
]

# Convert category codes to strings so they are encoded categorically
for col in categorical_columns:
    X[col] = X[col].astype(str)

# One-hot encode categorical features
X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

print("Original input shape:", X.shape)
print("Encoded input shape:", X_encoded.shape)

print("\nEncoded feature names:")
print(X_encoded.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values in encoded features:", X_encoded.isnull().sum().sum())

# Save prepared features for inspection
output_dir = Path(
    r"F:\AIML\email_campaign_prediction\data\processed"
)

X_encoded.to_csv(output_dir / "X_encoded.csv", index=False)
y.to_frame().to_csv(output_dir / "y_target.csv", index=False)

print("\nEncoded features saved to X_encoded.csv")
print("Target saved to y_target.csv")
