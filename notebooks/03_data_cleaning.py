
import pandas as pd
from pathlib import Path

# Input and output locations
input_path = Path(
    r"F:\AIML\email_campaign_prediction\data\raw\data_email_campaign.csv"
)

output_path = Path(
    r"F:\AIML\email_campaign_prediction\data\processed\email_campaign_cleaned.csv"
)

# Load the original dataset
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Step 1: Remove exact duplicate rows
df = df.drop_duplicates()

# Step 2: Remove duplicate email IDs, if any
df = df.drop_duplicates(subset=["Email_ID"], keep="first")

# Step 3: Check the target column
print("\nTarget values before cleaning:")
print(df["Email_Status"].value_counts(dropna=False))

# Rows without a target cannot be used for supervised training
df = df.dropna(subset=["Email_Status"])

# Step 4: Fill missing categorical location with an explicit label
df["Customer_Location"] = (
    df["Customer_Location"]
    .fillna("Unknown")
    .astype(str)
)

# Step 5: Fill missing numeric values with their median
numeric_missing_columns = [
    "Total_Past_Communications",
    "Total_Links",
    "Total_Images"
]

for col in numeric_missing_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Step 6: Confirm the results
print("\nShape after cleaning:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTarget distribution after cleaning:")
print(df["Email_Status"].value_counts())

# Step 7: Save the cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nCleaned data saved to:")
print(output_path)
