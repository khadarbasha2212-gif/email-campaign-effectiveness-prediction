
import pandas as pd
from pathlib import Path

file_path = Path(
    r"F:\AIML\email_campaign_prediction\data\raw\data_email_campaign.csv"
)

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# 1. Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Check duplicate email IDs
print("Duplicate Email_ID values:", df["Email_ID"].duplicated().sum())

# 3. Calculate missing-value percentages
missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

print("\nMissing-value report:")
print(missing[missing["Missing_Count"] > 0])

# 4. Inspect target classes
print("\nEmail_Status distribution:")
print(df["Email_Status"].value_counts(dropna=False))

print("\nEmail_Status percentages:")
print(
    (df["Email_Status"].value_counts(normalize=True) * 100)
    .round(2)
)

# 5. Check numeric ranges and unusual values
print("\nNumeric summary:")
print(df.describe().T)

# 6. Check unique values in categorical columns
categorical_columns = [
    "Email_Type",
    "Email_Source_Type",
    "Customer_Location",
    "Email_Campaign_Type",
    "Time_Email_sent_Category",
    "Email_Status"
]

print("\nCategorical value counts:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
