import pandas as pd
from pathlib import Path

file_path = Path(
    r"F:\AIML\email_campaign_prediction\data\raw\data_email_campaign.csv"
)

if not file_path.exists():
    print("CSV file not found:", file_path)
else:
    df = pd.read_csv(file_path)

    print("\n--- DATASET SHAPE ---")
    print(df.shape)

    print("\n--- ALL COLUMN NAMES ---")
    for i, column in enumerate(df.columns, start=1):
        print(i, repr(column))

    print("\n--- FIRST FIVE ROWS ---")
    print(df.head().to_string())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())