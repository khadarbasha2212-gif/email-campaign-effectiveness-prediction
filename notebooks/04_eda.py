
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned data
file_path = Path(
    r"F:\AIML\email_campaign_prediction\data\processed\email_campaign_cleaned.csv"
)

df = pd.read_csv(file_path)

# Create a folder for charts
output_dir = Path(
    r"F:\AIML\email_campaign_prediction\reports"
)
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Distribution of email status
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Email_Status")
plt.title("Distribution of Email Status")
plt.xlabel("Email Status")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig(output_dir / "01_email_status.png")
plt.show()

# 2. Email campaign type vs email status
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Email_Campaign_Type",
    hue="Email_Status"
)
plt.title("Email Campaign Type vs Email Status")
plt.xlabel("Campaign Type")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig(output_dir / "02_campaign_status.png")
plt.show()

# 3. Email status by customer location
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="Customer_Location",
    hue="Email_Status",
    order=sorted(df["Customer_Location"].unique())
)
plt.title("Customer Location vs Email Status")
plt.xlabel("Customer Location")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig(output_dir / "03_location_status.png")
plt.show()

# 4. Subject hotness score distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Subject_Hotness_Score",
    bins=30,
    kde=True
)
plt.title("Distribution of Subject Hotness Score")
plt.tight_layout()
plt.savefig(output_dir / "04_subject_hotness.png")
plt.show()

# 5. Word count by email status
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="Email_Status",
    y="Word_Count"
)
plt.title("Email Word Count by Email Status")
plt.tight_layout()
plt.savefig(output_dir / "05_word_count_status.png")
plt.show()

# 6. Average email status by campaign type
# Use proportions instead of raw counts to compare groups fairly.
campaign_status = pd.crosstab(
    df["Email_Campaign_Type"],
    df["Email_Status"],
    normalize="index"
) * 100

print("\nEmail status percentages within campaign type:")
print(campaign_status.round(2))

print("\nEDA charts saved to:", output_dir)

