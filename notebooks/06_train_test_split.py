
import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded features and target
X = pd.read_csv("data/processed/X_encoded.csv")
y = pd.read_csv("data/processed/y_target.csv")["Email_Status"]

print("Original feature shape:", X.shape)
print("Original target shape:", y.shape)

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the resulting shapes
print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2))

# Save the split datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nTraining and testing datasets saved successfully.")
