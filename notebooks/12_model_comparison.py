
import pandas as pd

# Results from our completed experiments
results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.6127,
        "Balanced Accuracy": 0.5183,
        "Macro F1": 0.39
    },
    {
        "Model": "Scaled Logistic Regression",
        "Accuracy": 0.6279,
        "Balanced Accuracy": 0.5267,
        "Macro F1": 0.40
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.6727,
        "Balanced Accuracy": 0.5113,
        "Macro F1": 0.44
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.8133,
        "Balanced Accuracy": 0.3837,
        "Macro F1": 0.39
    },
    {
        "Model": "Class-weighted XGBoost",
        "Accuracy": 0.6501,
        "Balanced Accuracy": 0.5350,
        "Macro F1": 0.43
    }
]

df = pd.DataFrame(results)

# Save report
df.to_csv("reports/model_comparison.csv", index=False)

print("MODEL COMPARISON")
print(df.to_string(index=False))

print("\nBest balanced accuracy:")
print(
    df.loc[df["Balanced Accuracy"].idxmax()]
    .to_string()
)

print("\nComparison saved to reports/model_comparison.csv")
