import pandas as pd
import matplotlib.pyplot as plt

# Load model results
results = pd.read_csv("model_results.csv")

print("===== MODEL COMPARISON =====")
print(results.to_string(index=False))

# Find best model based on F1 Score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\n===== BEST MODEL =====")
print("Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))

# Create comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("ML Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()

plt.savefig("graphs/model_comparison.png")
plt.show()

print("\nComparison graph saved successfully!")