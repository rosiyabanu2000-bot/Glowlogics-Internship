import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# STEP 5 — LOAD DATASET
# ============================================================

df = pd.read_csv("data/Mall_Customers.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())


# ============================================================
# STEP 6 — DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ============================================================
# STEP 7 — DATA UNDERSTANDING / EDA
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nUnique Values:")
print(df.nunique())


# ============================================================
# STEP 8 — VISUALIZATION
# ============================================================

# Age Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=10,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig("outputs/age_distribution.png")
plt.show()


# Annual Income vs Spending Score
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender",
    s=80
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()

plt.savefig("outputs/income_vs_spending.png")
plt.show()


# ============================================================
# STEP 9 — FEATURE SELECTION
# ============================================================

features = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\n" + "=" * 60)
print("SELECTED FEATURES")
print("=" * 60)

print(features.head())

print("\nFeature Shape:")
print(features.shape)


# ============================================================
# STEP 10 — DATA SCALING
# ============================================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

print("\n" + "=" * 60)
print("SCALED FEATURES")
print("=" * 60)

print(scaled_features[:5])


# ============================================================
# STEP 11 — ELBOW METHOD
# ============================================================

inertia = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_features)

    inertia.append(kmeans.inertia_)


# Plot Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/elbow_curve.png")
plt.show()


print("\n" + "=" * 60)
print("ELBOW METHOD COMPLETED")
print("=" * 60)

for k, value in zip(range(1, 11), inertia):
    print(f"K = {k} : Inertia = {value:.2f}")
    # ============================================================
# STEP 12 — K-MEANS CLUSTERING
# ============================================================

# Set optimal number of clusters
optimal_k = 4

# Create K-Means model
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Train the model
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original dataset
df["Cluster"] = cluster_labels

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING RESULTS")
print("=" * 60)

print("\nCluster Labels:")
print(df["Cluster"].value_counts().sort_index())

print("\nFirst 10 Customers with Clusters:")
print(df.head(10))
# ============================================================
# STEP 13 — CUSTOMER CLUSTER PROFILES
# ============================================================

cluster_summary = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print("\n" + "=" * 60)
print("CUSTOMER CLUSTER PROFILES")
print("=" * 60)

print(cluster_summary.round(2))
# ============================================================
# STEP 14 — CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()

# Save cluster visualization
plt.savefig("outputs/customer_clusters.png")
plt.show()
# ============================================================
# STEP 15 — BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

for cluster in cluster_summary.index:

    age = cluster_summary.loc[cluster, "Age"]
    income = cluster_summary.loc[cluster, "Annual Income (k$)"]
    spending = cluster_summary.loc[cluster, "Spending Score (1-100)"]

    print(f"\nCluster {cluster}:")
    print(f"  Average Age: {age:.2f}")
    print(f"  Average Annual Income: ${income:.2f}k")
    print(f"  Average Spending Score: {spending:.2f}")
    # ============================================================
# STEP 16 — CLUSTER EVALUATION
# ============================================================

# Calculate Silhouette Score
silhouette = silhouette_score(
    scaled_features,
    cluster_labels
)

print("\n" + "=" * 60)
print("CLUSTER EVALUATION")
print("=" * 60)

print(f"Silhouette Score: {silhouette:.4f}")
# ============================================================
# STEP 17 — SAVE RESULTS
# ============================================================

# Save customer data with cluster labels
df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# Save cluster summary
cluster_summary.round(2).to_csv(
    "outputs/cluster_summary.csv"
)

print("\n" + "=" * 60)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 60)

print("\nSaved files:")
print("1. outputs/customer_segments.csv")
print("2. outputs/cluster_summary.csv")