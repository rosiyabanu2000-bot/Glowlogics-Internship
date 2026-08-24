import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =====================================================
# STEP 1: LOAD DATA
# =====================================================

print("=" * 60)
print("DISEASE PREDICTION PROJECT")
print("=" * 60)

train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

print("\nTraining Shape:", train_df.shape)
print("Testing Shape:", test_df.shape)

print("\nFirst 5 Training Rows:")
print(train_df.head())


# =====================================================
# STEP 2: CLEAN DATA
# =====================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nMissing values in training data:")
print(train_df.isnull().sum().sum())

print("\nDuplicate rows:")
print(train_df.duplicated().sum())

train_df = train_df.drop_duplicates()

print("\nTraining shape after removing duplicates:")
print(train_df.shape)


# =====================================================
# STEP 3: TARGET COLUMN
# =====================================================

target_column = "prognosis"

print("\nTarget Column:", target_column)

print("\nNumber of diseases:")
print(train_df[target_column].nunique())

print("\nDisease distribution:")
print(train_df[target_column].value_counts())


# =====================================================
# STEP 4: EDA GRAPH
# =====================================================

plt.figure(figsize=(14, 7))

train_df[target_column].value_counts().plot(kind="bar")

plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Records")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("disease_distribution.png")

plt.show()


# =====================================================
# STEP 5: PREPARE FEATURES
# =====================================================

X = train_df.drop(columns=[target_column])
y = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# =====================================================
# STEP 6: ENCODE TARGET
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_test_encoded = label_encoder.transform(y_test)

print("\nNumber of diseases:", len(label_encoder.classes_))


# =====================================================
# STEP 7: TRAIN / VALIDATION SPLIT
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)


# =====================================================
# STEP 8: TRAIN MULTIPLE MODELS
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "KNN":
        KNeighborsClassifier(n_neighbors=5)
}


results = []


print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1

    })


# =====================================================
# STEP 9: MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# =====================================================
# STEP 10: MODEL COMPARISON GRAPH
# =====================================================

results_df.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()


# =====================================================
# STEP 11: BEST MODEL
# =====================================================

best_index = results_df["F1 Score"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_model = models[
    best_model_name
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model:", best_model_name)

print(
    "Accuracy:",
    results_df.loc[best_index, "Accuracy"]
)

print(
    "Precision:",
    results_df.loc[best_index, "Precision"]
)

print(
    "Recall:",
    results_df.loc[best_index, "Recall"]
)

print(
    "F1 Score:",
    results_df.loc[best_index, "F1 Score"]
)


# =====================================================
# STEP 12: TEST BEST MODEL
# =====================================================

best_model.fit(X, y_encoded)

test_predictions = best_model.predict(X_test)


print("\n" + "=" * 60)
print("TEST DATA PERFORMANCE")
print("=" * 60)

print(
    "Accuracy:",
    accuracy_score(
        y_test_encoded,
        test_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test_encoded,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test_encoded,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test_encoded,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)


# =====================================================
# STEP 13: CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test_encoded,
        test_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# =====================================================
# STEP 14: SAVE MODEL
# =====================================================

model_data = {

    "model": best_model,

    "label_encoder": label_encoder,

    "features": X.columns.tolist()

}


joblib.dump(
    model_data,
    "best_disease_model.pkl"
)


print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "File: best_disease_model.pkl"
)

print("\nPROJECT TRAINING COMPLETED!")