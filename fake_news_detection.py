import pandas as pd

# Load Fake News
fake_df = pd.read_csv("dataset/fake_news.csv")

# Load True/Real News
true_df = pd.read_csv("dataset/true_news.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display results
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nDataset Information:")
print(df.info())
import pandas as pd

# ==========================================
# STEP 4: Load and Combine Dataset
# ==========================================

# Load Fake News
fake_df = pd.read_csv("dataset/fake_news.csv")

# Load Real News
true_df = pd.read_csv("dataset/true_news.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nLabel Distribution:")
print(df["label"].value_counts())

# ==========================================
# STEP 5: Data Cleaning
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing title or text
df = df.dropna(subset=["title", "text"])

# Convert title and text to string
df["title"] = df["title"].astype(str)
df["text"] = df["text"].astype(str)

# Remove unnecessary spaces
df["title"] = df["title"].str.strip()
df["text"] = df["text"].str.strip()

# Remove empty text rows
df = df[df["text"].str.len() > 0]

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# ==========================================
# STEP 6: Exploratory Data Analysis (EDA)
# ==========================================

print("\n========== EDA ==========")

# Basic statistical information
print("\nBasic Statistics:")
print(df.describe(include="all"))

# Label distribution
print("\nFake vs Real News:")
print(df["label"].value_counts())

# Subject distribution
print("\nNews Subjects:")
print(df["subject"].value_counts())

# Number of unique subjects
print("\nNumber of Unique Subjects:")
print(df["subject"].nunique())

# Title length
df["title_length"] = df["title"].str.len()

# Text length
df["text_length"] = df["text"].str.len()

print("\nTitle Length Statistics:")
print(df["title_length"].describe())

print("\nText Length Statistics:")
print(df["text_length"].describe())
# ==========================================
# STEP 7: Data Visualization
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fake vs Real News
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="label")

plt.title("Fake vs Real News")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()


# 2. News Subject Distribution
plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    y="subject",
    order=df["subject"].value_counts().index
)

plt.title("News Articles by Subject")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")
plt.tight_layout()
plt.show()


# 3. Title Length Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="title_length",
    bins=50,
    kde=True
)

plt.title("Title Length Distribution")
plt.xlabel("Title Length")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()


# 4. Text Length Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="text_length",
    bins=50,
    kde=True
)

plt.title("Article Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
# ==========================================
# STEP 8: Prepare ML Features
# ==========================================

# Combine title and article text
df["content"] = df["title"] + " " + df["text"]

# Convert labels to numerical values
df["label_num"] = df["label"].map({
    "FAKE": 0,
    "REAL": 1
})

# Input features and target
X = df["content"]
y = df["label_num"]

print("\nFeature and Target Shapes:")
print("X:", X.shape)
print("y:", y.shape)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

# Transform text into numerical features

print("\nTF-IDF Feature Shape:")
# ==========================================
# STEP 9: Train Multiple ML Models
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Input and target
X = df["content"]
y = df["label_num"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)


# ==========================================
# Model 1: Logistic Regression
# ==========================================

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train_tfidf, y_train)

logistic_pred = logistic_model.predict(X_test_tfidf)


# ==========================================
# Model 2: Multinomial Naive Bayes
# ==========================================

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)


# ==========================================
# Model 3: Linear SVM
# ==========================================

svm_model = LinearSVC()

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)


print("\nAll models trained successfully!")
# ==========================================
# STEP 10: Model Evaluation
# ==========================================

def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# Evaluate all models
results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_pred
    )
)

results.append(
    evaluate_model(
        "Naive Bayes",
        y_test,
        nb_pred
    )
)

results.append(
    evaluate_model(
        "Linear SVM",
        y_test,
        svm_pred
    )
)


# Convert results into DataFrame
results_df = pd.DataFrame(results)

# Display results
print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Display values as percentages
print("\n========== MODEL COMPARISON (%) ==========")

results_percentage = results_df.copy()

for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    results_percentage[column] = (
        results_percentage[column] * 100
    ).round(2)

print(results_percentage.to_string(index=False))
# ==========================================
# STEP 11: Select Best Model
# ==========================================

# Find the model with the highest F1-score
best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Model"
]

best_f1_score = results_df["F1-Score"].max()

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)
print("Best F1-Score:", round(best_f1_score * 100, 2), "%")
import joblib

# Select the trained best model
if best_model_name == "Logistic Regression":
    best_model = logistic_model

elif best_model_name == "Naive Bayes":
    best_model = nb_model

elif best_model_name == "Linear SVM":
    best_model = svm_model

# Save model and TF-IDF vectorizer
joblib.dump(best_model, "best_fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nBest model saved successfully!")
print("TF-IDF vectorizer saved successfully!")