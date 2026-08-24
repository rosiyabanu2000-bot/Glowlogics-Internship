# 📰 Fake News Detection System

## 📌 Project Overview

The Fake News Detection System is a Machine Learning project designed to classify news articles as **Fake** or **Real** based on their textual content.

The system uses Natural Language Processing (NLP) techniques to convert news text into numerical features and applies Machine Learning classification algorithms to identify potentially fake news.

---

## 🎯 Objectives

* Detect whether a news article is Fake or Real.
* Perform data cleaning and exploratory data analysis.
* Convert textual data into numerical features using TF-IDF.
* Train and compare multiple Machine Learning models.
* Evaluate models using Accuracy, Precision, Recall, and F1-Score.
* Select the best-performing model.
* Build a prediction system for new news articles.

---

## 📊 Dataset

The project uses the **ISOT Fake and Real News Dataset**.

The dataset contains:

* Fake news articles
* Real news articles
* News titles
* Article text
* Subject/category
* Publication date

### Dataset Statistics

| Property       |  Value |
| -------------- | -----: |
| Total Articles | 46,962 |
| Fake Articles  | 23,481 |
| Real Articles  | 23,481 |
| Total Columns  |      5 |

### Dataset Columns

* `title` — News headline
* `text` — Full news article
* `subject` — News category
* `date` — Publication date
* `label` — FAKE or REAL

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* TF-IDF
* Natural Language Processing (NLP)

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Train/Test Split
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction System
```

---

## 🧹 Data Cleaning

The following preprocessing operations were performed:

* Checked missing values.
* Checked duplicate records.
* Removed duplicate rows.
* Removed rows with missing title or article text.
* Removed unnecessary spaces.
* Converted text columns into string format.
* Removed empty article text.

---

## 📈 Exploratory Data Analysis

The following analysis was performed:

* Fake vs Real news distribution.
* News subject distribution.
* Title length analysis.
* Article text length analysis.
* Basic statistical analysis.

---

## 📊 Visualizations

The project generates the following graphs:

1. Fake vs Real News
2. News Articles by Subject
3. Title Length Distribution
4. Article Text Length Distribution

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated:

### 1. Logistic Regression

A linear classification algorithm used to classify news articles into Fake and Real categories.

### 2. Multinomial Naive Bayes

A probabilistic classification algorithm that works effectively with text-based features.

### 3. Linear Support Vector Machine

A linear classification algorithm that is commonly effective for high-dimensional text classification problems.

---

## 🔢 Feature Extraction

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert news text into numerical feature vectors.

The title and article text are combined before feature extraction.

```text
News Title + Article Text
          ↓
       TF-IDF
          ↓
Numerical Feature Vector
          ↓
Machine Learning Model
```

---

## 📏 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

The model with the highest F1-Score is selected as the best model.

---

## 🔍 Prediction System

A separate Python script named `predict_news.py` is used to test new news articles.

Run:

```bash
python predict_news.py
```

The system asks for:

```text
Enter news title:
Enter news article text:
```

It then predicts:

```text
Result: FAKE NEWS
```

or

```text
Result: REAL NEWS
```

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── dataset/
│   ├── fake_news.csv
│   └── true_news.csv
│
├── screenshots/
│   ├── fake_vs_real.png
│   ├── subject_distribution.png
│   ├── title_length.png
│   └── text_length.png
│
├── .venv/
│
├── fake_news_detection.py
├── predict_news.py
├── best_fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate environment

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python fake_news_detection.py
```

### 5. Run the prediction system

```bash
python predict_news.py
```

---

## 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## ⚠️ Disclaimer

This system is a Machine Learning classification project. Its prediction is based on patterns learned from the training dataset and should not be treated as independent verification of whether a news story is factually true.

---

## 👩‍💻 Project Type

**Machine Learning / Natural Language Processing Project**

**Domain:** Fake News Detection

**Purpose:** Educational and internship project
