# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project performs customer segmentation using Machine Learning and K-Means Clustering.

The objective is to group customers based on their:

- Age
- Annual Income
- Spending Score

Customer segmentation helps businesses understand different customer groups and develop targeted marketing strategies.

---

## 🎯 Objectives

- Analyze customer demographic and spending data.
- Perform data cleaning and exploratory data analysis.
- Select relevant features for clustering.
- Standardize the selected features.
- Determine the optimal number of clusters using the Elbow Method.
- Apply K-Means Clustering.
- Visualize customer segments.
- Evaluate clustering quality using Silhouette Score.
- Generate business insights from customer segments.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

---

## 📂 Dataset

The project uses the Mall Customers dataset.

The dataset contains 200 customer records with the following columns:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Customer spending score |

---

## 🔄 Project Workflow

1. Dataset collection
2. Data loading
3. Data cleaning
4. Exploratory Data Analysis
5. Data visualization
6. Feature selection
7. Feature scaling
8. Elbow Method
9. K-Means Clustering
10. Cluster profiling
11. Cluster visualization
12. Silhouette Score evaluation
13. Results generation

---

## 🤖 Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data points into a predefined number of clusters.

In this project, K-Means is used to identify groups of customers with similar characteristics.

---

## 📉 Elbow Method

The Elbow Method is used to identify a suitable number of clusters by analyzing the Within-Cluster Sum of Squares (inertia).

The inertia values are calculated for different values of K and plotted to identify the elbow point.

---

## 📊 Visualizations

The project generates the following visualizations:

- Customer Age Distribution
- Annual Income vs Spending Score
- Elbow Curve
- Customer Clusters

---

## 📈 Model Evaluation

The clustering result is evaluated using the Silhouette Score.

A higher Silhouette Score generally indicates better-defined and better-separated clusters.

---

## 💡 Business Insights

Customer segmentation can help businesses:

- Identify high-value customers.
- Identify customers with high spending potential.
- Develop targeted marketing campaigns.
- Improve customer retention strategies.
- Personalize offers and promotions.
- Understand different customer groups.

---

## 📁 Project Structure

```text
Customer_Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── outputs/
│   ├── age_distribution.png
│   ├── income_vs_spending.png
│   ├── elbow_curve.png
│   ├── customer_clusters.png
│   ├── customer_segments.csv
│   └── cluster_summary.csv
│
├── screenshots/
│
├── .venv/
│
├── customer_segmentation.py
│
├── README.md
│
└── requirements.txt