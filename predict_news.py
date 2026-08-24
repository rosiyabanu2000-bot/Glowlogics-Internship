import joblib

# Load trained model
model = joblib.load("best_fake_news_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("===================================")
print("      FAKE NEWS DETECTION SYSTEM")
print("===================================")

# Get news article from user
title = input("\nEnter news title: ")
text = input("Enter news article text: ")

# Combine title and text
news_content = title + " " + text

# Convert text into TF-IDF features
news_vector = tfidf.transform([news_content])

# Predict
prediction = model.predict(news_vector)[0]

# Display result
print("\n========== PREDICTION ==========")

if prediction == 0:
    print("Result: FAKE NEWS ❌")
else:
    print("Result: REAL NEWS ✅")