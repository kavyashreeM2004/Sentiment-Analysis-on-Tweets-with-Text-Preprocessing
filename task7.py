# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')

# Sample tweet data (replace with your dataset)
data = {
    'tweet': [
        "I love this product! It's amazing. #happy",
        "This is the worst experience ever. Totally disappointed.",
        "Great job! Really enjoyed using this. :)",
        "I hate it. Completely useless and frustrating!",
        "Best purchase I've made, highly recommend it.",
        "Terrible! Will never buy again. Waste of money."
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

# Create DataFrame
df = pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(filtered_tokens)

# Apply preprocessing
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Define features and labels
X = df['cleaned_tweet']
y = df['sentiment']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example: Predict sentiment for new tweet
new_tweet = "Absolutely love this! Highly recommend."
new_tweet_clean = preprocess_text(new_tweet)
new_tweet_vec = vectorizer.transform([new_tweet_clean])
prediction = model.predict(new_tweet_vec)
print(f"\nNew tweet sentiment prediction: {prediction[0]}")
