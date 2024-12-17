import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load training data
train_data = pd.read_csv('data/processed_train.csv')

# Text and labels
X_train = train_data['cleaned_text']
y_train = train_data['label']

# Define an NLP pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Text vectorization
    ('classifier', LogisticRegression())  # Logistic Regression classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'model/nlp_model.pkl')

print("Model training completed successfully!")

 
