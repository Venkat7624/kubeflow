import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load test data
test_data = pd.read_csv('data/processed_test.csv')

# Text and labels
X_test = test_data['cleaned_text']
y_test = test_data['label']

# Load the trained model
pipeline = joblib.load('model/nlp_model.pkl')

# Make predictions
predictions = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, predictions))
