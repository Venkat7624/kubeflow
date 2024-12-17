import pandas as pd
from sklearn.model_selection import train_test_split
import re
# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text.strip()
# Load raw data
df = pd.read_csv('data/raw_text_data.csv')
# Clean text
df['cleaned_text'] = df['text'].apply(clean_text)
# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
# Save processed data
train.to_csv('data/processed_train.csv', index=False)
test.to_csv('data/processed_test.csv', index=False)
print("Data preprocessing completed successfully!")
 
has context menu
