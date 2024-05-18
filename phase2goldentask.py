import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Generate synthetic email messages
np.random.seed(42)

ham_messages = [
    "Hey, how are you?",
    "Let's meet for lunch tomorrow.",
    "Can you send me the report?",
    "Happy Birthday!",
    "Your order has been shipped.",
    "Meeting at 10 AM in Conference Room A.",
    "Looking forward to your presentation.",
    "Please find the attached document.",
    "Let's catch up over coffee.",
    "Thank you for your email."
]

spam_messages = [
    "Congratulations! You've won a $1000 gift card.",
    "Exclusive offer just for you. Click here to claim.",
    "Get rich quick with this one simple trick.",
    "You have a new message from your bank.",
    "Your account has been compromised. Click here to secure it.",
    "Limited time offer! Buy now and save 50%.",
    "Earn money from home. No experience needed.",
    "You've been selected for a free vacation.",
    "Act now to claim your free prize.",
    "This is not a scam. Click to find out more."
]

# Create a list of labels (1 for spam, 0 for ham)
labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

# Create messages based on labels
messages = [
    np.random.choice(ham_messages) if label == 0 else np.random.choice(spam_messages)
    for label in labels
]

# Create a DataFrame
df_synthetic = pd.DataFrame({
    'label': labels,
    'message': messages
})

# Map labels to 'ham' and 'spam'
df_synthetic['label'] = df_synthetic['label'].map({0: 'ham', 1: 'spam'})

# Display the first few rows
print(df_synthetic.head())

# Save to CSV (optional)
df_synthetic.to_csv('synthetic_email_data.csv', index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Generate synthetic dataset
num_samples = 1000

ham_messages = [
    "Hey, how are you?",
    "Let's meet for lunch tomorrow.",
    "Can you send me the report?",
    "Happy Birthday!",
    "Your order has been shipped.",
    "Meeting at 10 AM in Conference Room A.",
    "Looking forward to your presentation.",
    "Please find the attached document.",
    "Let's catch up over coffee.",
    "Thank you for your email."
]

spam_messages = [
    "Congratulations! You've won a $1000 gift card.",
    "Exclusive offer just for you. Click here to claim.",
    "Get rich quick with this one simple trick.",
    "You have a new message from your bank.",
    "Your account has been compromised. Click here to secure it.",
    "Limited time offer! Buy now and save 50%.",
    "Earn money from home. No experience needed.",
    "You've been selected for a free vacation.",
    "Act now to claim your free prize.",
    "This is not a scam. Click to find out more."
]

labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

messages = [
    np.random.choice(ham_messages) if label == 0 else np.random.choice(spam_messages)
    for label in labels
]

df_synthetic = pd.DataFrame({
    'label': labels,
    'message': messages
})

df_synthetic['label'] = df_synthetic['label'].map({0: 'ham', 1: 'spam'})

# Display the first few rows
print(df_synthetic.head())

# Save to CSV (optional)
df_synthetic.to_csv('synthetic_email_data.csv', index=False)

# Convert labels to binary (spam=1, ham=0)
df_synthetic['label'] = df_synthetic['label'].map({'spam': 1, 'ham': 0})

# Remove punctuation and numbers, convert to lower case
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove whitespace
    return text

df_synthetic['message'] = df_synthetic['message'].apply(preprocess_text)

# Remove stopwords and apply stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df_synthetic['message'] = df_synthetic['message'].apply(clean_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_synthetic['message'], df_synthetic['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
