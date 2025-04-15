import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        
        # Tokenization
        from nltk.tokenize import wordpunct_tokenize
        words = wordpunct_tokenize(text)
        
        # Remove stopwords and punctuation, and stem
        words = [self.ps.stem(word) for word in words if word not in self.stop_words and word not in string.punctuation]
        
        return ' '.join(words)
    
    def train(self, data_path):
        # Load data - The SMS Spam Collection dataset has a different format
        # It uses tab as delimiter and doesn't have a header
        try:
            # First attempt with tab delimiter (SMS Spam Collection format)
            df = pd.read_csv(data_path, encoding='latin-1', sep='\t', names=['label', 'text'], header=None)
            print(f"Loaded {len(df)} records from {data_path} using tab delimiter")
        except Exception as e:
            print(f"Error loading with tab delimiter: {e}")
            try:
                # Second attempt with comma delimiter
                df = pd.read_csv(data_path, encoding='latin-1')
                print(f"Loaded {len(df)} records from {data_path} using comma delimiter")
                
                # Check if we have the expected columns
                if len(df.columns) >= 2:
                    # Use the first two columns and rename them
                    df = df.iloc[:, 0:2]
                    df.columns = ['label', 'text']
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                raise ValueError(f"Could not load dataset at {data_path}. Please check the file format.")
        
        print(f"First few rows:\n{df.head()}")
        print(f"Columns: {df.columns}")
        
        # Map labels to numeric values
        if df['label'].dtype == 'object':
            # Convert labels to lowercase to handle case variations
            df['label'] = df['label'].str.lower()
            # Map 'ham' to 0 and 'spam' to 1
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Check if there are any NaN values after mapping
            if df['label'].isna().any():
                print(f"Warning: Found unmapped labels: {df.loc[df['label'].isna(), 'label'].unique()}")
                # Drop rows with unmapped labels
                df.dropna(subset=['label'], inplace=True)
        
        # Drop rows with missing text
        df.dropna(subset=['text'], inplace=True)
        
        # Confirm we have data after preprocessing
        if len(df) == 0:
            raise ValueError("After preprocessing, the dataset is empty. Please check the data format.")
        
        print(f"Dataset size after preprocessing: {len(df)} records")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, email):
        # Preprocess the input email
        processed_email = self.preprocess_text(email)
        
        # Vectorize
        email_vec = self.vectorizer.transform([processed_email])
        
        # Predict
        prediction = self.model.predict(email_vec)
        probability = self.model.predict_proba(email_vec)
        
        return prediction[0], probability[0]
    
    def save_model(self, model_path='spam_model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_model(self, model_path='spam_model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

def train_and_save_model():
    detector = SpamDetector()
    detector.train('data/spam.csv')
    detector.save_model()
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    train_and_save_model()