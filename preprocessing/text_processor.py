import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Download required NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

class TextProcessor:
    def __init__(self, max_features=5000):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def clean_text(self, text):
        """
        Preprocesses a single text document.
        Steps: lowercase, remove non-alphabetic chars, tokenization,
        stopword removal, and lemmatization.
        """
        # Lowercase
        text = str(text).lower()
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenization (split by space)
        tokens = text.split()
        # Stopword removal and lemmatization
        processed_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(processed_tokens)
        
    def fit_transform(self, texts):
        """
        Cleans a list of texts and fits the TF-IDF vectorizer.
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned_texts)
        
    def transform(self, texts):
        """
        Cleans a list of texts and transforms using the fitted TF-IDF vectorizer.
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

    def save_processor(self, filepath):
        """Saves the vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_processor(self, filepath):
        """Loads the vectorizer from disk."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
