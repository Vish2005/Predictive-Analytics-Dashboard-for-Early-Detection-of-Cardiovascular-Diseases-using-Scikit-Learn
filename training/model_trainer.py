from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class ModelTrainer:
    def __init__(self):
        # We use Logistic Regression for this example
        self.model = LogisticRegression(random_state=42, max_iter=1000)

    def train(self, X_train, y_train):
        """Trains the machine learning model."""
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return acc

    def save_model(self, filepath):
        """Saves the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        """Loads a model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
