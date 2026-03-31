# Sentiment Analysis System

This is a modular Natural Language Processing (NLP) project built using Python. The system analyzes text data and classifies it as positive, negative, or neutral.

## Project Structure
- `data/`: Contains sample dataset (`sample_dataset.csv`).
- `preprocessing/`: Contains `text_processor.py` for text cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization.
- `training/`: Contains `model_trainer.py` to train a Logistic Regression model via Scikit-learn.
- `prediction/`: Contains `predictor.py` which loads the saved model and preprocessing artifacts for inference.
- `interface/`: Contains `app.py` providing a simple interactive CLI interface.
- `main.py`: The entry point script to train the model and start the application.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python main.py --train
   ```

3. Run the interface:
   ```bash
   python main.py --run
   ```
