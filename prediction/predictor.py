from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
from explanation.explainer import SentimentExplainer

class SentimentPredictor:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        print(f"Loading BERT model: {model_name}...")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        self.explainer = SentimentExplainer(self.sentiment_pipeline)
        self.is_loaded = True
        print("Model loaded successfully.")

    def predict(self, text):
        if not text.strip():
            return {"error": "Empty text provided."}
            
        # Translate to English if it's not already
        try:
            detected_lang = detect(text)
            if detected_lang != 'en':
                translator = GoogleTranslator(source='auto', target='en')
                translated_text = translator.translate(text)
            else:
                translated_text = text
                
            # Map short code to full name using deep_translator
            lang_dict = GoogleTranslator().get_supported_languages(as_dict=True)
            inv_lang_dict = {v: k for k, v in lang_dict.items()}
            full_lang_name = inv_lang_dict.get(detected_lang, detected_lang).title()
        except Exception as e:
            print(f"Translation/Language Detection error: {e}")
            translated_text = text # Fallback
            full_lang_name = "Unknown"
            
        # Predict
        try:
            prediction_results = self.sentiment_pipeline(translated_text, top_k=None)
        except TypeError:
            # Fallback if top_k parameter is not supported by this version
            prediction_results = self.sentiment_pipeline(translated_text, return_all_scores=True)
        
        if isinstance(prediction_results, list) and len(prediction_results) > 0:
            if isinstance(prediction_results[0], list):
                results = prediction_results[0]
            else:
                results = prediction_results

            top_result = max(results, key=lambda x: x['score'])
            label = top_result['label']
            confidence = top_result['score']
        else:
            label = "UNKNOWN"
            confidence = 0.0

        confidence_pct = round(confidence * 100, 1)
        
        # Get LIME Explainability
        try:
            contributions = self.explainer.explain(translated_text)
        except Exception as e:
            print(f"LIME Error: {e}")
            contributions = []
            
        return {
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": full_lang_name,
            "sentiment": label,
            "confidence_pct": confidence_pct,
            "explainability": contributions
        }
