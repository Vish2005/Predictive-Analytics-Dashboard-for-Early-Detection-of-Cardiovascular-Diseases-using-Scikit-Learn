import numpy as np
from lime.lime_text import LimeTextExplainer

class SentimentExplainer:
    def __init__(self, predictor_pipeline):
        """
        predictor_pipeline: The transformers pipeline.
        We wrap it in a function that returns probabilities for LIME.
        """
        self.explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
        self.pipeline = predictor_pipeline

    def _predict_proba(self, texts):
        try:
            results = self.pipeline(texts, top_k=None)
        except TypeError:
            results = self.pipeline(texts, return_all_scores=True)
        
        probas = []
        for result in results:
            neg_score = neu_score = pos_score = 0
            for label_entry in result:
                label = label_entry['label'].lower()
                if label == 'negative':
                    neg_score = label_entry['score']
                elif label == 'neutral':
                    neu_score = label_entry['score']
                elif label == 'positive':
                    pos_score = label_entry['score']
            probas.append([neg_score, neu_score, pos_score])
            
        return np.array(probas)

    def explain(self, text, num_features=10):
        # Generate an explanation using LIME for the top predicted class
        exp = self.explainer.explain_instance(
            text, 
            self._predict_proba, 
            num_features=num_features, 
            num_samples=100,
            top_labels=1
        )
        
        # Get the word contributions for the top predicted class
        top_label_idx = exp.available_labels()[0]
        contributions = exp.as_list(label=top_label_idx)
        
        return contributions
