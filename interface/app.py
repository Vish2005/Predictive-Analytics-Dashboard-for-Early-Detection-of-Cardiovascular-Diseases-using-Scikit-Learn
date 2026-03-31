import sys
import os

# Ensure the root directory is on the path if run standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.predictor import SentimentPredictor

def run_interface():
    print("====================================")
    print("    Sentiment Analysis System")
    print("====================================")
    
    predictor = SentimentPredictor()
    
    if not predictor.is_loaded:
        print("Model is not initialized. Exiting interface.")
        return
        
    print("Type 'exit' to quit the application.\n")
    
    while True:
        try:
            user_input = input("Enter text for sentiment analysis: ")
            if user_input.lower().strip() == 'exit':
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            sentiment = predictor.predict(user_input)
            print(f"Predicted Sentiment: {sentiment.upper()}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
if __name__ == "__main__":
    run_interface()
