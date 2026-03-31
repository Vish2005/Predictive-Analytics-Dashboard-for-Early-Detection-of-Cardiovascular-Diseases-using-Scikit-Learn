import uvicorn

def run_interface():
    # Placeholder if you still want to run the CLI interface
    print("Interactive interface is currently disabled or must be updated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment Analysis Project (BERT Multi-lingual Suite)")
    parser.add_argument('--run', action='store_true', help="Run the interactive CLI interface")
    parser.add_argument('--serve', action='store_true', help="Start the FastAPI web application")
    
    args = parser.parse_args()
    
    if args.run:
        run_interface()
    elif args.serve:
        print("Starting FastAPI Web Server on http://127.0.0.1:8000")
        uvicorn.run("api.webapp:app", host="127.0.0.1", port=8000, reload=True)
    else:
        print("Please provide an argument: --run, or --serve.")
