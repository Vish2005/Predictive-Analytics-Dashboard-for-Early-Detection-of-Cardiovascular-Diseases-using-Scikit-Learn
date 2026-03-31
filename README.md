#link for web page: https://huggingface.co/spaces/vish2026/Multi-lingual-Sentiment-App
# 🧠 Multi-lingual Sentiment Intelligence Suite

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FF9D00)
![LIME](https://img.shields.io/badge/XAI-LIME-brightgreen)

An advanced, production-ready Natural Language Processing (NLP) web application that analyzes multi-lingual text and classifies it as **Positive, Neutral, or Negative**. 

Moving beyond traditional machine learning, this project utilizes a state-of-the-art **RoBERTa Transformer** model and integrates **Explainable AI (XAI)** to visually highlight exactly *why* the AI made its decision.

## ✨ Key Features
- **Transformer NLP Architecture:** Powered by `cardiffnlp/twitter-roberta-base-sentiment-latest` via Hugging Face.
- **Explainable AI (XAI):** Integrated `LIME` (Local Interpretable Model-Agnostic Explanations) to achieve "Glass-box" ML. The UI highlights specific words in green (supports prediction) or red (contradicts prediction) to establish trust in the AI's logic.
- **Multi-lingual Auto-Translation:** Automatically detects the source language (using `langdetect`) and translates it to English (via `deep-translator`) before passing it into the Transformer.
- **FastAPI Backend:** High-performance async server handling the model inference.
- **Modern Glassmorphism UI:** A fully responsive HTML/CSS frontend featuring dynamic logic, confidence gauges, and pastel styling.

## 📂 Project Structure
- `api/`: Contains `webapp.py` managing the FastAPI API routes and Jinja2 template rendering.
- `prediction/`: Contains `predictor.py`, which loads the RoBERTa Transformer pipeline and orchestrates translation.
- `explanation/`: Contains `explainer.py` to calculate word-level feature importance using LIME.
- `templates/` & `static/`: Contains the frontend `index.html` and modern `style.css`.
- `main.py`: The entry point script to start the server.

---

## 🚀 Setup & Installation Instructions

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Multi-lingual-Sentiment-Analyzer.git
cd Multi-lingual-Sentiment-Analyzer
```

**2. Install dependencies:**
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

**3. Run the local server:**
```bash
python main.py --serve
```
*Note: The first time you run this, it will automatically download the RoBERTa weights (~500MB) from Hugging Face.*

**4. Access the Web Interface:**
Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 🔮 Future Enhancements
- **Dockerization:** Containerizing the FastAPI app using Docker for seamless cloud deployment.
- **Fine-Tuning:** Adding a script to allow custom re-training of the RoBERTa layer on domain-specific data (e.g., Medical reviews, Financial sentiment).

## 🤝 Contributing
Feel free to open issues or submit pull requests. All contributions are welcome!
