from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prediction.predictor import SentimentPredictor
import os

app = FastAPI(title="Sentiment Analysis Web App")

# Create templates and static directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the predictor
predictor = SentimentPredictor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"result": None, "error": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_text(request: Request, text: str = Form(...)):
    try:
        result = predictor.predict(text)
        return templates.TemplateResponse(request=request, name="index.html", context={
            "result": result,
            "error": None
        })
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={
            "error": f"An error occurred: {str(e)}",
            "result": None
        })
