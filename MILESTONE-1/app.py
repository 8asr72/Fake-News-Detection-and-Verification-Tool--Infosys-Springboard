from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle

app = FastAPI()


templates = Jinja2Templates(directory="templates")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, news: str = Form(...)):

    
    X = vectorizer.transform([news])

    prediction = model.predict(X)[0]  
    probabilities = model.predict_proba(X)[0]


    real_conf = round(probabilities[1] * 100, 2)
    fake_conf = round(probabilities[0] * 100, 2)
    accuracy = 95  

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": "Real" if prediction == 1 else "Fake",
        "real_conf": real_conf,
        "fake_conf": fake_conf,
        "accuracy": accuracy,
        "news": news
    })
