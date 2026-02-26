from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

#Service Imports
from backend.services.data_service import (
    get_live_price,
    get_historical_data
)
from backend.services.model_service import (
    train_model,
    predict_next_day
)
from backend.services.evaluation_service import (
    evaluate_model,
    backtest_strategy,
    feature_importance
)
from backend.services.chatbot_service import llm_response

#App Initialization
app = FastAPI(
    title="Stock AI API",
    description="ML + LLM Powered Stock Analysis Platform",
    version="1.0.0"
)

#CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    #for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Root Endpoint
@app.get("/")
def root():
    return {
        "message": "Stock AI API is running 🚀",
        "endpoints": [
            "/price/{ticker}",
            "/history/{ticker}",
            "/train/{ticker}",
            "/predict/{ticker}",
            "/evaluate/{ticker}",
            "/backtest/{ticker}",
            "/feature-importance",
            "/chat/{ticker}"
        ]
    }

#Live Data Endpoints
@app.get("/price/{ticker}")
def price(ticker: str):
    logger.info(f"Live price requested for {ticker}")
    result = get_live_price(ticker)

    if result is None:
        return {"error": "Invalid ticker or no data found"}

    return result

@app.get("/history/{ticker}")
def history(ticker: str, period: str = "6mo"):
    logger.info(f"Historical data requested for {ticker}")
    result = get_historical_data(ticker, period)

    if result is None:
        return {"error": "Invalid ticker or no historical data found"}

    return result

#ML Endpoints
@app.get("/train/{ticker}")
def train(ticker: str):
    logger.info(f"Training model for {ticker}")
    return train_model(ticker)

@app.get("/predict/{ticker}")
def predict(ticker: str):
    logger.info(f"Prediction requested for {ticker}")
    return predict_next_day(ticker)

@app.get("/evaluate/{ticker}")
def evaluate(ticker: str):
    logger.info(f"Evaluation requested for {ticker}")
    return evaluate_model(ticker)

@app.get("/backtest/{ticker}")
def backtest(ticker: str):
    logger.info(f"Backtest requested for {ticker}")
    return backtest_strategy(ticker)

@app.get("/feature-importance")
def importance():
    logger.info("Feature importance requested")
    return feature_importance()

#LLM Chat Endpoint
@app.get("/chat/{ticker}")
def chat(ticker: str, question: str = Query(..., description="User question for chatbot")):
    logger.info(f"Chat request for {ticker}: {question}")
    generator = llm_response(ticker, question)
    return StreamingResponse(generator, media_type="text/plain")

#Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong. Please try again."}
    )

