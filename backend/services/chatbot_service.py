# backend/services/chatbot_service.py

import requests
import ollama

API_URL = "http://127.0.0.1:8000"


def get_stock_context(ticker: str):
    context = {}

    # Live price
    price_res = requests.get(f"{API_URL}/price/{ticker}")
    if price_res.status_code == 200:
        context["price"] = price_res.json()

    # Prediction
    pred_res = requests.get(f"{API_URL}/predict/{ticker}")
    if pred_res.status_code == 200:
        context["prediction"] = pred_res.json()

    # Evaluation
    eval_res = requests.get(f"{API_URL}/evaluate/{ticker}")
    if eval_res.status_code == 200:
        context["evaluation"] = eval_res.json()

    return context


def ask_llm(ticker: str, user_question: str):

    context = get_stock_context(ticker)

    prompt = f"""
You are a professional financial AI assistant.

Stock: {ticker}

Live Data:
{context.get("price")}

Model Prediction:
{context.get("prediction")}

Model Evaluation:
{context.get("evaluation")}

User Question:
{user_question}

Instructions:
- Use the provided data.
- Be concise but informative.
- Explain financial indicators if needed.
- Do NOT hallucinate data.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
