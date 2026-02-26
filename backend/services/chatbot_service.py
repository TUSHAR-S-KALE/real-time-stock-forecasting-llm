import requests
import ollama

API_URL = "http://127.0.0.1:8000"

def get_stock_context(ticker):
    context = {}

    price_res = requests.get(f"{API_URL}/price/{ticker}")
    if price_res.status_code == 200:
        context["price"] = price_res.json()

    pred_res = requests.get(f"{API_URL}/predict/{ticker}")
    if pred_res.status_code == 200:
        context["prediction"] = pred_res.json()

    eval_res = requests.get(f"{API_URL}/evaluate/{ticker}")
    if eval_res.status_code == 200:
        context["evaluation"] = eval_res.json()

    return context

def llm_response(ticker, user_question):

    context = get_stock_context(ticker)

    prompt = f"""
    You are a financial AI assistant.

    Stock: {ticker}
    Data: {context}

    Question: {user_question}

    Be concise. Do not hallucinate.
    """

    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if "message" in chunk:
            yield chunk["message"]["content"]

