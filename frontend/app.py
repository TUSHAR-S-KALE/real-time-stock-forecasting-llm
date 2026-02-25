import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock AI Dashboard", layout="wide")

st.title("Stock AI Prediction Dashboard")

#Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", "TSLA")
train_button = st.sidebar.button("Train Model")

#Period
period = st.sidebar.selectbox(
    "Select Period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

#Training model
if train_button:
    response = requests.get(f"{API_URL}/train/{ticker}", params={"period": period})
    st.sidebar.success(response.json())

#Fetching history
history_response = requests.get(f"{API_URL}/history/{ticker}", params={"period": period})

if history_response.status_code == 200:
    data = history_response.json()
    df = pd.DataFrame(data['data'])

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df.set_index("Date", inplace=True)

    # Plot candlestick chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.update_layout(
        title=f"{ticker} Live Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )

    st.plotly_chart(fig, width='stretch')

#Prediction Section
st.subheader("🔮 AI Prediction")

predict_response = requests.get(f"{API_URL}/predict/{ticker}", params={"period": period})

if predict_response.status_code == 200:
    prediction_data = predict_response.json()

    if "error" not in prediction_data:
        st.metric(
            label="Prediction",
            value=prediction_data["prediction"],
            delta=f"Confidence: {prediction_data['confidence']}"
        )
    else:
        st.warning(prediction_data["error"])

#Evaluation Section
st.subheader("📊 Model Evaluation")

eval_response = requests.get(f"{API_URL}/evaluate/{ticker}", params={"period": period})

if eval_response.status_code == 200:
    eval_data = eval_response.json()

    if "error" not in eval_data:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", eval_data["accuracy"])
        col2.metric("Precision", eval_data["precision"])
        col3.metric("Recall", eval_data["recall"])
        col4.metric("F1 Score", eval_data["f1_score"])

        st.write("Confusion Matrix:")
        st.write(eval_data["confusion_matrix"])
    else:
        st.warning(eval_data["error"])

#Feature Importance
st.subheader("🔥 Feature Importance")

importance_response = requests.get(f"{API_URL}/feature-importance", params={"period": period})

if importance_response.status_code == 200:
    importance_data = importance_response.json()

    if "error" not in importance_data:
        importance_df = pd.DataFrame(
            importance_data.items(),
            columns=["Feature", "Importance"]
        )

        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning(importance_data["error"])

#Backtest
st.subheader("📈 Backtest Strategy")

backtest_response = requests.get(f"{API_URL}/backtest/{ticker}", params={"period": period})

if backtest_response.status_code == 200:
    backtest_data = backtest_response.json()

    if "error" not in backtest_data:
        col1, col2, col3 = st.columns(3)

        col1.metric("Strategy Accuracy", backtest_data["strategy_accuracy"])
        col2.metric("Buy Signals", backtest_data["buy_signals"])
        col3.metric("Sell Signals", backtest_data["sell_signals"])
    else:
        st.warning(backtest_data["error"])

st.subheader("🤖 AI Financial Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about this stock..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.spinner("Thinking..."):
        response = requests.get(
            f"{API_URL}/chat/{ticker}",
            params={"question": prompt}
        )

        answer = response.json()["response"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()
