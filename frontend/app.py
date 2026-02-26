import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Stock AI Platform",
    layout="wide",
)

#Helper Functions
@st.cache_data(ttl=300)
def fetch_history(ticker):
    response = requests.get(f"{API_URL}/history/{ticker}")
    if response.status_code == 200:
        return response.json()
    return None

def fetch_prediction(ticker):
    return requests.get(f"{API_URL}/predict/{ticker}").json()

def fetch_evaluation(ticker):
    return requests.get(f"{API_URL}/evaluate/{ticker}").json()

def fetch_backtest(ticker):
    return requests.get(f"{API_URL}/backtest/{ticker}").json()

def fetch_importance():
    return requests.get(f"{API_URL}/feature-importance").json()

def stream_text(text, delay=0.015):
    placeholder = st.empty()
    full_text = ""
    for word in text.split():
        full_text += word + " "
        placeholder.markdown(full_text)
        time.sleep(delay)
    return full_text

#Sidebar Controls
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker", "TSLA")

if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        response = requests.get(f"{API_URL}/train/{ticker}")
        st.sidebar.success(response.json())

#Main Title
st.title("Stock AI Platform")
st.markdown("Machine Learning + LLM Powered Financial System")

#Tabs
tab1, tab2, tab3 = st.tabs([
    "Dashboard",
    "AI Prediction",
    "AI Financial Chatbot"
])

#TAB 1 — DASHBOARD
with tab1:
    st.subheader("Live Stock Chart")
    data = fetch_history(ticker)

    if data:
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.set_index("Date", inplace=True)

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
            height=600,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("No historical data available.")

    #Live price
    st.subheader("Live Price")

    price_res = requests.get(f"{API_URL}/price/{ticker}")
    if price_res.status_code == 200:
        price_data = price_res.json()
        if "error" not in price_data:
            st.metric("Current Price", f"${price_data['price']:.2f}")
        else:
            st.warning(price_data["error"])

#TAB 2 — AI PREDICTION
with tab2:
    st.subheader("Next Day Prediction")

    with st.spinner("Generating prediction..."):
        prediction_data = fetch_prediction(ticker)

    if "error" not in prediction_data:
        st.metric(
            "Predicted Direction",
            prediction_data["prediction"],
            f"Confidence: {prediction_data['confidence']}"
        )
    else:
        st.warning(prediction_data["error"])

    st.divider()

    st.subheader("Model Performance")

    eval_data = fetch_evaluation(ticker)

    if "error" not in eval_data:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", eval_data["accuracy"])
        col2.metric("Precision", eval_data["precision"])
        col3.metric("Recall", eval_data["recall"])
        col4.metric("F1 Score", eval_data["f1_score"])

        st.write("Confusion Matrix")
        cm_df = pd.DataFrame.from_dict(
            eval_data["confusion_matrix"],
            orient="index"
        )
        st.dataframe(cm_df)
    else:
        st.warning(eval_data["error"])

    st.divider()

    st.subheader("Feature Importance")

    importance_data = fetch_importance()

    if "error" not in importance_data:
        importance_df = pd.DataFrame(
            importance_data.items(),
            columns=["Feature", "Importance"]
        )
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning(importance_data["error"])

    st.divider()

    st.subheader("Backtest Strategy")

    backtest_data = fetch_backtest(ticker)

    if "error" not in backtest_data:
        col1, col2, col3 = st.columns(3)

        col1.metric("Strategy Accuracy", backtest_data["strategy_accuracy"])
        col2.metric("Buy Signals", backtest_data["buy_signals"])
        col3.metric("Sell Signals", backtest_data["sell_signals"])
    else:
        st.warning(backtest_data["error"])

#TAB 3 (AI FINANCIAL CHATBOT)
with tab3:
    st.subheader("Chat with AI Financial Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=500, border=True)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about this stock...")

    if prompt:

        #User message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        #Assistant response
        full_response = ""

        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                response = requests.get(
                    f"{API_URL}/chat/{ticker}",
                    params={"question": prompt},
                    stream=True
                )

                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        text = chunk.decode("utf-8")
                        full_response += text
                        message_placeholder.markdown(full_response)

        #Saving assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        st.rerun()
