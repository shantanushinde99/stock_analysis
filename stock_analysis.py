# Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta

# Configure the API key - IMPORTANT: Use Streamlit secrets or environment variables for security
GOOGLE_API_KEY = "AIzaSyDLBVOXnCYMe0tHIQ5JDEn_IRQx58fGbyE"  # REPLACE WITH YOUR ACTUAL API KEY SECURELY
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for multiple stock tickers
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Date range
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Technical indicators selection
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI", "MACD"],
    default=["20-Day SMA"]
)

# Chart type selection
chart_type = st.sidebar.selectbox("Chart Type:", ["Candlestick", "Line"])

# Indicator parameters
sma_period = st.sidebar.slider("SMA Period", min_value=5, max_value=50, value=20)
rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14)

# Check if parameters have changed
if "last_params" not in st.session_state:
    st.session_state["last_params"] = {}
current_params = {
    "tickers": tickers_input,
    "start_date": start_date,
    "end_date": end_date,
    "indicators": indicators,
    "sma_period": sma_period,
    "rsi_period": rsi_period
}
if st.session_state["last_params"] != current_params:
    if "stock_data" in st.session_state:
        del st.session_state["stock_data"]
    st.session_state["last_params"] = current_params

# Cache yfinance data
@st.cache_data
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

# Button to fetch data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching stock data..."):
        stock_data = {}
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            data = fetch_stock_data(ticker, start_date, end_date)
            if not data.empty:
                stock_data[ticker] = data
            else:
                st.warning(f"No data found for {ticker}.")
            progress_bar.progress((i + 1) / len(tickers))
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
        st.experimental_rerun()  # Force rerun to update charts

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    # Function to build chart and analyze
    def analyze_ticker(ticker, data):
        # Create figure
        fig = go.Figure()

        # Add chart based on user selection
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name="Close Price"
            ))

        # Calculate technical indicators
        data = data.copy()  # Avoid modifying original data
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name=f'SMA ({sma_period})'))
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=sma_period).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name=f'EMA ({sma_period})'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=sma_period).mean()
                std = data['Close'].rolling(window=sma_period).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
            elif indicator == "RSI":
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI'))
            elif indicator == "MACD":
                ema12 = data['Close'].ewm(span=12).mean()
                ema26 = data['Close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                fig.add_trace(go.Scatter(x=data.index, y=macd, mode='lines', name='MACD'))
                fig.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', name='Signal Line'))

        for ind in indicators:
            add_indicator(ind)

        fig.update_layout(xaxis_rangeslider_visible=False, height=600)

        # Save chart as PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        # Create image part
        image_part = {
            "data": image_bytes,
            "mime_type": "image/png"
        }

        # Analysis prompt
        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its {chart_type.lower()} chart and the displayed technical indicators: {', '.join(indicators)}. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        # Call Gemini API
        contents = [
            {"role": "user", "parts": [analysis_prompt]},
            {"role": "user", "parts": [image_part]}
        ]

        try:
            response = gen_model.generate_content(contents=contents)
            result_text = response.text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found")
        except Exception as e:
            result = {"action": "Error", "justification": f"Analysis failed: {str(e)}"}

        return fig, result

    # Create tabs
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # Overall results
    overall_results = []

    # Process each ticker
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        with st.spinner(f"Analyzing {ticker}..."):
            fig, result = analyze_ticker(ticker, data)
            overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
            with tabs[i + 1]:
                st.subheader(f"Analysis for {ticker}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.write("**Recommendation:**")
                    st.write(result.get("action", "N/A"))
                    st.write("**Justification:**")
                    st.write(result.get("justification", "No justification provided."))

    # Overall Summary tab
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
        # Download button
        csv = df_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary as CSV",
            data=csv,
            file_name="stock_analysis_summary.csv",
            mime="text/csv"
        )

else:
    st.info("Please fetch stock data using the sidebar.")
