import requests
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import streamlit as st

API_KEY = 'HI06RNQVLCJCPT6U'  #Alpha Vantage API Key

def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (1min)' not in data:
        raise ValueError(f"Error fetching data for {symbol}: {data.get('Note', 'No data found')}")
    
    df = pd.DataFrame(data['Time Series (1min)']).transpose()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

def preprocess_data(df):
    df['Pct_Change'] = df['Close'].pct_change()
    for lag in [1, 5, 10, 15, 30]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    df['Price_Change'] = df['Close'].diff()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / 
                         df['Close'].diff().apply(lambda x: -min(x, 0)).rolling(window=14).mean())))
    df['Bollinger_High'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Low'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    df['MACD'] = df['EMA_10'] - df['EMA_50']
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], 
                        abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])), axis=1).rolling(window=14).mean()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def train_rf_model(df):
    X = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 
             'Volatility', 'Pct_Change', 'RSI', 'MACD']].values
    y = df['Close'].values
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)
    return best_model, predictions

def train_xgboost_model(df):
    X = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 
             'Volatility', 'Pct_Change', 'RSI', 'MACD']].values
    y = df['Close'].values
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions

def train_sarima_model(df):
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    predictions = results.predict(start=0, end=len(df) - 1)
    return results, predictions

def train_prophet_model(df):
    df_prophet = df.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)

    future = prophet_model.make_future_dataframe(periods=30)  # Forecasting for 30 minutes
    forecast = prophet_model.predict(future)
    
    # Using the forecasted values for the next 30 time steps for comparison
    predictions = forecast['yhat'][-len(df):].values  # Ensure it matches the original length
    return prophet_model, predictions, forecast

def predict_real_time(df, model, model_type='rf'):
    X_real_time = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 
                       'EMA_50', 'Volatility', 'Pct_Change', 'RSI', 'MACD']].values[-1:]
    
    if model_type in ['xgb', 'rf']:
        prediction = model.predict(X_real_time)
    elif model_type == 'sarima':
        prediction = model.predict(start=len(df), end=len(df))
    elif model_type == 'prophet':
        prediction = model.predict(df)
    
    return prediction

# Streamlit UI Design
st.title('Comprehensive Stock Market Analysis and Trading Assistant')

symbol = st.text_input('Enter Stock Symbol', 'AAPL')

if symbol:
    try:
        latest_data = fetch_stock_data(symbol)
        
        # Title for raw data
        st.subheader("Raw Stock Data")
        st.write(latest_data)

        processed_data = preprocess_data(latest_data)

        # Training models and getting predictions
        rf_model, rf_predictions = train_rf_model(processed_data)
        xgb_model, xgb_predictions = train_xgboost_model(processed_data)
        sarima_model, sarima_predictions = train_sarima_model(processed_data)
        prophet_model, prophet_predictions, _ = train_prophet_model(processed_data)

        # Calculating MSE(Mean Squared Error) for each model
        rf_mse = mean_squared_error(processed_data['Close'], rf_predictions)
        xgb_mse = mean_squared_error(processed_data['Close'], xgb_predictions)
        sarima_mse = mean_squared_error(processed_data['Close'], sarima_predictions)
        prophet_mse = mean_squared_error(processed_data['Close'][-len(prophet_predictions):], prophet_predictions)

        # best model selection
        mse_dict = {
            'Random Forest': rf_mse,
            'XGBoost': xgb_mse,
            'SARIMA': sarima_mse,
            'Prophet': prophet_mse
        }
        best_model = min(mse_dict, key=mse_dict.get)

        #real-time predictions and best model
        st.subheader("Real-Time Predictions")
        st.write(f"Random Forest Prediction: {rf_predictions[-1]}")
        st.write(f"XGBoost Prediction: {xgb_predictions[-1]}")
        st.write(f"SARIMA Prediction: {sarima_predictions[-1]}")
        st.write(f"Prophet Prediction: {prophet_predictions[-1]}")
        st.write(f"Best Model: {best_model} with Mean Squared Error is: {mse_dict[best_model]}")

        # Plotting Close Price
        st.subheader(f"{symbol} Stock Close Price")
        fig_close = go.Figure()
        fig_close.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig_close)

        # Overlay Predictions on the Same Graph
        st.subheader("Predictions Overlaid on Actual Prices")
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], mode='lines', name='Actual Close Price'))
        fig_overlay.add_trace(go.Scatter(x=processed_data.index, y=rf_predictions, mode='lines', name='Random Forest Predictions'))
        fig_overlay.add_trace(go.Scatter(x=processed_data.index, y=xgb_predictions, mode='lines', name='XGBoost Predictions'))
        fig_overlay.add_trace(go.Scatter(x=processed_data.index, y=sarima_predictions, mode='lines', name='SARIMA Predictions'))
        fig_overlay.add_trace(go.Scatter(x=processed_data.index[-len(prophet_predictions):], y=prophet_predictions, mode='lines', name='Prophet Predictions'))
        fig_overlay.update_layout(title='Predictions Overlaid on Actual Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_overlay)

        # Technical Indicators Plot
        fig_indicators = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        # SMA subplot
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_10'], name="SMA 10"), row=1, col=1)
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_20'], name="SMA 20"), row=1, col=1)
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_50'], name="SMA 50"), row=1, col=1)

        # EMA subplot
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['EMA_10'], name="EMA 10"), row=2, col=1)
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['EMA_50'], name="EMA 50"), row=2, col=1)

        # MACD subplot
        fig_indicators.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MACD'], name="MACD"), row=3, col=1)
        
        fig_indicators.update_layout(height=600, title_text=f"{symbol} Technical Indicators")
        st.plotly_chart(fig_indicators)

        # Forecasting with Prophet
        st.subheader("Forecasting with Prophet")
        df_prophet = processed_data.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)

        future = prophet_model.make_future_dataframe(periods=30)  # Forecasting for 30 minutes
        forecast = prophet_model.predict(future)

        # Plotting Prophet Forecast
        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Close Price'))
        fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Close Price', line=dict(dash='dash')))
        fig_prophet.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_prophet)

        # Feature Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        corr = processed_data.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Feature Correlation")
        st.plotly_chart(fig_corr)

        # Bollinger Bands Plot
        st.subheader("Bollinger Bands")
        fig_bollinger = go.Figure()
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], mode='lines', name='Close Price'))
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(dash='dash')))
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(dash='dash')))
        fig_bollinger.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_bollinger)

        # RSI Plot
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=processed_data.index, y=processed_data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_color='red', line_dash='dash', annotation_text='Overbought', annotation_position='top right')
        fig_rsi.add_hline(y=30, line_color='green', line_dash='dash', annotation_text='Oversold', annotation_position='bottom right')
        fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
        st.plotly_chart(fig_rsi)

        # Volume Plot
        st.subheader("Volume")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=processed_data.index, y=processed_data['Volume'], name='Volume'))
        fig_volume.update_layout(title='Volume', xaxis_title='Date', yaxis_title='Volume')
        st.plotly_chart(fig_volume)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# streamlit run app.py--code to run app 
