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

API_KEY = 'HI06RNQVLCJCPT6U'  # Alpha Vantage API Key

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

def generate_signals(df):
    buy_signals = []
    sell_signals = []
    position = None

    for i in range(len(df)):
        if df['RSI'].iloc[i] < 30 and df['Close'].iloc[i] > df['SMA_10'].iloc[i]:  # Buy signal
            if position != 'buy':
                buy_signals.append(df['Close'].iloc[i])
                sell_signals.append(np.nan)
                position = 'buy'
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)
        elif df['RSI'].iloc[i] > 70 and df['Close'].iloc[i] < df['SMA_10'].iloc[i]:  # Sell signal
            if position != 'sell':
                buy_signals.append(np.nan)
                sell_signals.append(df['Close'].iloc[i])
                position = 'sell'
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
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

def future_predictions(df, model, periods=5):
    last_row = df.iloc[-1]
    predictions = []
    buy_sell_plan = []

    for i in range(periods):
        # Create a new DataFrame for prediction
        new_data = {
            'Open': last_row['Close'],
            'High': last_row['Close'],
            'Low': last_row['Close'],
            'Volume': last_row['Volume'],
            'SMA_10': last_row['SMA_10'],
            'SMA_20': last_row['SMA_20'],
            'SMA_50': last_row['SMA_50'],
            'EMA_10': last_row['EMA_10'],
            'EMA_50': last_row['EMA_50'],
            'Volatility': last_row['Volatility'],
            'Pct_Change': last_row['Pct_Change'],
            'RSI': last_row['RSI'],
            'MACD': last_row['MACD']
        }
        
        new_row = pd.DataFrame([new_data])
        prediction = model.predict(new_row.values)[0]

        # Buy/Sell logic based on comparison with last close price
        action = ''
        if prediction > last_row['Close']:
            action = 'Buy'
        elif prediction < last_row['Close']:
            action = 'Sell'

        predictions.append({
            'Predicted Price': prediction,
            'Action': action
        })
        
        # Record the date for future predictions
        prediction_date = pd.Timestamp.now() + pd.Timedelta(minutes=(i + 1) * 1)  # Predicting at 1-minute intervals
        buy_sell_plan.append({
            'Date': prediction_date,
            'Predicted Price': prediction,
            'Action': action
        })

        last_row['Close'] = prediction  # Update the close price for the next prediction

    return predictions, buy_sell_plan

# Streamlit UI
st.title("Stock Market Analysis and Trading Assistant")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if st.button("Fetch Data"):
    try:
        df = fetch_stock_data(symbol)
        df = preprocess_data(df)
        df = generate_signals(df)
        
        st.write("### Stock Data")
        st.write(df)

        rf_model, rf_predictions = train_rf_model(df)
        xgb_model, xgb_predictions = train_xgboost_model(df)
        sarima_model, sarima_predictions = train_sarima_model(df)

        # Calculating MSE(Mean Squared Error) for each model
        rf_mse = mean_squared_error(df['Close'], rf_predictions)
        xgb_mse = mean_squared_error(df['Close'], xgb_predictions)
        sarima_mse = mean_squared_error(df['Close'], sarima_predictions)

         # Best model selection
        mse_dict = {
            'Random Forest': rf_mse,
            'XGBoost': xgb_mse,
            'SARIMA': sarima_mse,
        }
        best_model = min(mse_dict, key=mse_dict.get)

         # Real-time predictions and best model
        st.subheader("Real-Time Predictions")
        st.write(f"Random Forest Prediction: {rf_predictions[-1]}")
        st.write(f"XGBoost Prediction: {xgb_predictions[-1]}")
        st.write(f"SARIMA Prediction: {sarima_predictions[-1]}")
        st.write(f"Best Model: {best_model} with Mean Squared Error is: {mse_dict[best_model]}")

         # Plotting Close Price
        st.subheader(f"{symbol} Stock Close Price")
        fig_close = go.Figure()
        fig_close.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig_close)

         # Overlay Predictions on the Same Graph
        st.subheader("Predictions Overlaid on Actual Prices")
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Close Price'))
        fig_overlay.add_trace(go.Scatter(x=df.index, y=rf_predictions, mode='lines', name='Random Forest Predictions'))
        fig_overlay.add_trace(go.Scatter(x=df.index, y=xgb_predictions, mode='lines', name='XGBoost Predictions'))
        fig_overlay.add_trace(go.Scatter(x=df.index, y=sarima_predictions, mode='lines', name='SARIMA Predictions'))
        fig_overlay.update_layout(title='Predictions Overlaid on Actual Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_overlay)


        # Future Predictions
        future_preds, buy_sell_plan = future_predictions(df, rf_model, periods=5)

        st.write("### Future Predictions")
        future_df = pd.DataFrame(future_preds)
        st.write(future_df)

        # Display Buy/Sell Plan
        st.write("### Buy/Sell Recommendations")
        buy_sell_df = pd.DataFrame(buy_sell_plan)
        st.write(buy_sell_df)

    except Exception as e:
        st.error(f"Error: {e}")

# streamlit run app.py--code to run app 
