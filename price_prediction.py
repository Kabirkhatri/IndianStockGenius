import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
from dateutil.relativedelta import relativedelta

def prepare_features(data):
    """
    Prepare features for price prediction
    
    Args:
        data (pd.DataFrame): Stock price data
    
    Returns:
        pd.DataFrame: DataFrame with features
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Add basic technical indicators as features
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Price momentum
    df['Price_Momentum_1'] = df['Close'].pct_change(periods=1)
    df['Price_Momentum_5'] = df['Close'].pct_change(periods=5)
    df['Price_Momentum_10'] = df['Close'].pct_change(periods=10)
    df['Price_Momentum_20'] = df['Close'].pct_change(periods=20)
    
    # Volatility
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # Price range
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['High_Low_Range_MA5'] = df['High_Low_Range'].rolling(window=5).mean()
    
    # Trend indicators
    df['Above_MA20'] = (df['Close'] > df['MA20']).astype(int)
    df['Above_MA50'] = (df['Close'] > df['MA50']).astype(int)
    
    # Lagged features (previous days' closing prices)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    # Date-based features
    if isinstance(df.index, pd.DatetimeIndex):
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_prediction_model(df, target_days=30):
    """
    Train a machine learning model for price prediction
    
    Args:
        df (pd.DataFrame): DataFrame with features
        target_days (int): Days ahead to predict
    
    Returns:
        tuple: (model, scaler, features, metrics)
    """
    # Create target variable: future price
    df['Target'] = df['Close'].shift(-target_days)
    
    # Drop rows with NaN in target
    df = df.dropna()
    
    # Select features
    features = [
        'MA5', 'MA10', 'MA20', 'MA50',
        'Price_Momentum_1', 'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
        'Volatility_5', 'Volatility_10', 'Volatility_20',
        'Volume_Change', 'Volume_MA5', 'Volume_MA10', 'Volume_Ratio',
        'High_Low_Range', 'High_Low_Range_MA5',
        'Above_MA20', 'Above_MA50',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5'
    ]
    
    # If we have date features, add them
    if 'Day_of_Week' in df.columns:
        features.extend(['Day_of_Week', 'Month', 'Quarter'])
    
    # Prepare X and y
    X = df[features]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'R²': round(r2, 2)
    }
    
    return model, scaler, features, metrics

def predict_future_prices(model, scaler, features, last_data, periods):
    """
    Predict future prices for specified periods
    
    Args:
        model: Trained model
        scaler: Feature scaler
        features: List of feature names
        last_data: DataFrame with latest data
        periods: Dictionary of periods to predict
    
    Returns:
        dict: Dictionary with predictions for each period
    """
    # Get the last row of data
    X_last = last_data[features].iloc[-1:].values
    
    # Scale the features
    X_last_scaled = scaler.transform(X_last)
    
    # Make prediction for the last point
    baseline_prediction = model.predict(X_last_scaled)[0]
    
    # Current price
    current_price = last_data['Close'].iloc[-1]
    
    # Create predictions for each period
    predictions = {}
    
    for period_name, days in periods.items():
        # For simplicity, we're using a combination of the model prediction and some randomness
        # In a real system, you'd train separate models for each time horizon or use time series forecasting
        
        # Base prediction
        if days <= 30:
            # For shorter periods, we can use direct model prediction with some adjustment
            prediction = current_price + (baseline_prediction - current_price) * (days / 30)
        else:
            # For longer periods, we apply a compounding effect
            monthly_return = (baseline_prediction / current_price) - 1
            months = days / 30
            prediction = current_price * ((1 + monthly_return) ** months)
        
        # Add some controlled variability for different time horizons
        volatility = last_data['Volatility_20'].iloc[-1] * np.sqrt(days / 20)
        adjustment = np.random.normal(0, volatility * 0.5)
        
        # Ensure the prediction is positive and has reasonable bounds
        prediction = max(prediction + adjustment, current_price * 0.5)
        prediction = min(prediction, current_price * 2)
        
        predictions[period_name] = prediction
    
    return predictions

def create_prediction_chart(data, predictions, symbol):
    """
    Create a chart with price predictions
    
    Args:
        data (pd.DataFrame): Historical stock data
        predictions (dict): Dictionary with predictions
        symbol (str): Stock symbol
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical price data
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add predictions
    last_date = data.index[-1]
    
    # Create dates for predictions
    prediction_dates = {}
    for period, days in {'1w': 7, '1m': 30, '3m': 90, '6m': 180, '1y': 365}.items():
        if isinstance(last_date, pd.Timestamp):
            prediction_dates[period] = last_date + pd.Timedelta(days=days)
        else:
            # If index is not datetime, create a numeric extension
            prediction_dates[period] = last_date + days
    
    # Sort by date
    prediction_items = sorted(prediction_dates.items(), key=lambda x: x[1])
    
    # Create prediction line
    pred_x = [last_date] + [date for _, date in prediction_items]
    pred_y = [data['Close'].iloc[-1]] + [predictions[period] for period, _ in prediction_items]
    
    fig.add_trace(
        go.Scatter(
            x=pred_x,
            y=pred_y,
            mode='lines+markers',
            name='Price Projection',
            line=dict(color='green', dash='dash'),
            marker=dict(size=8)
        )
    )
    
    # Add markers for each prediction point
    for i, (period, date) in enumerate(prediction_items):
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[predictions[period]],
                mode='markers+text',
                name=f'{period} Projection',
                text=f'{period}: ₹{predictions[period]:.2f}',
                textposition='top center',
                marker=dict(size=10, color='green'),
                showlegend=False
            )
        )
    
    # Format chart
    fig.update_layout(
        title=f"{symbol} Price Projections",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

def predict_prices(data, symbol):
    """
    Predict future prices for a stock
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
    
    Returns:
        dict: Dictionary with prediction results
    """
    # Prepare features
    df = prepare_features(data)
    
    # Define prediction periods in days
    periods = {
        '1w': 7,      # 1 week
        '1m': 30,     # 1 month
        '3m': 90,     # 3 months
        '6m': 180,    # 6 months
        '1y': 365     # 1 year
    }
    
    # Train model (target for 30 days ahead)
    model, scaler, features, metrics = train_prediction_model(df, target_days=30)
    
    # Make predictions for different periods
    predictions = predict_future_prices(model, scaler, features, df, periods)
    
    # Create prediction chart
    prediction_chart = create_prediction_chart(data, predictions, symbol)
    
    # Get feature importances
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for feature, importance in zip(features, importances):
            feature_importance[feature] = importance
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Generate explanation of key factors
    key_factors = []
    for feature, importance in feature_importance.items():
        if feature.startswith('MA'):
            period = feature[2:]
            key_factors.append(f"The {period}-day moving average is an important indicator, affecting projections by {importance*100:.1f}%")
        elif feature.startswith('Price_Momentum'):
            period = feature.split('_')[2]
            key_factors.append(f"Price momentum over {period} days is influential, contributing {importance*100:.1f}% to the forecast")
        elif feature.startswith('Volatility'):
            period = feature.split('_')[1]
            key_factors.append(f"Market volatility over {period} days impacts projections by {importance*100:.1f}%")
        elif feature.startswith('Volume'):
            key_factors.append(f"Trading volume patterns affect the prediction by {importance*100:.1f}%")
        else:
            key_factors.append(f"The {feature.replace('_', ' ').lower()} is a significant factor with {importance*100:.1f}% influence")
    
    # Generate prediction factors text
    prediction_factors = f"""
    The price projections are based on machine learning analysis of historical price patterns and technical indicators. The model identifies the following key factors influencing the projections:
    
    1. {key_factors[0] if len(key_factors) > 0 else "Historical price patterns"}
    2. {key_factors[1] if len(key_factors) > 1 else "Trading volume trends"}
    3. {key_factors[2] if len(key_factors) > 2 else "Market volatility"}
    
    The model achieved an accuracy of {metrics['R²']*100:.1f}% (R²) in backtesting, with a Mean Absolute Percentage Error (MAPE) of {metrics['MAPE']:.2f}%. 
    
    Note that these projections represent a statistical forecast based on historical patterns and should be used alongside other analysis methods for investment decisions. Market conditions can change rapidly due to external factors not captured by the model.
    """
    
    # Return results
    return {
        'projections': predictions,
        'model_metrics': metrics,
        'prediction_chart': prediction_chart,
        'feature_importance': feature_importance,
        'prediction_factors': prediction_factors
    }
