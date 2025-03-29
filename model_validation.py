import pandas as pd
import numpy as np
import plotly.graph_objects as go
from price_prediction import prepare_features, train_prediction_model, predict_future_prices
from datetime import datetime, timedelta
import streamlit as st

def validate_model_with_past_data(data, periods_back=6, prediction_days=30):
    """
    Validate the model's accuracy by comparing predictions against actual historical prices
    
    Args:
        data (pd.DataFrame): Full historical stock data
        periods_back (int): Number of past periods to validate against
        prediction_days (int): Number of days ahead to predict
    
    Returns:
        dict: Dictionary with validation results and visualization
    """
    if len(data) < (periods_back * prediction_days + 60):
        # Not enough data for validation
        return {
            "error": "Not enough historical data for validation",
            "accuracy": None,
            "validation_chart": None,
            "validation_results": None
        }
    
    # Create copies to avoid modifying the original data
    full_data = data.copy()
    
    # Results storage
    actual_prices = []
    predicted_prices = []
    dates = []
    accuracy_values = []
    
    # For each validation period
    for i in range(periods_back):
        # Calculate the cutoff point for this validation period
        cutoff_idx = len(full_data) - (i * prediction_days) - 1
        
        if cutoff_idx <= 60:  # Ensure we have enough data to train
            break
            
        # Split the data at the cutoff point
        training_data = full_data.iloc[:cutoff_idx].copy()
        
        # Actual future price at prediction_days ahead
        try:
            future_idx = min(cutoff_idx + prediction_days, len(full_data) - 1)
            actual_price = float(full_data['Close'].iloc[future_idx])
            actual_date = full_data.index[future_idx]
            
            # Prepare features
            try:
                df = prepare_features(training_data)
                # Train model
                model, scaler, features, metrics = train_prediction_model(df, target_days=prediction_days)
                
                # Make prediction
                predictions = predict_future_prices(
                    model, 
                    scaler, 
                    features, 
                    df, 
                    {'validation': prediction_days}
                )
                
                predicted_price = float(predictions['validation'])
                
                # Calculate accuracy for this prediction
                error_pct = abs((predicted_price - actual_price) / actual_price) * 100
                accuracy = 100 - error_pct
                
                # Store results
                actual_prices.append(actual_price)
                predicted_prices.append(predicted_price)
                dates.append(actual_date)
                accuracy_values.append(accuracy)
                
            except Exception as e:
                # Skip this period if we encounter an error
                continue
                
        except IndexError:
            # Skip if we don't have actual data for this future point
            continue
    
    # If we have no successful validations, return error
    if not actual_prices:
        return {
            "error": "Could not validate on any periods",
            "accuracy": None, 
            "validation_chart": None,
            "validation_results": None
        }
    
    # Create validation chart
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual_prices,
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='blue'),
            marker=dict(size=8)
        )
    )
    
    # Add predicted prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', dash='dash'),
            marker=dict(size=8)
        )
    )
    
    # Format chart
    fig.update_layout(
        title=f"Model Prediction Validation: {prediction_days}-Day Forecasts",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        height=500,
        showlegend=True
    )
    
    # Create validation results table
    validation_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': [f"₹{price:.2f}" for price in actual_prices],
        'Predicted Price': [f"₹{price:.2f}" for price in predicted_prices],
        'Accuracy': [f"{acc:.1f}%" for acc in accuracy_values]
    })
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(accuracy_values)
    
    return {
        "error": None,
        "accuracy": overall_accuracy,
        "validation_chart": fig,
        "validation_results": validation_df
    }

def compare_prediction_timeframes(data, symbol, timeframes=[7, 30, 90]):
    """
    Compare model accuracy across different prediction timeframes
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
        timeframes (list): List of prediction timeframes to test
    
    Returns:
        dict: Dictionary with comparison results and visualization
    """
    results = {}
    accuracy_by_timeframe = {}
    
    for days in timeframes:
        # Run validation for this timeframe
        validation = validate_model_with_past_data(data, periods_back=4, prediction_days=days)
        
        if validation["error"] is None:
            accuracy_by_timeframe[days] = validation["accuracy"]
            results[days] = validation
    
    # Create comparison chart
    if accuracy_by_timeframe:
        fig = go.Figure()
        
        days = list(accuracy_by_timeframe.keys())
        accuracies = list(accuracy_by_timeframe.values())
        
        fig.add_trace(
            go.Bar(
                x=[f"{d} days" for d in days],
                y=accuracies,
                text=[f"{acc:.1f}%" for acc in accuracies],
                textposition='auto',
                marker_color='skyblue'
            )
        )
        
        fig.update_layout(
            title=f"Prediction Accuracy by Timeframe for {symbol}",
            xaxis_title="Prediction Timeframe",
            yaxis_title="Accuracy (%)",
            template="plotly_white",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        return {
            "error": None,
            "comparison_chart": fig,
            "detailed_results": results,
            "best_timeframe": max(accuracy_by_timeframe.items(), key=lambda x: x[1])[0] if accuracy_by_timeframe else None
        }
    else:
        return {
            "error": "Could not generate timeframe comparison",
            "comparison_chart": None,
            "detailed_results": None,
            "best_timeframe": None
        }