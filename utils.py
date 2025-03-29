import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Define time periods for predictions
time_periods = {
    '1w': '1 Week',
    '1m': '1 Month',
    '3m': '3 Months',
    '6m': '6 Months',
    '1y': '1 Year'
}

def plot_stock_data(data, symbol):
    """
    Plot stock price data with volume
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
    
    Returns:
        None (displays plot in Streamlit)
    """
    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Price', 'Volume'))
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Add moving averages
    ma_periods = [20, 50, 200]
    ma_colors = ['rgba(255, 0, 0, 0.7)', 'rgba(0, 255, 0, 0.7)', 'rgba(0, 0, 255, 0.7)']
    
    for period, color in zip(ma_periods, ma_colors):
        ma = data['Close'].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma,
                name=f'MA {period}',
                line=dict(color=color, width=1)
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_white'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Display the figure
    return fig

def plot_technical_indicators(data, symbol):
    """
    Plot technical indicators
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
    
    Returns:
        None (displays plot in Streamlit)
    """
    # Create figure with subplots for different indicators
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=(f'{symbol} Price', 'MACD', 'RSI', 'Bollinger Bands'))
    
    # Add price with candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Calculate and add MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=macd,
            name="MACD",
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=signal,
            name="Signal",
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=histogram,
            name="Histogram",
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Calculate and add RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi,
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        row=3, col=1
    )
    
    # Add RSI reference lines
    fig.add_shape(
        type="line",
        xref="x", yref="y3",
        x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
        line=dict(color="red", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        xref="x", yref="y3",
        x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
        line=dict(color="green", width=1, dash="dash")
    )
    
    # Calculate and add Bollinger Bands
    sma20 = data['Close'].rolling(window=20).mean()
    std20 = data['Close'].rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=upper_band,
            name="Upper BB",
            line=dict(color='rgba(0, 255, 0, 0.7)', width=1)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=sma20,
            name="SMA 20",
            line=dict(color='rgba(255, 165, 0, 0.7)', width=1)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=lower_band,
            name="Lower BB",
            line=dict(color='rgba(255, 0, 0, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.1)'
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Close",
            line=dict(color='blue', width=1)
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Indicators',
        xaxis_title='Date',
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Price (₹)", row=4, col=1)
    
    # Update x-axis
    fig.update_xaxes(rangeslider_visible=False)
    
    # Display the figure
    return fig
