import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define helper functions for technical indicators
def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # For periods beyond the initial window
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    # Calculate %K
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (3-day SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_adx(data, period=14):
    """Calculate Average Directional Index (ADX)"""
    # True Range (TR)
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Directional Movement (DM)
    data['up_move'] = data['High'] - data['High'].shift(1)
    data['down_move'] = data['Low'].shift(1) - data['Low']
    
    data['+DM'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
    data['-DM'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
    
    # Smoothed TR, +DM, -DM
    data['ATR'] = data['TR'].rolling(window=period).mean()
    data['+DI'] = 100 * (data['+DM'].rolling(window=period).mean() / data['ATR'])
    data['-DI'] = 100 * (data['-DM'].rolling(window=period).mean() / data['ATR'])
    
    # Directional Index (DX)
    data['DI_diff'] = abs(data['+DI'] - data['-DI'])
    data['DI_sum'] = data['+DI'] + data['-DI']
    data['DX'] = 100 * (data['DI_diff'] / data['DI_sum'])
    
    # Average Directional Index (ADX)
    data['ADX'] = data['DX'].rolling(window=period).mean()
    
    return data['ADX']

def calculate_obv(data):
    """Calculate On-Balance Volume (OBV)"""
    obv = pd.Series(index=data.index)
    obv.iloc[0] = 0

    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
            
    return obv

def calculate_pivot_points(data):
    """Calculate Pivot Points (traditional method)"""
    pivot = (data['High'].iloc[-1] + data['Low'].iloc[-1] + data['Close'].iloc[-1]) / 3
    s1 = (2 * pivot) - data['High'].iloc[-1]
    s2 = pivot - (data['High'].iloc[-1] - data['Low'].iloc[-1])
    r1 = (2 * pivot) - data['Low'].iloc[-1]
    r2 = pivot + (data['High'].iloc[-1] - data['Low'].iloc[-1])
    
    return {
        'pivot': pivot,
        'support1': s1,
        'support2': s2,
        'resistance1': r1,
        'resistance2': r2
    }

def get_technical_indicators(data):
    """
    Calculate and return technical indicators for a stock
    
    Args:
        data (pd.DataFrame): Stock price data
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    # Calculate indicators
    sma_20 = calculate_sma(data, 20).iloc[-1]
    sma_50 = calculate_sma(data, 50).iloc[-1]
    sma_200 = calculate_sma(data, 200).iloc[-1]
    
    ema_20 = calculate_ema(data, 20).iloc[-1]
    ema_50 = calculate_ema(data, 50).iloc[-1]
    
    rsi = calculate_rsi(data).iloc[-1]
    
    macd_line, signal_line, histogram = calculate_macd(data)
    macd_val = macd_line.iloc[-1]
    macd_signal = signal_line.iloc[-1]
    macd_hist = histogram.iloc[-1]
    
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
    bb_upper = upper_band.iloc[-1]
    bb_middle = middle_band.iloc[-1]
    bb_lower = lower_band.iloc[-1]
    
    k, d = calculate_stochastic(data)
    stoch_k = k.iloc[-1]
    stoch_d = d.iloc[-1]
    
    current_close = data['Close'].iloc[-1]
    
    # Build DataFrame
    indicators = {
        'Indicator': [
            'SMA(20)', 'SMA(50)', 'SMA(200)', 'EMA(20)', 'EMA(50)',
            'RSI(14)', 'MACD', 'MACD Signal', 'MACD Histogram',
            'Bollinger Upper', 'Bollinger Middle', 'Bollinger Lower',
            'Stochastic %K', 'Stochastic %D'
        ],
        'Value': [
            round(sma_20, 2), round(sma_50, 2), round(sma_200, 2),
            round(ema_20, 2), round(ema_50, 2), round(rsi, 2),
            round(macd_val, 2), round(macd_signal, 2), round(macd_hist, 2),
            round(bb_upper, 2), round(bb_middle, 2), round(bb_lower, 2),
            round(stoch_k, 2), round(stoch_d, 2)
        ],
        'Signal': [
            'Bullish' if float(current_close) > float(sma_20) else 'Bearish',
            'Bullish' if float(current_close) > float(sma_50) else 'Bearish',
            'Bullish' if float(current_close) > float(sma_200) else 'Bearish',
            'Bullish' if float(current_close) > float(ema_20) else 'Bearish',
            'Bullish' if float(current_close) > float(ema_50) else 'Bearish',
            'Overbought' if float(rsi) > 70 else 'Oversold' if float(rsi) < 30 else 'Neutral',
            'Bullish' if float(macd_val) > 0 else 'Bearish',
            'Bullish' if float(macd_val) > float(macd_signal) else 'Bearish',
            'Bullish' if float(macd_hist) > 0 else 'Bearish',
            'Near Resistance' if float(current_close) > float(bb_middle) and float(current_close) < float(bb_upper) else 'Neutral',
            'Neutral',
            'Near Support' if float(current_close) < float(bb_middle) and float(current_close) > float(bb_lower) else 'Neutral',
            'Overbought' if float(stoch_k) > 80 else 'Oversold' if float(stoch_k) < 20 else 'Neutral',
            'Overbought' if float(stoch_d) > 80 else 'Oversold' if float(stoch_d) < 20 else 'Neutral'
        ]
    }
    
    return pd.DataFrame(indicators)

def perform_technical_analysis(data):
    """
    Perform comprehensive technical analysis on stock data
    
    Args:
        data (pd.DataFrame): Stock price data
    
    Returns:
        dict: Dictionary with technical analysis results
    """
    # Get technical indicators
    indicators_df = get_technical_indicators(data)
    
    # Calculate additional indicators
    adx = calculate_adx(data.copy()).iloc[-1]  # Make copy for ADX calculation
    obv = calculate_obv(data).iloc[-1]
    obv_prev = calculate_obv(data).iloc[-2] if len(data) > 1 else 0
    pivot_points = calculate_pivot_points(data)
    
    # Current price
    current_price = data['Close'].iloc[-1]
    
    # Determine signals
    signals = {}
    
    # Moving Averages
    signals['SMA 20'] = 'BUY' if current_price > float(indicators_df[indicators_df['Indicator'] == 'SMA(20)']['Value']) else 'SELL'
    signals['SMA 50'] = 'BUY' if current_price > float(indicators_df[indicators_df['Indicator'] == 'SMA(50)']['Value']) else 'SELL'
    signals['SMA 200'] = 'BUY' if current_price > float(indicators_df[indicators_df['Indicator'] == 'SMA(200)']['Value']) else 'SELL'
    signals['EMA 20'] = 'BUY' if current_price > float(indicators_df[indicators_df['Indicator'] == 'EMA(20)']['Value']) else 'SELL'
    signals['EMA 50'] = 'BUY' if current_price > float(indicators_df[indicators_df['Indicator'] == 'EMA(50)']['Value']) else 'SELL'
    
    # Golden/Death Cross
    sma_50 = float(indicators_df[indicators_df['Indicator'] == 'SMA(50)']['Value'])
    sma_200 = float(indicators_df[indicators_df['Indicator'] == 'SMA(200)']['Value'])
    signals['MA Cross'] = 'GOLDEN CROSS (BUY)' if sma_50 > sma_200 else 'DEATH CROSS (SELL)'
    
    # RSI
    rsi = float(indicators_df[indicators_df['Indicator'] == 'RSI(14)']['Value'])
    if rsi > 70:
        signals['RSI'] = 'SELL (Overbought)'
    elif rsi < 30:
        signals['RSI'] = 'BUY (Oversold)'
    else:
        signals['RSI'] = 'NEUTRAL'
    
    # MACD
    macd = float(indicators_df[indicators_df['Indicator'] == 'MACD']['Value'])
    macd_signal = float(indicators_df[indicators_df['Indicator'] == 'MACD Signal']['Value'])
    macd_hist = float(indicators_df[indicators_df['Indicator'] == 'MACD Histogram']['Value'])
    
    if macd > macd_signal and macd_hist > 0:
        signals['MACD'] = 'BUY'
    elif macd < macd_signal and macd_hist < 0:
        signals['MACD'] = 'SELL'
    else:
        signals['MACD'] = 'NEUTRAL'
    
    # Bollinger Bands
    bb_upper = float(indicators_df[indicators_df['Indicator'] == 'Bollinger Upper']['Value'])
    bb_middle = float(indicators_df[indicators_df['Indicator'] == 'Bollinger Middle']['Value'])
    bb_lower = float(indicators_df[indicators_df['Indicator'] == 'Bollinger Lower']['Value'])
    
    if current_price > bb_upper:
        signals['Bollinger Bands'] = 'SELL (Upper Band Resistance)'
    elif current_price < bb_lower:
        signals['Bollinger Bands'] = 'BUY (Lower Band Support)'
    else:
        signals['Bollinger Bands'] = 'NEUTRAL'
    
    # Stochastic
    stoch_k = float(indicators_df[indicators_df['Indicator'] == 'Stochastic %K']['Value'])
    stoch_d = float(indicators_df[indicators_df['Indicator'] == 'Stochastic %D']['Value'])
    
    if stoch_k > 80 and stoch_d > 80:
        signals['Stochastic'] = 'SELL (Overbought)'
    elif stoch_k < 20 and stoch_d < 20:
        signals['Stochastic'] = 'BUY (Oversold)'
    else:
        signals['Stochastic'] = 'NEUTRAL'
    
    # ADX
    if adx > 25:
        signals['ADX'] = 'STRONG TREND'
    else:
        signals['ADX'] = 'WEAK TREND'
    
    # OBV
    signals['OBV'] = 'INCREASING (Bullish)' if obv > obv_prev else 'DECREASING (Bearish)'
    
    # Pivot Points
    if current_price > pivot_points['pivot']:
        if current_price > pivot_points['resistance1']:
            signals['Pivot Points'] = 'ABOVE R1 (Bullish)'
        else:
            signals['Pivot Points'] = 'ABOVE PIVOT (Bullish)'
    else:
        if current_price < pivot_points['support1']:
            signals['Pivot Points'] = 'BELOW S1 (Bearish)'
        else:
            signals['Pivot Points'] = 'BELOW PIVOT (Bearish)'
    
    # Generate summary based on signals
    bull_count = sum(1 for signal in signals.values() if 'BUY' in signal or 'Bullish' in signal or 'GOLDEN CROSS' in signal)
    bear_count = sum(1 for signal in signals.values() if 'SELL' in signal or 'Bearish' in signal or 'DEATH CROSS' in signal)
    
    # Weighted signals (give more importance to certain indicators)
    weighted_bull = bull_count
    weighted_bear = bear_count
    
    # Add weights to important indicators
    for key, value in signals.items():
        if key in ['MACD', 'RSI', 'MA Cross'] and ('BUY' in value or 'GOLDEN CROSS' in value or 'Bullish' in value):
            weighted_bull += 1
        elif key in ['MACD', 'RSI', 'MA Cross'] and ('SELL' in value or 'DEATH CROSS' in value or 'Bearish' in value):
            weighted_bear += 1
    
    # Determine overall sentiment
    if weighted_bull > weighted_bear + 2:
        sentiment = "Strongly Bullish"
    elif weighted_bull > weighted_bear:
        sentiment = "Moderately Bullish"
    elif weighted_bear > weighted_bull + 2:
        sentiment = "Strongly Bearish"
    elif weighted_bear > weighted_bull:
        sentiment = "Moderately Bearish"
    else:
        sentiment = "Neutral"
    
    # Create detailed summary
    summary = f"""
    Technical Analysis Summary: {sentiment}
    
    The technical indicators show a {sentiment.lower()} outlook for this stock. Out of {len(signals)} indicators analyzed, 
    {bull_count} are giving bullish signals and {bear_count} are giving bearish signals.
    
    Key findings:
    - Moving Averages: The stock is {'above' if signals['SMA 50'] == 'BUY' else 'below'} its 50-day SMA and {'above' if signals['SMA 200'] == 'BUY' else 'below'} its 200-day SMA.
    - RSI: Currently at {rsi:.2f}, indicating the stock is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neither overbought nor oversold'}.
    - MACD: The MACD line is {'above' if macd > macd_signal else 'below'} the signal line, suggesting {'bullish' if macd > macd_signal else 'bearish'} momentum.
    - Bollinger Bands: The price is {'above the upper band' if current_price > bb_upper else 'below the lower band' if current_price < bb_lower else 'within the bands'}.
    
    The ADX value of {adx:.2f} indicates a {'strong' if adx > 25 else 'weak'} trend in the market, suggesting that the signals from trend-following indicators should be given {'higher' if adx > 25 else 'lower'} weight.
    """
    
    # Return results
    return {
        'indicators': indicators_df,
        'signals': signals,
        'sentiment': sentiment,
        'summary': summary,
        'pivot_points': pivot_points,
        'adx': adx,
        'obv': obv
    }
