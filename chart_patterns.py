import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def identify_double_top(data, window=20, threshold=0.03):
    """
    Identify double top pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for peaks
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window * 2:
        return False
    
    # Get recent price data
    recent_data = data[-window*2:].copy()
    
    # Find peaks
    peaks = []
    for i in range(1, len(recent_data) - 1):
        high_i = float(recent_data['High'].iloc[i])
        high_prev = float(recent_data['High'].iloc[i-1])
        high_next = float(recent_data['High'].iloc[i+1])
        
        if high_i > high_prev and high_i > high_next:
            peaks.append((i, high_i))
    
    # Need at least 2 peaks
    if len(peaks) < 2:
        return False
    
    # Check for double top pattern
    for i in range(len(peaks) - 1):
        peak1_idx, peak1_val = peaks[i]
        peak2_idx, peak2_val = peaks[i+1]
        
        # Peaks should be separated
        if peak2_idx - peak1_idx < window / 2:
            continue
            
        # Peaks should be at similar levels
        if abs(peak1_val - peak2_val) / peak1_val > threshold:
            continue
            
        # Check for trough between peaks
        trough_idx = peak1_idx + np.argmin(recent_data['Low'].iloc[peak1_idx:peak2_idx].values)
        trough_val = float(recent_data['Low'].iloc[trough_idx])
        
        # Trough should be significantly lower
        if (peak1_val - trough_val) / peak1_val < threshold * 2:
            continue
            
        # We found a valid double top
        return True
    
    return False

def identify_double_bottom(data, window=20, threshold=0.03):
    """
    Identify double bottom pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for bottoms
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window * 2:
        return False
    
    # Get recent price data
    recent_data = data[-window*2:].copy()
    
    # Find bottoms
    bottoms = []
    for i in range(1, len(recent_data) - 1):
        low_i = float(recent_data['Low'].iloc[i])
        low_prev = float(recent_data['Low'].iloc[i-1])
        low_next = float(recent_data['Low'].iloc[i+1])
        
        if low_i < low_prev and low_i < low_next:
            bottoms.append((i, low_i))
    
    # Need at least 2 bottoms
    if len(bottoms) < 2:
        return False
    
    # Check for double bottom pattern
    for i in range(len(bottoms) - 1):
        bottom1_idx, bottom1_val = bottoms[i]
        bottom2_idx, bottom2_val = bottoms[i+1]
        
        # Bottoms should be separated
        if bottom2_idx - bottom1_idx < window / 2:
            continue
            
        # Bottoms should be at similar levels
        if abs(bottom1_val - bottom2_val) / bottom1_val > threshold:
            continue
            
        # Check for peak between bottoms
        peak_idx = bottom1_idx + np.argmax(recent_data['High'].iloc[bottom1_idx:bottom2_idx].values)
        peak_val = float(recent_data['High'].iloc[peak_idx])
        
        # Peak should be significantly higher
        if (peak_val - bottom1_val) / bottom1_val < threshold * 2:
            continue
            
        # We found a valid double bottom
        return True
    
    return False

def identify_head_and_shoulders(data, window=30, threshold=0.03):
    """
    Identify head and shoulders pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for pattern
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window * 2:
        return False
    
    # Get recent price data
    recent_data = data[-window*2:].copy()
    
    # Find peaks
    peaks = []
    for i in range(2, len(recent_data) - 2):
        high_i = float(recent_data['High'].iloc[i])
        high_prev1 = float(recent_data['High'].iloc[i-1])
        high_prev2 = float(recent_data['High'].iloc[i-2])
        high_next1 = float(recent_data['High'].iloc[i+1])
        high_next2 = float(recent_data['High'].iloc[i+2])
        
        if high_i > high_prev1 and high_i > high_prev2 and \
           high_i > high_next1 and high_i > high_next2:
            peaks.append((i, high_i))
    
    # Need at least 3 peaks
    if len(peaks) < 3:
        return False
    
    # Check for head and shoulders pattern
    for i in range(len(peaks) - 2):
        left_idx, left_val = peaks[i]
        head_idx, head_val = peaks[i+1]
        right_idx, right_val = peaks[i+2]
        
        # Head should be higher than shoulders
        if head_val <= left_val or head_val <= right_val:
            continue
            
        # Shoulders should be at similar levels
        if abs(left_val - right_val) / left_val > threshold:
            continue
            
        # Check spacing
        if head_idx - left_idx < window / 5 or right_idx - head_idx < window / 5:
            continue
            
        # We found a valid head and shoulders
        return True
    
    return False

def identify_inverse_head_and_shoulders(data, window=30, threshold=0.03):
    """
    Identify inverse head and shoulders pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for pattern
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window * 2:
        return False
    
    # Get recent price data
    recent_data = data[-window*2:].copy()
    
    # Find bottoms
    bottoms = []
    for i in range(2, len(recent_data) - 2):
        low_i = float(recent_data['Low'].iloc[i])
        low_prev1 = float(recent_data['Low'].iloc[i-1])
        low_prev2 = float(recent_data['Low'].iloc[i-2])
        low_next1 = float(recent_data['Low'].iloc[i+1])
        low_next2 = float(recent_data['Low'].iloc[i+2])
        
        if low_i < low_prev1 and low_i < low_prev2 and \
           low_i < low_next1 and low_i < low_next2:
            bottoms.append((i, low_i))
    
    # Need at least 3 bottoms
    if len(bottoms) < 3:
        return False
    
    # Check for inverse head and shoulders pattern
    for i in range(len(bottoms) - 2):
        left_idx, left_val = bottoms[i]
        head_idx, head_val = bottoms[i+1]
        right_idx, right_val = bottoms[i+2]
        
        # Head should be lower than shoulders
        if head_val >= left_val or head_val >= right_val:
            continue
            
        # Shoulders should be at similar levels
        if abs(left_val - right_val) / left_val > threshold:
            continue
            
        # Check spacing
        if head_idx - left_idx < window / 5 or right_idx - head_idx < window / 5:
            continue
            
        # We found a valid inverse head and shoulders
        return True
    
    return False

def identify_bullish_flag(data, window=20, threshold=0.03):
    """
    Identify bullish flag pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for pattern
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window:
        return False
    
    # Get recent price data
    recent_data = data[-window:].copy()
    
    # Check for strong uptrend before consolidation
    uptrend_window = min(window // 2, 10)
    uptrend_data = recent_data[:uptrend_window]
    
    # Calculate trend strength
    uptrend_start = float(uptrend_data['Close'].iloc[0])
    uptrend_end = float(uptrend_data['Close'].iloc[-1])
    uptrend_strength = (uptrend_end - uptrend_start) / uptrend_start
    
    # Must have strong uptrend
    if uptrend_strength < 0.05:
        return False
    
    # Check for consolidation after uptrend
    consolidation_data = recent_data[uptrend_window:]
    
    # Calculate highest and lowest in consolidation
    high = float(consolidation_data['High'].max())
    low = float(consolidation_data['Low'].min())
    
    # Consolidation range should be narrow
    if (high - low) / low > threshold * 2:
        return False
    
    # Current close should be near the bottom of range (ready to breakout)
    current_close = float(recent_data['Close'].iloc[-1])
    if (current_close - low) / (high - low) > 0.7:
        return False
    
    # We found a valid bullish flag
    return True

def identify_bearish_flag(data, window=20, threshold=0.03):
    """
    Identify bearish flag pattern
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Window to look for pattern
        threshold (float): Price threshold for pattern
    
    Returns:
        bool: True if pattern found
    """
    if len(data) < window:
        return False
    
    # Get recent price data
    recent_data = data[-window:].copy()
    
    # Check for strong downtrend before consolidation
    downtrend_window = min(window // 2, 10)
    downtrend_data = recent_data[:downtrend_window]
    
    # Calculate trend strength
    downtrend_start = float(downtrend_data['Close'].iloc[0])
    downtrend_end = float(downtrend_data['Close'].iloc[-1])
    downtrend_strength = (downtrend_start - downtrend_end) / downtrend_start
    
    # Must have strong downtrend
    if downtrend_strength < 0.05:
        return False
    
    # Check for consolidation after downtrend
    consolidation_data = recent_data[downtrend_window:]
    
    # Calculate highest and lowest in consolidation
    high = float(consolidation_data['High'].max())
    low = float(consolidation_data['Low'].min())
    
    # Consolidation range should be narrow
    if (high - low) / low > threshold * 2:
        return False
    
    # Current close should be near the top of range (ready to breakdown)
    current_close = float(recent_data['Close'].iloc[-1])
    if (high - current_close) / (high - low) > 0.7:
        return False
    
    # We found a valid bearish flag
    return True

def identify_chart_patterns(data):
    """
    Identify chart patterns in price data
    
    Args:
        data (pd.DataFrame): Price data
    
    Returns:
        dict: Dictionary with pattern analysis results
    """
    patterns = {}
    
    # Check for various patterns
    patterns['Double Top'] = 'Bearish' if identify_double_top(data) else None
    patterns['Double Bottom'] = 'Bullish' if identify_double_bottom(data) else None
    patterns['Head and Shoulders'] = 'Bearish' if identify_head_and_shoulders(data) else None
    patterns['Inverse Head and Shoulders'] = 'Bullish' if identify_inverse_head_and_shoulders(data) else None
    patterns['Bullish Flag'] = 'Bullish' if identify_bullish_flag(data) else None
    patterns['Bearish Flag'] = 'Bearish' if identify_bearish_flag(data) else None
    
    # Remove None values
    patterns = {k: v for k, v in patterns.items() if v is not None}
    
    # Generate summary
    bullish_patterns = [k for k, v in patterns.items() if v == 'Bullish']
    bearish_patterns = [k for k, v in patterns.items() if v == 'Bearish']
    
    if bullish_patterns and bearish_patterns:
        pattern_bias = "Mixed"
        summary = f"The chart shows mixed patterns with {len(bullish_patterns)} bullish and {len(bearish_patterns)} bearish formations. Bullish patterns include {', '.join(bullish_patterns)}, while bearish patterns include {', '.join(bearish_patterns)}. This suggests uncertainty in the price direction, and traders should look for additional confirmation signals."
    elif bullish_patterns:
        pattern_bias = "Bullish"
        summary = f"The chart shows bullish chart patterns including {', '.join(bullish_patterns)}. These patterns typically suggest potential upward movement in the near term. Traders might consider buying on breakouts with appropriate stop losses."
    elif bearish_patterns:
        pattern_bias = "Bearish"
        summary = f"The chart shows bearish chart patterns including {', '.join(bearish_patterns)}. These patterns typically suggest potential downward movement in the near term. Traders might consider selling on breakdowns with appropriate stop losses."
    else:
        pattern_bias = "Neutral"
        summary = "No significant chart patterns were identified in the current timeframe. The price may continue its current trend or move sideways. Traders should look for other technical indicators to guide their decisions."
    
    # Create chart visualization
    fig = create_pattern_chart(data, patterns)
    
    # Return results
    return {
        'patterns': patterns,
        'pattern_bias': pattern_bias,
        'summary': summary,
        'pattern_chart': fig
    }

def create_pattern_chart(data, patterns):
    """
    Create a chart visualizing identified patterns
    
    Args:
        data (pd.DataFrame): Price data
        patterns (dict): Identified patterns
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        )
    )
    
    # Format chart
    fig.update_layout(
        title="Chart Pattern Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    # Add pattern annotations
    if patterns:
        annotations = []
        high_max = float(data['High'].max())
        y_pos = high_max * 1.05
        
        for i, (pattern, bias) in enumerate(patterns.items()):
            color = "green" if bias == "Bullish" else "red"
            annotations.append(
                dict(
                    x=data.index[-1],
                    y=y_pos - (i * high_max * 0.03),
                    xref="x",
                    yref="y",
                    text=f"{pattern} (⯀ {bias})",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=-60,
                    ay=0,
                    font=dict(
                        size=12,
                        color=color
                    )
                )
            )
        
        fig.update_layout(annotations=annotations)
    
    return fig
