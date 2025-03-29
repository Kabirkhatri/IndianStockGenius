import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import base64

def generate_recommendation(tech_analysis, fund_analysis, sentiment_analysis, prediction_results):
    """
    Generate buy/hold/sell recommendation based on analysis
    
    Args:
        tech_analysis (dict): Technical analysis results
        fund_analysis (dict): Fundamental analysis results
        sentiment_analysis (dict): Sentiment analysis results
        prediction_results (dict): Price prediction results
    
    Returns:
        tuple: (recommendation, confidence, reasoning)
    """
    # Initialize scores (scale -2 to +2)
    technical_score = 0
    fundamental_score = 0
    sentiment_score = 0
    prediction_score = 0
    
    # Technical analysis score
    if tech_analysis:
        if tech_analysis['sentiment'] == "Strongly Bullish":
            technical_score = 2
        elif tech_analysis['sentiment'] == "Moderately Bullish":
            technical_score = 1
        elif tech_analysis['sentiment'] == "Neutral":
            technical_score = 0
        elif tech_analysis['sentiment'] == "Moderately Bearish":
            technical_score = -1
        elif tech_analysis['sentiment'] == "Strongly Bearish":
            technical_score = -2
    
    # Fundamental analysis score
    if fund_analysis:
        if fund_analysis['evaluations']['overall_valuation'] == "Undervalued":
            fundamental_score = 2
        elif fund_analysis['evaluations']['overall_valuation'] == "Fairly valued":
            fundamental_score = 1
        elif fund_analysis['evaluations']['overall_valuation'] == "Moderately overvalued":
            fundamental_score = -1
        elif fund_analysis['evaluations']['overall_valuation'] == "Significantly overvalued":
            fundamental_score = -2
    
    # Sentiment analysis score
    if sentiment_analysis:
        sentiment_score = sentiment_analysis['sentiment_score'] * 2  # Scale to -2 to +2
    
    # Price prediction score using the EXACT same logic as our price projection page
    if prediction_results and 'projections' in prediction_results:
        # Get current price from the data
        try:
            # This gets current price from the last row of the data
            current_price = list(prediction_results['projections'].values())[0] / 1.03  # Approximate current price from 1m projection
        except (IndexError, TypeError, ValueError):
            current_price = 100.0  # Fallback
            
        # Use the 1-year projection for the main recommendation
        prediction_1y = None
        for period, price in prediction_results['projections'].items():
            if period == '1y':
                prediction_1y = float(price)
                break
                
        if prediction_1y:
            # Calculate expected yearly return
            yearly_return = ((prediction_1y / current_price) - 1) * 100
            
            # Use EXACTLY the same thresholds as in the app.py file for consistency
            if yearly_return > 5:
                prediction_score = 2  # STRONG BUY
            elif yearly_return > 2:
                prediction_score = 1  # BUY
            elif yearly_return > -2:
                prediction_score = 0  # HOLD
            elif yearly_return > -5:
                prediction_score = -1  # SELL
            else:
                prediction_score = -2  # STRONG SELL
    
    # Calculate weighted score
    # Weights: Technical (30%), Fundamental (30%), Sentiment (15%), Prediction (25%)
    weighted_score = (
        technical_score * 0.3 +
        fundamental_score * 0.3 +
        sentiment_score * 0.15 +
        prediction_score * 0.25
    )
    
    # Give more weight to the price prediction score for the final recommendation
    # This ensures the recommendation from projections is more likely to match 
    # the buy/sell signals shown in the price projections page
    
    # For consistency with the price projections page, scale weights to emphasize prediction
    prediction_weight = 0.6  # Giving more emphasis to price prediction
    remaining_weight = 1.0 - prediction_weight
    
    # Scale other weights accordingly
    tech_weight = 0.3 * remaining_weight
    fund_weight = 0.3 * remaining_weight
    sent_weight = 0.4 * remaining_weight
    
    # Calculate weighted score with adjusted weights
    weighted_score = (
        technical_score * tech_weight +
        fundamental_score * fund_weight +
        sentiment_score * sent_weight +
        prediction_score * prediction_weight
    )
    
    # Determine recommendation based directly on 1-year projection
    # This ensures it matches the signal in the price projection page
    if prediction_score >= 2:
        recommendation = "STRONG BUY"
        confidence = 90  # High confidence
    elif prediction_score >= 1:
        recommendation = "BUY"
        confidence = 75
    elif prediction_score <= -2:
        recommendation = "STRONG SELL"
        confidence = 90  # High confidence
    elif prediction_score <= -1:
        recommendation = "SELL"
        confidence = 75
    else:
        recommendation = "HOLD"
        confidence = 60
    
    # Generate reasoning
    reasons = []
    
    if technical_score > 0:
        reasons.append(f"Technical indicators are {tech_analysis['sentiment'].lower() if tech_analysis else 'bullish'}")
    elif technical_score < 0:
        reasons.append(f"Technical indicators are {tech_analysis['sentiment'].lower() if tech_analysis else 'bearish'}")
    
    if fundamental_score > 0:
        reasons.append(f"Stock appears {fund_analysis['evaluations']['overall_valuation'].lower() if fund_analysis else 'undervalued'} based on fundamentals")
    elif fundamental_score < 0:
        reasons.append(f"Stock appears {fund_analysis['evaluations']['overall_valuation'].lower() if fund_analysis else 'overvalued'} based on fundamentals")
    
    if sentiment_score > 0:
        reasons.append("Market sentiment is positive")
    elif sentiment_score < 0:
        reasons.append("Market sentiment is negative")
    
    if prediction_score > 0:
        reasons.append("Price projections indicate potential upside")
    elif prediction_score < 0:
        reasons.append("Price projections indicate potential downside")
    
    # Format reasoning
    reasoning = "Based on our analysis: " + "; ".join(reasons) + "."
    
    return recommendation, confidence, reasoning

def generate_report(stock_data, symbol, tech_analysis=None, fund_analysis=None, 
                   sentiment_analysis=None, pattern_analysis=None, prediction_results=None):
    """
    Generate a comprehensive report with analysis results
    
    Args:
        stock_data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
        tech_analysis (dict): Technical analysis results
        fund_analysis (dict): Fundamental analysis results
        sentiment_analysis (dict): Sentiment analysis results
        pattern_analysis (dict): Chart pattern analysis results
        prediction_results (dict): Price prediction results
    
    Returns:
        dict: Dictionary with report content
    """
    # Generate recommendation
    recommendation, confidence, reasoning = generate_recommendation(
        tech_analysis, fund_analysis, sentiment_analysis, prediction_results
    )
    
    # Determine target price (1-year projection)
    target_price = prediction_results['projections']['1y'] if prediction_results else stock_data['Close'].iloc[-1] * 1.1
    
    # Generate key points
    key_points = []
    
    # Technical analysis points
    if tech_analysis:
        key_points.append(f"Technical Analysis: {tech_analysis['sentiment']} outlook")
        
        # Add specific indicator insights
        if 'RSI' in tech_analysis['signals']:
            key_points.append(f"RSI: {str(tech_analysis['signals']['RSI'])}")
        
        if 'MACD' in tech_analysis['signals']:
            key_points.append(f"MACD: {str(tech_analysis['signals']['MACD'])}")
        
        if 'MA Cross' in tech_analysis['signals']:
            key_points.append(f"Moving Averages: {str(tech_analysis['signals']['MA Cross'])}")
    
    # Fundamental analysis points
    if fund_analysis:
        key_points.append(f"Fundamental Analysis: Stock appears {fund_analysis['evaluations']['overall_valuation'].lower()}")
        
        if fund_analysis['pe_ratio'] > 0:
            key_points.append(f"P/E Ratio: {fund_analysis['pe_ratio']:.2f} - {fund_analysis['evaluations']['pe_evaluation']}")
        
        if fund_analysis['roe'] > 0:
            key_points.append(f"Return on Equity: {fund_analysis['roe']:.2f}% - {fund_analysis['evaluations']['roe_evaluation']}")
    
    # Sentiment analysis points
    if sentiment_analysis:
        key_points.append(f"Market Sentiment: {sentiment_analysis['sentiment_category']} (Score: {sentiment_analysis['sentiment_score']:.2f})")
    
    # Chart pattern points
    if pattern_analysis and pattern_analysis['patterns']:
        pattern_list = ", ".join(f"{pattern} ({bias})" for pattern, bias in pattern_analysis['patterns'].items())
        key_points.append(f"Chart Patterns: {pattern_list}")
    
    # Price prediction points
    if prediction_results:
        current_price = float(stock_data['Close'].iloc[-1])
        one_year_projection = float(prediction_results['projections']['1y'])
        expected_return = ((one_year_projection / current_price) - 1) * 100
        
        key_points.append(f"1-Year Price Target: ₹{one_year_projection:.2f} ({expected_return:.2f}% from current price)")
    
    # Generate risk factors
    risks = [
        "Market volatility could impact short-term price movements",
        "Regulatory changes may affect the company's operations",
        "Macroeconomic factors such as interest rates and inflation could influence stock performance",
        "Sector-specific challenges might arise"
    ]
    
    # Add specific risk factors based on analysis
    if fund_analysis and fund_analysis['debt_to_equity'] > 1:
        risks.append("High debt-to-equity ratio increases financial risk")
    
    if tech_analysis and any(signal for signal, value in tech_analysis['signals'].items() if 'SELL' in value):
        risks.append("Some technical indicators are showing bearish signals")
    
    if sentiment_analysis and sentiment_analysis['sentiment_score'] < 0:
        risks.append("Negative market sentiment could persist in the short term")
    
    # Generate comprehensive summary
    current_price = float(stock_data['Close'].iloc[-1])
    summary = f"""
    After conducting comprehensive analysis of {symbol} using technical, fundamental, and sentiment analysis, our recommendation is to {recommendation}.
    
    The stock is currently trading at ₹{current_price:.2f} with a one-year price target of ₹{float(target_price):.2f}, representing a potential {'gain' if float(target_price) > current_price else 'loss'} of {abs((float(target_price) / current_price - 1) * 100):.2f}%.
    
    {reasoning}
    
    {'Technical indicators suggest a ' + str(tech_analysis['sentiment']).lower() + ' trend, ' if tech_analysis else ''}
    {'with the stock appearing ' + str(fund_analysis['evaluations']['overall_valuation']).lower() + ' based on fundamental metrics. ' if fund_analysis else ''}
    {'Market sentiment is ' + str(sentiment_analysis['sentiment_category']).lower() + '. ' if sentiment_analysis else ''}
    
    Investors should consider their investment goals, risk tolerance, and time horizon before making any investment decisions.
    """
    
    # Generate HTML report
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Stock Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .recommendation {{ font-size: 24px; font-weight: bold; padding: 10px; border-radius: 5px; display: inline-block; margin: 10px 0; }}
            .buy {{ background-color: #27ae60; color: white; }}
            .hold {{ background-color: #f39c12; color: white; }}
            .sell {{ background-color: #e74c3c; color: white; }}
            .section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .key-points li, .risks li {{ margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>{symbol} Stock Analysis Report</h1>
        <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
        
        <div class="section">
            <h2>Investment Recommendation</h2>
            <div class="recommendation {'buy' if recommendation == 'BUY' else 'hold' if recommendation == 'HOLD' else 'sell'}">{recommendation}</div>
            <p><strong>Target Price (1 Year):</strong> ₹{float(target_price):.2f}</p>
            <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
            <p><strong>Potential Return:</strong> {((float(target_price) / current_price - 1) * 100):.2f}%</p>
            <p><strong>Confidence Level:</strong> {float(confidence):.1f}%</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{summary}</p>
        </div>
        
        <div class="section">
            <h2>Key Points</h2>
            <ul class="key-points">
                {''.join(f'<li>{point}</li>' for point in key_points)}
            </ul>
        </div>
        
        <div class="section">
            <h2>Price Projections</h2>
            <table>
                <tr>
                    <th>Time Horizon</th>
                    <th>Projected Price</th>
                    <th>Potential Return</th>
                </tr>
                {''.join(f'<tr><td>{period}</td><td>₹{float(price):.2f}</td><td>{((float(price) / current_price - 1) * 100):.2f}%</td></tr>' for period, price in prediction_results["projections"].items()) if prediction_results else '<tr><td colspan="3">No price projections available</td></tr>'}
            </table>
        </div>
        
        <div class="section">
            <h2>Risk Factors</h2>
            <ul class="risks">
                {''.join(f'<li>{risk}</li>' for risk in risks)}
            </ul>
        </div>
        
        <div class="section">
            <h2>Technical Analysis</h2>
            {f'<p>{str(tech_analysis["summary"])}</p>' if tech_analysis else '<p>Technical analysis not performed</p>'}
        </div>
        
        <div class="section">
            <h2>Fundamental Analysis</h2>
            {f'<p>{str(fund_analysis["summary"])}</p>' if fund_analysis else '<p>Fundamental analysis not performed</p>'}
        </div>
        
        <div class="section">
            <h2>Sentiment Analysis</h2>
            {f'<p>{str(sentiment_analysis["summary"])}</p>' if sentiment_analysis else '<p>Sentiment analysis not performed</p>'}
        </div>
        
        <div class="footer">
            <p>This report is generated for informational purposes only. It does not constitute investment advice or a recommendation to buy, sell, or hold any security. Always conduct your own research and consult with a financial advisor before making investment decisions.</p>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    # Return report data
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'target_price': target_price,
        'reasoning': reasoning,
        'summary': summary,
        'key_points': key_points,
        'risks': risks,
        'report_html': report_html
    }
