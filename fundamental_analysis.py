import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import random
from data_fetcher import fetch_stock_info, fetch_financial_data

def perform_fundamental_analysis(symbol):
    """
    Perform fundamental analysis on a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Dictionary with fundamental analysis results
    """
    # Fetch basic stock info
    stock_info = fetch_stock_info(symbol)
    
    # Fetch financial data
    financial_data = fetch_financial_data(symbol)
    
    # Calculate additional ratios and metrics
    try:
        # Try with yfinance for more comprehensive data
        stock = yf.Ticker(f"{symbol}.NS")
        info = stock.info
        
        # Extract key metrics
        pe_ratio = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        ps_ratio = info.get('priceToSalesTrailing12Months', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        peg_ratio = info.get('pegRatio', 0)
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
        operating_margin = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
        roa = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        
        # Debt ratios
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        total_revenue = info.get('totalRevenue', 0)
        market_cap = info.get('marketCap', 0)
        
        quick_ratio = info.get('quickRatio', 0)
        current_ratio = info.get('currentRatio', 0)
        debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
        
        # Fill missing data from financial_data if needed
        if roe == 0 and financial_data['roe'] != 0:
            roe = financial_data['roe'] * 100
        if roa == 0 and financial_data['roa'] != 0:
            roa = financial_data['roa'] * 100
        if debt_to_equity == 0 and financial_data['debt_to_equity'] != 0:
            debt_to_equity = financial_data['debt_to_equity']
            
    except:
        # Fall back to basic values if detailed info isn't available
        pe_ratio = stock_info.get('pe_ratio', 0)
        forward_pe = 0
        pb_ratio = stock_info.get('pb_ratio', 0)
        ps_ratio = 0
        dividend_yield = stock_info.get('dividend_yield', 0)
        peg_ratio = 0
        ev_ebitda = 0
        profit_margin = 0
        operating_margin = 0
        roa = financial_data.get('roa', 0) * 100
        roe = financial_data.get('roe', 0) * 100
        
        total_debt = financial_data.get('total_debt', 0)
        total_cash = 0
        total_revenue = financial_data.get('revenue', 0)
        market_cap = stock_info.get('market_cap', 0)
        
        quick_ratio = 0
        current_ratio = 0
        debt_to_equity = financial_data.get('debt_to_equity', 0)
    
    # Sector comparison (would ideally use real sector averages, using approximate ranges for Indian market)
    sector_pe_ranges = {
        'Technology': (20, 40),
        'Financial Services': (12, 25),
        'Healthcare': (18, 35),
        'Consumer Cyclical': (15, 30),
        'Consumer Defensive': (14, 28),
        'Industrials': (15, 35),
        'Basic Materials': (10, 20),
        'Energy': (8, 18),
        'Real Estate': (10, 25),
        'Utilities': (12, 20),
        'Communication Services': (15, 30),
        'Unknown': (15, 30)  # Default range
    }
    
    sector = stock_info.get('sector', 'Unknown')
    sector_pe_range = sector_pe_ranges.get(sector, sector_pe_ranges['Unknown'])
    
    # Evaluate fundamentals
    # PE ratio evaluation
    if pe_ratio == 0 or np.isnan(pe_ratio):
        pe_evaluation = "Unavailable"
    elif pe_ratio < sector_pe_range[0]:
        pe_evaluation = "Undervalued compared to sector average"
    elif pe_ratio > sector_pe_range[1]:
        pe_evaluation = "Overvalued compared to sector average"
    else:
        pe_evaluation = "Within sector average range"
    
    # PB ratio evaluation
    if pb_ratio == 0 or np.isnan(pb_ratio):
        pb_evaluation = "Unavailable"
    elif pb_ratio < 1:
        pb_evaluation = "Potentially undervalued (trading below book value)"
    elif pb_ratio < 3:
        pb_evaluation = "Reasonably valued"
    else:
        pb_evaluation = "Premium valuation"
    
    # ROE evaluation
    if roe == 0 or np.isnan(roe):
        roe_evaluation = "Unavailable"
    elif roe < 10:
        roe_evaluation = "Below average"
    elif roe < 20:
        roe_evaluation = "Good"
    else:
        roe_evaluation = "Excellent"
    
    # Debt to Equity evaluation
    if debt_to_equity == 0 or np.isnan(debt_to_equity):
        debt_evaluation = "Unavailable"
    elif debt_to_equity < 0.5:
        debt_evaluation = "Low debt (conservative)"
    elif debt_to_equity < 1.5:
        debt_evaluation = "Moderate debt"
    else:
        debt_evaluation = "High debt (risky)"
    
    # Generate summary
    # Determine overall valuation
    valuation_points = 0
    total_metrics = 0
    
    # PE ratio points
    if pe_evaluation == "Undervalued compared to sector average":
        valuation_points += 2
        total_metrics += 1
    elif pe_evaluation == "Within sector average range":
        valuation_points += 1
        total_metrics += 1
    elif pe_evaluation == "Overvalued compared to sector average":
        total_metrics += 1
    
    # PB ratio points
    if pb_evaluation == "Potentially undervalued (trading below book value)":
        valuation_points += 2
        total_metrics += 1
    elif pb_evaluation == "Reasonably valued":
        valuation_points += 1
        total_metrics += 1
    elif pb_evaluation == "Premium valuation":
        total_metrics += 1
    
    # Financial health points
    if roe_evaluation == "Excellent":
        valuation_points += 2
        total_metrics += 1
    elif roe_evaluation == "Good":
        valuation_points += 1
        total_metrics += 1
    elif roe_evaluation == "Below average":
        total_metrics += 1
    
    if debt_evaluation == "Low debt (conservative)":
        valuation_points += 2
        total_metrics += 1
    elif debt_evaluation == "Moderate debt":
        valuation_points += 1
        total_metrics += 1
    elif debt_evaluation == "High debt (risky)":
        total_metrics += 1
    
    # Calculate overall score
    if total_metrics > 0:
        overall_score = valuation_points / total_metrics
        
        if overall_score >= 1.5:
            overall_valuation = "Undervalued"
        elif overall_score >= 1.0:
            overall_valuation = "Fairly valued"
        elif overall_score >= 0.5:
            overall_valuation = "Moderately overvalued"
        else:
            overall_valuation = "Significantly overvalued"
    else:
        overall_valuation = "Unable to determine valuation due to insufficient data"
    
    # Format the summary
    summary = f"""
    Fundamental Analysis Summary for {symbol}:
    
    Overall Assessment: {overall_valuation}
    
    Valuation Metrics:
    - P/E Ratio: {pe_ratio:.2f} - {pe_evaluation}
    - P/B Ratio: {pb_ratio:.2f} - {pb_evaluation}
    - Dividend Yield: {dividend_yield:.2f}%
    
    Financial Health:
    - Return on Equity (ROE): {roe:.2f}% - {roe_evaluation}
    - Debt to Equity: {debt_to_equity:.2f} - {debt_evaluation}
    
    The company operates in the {sector} sector. Based on the available fundamental data, the stock appears to be {overall_valuation.lower()}. 
    {'The high ROE indicates efficient use of shareholder equity.' if roe_evaluation == "Excellent" else ''}
    {'The low debt level reduces financial risk.' if debt_evaluation == "Low debt (conservative)" else ''}
    {'The high debt level increases financial risk.' if debt_evaluation == "High debt (risky)" else ''}
    """
    
    # Return complete analysis
    return {
        'pe_ratio': pe_ratio,
        'forward_pe': forward_pe,
        'pb_ratio': pb_ratio,
        'ps_ratio': ps_ratio,
        'dividend_yield': dividend_yield,
        'peg_ratio': peg_ratio,
        'ev_ebitda': ev_ebitda,
        'profit_margin': profit_margin,
        'operating_margin': operating_margin,
        'roa': roa,
        'roe': roe,
        'debt_to_equity': debt_to_equity,
        'quick_ratio': quick_ratio,
        'current_ratio': current_ratio,
        'market_cap': market_cap,
        'sector': sector,
        'evaluations': {
            'pe_evaluation': pe_evaluation,
            'pb_evaluation': pb_evaluation,
            'roe_evaluation': roe_evaluation,
            'debt_evaluation': debt_evaluation,
            'overall_valuation': overall_valuation
        },
        'summary': summary
    }
