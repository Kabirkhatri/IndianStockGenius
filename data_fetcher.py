import pandas as pd
import numpy as np
import os
import yfinance as yf
from nsepy import get_history
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

# Base indices for Indian market
nifty50_url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
nifty500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# Cache for stock data to reduce API calls
stock_data_cache = {}
stock_list_cache = None

def get_available_stocks():
    """
    Get a list of available stocks from NSE
    """
    global stock_list_cache
    
    if stock_list_cache is not None:
        return stock_list_cache
    
    try:
        # Try to fetch Nifty 500 list
        df = pd.read_csv(nifty500_url)
        df = df.rename(columns={"Symbol": "Symbol", "Company Name": "Company Name"})
        stock_list_cache = df[["Symbol", "Company Name"]]
    except:
        try:
            # Fallback to Nifty 50
            df = pd.read_csv(nifty50_url)
            df = df.rename(columns={"Symbol": "Symbol", "Company Name": "Company Name"})
            stock_list_cache = df[["Symbol", "Company Name"]]
        except:
            # Create sample dataframe if both fail
            stock_list_cache = pd.DataFrame({
                "Symbol": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
                "Company Name": ["Reliance Industries Ltd.", "Tata Consultancy Services Ltd.", 
                                "HDFC Bank Ltd.", "Infosys Ltd.", "ICICI Bank Ltd."]
            })
    
    return stock_list_cache

def search_stock(query):
    """
    Search for stocks based on a query string
    
    Args:
        query (str): Search query (company name or symbol)
    
    Returns:
        pd.DataFrame: DataFrame with matching stocks
    """
    # Get the list of available stocks
    all_stocks = get_available_stocks()
    
    # Convert query to uppercase for case-insensitive search
    query = query.upper()
    
    # Search in symbols and company names
    symbol_matches = all_stocks[all_stocks["Symbol"].str.contains(query)]
    name_matches = all_stocks[all_stocks["Company Name"].str.upper().str.contains(query)]
    
    # Combine results
    results = pd.concat([symbol_matches, name_matches]).drop_duplicates()
    
    return results

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data for a given symbol
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
    
    Returns:
        pd.DataFrame: DataFrame with stock data
    """
    # Convert dates to string format for cache key
    cache_key = f"{symbol}_{start_date}_{end_date}"
    
    # Check if data is in cache
    if cache_key in stock_data_cache:
        return stock_data_cache[cache_key]
    
    try:
        # Try fetching data from NSEpy first
        stock_data = get_history(symbol=symbol, 
                                start=start_date, 
                                end=end_date)
        
        # If data is empty, try with yfinance
        if stock_data.empty:
            # Add .NS suffix for NSE stocks in yfinance
            yf_symbol = f"{symbol}.NS"
            stock_data = yf.download(yf_symbol, start=start_date, end=end_date)
    except:
        # If NSEpy fails, try with yfinance
        try:
            # Add .NS suffix for NSE stocks in yfinance
            yf_symbol = f"{symbol}.NS"
            stock_data = yf.download(yf_symbol, start=start_date, end=end_date)
        except:
            # If both fail, return empty DataFrame
            return pd.DataFrame()
    
    # Ensure the DataFrame has the expected columns
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Add missing columns if needed
    for col in expected_columns:
        if col not in stock_data.columns:
            stock_data[col] = np.nan
    
    # Reset index if date is in the index
    if 'Date' not in stock_data.columns and isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data = stock_data.reset_index()
    
    # Cache the data
    stock_data_cache[cache_key] = stock_data
    
    return stock_data

def fetch_stock_info(symbol):
    """
    Fetch general information about a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Dictionary with stock information
    """
    try:
        # Try with yfinance
        yf_symbol = f"{symbol}.NS"
        stock = yf.Ticker(yf_symbol)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            'symbol': symbol,
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'beta': info.get('beta', 0),
            'website': info.get('website', ''),
        }
        
        return stock_info
    except:
        # If yfinance fails, return basic info
        return {
            'symbol': symbol,
            'name': symbol,
            'sector': '',
            'industry': '',
            'market_cap': 0,
            'pe_ratio': 0,
            'pb_ratio': 0,
            'dividend_yield': 0,
            'beta': 0,
            'website': '',
        }

def fetch_financial_data(symbol):
    """
    Fetch financial data for a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Dictionary with financial data
    """
    try:
        # Try with yfinance
        yf_symbol = f"{symbol}.NS"
        stock = yf.Ticker(yf_symbol)
        
        # Get financial statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Extract key metrics
        if not income_stmt.empty and not balance_sheet.empty:
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            
            revenue = latest_income.get('Total Revenue', 0)
            net_income = latest_income.get('Net Income', 0)
            total_assets = latest_balance.get('Total Assets', 0)
            total_equity = latest_balance.get('Total Stockholder Equity', 0)
            total_debt = latest_balance.get('Total Debt', 0)
            
            # Calculate ratios
            roe = net_income / total_equity if total_equity else 0
            roa = net_income / total_assets if total_assets else 0
            debt_to_equity = total_debt / total_equity if total_equity else 0
            
            return {
                'revenue': revenue,
                'net_income': net_income,
                'total_assets': total_assets,
                'total_equity': total_equity,
                'total_debt': total_debt,
                'roe': roe,
                'roa': roa,
                'debt_to_equity': debt_to_equity
            }
    except:
        # If yfinance fails, return empty data
        pass
    
    # Return default values if data fetch fails
    return {
        'revenue': 0,
        'net_income': 0,
        'total_assets': 0,
        'total_equity': 0,
        'total_debt': 0,
        'roe': 0,
        'roa': 0,
        'debt_to_equity': 0
    }

def fetch_index_data(index_name='NIFTY 50', start_date=None, end_date=None):
    """
    Fetch index data for benchmarking
    
    Args:
        index_name (str): Index name
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        pd.DataFrame: DataFrame with index data
    """
    if not start_date:
        start_date = datetime.now() - timedelta(days=365)
    if not end_date:
        end_date = datetime.now()
        
    try:
        # Map index names to yfinance symbols
        index_map = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY IT': '^CNXIT',
            'NIFTY PHARMA': '^CNXPHARMA'
        }
        
        yf_symbol = index_map.get(index_name, '^NSEI')  # Default to NIFTY 50
        index_data = yf.download(yf_symbol, start=start_date, end=end_date)
        
        return index_data
    except:
        # Return empty DataFrame on error
        return pd.DataFrame()
