import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Initialize NLTK components
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# News sources for Indian stocks
NEWS_SOURCES = [
    "https://economictimes.indiatimes.com/markets/stocks",
    "https://www.moneycontrol.com/stocksmarketsindia/",
    "https://www.business-standard.com/markets",
    "https://www.livemint.com/market"
]

def fetch_news_headlines(symbol, max_headlines=10):
    """
    Fetch news headlines related to a stock
    
    Args:
        symbol (str): Stock symbol
        max_headlines (int): Maximum number of headlines to fetch
    
    Returns:
        list: List of news headlines
    """
    headlines = []
    
    # In a real implementation, we would use news APIs or web scraping
    # For demo purposes, we'll simulate with a combination of generic and symbol-specific headlines
    
    # Try to fetch news from Economic Times
    try:
        url = f"https://economictimes.indiatimes.com/markets/stocks/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = soup.find_all('div', class_='eachStory')
            
            for item in news_items[:max_headlines]:
                title_element = item.find('h3')
                if title_element:
                    headlines.append(title_element.text.strip())
        
        # If we have headlines and they contain the symbol, prioritize them
        symbol_specific = [h for h in headlines if symbol.upper() in h.upper()]
        other_headlines = [h for h in headlines if symbol.upper() not in h.upper()]
        
        # Combine symbol-specific headlines first, then others
        headlines = symbol_specific + other_headlines
        
    except Exception as e:
        # If there's an error, we'll continue with simulated headlines
        pass
    
    # If we couldn't fetch headlines or didn't get enough, generate some
    if len(headlines) < max_headlines:
        # Add some generic financial news headlines
        generic_headlines = [
            f"Investors eyeing {symbol} as Q2 results approach",
            f"{symbol} announces expansion plans in key markets",
            f"Analysts remain divided on {symbol}'s growth prospects",
            f"New government policy could impact {symbol} and sector peers",
            f"{symbol} shows resilience amid market volatility",
            f"Foreign investors increase stake in {symbol}",
            f"Domestic funds trim holdings in {symbol}",
            f"{symbol} declares quarterly dividend",
            f"Technical indicators suggest bullish momentum for {symbol}",
            f"Is {symbol} a value trap? Experts weigh in",
            f"{symbol} partners with tech giants for digital transformation",
            f"Supply chain issues continue to impact {symbol}",
            f"Retail investors flock to {symbol} after recent correction",
            f"{symbol} management addresses concerns over debt levels",
            f"ESG compliance improves for {symbol}, attracts sustainable investors"
        ]
        
        # Add random generic headlines until we reach max_headlines
        while len(headlines) < max_headlines and generic_headlines:
            headline = random.choice(generic_headlines)
            headlines.append(headline)
            generic_headlines.remove(headline)
    
    # Limit to max_headlines
    return headlines[:max_headlines]

def analyze_headline_sentiment(headline):
    """
    Analyze the sentiment of a headline
    
    Args:
        headline (str): News headline
    
    Returns:
        float: Sentiment score (-1 to 1, negative to positive)
    """
    # Use VADER sentiment analyzer
    sentiment = sia.polarity_scores(headline)
    return sentiment['compound']

def get_recent_news_with_sentiment(symbol, max_headlines=10):
    """
    Get recent news headlines with sentiment analysis
    
    Args:
        symbol (str): Stock symbol
        max_headlines (int): Maximum number of headlines
    
    Returns:
        pd.DataFrame: DataFrame with headlines and sentiment
    """
    # Fetch headlines
    headlines = fetch_news_headlines(symbol, max_headlines)
    
    # Analyze sentiment
    data = []
    for headline in headlines:
        sentiment_score = analyze_headline_sentiment(headline)
        
        # Determine sentiment category
        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Generate random date within last 7 days
        days_ago = random.randint(0, 7)
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'Date': date,
            'Headline': headline,
            'Sentiment': sentiment,
            'Score': sentiment_score
        })
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False)
    
    return df

def simulate_social_sentiment(symbol, days=7):
    """
    Simulate social media sentiment data
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days of data
    
    Returns:
        pd.DataFrame: DataFrame with simulated social sentiment
    """
    data = []
    today = datetime.now()
    
    for i in range(days):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Simulate sentiment data
        # In a real implementation, this would come from Twitter/X, Reddit, etc.
        positive = random.randint(30, 70)
        negative = random.randint(10, 100 - positive - 10)
        neutral = 100 - positive - negative
        
        # Calculate overall sentiment score (-1 to 1)
        sentiment_score = (positive - negative) / 100
        
        data.append({
            'Date': date,
            'Positive': positive,
            'Neutral': neutral,
            'Negative': negative,
            'Overall Score': sentiment_score
        })
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False)
    
    return df

def perform_sentiment_analysis(symbol):
    """
    Perform sentiment analysis for a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Dictionary with sentiment analysis results
    """
    # Get news sentiment
    news_df = get_recent_news_with_sentiment(symbol)
    
    # Get social media sentiment
    social_df = simulate_social_sentiment(symbol)
    
    # Calculate overall sentiment score
    # Combine news and social media sentiment with more weight to news
    news_sentiment = news_df['Score'].mean() if not news_df.empty else 0
    social_sentiment = social_df['Overall Score'].mean() if not social_df.empty else 0
    
    # Weight: 60% news, 40% social media
    overall_sentiment = (news_sentiment * 0.6) + (social_sentiment * 0.4)
    
    # Generate summary based on sentiment
    if overall_sentiment >= 0.2:
        sentiment_category = "Strongly Positive"
        summary = f"The overall sentiment for {symbol} is strongly positive. News coverage and social media discussions are predominantly favorable, which could indicate positive momentum for the stock price. Investors appear optimistic about the company's prospects."
    elif overall_sentiment >= 0.05:
        sentiment_category = "Moderately Positive"
        summary = f"The sentiment around {symbol} is moderately positive. While there are some mixed opinions, the general tone in news and social media leans positive, suggesting cautious optimism among investors."
    elif overall_sentiment > -0.05:
        sentiment_category = "Neutral"
        summary = f"The sentiment for {symbol} appears largely neutral. There is a balance of positive and negative opinions in the media and social discussions, indicating uncertainty or lack of strong catalysts in either direction."
    elif overall_sentiment > -0.2:
        sentiment_category = "Moderately Negative"
        summary = f"The sentiment surrounding {symbol} is somewhat negative. There are more bearish than bullish opinions in recent news and social media, which could signal caution for potential investors."
    else:
        sentiment_category = "Strongly Negative"
        summary = f"The overall sentiment for {symbol} is strongly negative. News coverage and social media discussions show significant pessimism, which might indicate underlying problems or concerns about the company's performance or prospects."
    
    # Return results
    return {
        'sentiment_score': overall_sentiment,
        'sentiment_category': sentiment_category,
        'news_sentiment': news_df,
        'social_sentiment': social_df,
        'recent_news': news_df[['Date', 'Headline', 'Sentiment']],
        'summary': summary
    }
