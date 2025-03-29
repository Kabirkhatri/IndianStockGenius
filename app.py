import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta

# Import custom modules
from data_fetcher import fetch_stock_data, search_stock, get_available_stocks
from technical_analysis import perform_technical_analysis, get_technical_indicators
from fundamental_analysis import perform_fundamental_analysis
from sentiment_analysis import perform_sentiment_analysis
from chart_patterns import identify_chart_patterns
from price_prediction import predict_prices
from report_generator import generate_report
from utils import plot_stock_data, plot_technical_indicators, time_periods

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("Indian Stock Market Analysis Tool")
st.write("An AI-powered analysis tool for Indian stocks with technical, fundamental, and sentiment analysis")

# Sidebar for inputs
st.sidebar.header("Stock Selection")

# Stock search and selection
stock_search = st.sidebar.text_input("Search for a stock (company name or symbol)")
if stock_search:
    search_results = search_stock(stock_search)
    if not search_results.empty:
        selected_stock = st.sidebar.selectbox(
            "Select a stock",
            options=search_results['Symbol'].tolist(),
            format_func=lambda x: f"{x} - {search_results[search_results['Symbol'] == x]['Company Name'].values[0]}"
        )
    else:
        st.sidebar.warning("No stocks found matching your search")
        selected_stock = None
else:
    # Default stock list (top 20 NSE stocks by market cap)
    default_stocks = get_available_stocks()
    selected_stock = st.sidebar.selectbox(
        "Select a stock",
        options=default_stocks['Symbol'].tolist(),
        format_func=lambda x: f"{x} - {default_stocks[default_stocks['Symbol'] == x]['Company Name'].values[0]}"
    )

# Date range selection
st.sidebar.header("Analysis Period")
end_date = datetime.date.today()
start_date = end_date - relativedelta(years=1)

start_date = st.sidebar.date_input("Start date", value=start_date)
end_date = st.sidebar.date_input("End date", value=end_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")

# Analysis options
st.sidebar.header("Analysis Options")
perform_tech_analysis = st.sidebar.checkbox("Technical Analysis", value=True)
perform_fund_analysis = st.sidebar.checkbox("Fundamental Analysis", value=True)
perform_sent_analysis = st.sidebar.checkbox("Sentiment Analysis", value=True)
perform_pattern_analysis = st.sidebar.checkbox("Chart Pattern Analysis", value=True)

# Analysis button
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

# Main section
if selected_stock and analyze_button:
    # Show loading message
    with st.spinner(f"Analyzing {selected_stock}... This may take a moment."):
        try:
            # Fetch data
            stock_data = fetch_stock_data(selected_stock, start_date, end_date)
            
            if stock_data.empty:
                st.error(f"Could not fetch data for {selected_stock}. Please try another stock.")
            else:
                # Display basic stock information
                st.header(f"{selected_stock} Analysis")
                
                # Create tabs for different analysis views
                tabs = st.tabs(["Overview", "Technical Analysis", "Fundamental Analysis", 
                               "Sentiment Analysis", "Chart Patterns", "Price Predictions", "Report"])
                
                # Overview tab
                with tabs[0]:
                    st.subheader("Stock Price Overview")
                    plot_stock_data(stock_data, selected_stock)
                    
                    # Display basic stats
                    col1, col2, col3, col4 = st.columns(4)
                    current_price = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else None
                    day_change = None if prev_close is None else (current_price - prev_close)
                    day_change_pct = None if prev_close is None else (day_change / prev_close * 100)
                    
                    col1.metric("Current Price", f"₹{float(current_price):.2f}")
                    
                    if day_change is not None:
                        col2.metric("Day Change", 
                                   f"₹{float(day_change):.2f}", 
                                   f"{float(day_change_pct):.2f}%")
                    
                    col3.metric("52W High", f"₹{float(stock_data['High'].max()):.2f}")
                    col4.metric("52W Low", f"₹{float(stock_data['Low'].min()):.2f}")
                
                # Technical Analysis tab
                with tabs[1]:
                    if perform_tech_analysis:
                        st.subheader("Technical Analysis")
                        tech_analysis_results = perform_technical_analysis(stock_data)
                        
                        # Display technical indicators
                        st.write("Technical Indicators")
                        indicators_table = get_technical_indicators(stock_data)
                        st.dataframe(indicators_table)
                        
                        # Plot selected indicators
                        st.write("Indicator Visualization")
                        plot_technical_indicators(stock_data, selected_stock)
                        
                        # Technical Analysis Summary
                        st.write("Technical Analysis Summary")
                        st.write(tech_analysis_results['summary'])
                        
                        # Signals
                        st.subheader("Technical Signals")
                        signal_df = pd.DataFrame({
                            'Indicator': tech_analysis_results['signals'].keys(),
                            'Signal': tech_analysis_results['signals'].values()
                        })
                        st.dataframe(signal_df)
                    else:
                        st.info("Technical analysis not selected. Enable it in the sidebar options.")
                
                # Fundamental Analysis tab
                with tabs[2]:
                    if perform_fund_analysis:
                        st.subheader("Fundamental Analysis")
                        fund_analysis_results = perform_fundamental_analysis(selected_stock)
                        
                        # Display key metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Valuation Metrics")
                            valuation_df = pd.DataFrame({
                                'Metric': ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'Dividend Yield'],
                                'Value': [
                                    fund_analysis_results['pe_ratio'],
                                    fund_analysis_results['pb_ratio'],
                                    fund_analysis_results['ev_ebitda'],
                                    fund_analysis_results['dividend_yield']
                                ]
                            })
                            st.dataframe(valuation_df)
                        
                        with col2:
                            st.write("Financial Health")
                            health_df = pd.DataFrame({
                                'Metric': ['ROE', 'ROA', 'Debt to Equity', 'Quick Ratio'],
                                'Value': [
                                    fund_analysis_results['roe'],
                                    fund_analysis_results['roa'],
                                    fund_analysis_results['debt_to_equity'],
                                    fund_analysis_results['quick_ratio']
                                ]
                            })
                            st.dataframe(health_df)
                        
                        # Fundamental Analysis Summary
                        st.write("Fundamental Analysis Summary")
                        st.write(fund_analysis_results['summary'])
                    else:
                        st.info("Fundamental analysis not selected. Enable it in the sidebar options.")
                
                # Sentiment Analysis tab
                with tabs[3]:
                    if perform_sent_analysis:
                        st.subheader("Sentiment Analysis")
                        sentiment_results = perform_sentiment_analysis(selected_stock)
                        
                        # Overall sentiment
                        st.write("Overall Market Sentiment")
                        sentiment_score = sentiment_results['sentiment_score']
                        
                        # Create sentiment gauge
                        sentiment_color = "red" if sentiment_score < -0.3 else "green" if sentiment_score > 0.3 else "orange"
                        sentiment_label = "Bearish" if sentiment_score < -0.3 else "Bullish" if sentiment_score > 0.3 else "Neutral"
                        
                        st.progress(sentiment_score/2 + 0.5, text=f"{sentiment_label} ({sentiment_score:.2f})")
                        
                        # News and social media sentiment
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("News Sentiment")
                            st.dataframe(sentiment_results['news_sentiment'])
                        
                        with col2:
                            st.write("Social Media Sentiment")
                            st.dataframe(sentiment_results['social_sentiment'])
                        
                        # Recent news
                        st.write("Recent News Headlines")
                        st.dataframe(sentiment_results['recent_news'])
                    else:
                        st.info("Sentiment analysis not selected. Enable it in the sidebar options.")
                
                # Chart Patterns tab
                with tabs[4]:
                    if perform_pattern_analysis:
                        st.subheader("Chart Pattern Analysis")
                        pattern_results = identify_chart_patterns(stock_data)
                        
                        # Display identified patterns
                        st.write("Identified Chart Patterns")
                        if pattern_results['patterns']:
                            pattern_df = pd.DataFrame({
                                'Pattern': list(pattern_results['patterns'].keys()),
                                'Signal': list(pattern_results['patterns'].values())
                            })
                            st.dataframe(pattern_df)
                        else:
                            st.write("No significant chart patterns identified in the current timeframe.")
                        
                        # Pattern analysis summary
                        st.write("Pattern Analysis Summary")
                        st.write(pattern_results['summary'])
                        
                        # Display pattern visualization
                        if 'pattern_chart' in pattern_results:
                            st.plotly_chart(pattern_results['pattern_chart'])
                    else:
                        st.info("Chart pattern analysis not selected. Enable it in the sidebar options.")
                
                # Price Predictions tab
                with tabs[5]:
                    st.subheader("Price Predictions")
                    
                    prediction_results = predict_prices(stock_data, selected_stock)
                    
                    # Display price predictions for different time periods
                    st.write("Price Projections")
                    
                    projection_df = pd.DataFrame({
                        'Time Period': [time_periods[period] for period in prediction_results['projections'].keys()],
                        'Projected Price (₹)': [f"₹{float(price):.2f}" for price in prediction_results['projections'].values()],
                        'Change (%)': [f"{float(((price / stock_data['Close'].iloc[-1]) - 1) * 100):.2f}%" 
                                      for price in prediction_results['projections'].values()]
                    })
                    st.dataframe(projection_df)
                    
                    # Display prediction chart
                    st.plotly_chart(prediction_results['prediction_chart'])
                    
                    # Model metrics
                    st.write("Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(prediction_results['model_metrics'].keys()),
                        'Value': list(prediction_results['model_metrics'].values())
                    })
                    st.dataframe(metrics_df)
                    
                    # Prediction factors
                    st.write("Key Factors Influencing Predictions")
                    st.write(prediction_results['prediction_factors'])
                
                # Report tab
                with tabs[6]:
                    st.subheader("Comprehensive Analysis Report")
                    
                    # Generate comprehensive report
                    report = generate_report(
                        stock_data, 
                        selected_stock,
                        tech_analysis_results if perform_tech_analysis else None,
                        fund_analysis_results if perform_fund_analysis else None,
                        sentiment_results if perform_sent_analysis else None,
                        pattern_results if perform_pattern_analysis else None,
                        prediction_results
                    )
                    
                    # Display recommendation
                    recommendation = report['recommendation']
                    rec_color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "orange"
                    
                    st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>Recommendation: {recommendation}</h3>", 
                               unsafe_allow_html=True)
                    
                    # Display target price
                    st.markdown("<h4 style='text-align: center;'>Target Price (1 Year): "
                               f"₹{float(report['target_price']):.2f}</h4>", unsafe_allow_html=True)
                    
                    # Display comprehensive analysis
                    st.write("Analysis Summary")
                    st.write(report['summary'])
                    
                    # Display key points
                    st.write("Key Points")
                    for point in report['key_points']:
                        st.write(f"• {point}")
                    
                    # Display risks
                    st.write("Risk Factors")
                    for risk in report['risks']:
                        st.write(f"• {risk}")
                    
                    # Download report
                    st.download_button(
                        label="Download Full Report",
                        data=report['report_html'],
                        file_name=f"{selected_stock}_Analysis_Report.html",
                        mime="text/html"
                    )
                    
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)

else:
    # Display welcome message and instructions
    st.write("""
    ## Welcome to the Indian Stock Market Analysis Tool
    
    This AI-powered tool provides comprehensive analysis of Indian stocks using:
    
    - **Technical Analysis**: Indicators, oscillators, moving averages, pivots, etc.
    - **Fundamental Analysis**: Financial ratios, earnings, growth metrics
    - **Sentiment Analysis**: News, social media sentiment
    - **Chart Pattern Recognition**: Identifying common patterns
    - **Price Projections**: Forecasting for multiple timeframes (1w, 1m, 3m, 6m, 1y)
    
    ### How to use:
    1. Select a stock from the sidebar or search for one
    2. Choose your analysis options
    3. Click "Analyze Stock" to generate insights
    4. Review the analysis in the different tabs
    5. Download a comprehensive report with trading recommendations
    
    Get started by selecting a stock from the sidebar!
    """)
