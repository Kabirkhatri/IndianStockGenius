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

# Fixed one-year analysis period
st.sidebar.header("Analysis Period")
end_date = datetime.date.today()
start_date = end_date - relativedelta(years=1)

# Show the fixed one-year period without allowing changes
st.sidebar.info(f"Analysis period: One year from {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")

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
            
            # Initialize analysis result variables to None
            tech_analysis_results = None
            fund_analysis_results = None
            sentiment_results = None
            pattern_results = None
            prediction_results = None
            
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
                    current_price = float(stock_data['Close'].iloc[-1])
                    
                    # Calculate day change and percentage safely
                    if len(stock_data) > 1:
                        prev_close = float(stock_data['Close'].iloc[-2])
                        day_change = float(current_price - prev_close)
                        day_change_pct = float(day_change / prev_close * 100)
                    else:
                        prev_close = None
                        day_change = None
                        day_change_pct = None
                    
                    col1.metric("Current Price", f"₹{current_price:.2f}")
                    
                    if day_change is not None:
                        col2.metric("Day Change", 
                                   f"₹{day_change:.2f}", 
                                   f"{day_change_pct:.2f}%")
                    
                    high_52w = float(stock_data['High'].max())
                    low_52w = float(stock_data['Low'].min())
                    col3.metric("52W High", f"₹{high_52w:.2f}")
                    col4.metric("52W Low", f"₹{low_52w:.2f}")
                
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
                    
                    # Calculate and display model accuracy prominently
                    mape = prediction_results['model_metrics']['MAPE']
                    accuracy_percentage = max(0, min(100, 100 - mape))
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(
                            label="Model Accuracy", 
                            value=f"{accuracy_percentage:.1f}%",
                            delta="Based on backtesting"
                        )
                    
                    # Display price predictions for different time periods with buy/sell recommendations
                    st.write("Price Projections with Buy/Sell Signals")
                    
                    current_close = float(stock_data['Close'].iloc[-1])
                    projection_periods = list(prediction_results['projections'].keys())
                    projection_prices = [float(price) for price in prediction_results['projections'].values()]
                    
                    # Calculate percent changes
                    percent_changes = [((price / current_close) - 1) * 100 for price in projection_prices]
                    
                    # Determine buy/sell signals based on percent change
                    signals = []
                    signal_colors = []
                    
                    for change in percent_changes:
                        if change > 5:
                            signals.append("STRONG BUY")
                            signal_colors.append("green")
                        elif change > 2:
                            signals.append("BUY")
                            signal_colors.append("lightgreen")
                        elif change > -2:
                            signals.append("HOLD")
                            signal_colors.append("orange")
                        elif change > -5:
                            signals.append("SELL")
                            signal_colors.append("lightcoral")
                        else:
                            signals.append("STRONG SELL")
                            signal_colors.append("red")
                    
                    # Create DataFrame
                    projection_df = pd.DataFrame({
                        'Time Period': [time_periods[period] for period in projection_periods],
                        'Projected Price (₹)': [f"₹{price:.2f}" for price in projection_prices],
                        'Change (%)': [f"{change:.2f}%" for change in percent_changes],
                        'Signal': signals
                    })
                    
                    # Display as a styled dataframe
                    # Create a styling function
                    def color_signal(val):
                        if val == "STRONG BUY":
                            return 'background-color: darkgreen; color: white'
                        elif val == "BUY":
                            return 'background-color: green; color: white'
                        elif val == "HOLD":
                            return 'background-color: orange; color: white'
                        elif val == "SELL":
                            return 'background-color: red; color: white'
                        elif val == "STRONG SELL":
                            return 'background-color: darkred; color: white'
                        return ''
                    
                    # Apply styling and display
                    styled_df = projection_df.style.applymap(color_signal, subset=['Signal'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Chart removed as requested
                    # st.plotly_chart(prediction_results['prediction_chart'])
                    
                    # Model metrics with clearer display
                    st.subheader("Model Performance Metrics")
                    
                    # Format the metrics for better readability
                    metrics = prediction_results['model_metrics']
                    formatted_values = []
                    
                    for k, v in metrics.items():
                        if k == 'R²':
                            formatted_values.append(f"{v*100:.1f}%")
                        elif k == 'MAPE':
                            formatted_values.append(f"{v:.2f}%")
                        elif k == 'RMSE':
                            formatted_values.append(f"{v:.2f}")
                        else:
                            formatted_values.append(f"{v}")
                    
                    # Display metrics in a nicer table
                    metrics_df = pd.DataFrame({
                        'Metric': [
                            'Accuracy',
                            'R² Score (Statistical Fit)',
                            'Mean Absolute Percentage Error',
                            'Root Mean Squared Error'
                        ],
                        'Value': [
                            f"{accuracy_percentage:.1f}%",
                            formatted_values[0],
                            formatted_values[1],
                            formatted_values[2]
                        ],
                        'Interpretation': [
                            'Higher is better. Represents model prediction accuracy.',
                            'Higher is better. A measure of how well the model fits the data.',
                            'Lower is better. Average percentage error in predictions.',
                            'Lower is better. Error magnitude in price units.'
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
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
                    target_price = float(report['target_price'])
                    st.markdown("<h4 style='text-align: center;'>Target Price (1 Year): "
                               f"₹{target_price:.2f}</h4>", unsafe_allow_html=True)
                    
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
