"""
plotting.py

Imports essential libraries and utility functions for financial data plotting and analysis.
Includes data fetching (yfinance), data manipulation (pandas, numpy), and plotting tools 
(matplotlib, seaborn) with formatting utilities for clean visualization.

Dependencies:
- matplotlib
- seaborn
- yfinance
- pandas
- numpy

Also imports core utility functions and configuration constants for consistent data handling.

Author: JSOSIF Quant Team 
Verified: 2025-06-27
"""

import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from functions import get_log_return_df, get_price_df
from config import ONE_YR, TWO_YRS, FIVE_YRS, NOW

# Reviewed: 2025-06-27
def plot_df(df,title="Quick Plot", ypct = True, download_graph=False):
    """
    Quickly plot a DataFrame or Series with optional percent y-axis formatting.
    
    Parameters:
        df (pd.DataFrame or pd.Series): Data to plot.
        title (str): Plot title.
        ypct (bool): Format y-axis as percentage if True.
        download_graph (bool): Save plot as PNG if True.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    plt.figure(figsize=(10, 6))
    plt.plot(df, color='#0e5690')
    plt.title(title, fontsize=14, fontfamily='serif')
    plt.xlabel('Date', fontsize=12, fontfamily='serif')
    plt.ylabel('Values', fontsize=12, fontfamily='serif')
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    if ypct:
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if download_graph:
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Reviewed: 2025-06-27
def show_rolling_volatility_graph(ticker, days:int = 360, start=FIVE_YRS, end=NOW, download_graph=False):
    """
    Plot rolling annualized volatility for a single ticker.
    
    Parameters:
        ticker (str): Ticker symbol.
        days (int): Rolling window size in days.
        start (datetime): Start date.
        end (datetime): End date.
        download_graph (bool): Save plot if True.
    """
    return_df = pd.DataFrame(get_log_return_df(ticker))

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    rolling_volatility = rolling_volatility.loc[start:end]

    plot_df(rolling_volatility,f"{ticker} {days}-Day Rolling Volatility",download_graph=download_graph)

# Reviewed: 2025-06-27
def show_rolling_volatility_comparison(tickers, highlights=[], days:int = 360, start=FIVE_YRS, end=NOW):
    """
    Plot rolling annualized volatility comparison for multiple tickers,
    highlighting specified tickers in red.
    
    Parameters:
        tickers (list): List of ticker symbols.
        highlights (list): Tickers to highlight.
        days (int): Rolling window size in days.
        start (datetime): Start date.
        end (datetime): End date.
    """
    return_df = pd.DataFrame(get_log_return_df(tickers))

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    rolling_volatility = rolling_volatility.loc[start:end]

    if isinstance(rolling_volatility, pd.Series):
        rolling_volatility = rolling_volatility.to_frame()

    colors = {col: '#0e5690' if col not in highlights else 'red' for col in rolling_volatility.columns}

    for col in rolling_volatility.columns:
        if col not in highlights:
            plt.plot(rolling_volatility.index, rolling_volatility[col], color=colors[col], label=col, zorder=1)

    for col in highlights:
        plt.plot(rolling_volatility.index, rolling_volatility[col], color='red', label=col, linewidth=2, zorder=2)

    plt.title(f"{days}-Day Rolling Volatility Comparison", fontsize=14, fontfamily='serif')
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Date', fontsize=12, fontfamily='serif')
    plt.ylabel('Values', fontsize=12, fontfamily='serif')

    plt.show()

# Reviewed: 2025-06-27
def plot_stock_returns(tickers, start=FIVE_YRS, end=NOW):
    """
    Plot cumulative stock returns normalized to 0% at the start date.
    
    Parameters:
        tickers (list or str): Ticker(s) to plot.
        start (datetime): Start date.
        end (datetime): End date.
    """
    df = get_price_df(tickers, start=start, end=end)
    df = df / df.iloc[0] - 1

    plt.figure(figsize=(12, 4))

    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], label=column)

        total_return = (df[column].iloc[-1] - df[column].iloc[0]) * 100

        plt.text(df.index[-1], df[column].iloc[-1], f'{total_return:.2f}%',
                ha='left', va='center', fontsize=10)

    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    plt.title('Stock Performance', fontsize=14)
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.show()

# Reviewed: 2025-06-27
def plot_log_returns_histogram(tickers, start=TWO_YRS, end=NOW):
    """
    Plot histograms of log returns for given tickers over a specified date range.
    
    Parameters:
        tickers (list or str): Tickers to include.
        start (datetime): Start date for price data.
        end (datetime): End date for price data.
    """
    df = get_price_df(tickers=tickers, start=start, end=end)
    df = df.dropna()
    log_returns = np.log(df / df.shift(1)).dropna()

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    plt.figure(figsize=(12, 4))
    for i, ticker in enumerate(log_returns.columns):
        data = log_returns[ticker]
        mu = data.mean()
        sigma = data.std()

        plt.subplot(1, len(tickers), i + 1)
        plt.hist(data, bins=50, alpha=0.75, edgecolor='black')
        plt.title(f'{ticker} Log Returns')
        plt.xlabel('Log Return')
        plt.ylabel('Frequency')

        stats_label = f'μ = {mu:.4f}\nσ = {sigma:.4f}'
        plt.text(0.95, 0.95, stats_label,
                 ha='right', va='top',
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Reviewed: 2025-06-27
def plot_stock_chart(tickers, start=FIVE_YRS, end=NOW):
    """
    Plot price chart for given ticker(s) over a specified date range.
    
    Parameters:
        tickers (list or str): Tickers to plot.
        start (datetime): Start date for price data.
        end (datetime): End date for price data.
    """
    df = get_price_df(tickers=tickers, start=start, end=end)

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df.values, linewidth=2, color='#007ACC')
    plt.title(f"{tickers} — 5-Year Price Chart", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()