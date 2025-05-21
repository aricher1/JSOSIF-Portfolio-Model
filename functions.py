# === functions.py

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import math
from scipy.stats import kurtosis

from config import FIVE_YRS, NOW, ONE_YR, price_dfs, equity_betas
from data_processing import Holding, Portfolio
from data_utils import get_price_df

def get_simple_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """Gets a simple return df for 1 or more tickers"""
    price_df = get_price_df(tickers,freq,start,end)
    return price_df.pct_change(fill_method=None).dropna()
def get_log_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """Gets a log return df for 1 or more tickers"""
    return np.log(1 + get_simple_return_df(tickers,freq,start,end))
def get_avg_rf(log:bool=False, rf_ticker="^TNX", freq='ME', start=FIVE_YRS, end=NOW):
    """Gets the avg risk free rate"""
    rf = get_price_df(rf_ticker, freq, start, end).mean().item() / 100
    return np.log(1+rf) if log else rf
def get_df_beta(df1, df2):
    """Gets the beta of df1 relative to df2"""
    combined_df = pd.concat([df1,df2],axis=1).dropna()

    cov_matrix = np.cov(combined_df.iloc[:,0], combined_df.iloc[:,1])
    return cov_matrix[1, 0] / cov_matrix[1, 1]
def get_raw_beta(ticker, benchmark="^GSPC", freq='ME', start=FIVE_YRS, end=NOW):
    """Gets the raw beta for a given ticker"""
    #Checks if raw beta has already been calculated for ticker
    if (ticker,'RAW',benchmark,freq,start,end) in equity_betas:
        return equity_betas[(ticker,'RAW',benchmark,freq,start,end)]

    return_df = get_log_return_df(ticker,freq,start,end)
    benchmark_return_df = get_log_return_df(benchmark,freq,start,end)

    beta = get_df_beta(return_df,benchmark_return_df)

    equity_betas[(ticker,'RAW',benchmark,freq,start,end)] = beta

    return beta
def get_adj_beta(ticker, benchmark="^GSPC", method = "BLUME", freq='ME', start=FIVE_YRS, end=NOW):
    """Gets the adjusted beta using the given method"""
    #Checks if adj beta has already been calculated for ticker
    if (ticker,method,benchmark,freq,start,end) in equity_betas:
        return equity_betas[(ticker,method,benchmark,freq,start,end)]

    if method == "BLUME":
        beta = get_raw_beta(ticker, benchmark, freq, start, end)*2/3+1/3

        equity_betas[(ticker,method,benchmark,freq,start,end)] = beta

        return beta
    return
def get_dual_betas(ticker, benchmark="^GSPC", freq='ME', start=FIVE_YRS, end=NOW):
    """Gets a dict of the beta+ & beta- for a given ticker"""
    #Checks if dual betas have already been calculated for ticker
    if (ticker,'PLUS',benchmark,freq,start,end) in equity_betas:
        return {
            'PLUS':equity_betas[(ticker,'PLUS',benchmark,freq,start,end)],
            'MINUS':equity_betas[(ticker,'MINUS',benchmark,freq,start,end)]
        }

    return_df = get_log_return_df(ticker,freq,start,end)
    benchmark_return_df = get_log_return_df(benchmark,freq,start,end)

    #Creates a mask for the upside & downside
    plus_mask = benchmark_return_df.iloc[:,0] > 0
    minus_mask = benchmark_return_df.iloc[:,0] < 0

    #Applies the mask to the equity & market dfs
    plus_return_df = return_df[plus_mask]
    plus_benchmark_return_df = benchmark_return_df[plus_mask]
    minus_return_df = return_df[minus_mask]
    minus_benchmark_return_df = benchmark_return_df[minus_mask]

    dual_betas = {
        "PLUS":get_df_beta(plus_return_df,plus_benchmark_return_df),
        "MINUS":get_df_beta(minus_return_df,minus_benchmark_return_df)
    }

    equity_betas[(ticker,'PLUS',benchmark,freq,start,end)] = dual_betas['PLUS']
    equity_betas[(ticker,'MINUS',benchmark,freq,start,end)] = dual_betas['MINUS']

    return dual_betas
def get_betas(ticker, benchmark="^GSPC", freq='ME', start=FIVE_YRS, end=NOW):
    """Gets a dictionary of raw, adjusted, & + / - betas of a given ticker"""
    return {
        "RAW":get_raw_beta(ticker,benchmark,freq,start,end),
        "ADJ":get_adj_beta(ticker,benchmark,"BLUME",freq,start,end)
    } | get_dual_betas(ticker,benchmark,freq,start,end)
def get_volatility(ticker, days:int = 360):
    """Gets the annualized volatility from a given number of trading days"""
    return_df = get_log_return_df(ticker)

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    return rolling_volatility.iloc[-1,0]
def get_kurtosis(ticker):
    """Gets the kurtosis of a given ticker"""
    df = get_log_return_df(ticker)
    return kurtosis(df, fisher=True)  # Fisher=True gives excess kurtosis
def get_skewness(ticker):
    """Gets the skewness of a given ticker"""
    df = get_log_return_df(ticker)
    return kurtosis(df, fisher=True)  # Fisher=True gives excess kurtosis
def get_shannon_entropy(ticker, bins=50):
    """Calculates the Shannon entropy of the log returns of a given ticker"""
    df = get_log_return_df(ticker)
    hist, bin_edges = np.histogram(df['Log Returns'], bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities to avoid log(0)
    entropy = -np.sum(hist * np.log(hist))
    return entropy
def get_df_correlation(df1, df2):
    """Gets the correlation between 2 given dataframes (only for the first col of each)"""
    df1 = pd.DataFrame(df1) if isinstance(df1, pd.Series) else df1
    df2 = pd.DataFrame(df2) if isinstance(df2, pd.Series) else df2
    return df1.iloc[:, 0].corr(df2.iloc[:, 0])
def get_ticker_correlation(ticker1, ticker2, freq='W', start=FIVE_YRS, end=NOW):
    """Returns the correlation between 2 tickers"""
    return_df = get_log_return_df([ticker1,ticker2],freq,start,end)
    return get_df_correlation(return_df[ticker1],return_df[ticker2])
def get_correlations(holding:Holding, portfolio:Portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """Returns a dictionary of various correlation numbers for the given holding and portfolio"""
    correlations = {}
    return_df = get_log_return_df(holding.ticker,freq,start,end)

    #Gets the correlation between the holding and the portfolio
    correlations["PORT"] = get_df_correlation(return_df, portfolio.get_port_return_df(False,freq,start,end))

    #Gets the correlations between the holding and each sector in the portfolio
    correlations["SECTORS"] = {}
    for sector,df in portfolio.get_sector_return_dfs(freq,start,end).items():
        correlations["SECTORS"][sector] = get_df_correlation(return_df,df)

    #Gets the correlations between the holding and all individual holdings in the portfolio
    correlations["HOLDINGS"] = {}
    for ticker,df in get_log_return_df([h.ticker for h in portfolio.holdings],freq,start,end).items():
        correlations["HOLDINGS"][ticker] = get_df_correlation(return_df,df)
    correlations["HOLDINGS"] = dict(sorted(correlations["HOLDINGS"].items(), key=lambda item: item[1],reverse=True))

    return correlations
def get_risk_tier(holding:Holding):
    """Gets the risk tier for a given ticker"""
    dual_betas = get_dual_betas(holding.ticker)
    vol = get_volatility(holding.ticker)

    vol_tier = 1 if vol >= 0.35 else 3 if vol < 0.25 else 2
    dual_betas_tier = 3 if dual_betas["MINUS"] <= 1.1 and dual_betas["PLUS"] >= 0.7 else 2 if dual_betas["MINUS"] <= 0.7 and dual_betas["PLUS"] >= 0.7 else 1
    qual_tier = holding.qual_risk_tier if holding.qual_risk_tier != 0 else (vol_tier + dual_betas_tier) / 2

    return round((vol_tier + dual_betas_tier + qual_tier) / 3)
def get_buy_amounts(holding:Holding, portfolio:Portfolio) -> dict:
    """Gets the minimum and maximum buy amounts"""
    port_value = portfolio.port_value_df.iloc[-1,:].sum()
    stock_price = get_price_df(holding.ticker).iloc[-1,0]
    risk_tier = get_risk_tier(holding)

    if risk_tier == 1:
        return {"MIN":math.ceil(port_value*0.02/stock_price),"MED":round(port_value*0.025/stock_price),"MAX":math.floor(port_value*0.03/stock_price)}
    elif risk_tier == 2:
        return {"MIN":math.ceil(port_value*0.03/stock_price),"MED":round(port_value*0.035/stock_price),"MAX":math.floor(port_value*0.04/stock_price)}
    elif risk_tier == 3:
        return {"MIN":math.ceil(port_value*0.04/stock_price),"MED":round(port_value*0.05/stock_price),"MAX":math.floor(port_value*0.06/stock_price)}