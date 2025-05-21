import numpy as np
import pandas as pd
import yfinance as yf

from config import NOW, FIVE_YRS, price_dfs

def get_price_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """Gets the price df for 1 or more tickers"""
    price_df = pd.DataFrame()
    tickers = [tickers] if not isinstance(tickers, list) else tickers

    for ticker in tickers:
        if ticker in price_dfs:
            price_df[ticker] = price_dfs[ticker].copy()

    new_tickers = [t for t in tickers if t not in price_dfs]
    if new_tickers:
        new_df = yf.download(new_tickers, start=start, end=end, auto_adjust=False).ffill().bfill()
        if 'Adj Close' in new_df.columns:
            new_df = new_df['Adj Close']
        else:
            new_df = new_df['Close']
        price_df = pd.concat([price_df, new_df], axis=1)

    for ticker in new_tickers:
        price_dfs[ticker] = price_df[[ticker]].copy()

    if fx_adj:
        fx_df = price_dfs.get("USDCAD=X", None)
        if fx_df is None:
            fx_df = yf.download("USDCAD=X", start=start, end=end)["Close"]
            price_dfs["USDCAD=X"] = fx_df
        fx_df = fx_df.reindex(price_df.index, method='ffill')
        for ticker in price_df.columns:
            if ticker.endswith(".TO") or ticker == "USDCAD=X":
                continue
            price_df[ticker] *= fx_df.iloc[:, 0]

    price_df.index = pd.to_datetime(price_df.index).tz_localize("UTC")
    return price_df.ffill().bfill()
def get_simple_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW):
    """Gets a simple return df for 1 or more tickers"""
    price_df = get_price_df(tickers, freq, start, end)
    return price_df.pct_change().dropna()
def get_log_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW):
    """Gets a log return df for 1 or more tickers"""
    return np.log(1 + get_simple_return_df(tickers, freq, start, end))
def resample_df(df, freq, start=FIVE_YRS, end=NOW):
    """Returns a resampled data frame within the date range passed"""
    df = df.loc[start:end]
    return df if freq == 'D' else df.resample(freq).last().dropna()
