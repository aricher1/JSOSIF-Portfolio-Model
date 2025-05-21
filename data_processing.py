# === data_processing.py === 
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
import uuid 
import warnings

from config import NOW, FIVE_YRS, ONE_YR, price_dfs
from data_utils import get_price_df, get_log_return_df
from data_utils import resample_df

warnings.simplefilter(action='ignore', category=FutureWarning)

price_dfs = {}
equity_betas = {}
class Holding:
    def __init__(self, ticker, sector='NONE', shares=0, qual_risk_tier=0):
        self.ticker = ticker
        self.sector = sector
        self.shares = shares
        self.qual_risk_tier = qual_risk_tier
class Transaction:
    def __init__(self, holding:Holding, side):
        self.holding = holding
        self.side = side.upper()

class Portfolio:
    def __init__(self, holdings:list[Holding], cash=0, benchmark="^GSPC"):
        #Initializing instance variables
        self.guid = str(uuid.uuid4())
        self.equities_guid = str(uuid.uuid4())
        self.holdings = holdings
        self.cash = cash
        self.benchmark = benchmark
        self.price_df = get_price_df([h.ticker for h in holdings])
        self.log_return_df = get_log_return_df([h.ticker for h in holdings])
        self.port_value_df = self.get_port_value_df()
        self.port_return_df = self.get_port_return_df()
        self.sector_value_dfs = self.get_sector_value_dfs()
        self.sector_return_df = self.get_sector_return_dfs()

        #Adds the portfolios summed value df to price_dfs
        sum_value_df = self.port_value_df.sum(axis=1).to_frame()
        price_dfs[self.guid] = sum_value_df.set_index(sum_value_df.index.tz_localize(None))
        equity_value_df = sum_value_df.copy()
        equity_value_df.iloc[:, 0] = sum_value_df.iloc[:, 0] - self.cash
        price_dfs[self.equities_guid] = equity_value_df.set_index(equity_value_df.index.tz_localize(None))
   #Dataframe methods
    def get_port_value_df(self, include_cash:bool=True):
        """Returns a dataframe of each stocks historical holding prices * shares owned * historical FX rate"""
        port_value_df = self.price_df.copy()

        #Multiplies each holdings price by its shares
        for h in self.holdings:
            port_value_df.loc[:,h.ticker] *= h.shares

        #Adds the portfolios cash balance as a constant column
        if include_cash:
            port_value_df["CASH"] = self.cash

        return port_value_df

    def get_port_return_df(self, include_cash:bool=True, freq = 'D', start=FIVE_YRS, end=NOW):
        """Returns a data frame of log returns for the portfolio's value"""
        return np.log(1+resample_df(self.get_port_value_df(include_cash),freq,start,end).sum(axis=1).pct_change().dropna())

    def get_sector_value_dfs(self):
        """Returns a dictionary of slices of the port_value_df for each sector"""
        sector_value_dfs = {}

        #All unique sectors in the portfolio's holdings
        sectors = list(set([h.sector for h in self.holdings]))

        #Adds every sector with a df of all of it's stocks value columns to the dict
        for sector in sectors:
            sector_value_dfs[sector] = self.port_value_df.loc[:,[h.ticker for h in self.holdings if h.sector == sector]]

        return sector_value_dfs
    def get_sector_return_dfs(self, freq = 'D', start=FIVE_YRS, end=NOW):
        """Returns a dictionary of each sectors log returns"""
        sector_return_dfs = {}

        #Adds every sector with a df of its historical return in the portfolio to a dict
        for sector,value_df in self.sector_value_dfs.items():
            sector_return_dfs[sector] = np.log(1+resample_df(value_df,freq,start,end).sum(axis=1).pct_change().dropna())

        return sector_return_dfs
    #Metrics
    def get_holding_weights(self, include_cash:bool=False):
        """Returns a dictionary of each holding's weight in the portfolio"""
        holding_weights = {}
        value_df = self.get_port_value_df(include_cash)

        for h in self.holdings:
            holding_weights[h.ticker] = value_df.iloc[-1,value_df.columns.get_loc(h.ticker)] / value_df.iloc[-1,:].sum()

        #Adds weight of cash
        if include_cash:
            holding_weights["CASH"] = 1-sum(holding_weights.values())

        return holding_weights
    def get_stdev(self, historical:bool=False, include_cash:bool=True, annualize:bool=True, start=ONE_YR, end=NOW, days=None):
        """Calculate the std dev of the portfolio"""
        #Returns historical std dev
        if historical:
            df = self.get_port_return_df(include_cash)
            std = df.loc[start:end].std() if days is None else df.tail(days).std()
            return std * (np.sqrt(252) if annualize else 1)

        return_df = self.log_return_df.loc[start:end].copy() if days is None else self.log_return_df.tail(days).copy()

        weights_dict = self.get_holding_weights(include_cash)

        #Adds cash as a synthetic holding to the df
        if include_cash:
            return_df["CASH"] = 0

        #Arranges weights in the same order as the df columns for the dot function
        weights = np.array([weights_dict[ticker] for ticker in return_df.columns])

        cov_matrix = return_df.cov()

        variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        return np.sqrt(variance * (252 if annualize else 1))
    def get_sharpe_ratio(self, include_cash:bool=True, freq='ME', start=ONE_YR, end=NOW, rf_override=None):
        """Gets the sharpe ratio for the portfolio"""
        return_df = self.get_port_return_df(include_cash, freq, start, end)
        rf = get_avg_rf(log=True,freq=freq, start=start, end=end) if rf_override is None else rf_override

        #Sets the annualizing factor relative to the data frequency
        freq = freq.upper()
        ann = 252 if freq=='D' else 52 if freq=='W' else 12 if freq=='ME' else 1

        return (return_df.mean() * ann - rf) / (return_df.std() * np.sqrt(ann))