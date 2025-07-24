"""
functions.py

Utility functions for financial data retrieval, return calculations,
risk metrics (beta, volatility, skewness), correlations, and portfolio analytics.
Includes plotting tools and caching mechanisms for efficiency.

Designed for quantitative equity analysis and portfolio management workflows.

Dependencies:
- numpy, pandas, yfinance, matplotlib, seaborn, scipy, dateutil

Globals:
- price_dfs: cached price data
- equity_betas: cached beta calculations

Constants imported from config:
- FIVE_YRS, NOW, ONE_YR

Author: JSOSIF Quant Team 
Verified: 2025-06-27
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import math
import uuid
import warnings
from dateutil.relativedelta import relativedelta
from scipy.stats import skew, kurtosis

from config import FIVE_YRS, NOW, ONE_YR, price_dfs, equity_betas

warnings.simplefilter(action='ignore', category=FutureWarning)

price_dfs = {}
equity_betas = {}

# Reviewed: 2025-06-27
class Holding:
    """
    Represents a single stock holding in a portfolio.

    Attributes:
        ticker (str): Stock ticker symbol, e.g., 'AAPL'.
        sector (str): Sector name of the holding, e.g., 'TMT', defaults to 'NONE'.
        shares (int): Number of shares held, defaults to 0.
        qual_risk_tier (int): Qualitative risk tier (custom scale), defaults to 0.
    """
    def __init__(self, ticker, sector='NONE', shares=0, qual_risk_tier=0):
        self.ticker = ticker
        self.sector = sector
        self.shares = shares
        self.qual_risk_tier = qual_risk_tier

# Reviewed: 2025-06-27
class Transaction:
    """
    Represents a transaction involving a holding (buy or sell).

    Attributes:
        holding (Holding): The holding involved in the transaction.
        side (str): 'BUY' or 'SELL', case-insensitive input converted to uppercase.
    """
    def __init__(self, holding:Holding, side):
        self.holding = holding
        self.side = side.upper()

# Reviewed: 2025-06-27
class Portfolio:
    """
    Represents an investment portfolio consisting of multiple holdings and cash.

    Attributes:
        guid (str): Unique identifier for the portfolio instance.
        equities_guid (str): Unique identifier for the equities subset (excluding cash).
        holdings (list[Holding]): List of Holding objects in the portfolio.
        cash (float): Cash position in the portfolio.
        benchmark (str): Benchmark ticker symbol for portfolio comparison (default: '^GSPC').
        price_df (pd.DataFrame): Price data frame for all holdings.
        log_return_df (pd.DataFrame): Log returns data frame for all holdings.
        port_value_df (pd.DataFrame): Portfolio value over time (including cash).
        port_return_df (pd.DataFrame): Portfolio return time series.
        sector_value_dfs (dict): Mapping sector names to their value data frames.
        sector_return_df (pd.DataFrame): Sector-level return time series.
    """
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
    
    # Reviewed: 2025-06-27
    def get_port_value_df(self, include_cash:bool=True):
        """
        Calculate historical portfolio value over time.

        Multiplies each holding's historical prices by shares owned and (if applicable) adds cash as a constant column.

        Args:
            include_cash (bool): Whether to include cash as a constant value column. Default True.

        Returns:
            pd.DataFrame: DataFrame with dates as index and columns for each holding's value plus optional cash.
        """
        port_value_df = self.price_df.copy()

        #Multiplies each holdings price by its shares
        for h in self.holdings:
            port_value_df.loc[:,h.ticker] *= h.shares

        #Adds the portfolios cash balance as a constant column
        if include_cash:
            port_value_df["CASH"] = self.cash

        return port_value_df

    # Reviewed: 2025-06-27
    def get_port_return_df(self, include_cash: bool = True, freq='D', start=FIVE_YRS, end=NOW):
        """
        Calculate portfolio log returns over time at specified frequency.

        Args:
            include_cash (bool): Whether to include cash in portfolio value. Default True.
            freq (str): Resample frequency ('D', 'W', etc.). Default daily.
            start (datetime): Start date for returns calculation.
            end (datetime): End date for returns calculation.

        Returns:
            pd.Series: Log returns of total portfolio value.
        """
        port_val = self.get_port_value_df(include_cash)
        port_val = resample_df(port_val.sum(axis=1), freq, start, end)
        returns = np.log(port_val / port_val.shift(1))
        return returns.dropna()

    # Reviewed: 2025-06-27
    def get_sector_value_dfs(self):
        """
        Get historical value DataFrames grouped by sector.

        Returns:
            dict: Keys are sector names, values are DataFrames of portfolio values for holdings in that sector.
        """
        sector_value_dfs = {}

        sectors = list(set([h.sector for h in self.holdings]))

        for sector in sectors:
            sector_value_dfs[sector] = self.port_value_df.loc[:,[h.ticker for h in self.holdings if h.sector == sector]]

        return sector_value_dfs
    
    # Reviewed: 2025-06-27
    def get_sector_return_dfs(self, freq = 'D', start=FIVE_YRS, end=NOW):
        """
        Calculate log returns for each sector in the portfolio.

        Args:
            freq (str): Resample frequency (e.g., 'D', 'W').
            start (datetime): Start date for returns calculation.
            end (datetime): End date for returns calculation.

        Returns:
            dict: Keys are sectors, values are Series of log returns for each sector.
        """
        sector_return_dfs = {}

        for sector,value_df in self.sector_value_dfs.items():
            sector_return_dfs[sector] = np.log(1+resample_df(value_df,freq,start,end).sum(axis=1).pct_change().dropna())

        return sector_return_dfs

    # Reviewed: 2025-06-27
    def get_holding_weights(self, include_cash:bool=False):
        """
        Calculate current portfolio weights for each holding and optionally cash.

        Args:
            include_cash (bool): Whether to include cash weight. Default False.

        Returns:
            dict: Mapping ticker symbols (and 'CASH' if included) to weights summing to 1.
        """
        holding_weights = {}
        value_df = self.get_port_value_df(include_cash)

        for h in self.holdings:
            holding_weights[h.ticker] = value_df.iloc[-1,value_df.columns.get_loc(h.ticker)] / value_df.iloc[-1,:].sum()

        if include_cash:
            holding_weights["CASH"] = 1-sum(holding_weights.values())

        return holding_weights
    
    # Reviewed: 2025-06-27
    def get_stdev(self, historical:bool=False, include_cash:bool=True, annualize:bool=True, start=ONE_YR, end=NOW, days=None):
        """
        Calculate the standard deviation (volatility) of the portfolio returns.

        Args:
            historical (bool): Whether to calculate using historical portfolio returns or covariance matrix.
            include_cash (bool): Include cash in portfolio weighting.
            annualize (bool): Annualize the standard deviation.
            start (datetime): Start date for historical returns.
            end (datetime): End date for historical returns.
            days (int, optional): Number of most recent days to consider (overrides start/end if given).

        Returns:
            float: Portfolio standard deviation (annualized if requested).
        """
        if historical:
            df = self.get_port_return_df(include_cash)
            std = df.loc[start:end].std() if days is None else df.tail(days).std()
            return std * (np.sqrt(252) if annualize else 1)

        return_df = self.log_return_df.loc[start:end].copy() if days is None else self.log_return_df.tail(days).copy()

        weights_dict = self.get_holding_weights(include_cash)

        if include_cash:
            return_df["CASH"] = 0

        weights = np.array([weights_dict[ticker] for ticker in return_df.columns])

        cov_matrix = return_df.cov()

        variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        return np.sqrt(variance * (252 if annualize else 1))
    
    # Reviewed: 2025-06-27
    def get_sharpe_ratio(self, include_cash:bool=True, freq='M', start=ONE_YR, end=NOW, rf_override=None):
        """
        Calculate the portfolio's Sharpe ratio over a given time period.

        Args:
            include_cash (bool): Whether to include cash in portfolio returns.
            freq (str): Frequency of returns data ('D' = daily, 'W' = weekly, 'M' = monthly).
            start (datetime): Start date of returns period.
            end (datetime): End date of returns period.
            rf_override (float, optional): Override for risk-free rate (log scale).

        Returns:
            float: Sharpe ratio (annualized).
        """
        return_df = self.get_port_return_df(include_cash, freq, start, end)
        rf = get_avg_rf(log=True,freq=freq, start=start, end=end) if rf_override is None else rf_override

        freq = freq.upper() #Sets the annualizing factor relative to the data frequency
        ann = 252 if freq=='D' else 52 if freq=='W' else 12 if freq=='M' else 1

        return (return_df.mean() * ann - rf) / (return_df.std() * np.sqrt(ann))
    
    # Reviewed: 2025-06-27
    def get_betas(self, historical:bool=False, include_cash:bool=True, freq='M', start=FIVE_YRS, end=NOW):
        """
        Calculate portfolio beta values relative to benchmark.

        Args:
            historical (bool): Whether to calculate historical beta from portfolio ID.
            include_cash (bool): Include cash in portfolio weighting.
            freq (str): Frequency of return data.
            start (datetime): Start date for beta calculation.
            end (datetime): End date for beta calculation.

        Returns:
            dict: Dictionary containing beta metrics:
            - "RAW": weighted raw beta
            - "ADJ": adjusted beta (2/3 raw + 1/3 1)
            - "PLUS": weighted positive beta component
            - "MINUS": weighted negative beta component
        """
        if not historical:
            weights = self.get_holding_weights(include_cash)

            beta = sum([get_raw_beta(h.ticker,self.benchmark,freq,start,end)*weights[h.ticker] for h in self.holdings])
            dual_beta = {'PLUS':0,'MINUS':0}
            for h in self.holdings:
                dual_betas = get_dual_betas(h.ticker,self.benchmark,freq,start,end)
                dual_beta['PLUS'] += dual_betas['PLUS'] * weights[h.ticker]
                dual_beta['MINUS'] += dual_betas['MINUS'] * weights[h.ticker]

            return {
                "RAW":beta,
                "ADJ":beta*2/3+1/3
            } | dual_beta

        return get_betas(self.guid if include_cash else self.equities_guid, self.benchmark, freq, start, end)
    
    # Reviewed: 2025-06-27
    def get_avg_pairwise_corr(self, freq='W', start=FIVE_YRS, end=NOW):
       """
        Calculate average pairwise correlation among portfolio holdings.

        Args:
            freq (str): Frequency of return data for correlation calculation.
            start (datetime): Start date for correlation calculation.
            end (datetime): End date for correlation calculation.

        Returns:
            float: Mean of upper-triangle pairwise correlations.
        """
       return_df = get_log_return_df([h.ticker for h in self.holdings], freq, start, end)
       corr_matrix = return_df.corr()
       upper_tri_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()

       return upper_tri_values.mean()
    
    # Reviewed: 2025-06-27
    def get_r_squared(self, include_cash:bool=True, freq='W', start=FIVE_YRS, end=NOW):
        """
        Calculate the R-squared value of the portfolio returns relative to its benchmark.

        Args:
            include_cash (bool): Whether to include cash in portfolio returns.
            freq (str): Frequency of return data ('D', 'W', etc.).
            start (datetime): Start date for return calculation.
            end (datetime): End date for return calculation.

        Returns:
            float: R-squared value (coefficient of determination) between portfolio and benchmark returns.
        """
        price_df = resample_df(self.get_port_value_df(include_cash).sum(axis=1),freq,start,end)
        return_df = np.log(1+price_df.pct_change(fill_method=None).dropna())
        market_df = get_log_return_df(self.benchmark,freq,start,end,True).squeeze()

        return return_df.corr(market_df)**2
    
    # Reviewed: 2025-06-27
    def show_corr_matrix(self, freq='W', start=FIVE_YRS, end=NOW, download_graph=False):
        """
        Display a heatmap of the portfolio holdings' correlation matrix.

        Args:
            freq (str): Frequency of return data.
            start (datetime): Start date for return data.
            end (datetime): End date for return data.
            download_graph (bool): If True, save the heatmap as 'corr_matrix.png'.

        Returns:
            None: Displays the heatmap plot.
        """
        return_df = get_log_return_df([h.ticker for h in self.holdings], freq, start, end)
        corr_matrix = (return_df*100).corr()
        plt.figure(figsize=(20, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='inferno', fmt=".2f")
        plt.title('Correlation Matrix')

        if download_graph:
            plt.savefig(f'corr_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Reviewed: 2025-06-27
    def get_compliance_report(self):
        """
        Generate a compliance report for each holding.

        Returns:
            pd.DataFrame: Report containing risk tier, minimum and maximum shares,
                          actual shares, actual percentage weight, and suggested action.
        """
        report = pd.DataFrame()
        weights = self.get_holding_weights(True)

        for h in self.holdings:
            tier = get_risk_tier(h)
            amts = get_buy_amounts(h,self)
            min = amts["MIN"]
            max = amts["MAX"]
            action = min - h.shares if h.shares < min else max - h.shares if h.shares > max else 0

            report[h.ticker] = {
                "TIER":tier,
                "MIN":min,
                "MAX":max,
                "ACTUAL %":round(weights[h.ticker],4),
                "ACTUAL":h.shares,
                "ACTION":action}

        return report
    
    # Reviewed: 2025-06-27
    def get_contribution_report(self, historical:bool=False, include_cash:bool=False):
        """
        Generate a report showing each holding's contribution to portfolio metrics.

        Args:
            historical (bool): Use historical data if True.
            include_cash (bool): Include cash in calculations if True.

        Returns:
            pd.DataFrame: Report including beta, volatility, correlation contributions, and their changes when excluding each holding.
        """
        report = pd.DataFrame()

        betas = self.get_betas(historical, include_cash)
        vol = self.get_stdev(historical, include_cash, days=360)
        corr = self.get_avg_pairwise_corr()

        for h in self.holdings: #Creates a synthetic portfolio excluding each holding to meausure it's contribution
            synth_port = Portfolio([x for x in self.holdings if x != h],self.cash,self.benchmark)
            synth_betas = synth_port.get_betas(historical, include_cash)
            synth_vol = synth_port.get_stdev(historical, include_cash, days=360)
            synth_corr = synth_port.get_avg_pairwise_corr()

            report[h.ticker] = {
                "Beta":round(synth_betas["RAW"],4),
                "Beta Contribution":round(betas["RAW"] - synth_betas["RAW"],4),
                "Beta+":round(synth_betas["PLUS"],4),
                "Beta+ Contribution":round(betas["PLUS"] - synth_betas["PLUS"],4),
                "Beta-":round(synth_betas["MINUS"],4),
                "Beta- Contribution":round(betas["MINUS"] - synth_betas["MINUS"],4),
                "360D Vol":f"{synth_vol:.3%}",
                "360D Vol Contribution":f"{vol - synth_vol:.3%}",
                "Avg Pairwise Corr":f"{synth_corr:.3%}",
                "Avg Pairwise Corr Contribution":f"{corr - synth_corr:.3%}"
            }

            synth_port.destroy()

        return report
    
    # Reviewed: 2025-06-27
    def get_metrics_report(self, historical:bool=False, download_graphs:bool=False, show_graphs:bool=False):
        """
        Generate combined metrics report for portfolio and sectors.

        Args:
            historical (bool): Use historical data if True.
            download_graphs (bool): Save graphs as files if True.
            show_graphs (bool): Display graphs if True.

        Returns:
            pd.DataFrame: Metrics including beta, volatility, average pairwise correlation, Sharpe ratio, and R-squared.
        """
        report = pd.DataFrame()

        sectors = list(set([h.sector for h in self.holdings]))

        betas = self.get_betas(historical)
        report["Portfolio"] = {
            "Beta":round(betas["RAW"],3),
            "Beta+":round(betas["PLUS"],3),
            "Beta-":round(betas["MINUS"],3),
            "360D Vol":f"{self.get_stdev(historical,days=360):.2%}",
            "Avg Pairwise Corr":f"{self.get_avg_pairwise_corr():.2%}",
            "Sharpe 1y":round(self.get_sharpe_ratio(),3),
            "R Squared":f"{self.get_r_squared():.2%}"
        }
        if download_graphs or show_graphs:
            plot_beta_scatter(self.guid,self.benchmark,'M',FIVE_YRS,NOW,'Portfolio',download_graphs)

        betas = self.get_betas(historical,False)
        report["Equities"] = {
            "Beta":round(betas["RAW"],3),
            "Beta+":round(betas["PLUS"],3),
            "Beta-":round(betas["MINUS"],3),
            "360D Vol":f"{self.get_stdev(historical,False,days=360):.2%}",
            "Avg Pairwise Corr":f"{self.get_avg_pairwise_corr():.2%}",
            "Sharpe 1y":round(self.get_sharpe_ratio(False),3),
            "R Squared":f"{self.get_r_squared(False):.2%}"
        }
        if download_graphs or show_graphs:
            plot_beta_scatter(self.equities_guid,self.benchmark,'M',FIVE_YRS,NOW,'Equities',download_graphs)

        for s in sectors:
            sector_port = Portfolio([h for h in self.holdings if h.sector == s],0,self.benchmark)

            betas = sector_port.get_betas(historical,False)
            report[s] = {
                "Beta":round(betas["RAW"],3),
                "Beta+":round(betas["PLUS"],3),
                "Beta-":round(betas["MINUS"],3),
                "360D Vol":f"{sector_port.get_stdev(days=360,include_cash=False):.2%}",
                "Avg Pairwise Corr":f"{sector_port.get_avg_pairwise_corr():.2%}",
                "Sharpe 1y":round(sector_port.get_sharpe_ratio(False),3),
                "R Squared":f"{sector_port.get_r_squared():.2%}"
            }
            if download_graphs or show_graphs:
                plot_beta_scatter(sector_port.guid,sector_port.benchmark,'M',FIVE_YRS,NOW,s,download_graphs)

            sector_port.destroy()

        return report
    
    # Reviewed: 2025-06-27
    def get_holding_returns(self, start=NOW-relativedelta(days=7), end=NOW):
        """
        Get returns for each holding over a specified time frame.

        Args:
            start (datetime): Start date for return calculation.
            end (datetime): End date for return calculation.

        Returns:
            pd.DataFrame: Returns data including percent and dollar changes, sorted descending by percent change.
        """
        df = self.port_value_df.copy()
        df = df.loc[start:end]
        df.loc["%Chg"] = df.iloc[-1] / df.iloc[1] - 1
        df.loc["$Chg"] = df.iloc[-2] - df.iloc[1]
        df = df.T.sort_values(by=['%Chg'], ascending=False).T

        df.loc["%Chg"] = df.loc["%Chg"].apply(lambda x: f"{x:.2%}")
        df.loc["$Chg"] = df.loc["$Chg"].apply(lambda x: f"${x:,.2f}")

        return df.loc[df.index[-2:]].T
    
    # Reviewed: 2025-06-27
    def get_portfolio_returns(self, start=NOW-relativedelta(days=7), end=NOW):
        """
        Get portfolio returns grouped by sector over a specified time frame.

        Args:
            start (datetime): Start date for return calculation.
            end (datetime): End date for return calculation.

        Returns:
            pd.DataFrame: Sector-level and equity portfolio returns including percent and dollar changes, sorted descending by percent change.
        """
        df = self.port_value_df.copy()
        df = df.loc[start:end]
        ticker_to_sector = {h.ticker: h.sector for h in self.holdings}
        sector_mapping = pd.Series(ticker_to_sector)
        df_grouped = (
            df.groupby(sector_mapping, axis=1)  # Group by the sector mapping
              .sum()                            # Sum up values row-wise
        )
        df_grouped["Equities"] = df_grouped.sum(axis=1)

        df_grouped.loc["%Chg"] = df_grouped.iloc[-1] / df_grouped.iloc[1] - 1
        df_grouped.loc["$Chg"] = df_grouped.iloc[-2] - df_grouped.iloc[1]
        df_grouped = df_grouped.T.sort_values(by=['%Chg'], ascending=False).T

        df_grouped.loc["%Chg"] = df_grouped.loc["%Chg"].apply(lambda x: f"{x:.2%}")
        df_grouped.loc["$Chg"] = df_grouped.loc["$Chg"].apply(lambda x: f"${x:,.2f}")

        return df_grouped.loc[df_grouped.index[-2:]].T
    
    # Reviewed: 2025-06-27
    def get_dividends(self, start=NOW-relativedelta(days=7), end=NOW):
        """
        Returns a DataFrame of dividends paid for each holding over the specified timeframe.

        Args:
            start (datetime): Start date of the period.
            end (datetime): End date of the period.

        Returns:
            pd.DataFrame: Indexed by ticker with columns for dividend amount, shares, FX rate,
                          total value, and percentage of holding value.
        """
        dividends = pd.DataFrame(columns=['dividend' , 'shares', 'FX', 'total','%'])

        for h in self.holdings:
            try:
                stock = yf.Ticker(h.ticker)
                divs = stock.dividends.loc[start:end]
                if not divs.empty:
                    fx = 1 if h.ticker.endswith('.TO') else get_price_df('USDCAD=X').iloc[-1,0]
                    div = divs.sum()
                    sh = h.shares
                    val = fx*div*sh
                    pct = val / self.port_value_df[h.ticker].iloc[-1]

                    dividends.loc[h.ticker] = [f"${div:,.2f}",sh,fx,f"${val:,.2f}",f"{pct:.2%}"]
            except Exception as e:
                print(f"Error processing {h.ticker}: {e}")

        return dividends
    
    # Reviewed: 2025-06-27
    def get_weekly_update(self, start = NOW-relativedelta(days=7), end=NOW):
        """
        Prints a weekly summary including holding returns, portfolio returns, and dividend records.

        Args:
            start (datetime): Start date of the update period.
            end (datetime): End date of the update period.
        """
        print("-- HOLDING RETURNS --")
        print(self.get_holding_returns(start,end))
        print("")
        print("-- PORTFOLIO RETURNS --")
        print(self.get_portfolio_returns(start,end))
        print("")
        print("-- DIVIDENDS RECORDS --")
        print(self.get_dividends(start,end))

    # Reviewed: 2025-06-27
    def simulate_transactions(self, transactions:list[Transaction],cash=0):
        """
        Simulates a list of buy/sell transactions, updates holdings and cash,
        and outputs new portfolio metrics and cash balance.

        Args:
            transactions (list[Transaction]): List of transaction objects to simulate.
            cash (float): Optional cash balance override.

        Returns:
            Portfolio: New portfolio object reflecting the simulated transactions.
        """
        new_holdings = self.holdings[:]
        new_cash = self.cash

        for tr in transactions:
            price = get_price_df(tr.holding.ticker).iloc[-1,0]

            if tr.side == 'BUY':
                shares = tr.holding.shares if tr.holding.shares > 0 else get_buy_amounts(tr.holding,self)['MED']
                value = price * shares
                if value <= self.cash:
                    tr.holding.shares = shares
                    new_holdings.append(tr.holding)
                    new_cash -= value
                    print(f'Bought {shares} shares of {tr.holding.ticker} @ ${price:,.2f} for a total of ${value:,.2f}.')
                else:
                    print(f'Error buying {tr.holding.ticker}: {shares} shares @ ${price:,.2f} (${value:,.2f}) is greater than cash balance (${new_cash:,.2f}).')
                    return
            else:
                #Prevents selling unheld holdings
                if tr.holding.ticker not in [h.ticker for h in new_holdings]:
                    print(f'Error selling {tr.holding.ticker}: Not in portfolio.')
                    return

                position = next((h for h in new_holdings if h.ticker == tr.holding.ticker), None)
                shares = tr.holding.shares if tr.holding.shares > 0 else position.shares
                value = price * shares

                if shares <= position.shares:
                    new_holdings.remove(position)
                    new_cash += value
                    print(f'Sold {shares} shares of {tr.holding.ticker} @ ${price:,.2f} for a total of ${value:,.2f}.')
                else:
                    print(f'Error selling {tr.holding.ticker}: {shares} shares is greater than what is held: {position.shares} shares')
                    return

        new_port = Portfolio(new_holdings, round(new_cash,2), self.benchmark)
        cash_change = new_port.cash - self.cash
        portfolios = ['Equities'] + list(set([t.holding.sector for t in transactions]))
        print(f'-- Old metrics --')
        print(self.get_metrics_report()[portfolios])
        print(f'-- New metrics --')
        print(new_port.get_metrics_report()[portfolios])
        print('-- Cash --')
        print(f'Change in cash: ${cash_change:,.2f}')
        print(f'Remaining cash: ${new_cash if cash == 0 else cash + cash_change:,.2f}')
        return new_port
    
    # Reviewed: 2025-06-27
    def destroy(self):
        """
        Cleans up by removing portfolio dataframes from global caches.
        """
        price_dfs.pop(self.guid)
        price_dfs.pop(self.equities_guid)

# Reviewed: 2025-06-27 
def get_simple_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """
    Returns a DataFrame of simple (percentage) returns for given tickers over the specified period.

    Args:
        tickers (list or str): Tickers to fetch returns for.
        freq (str): Frequency of data (e.g., 'D', 'W').
        start (datetime): Start date.
        end (datetime): End date.
        fx_adj (bool): Flag to adjust for FX if applicable.

    Returns:
        pd.DataFrame: Simple returns indexed by date.
    """
    price_df = get_price_df(tickers,freq,start,end)
    return price_df.pct_change(fill_method=None).dropna()

# Reviewed: 2025-06-27
def get_log_return_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """
    Returns a DataFrame of log returns for given tickers over the specified period.

    Args:
        tickers (list or str): Tickers to fetch returns for.
        freq (str): Frequency of data.
        start (datetime): Start date.
        end (datetime): End date.
        fx_adj (bool): Flag to adjust for FX if applicable.

    Returns:
        pd.DataFrame: Log returns indexed by date.
    """
    return np.log(1 + get_simple_return_df(tickers,freq,start,end))

# Reviewed: 2025-06-27
def get_avg_rf(log:bool=False, rf_ticker="^TNX", freq='M', start=FIVE_YRS, end=NOW):
    """
    Returns the average risk-free rate over a specified period.

    Args:
        log (bool): Whether to return the log of the risk-free rate.
        rf_ticker (str): Ticker symbol for the risk-free rate proxy.
        freq (str): Data frequency.
        start (datetime): Start date.
        end (datetime): End date.

    Returns:
        float: Average risk-free rate or its log.
    """
    rf = get_price_df(rf_ticker, freq, start, end).mean().item() / 100
    return np.log(1+rf) if log else rf

# Reviewed: 2025-06-27
def get_df_beta(df1, df2):
    """
    Calculates the beta of one returns series relative to another.

    Args:
        df1 (pd.Series or DataFrame): Returns series of asset.
        df2 (pd.Series or DataFrame): Returns series of benchmark.

    Returns:
        float: Beta value.
    """
    combined_df = pd.concat([df1,df2],axis=1).dropna()

    cov_matrix = np.cov(combined_df.iloc[:,0], combined_df.iloc[:,1])
    return cov_matrix[1, 0] / cov_matrix[1, 1]

# Reviewed: 2025-06-27
def get_raw_beta(ticker, benchmark="^GSPC", freq='M', start=FIVE_YRS, end=NOW):
    """
    Retrieves or calculates the raw beta of a ticker relative to a benchmark.

    Args:
        ticker (str): Asset ticker symbol.
        benchmark (str): Benchmark ticker symbol.
        freq (str): Frequency of returns data.
        start (datetime): Start date.
        end (datetime): End date.

    Returns:
        float: Beta value.
    """
    if (ticker,'RAW',benchmark,freq,start,end) in equity_betas: #checks if raw beta has already been calculated for ticker
        return equity_betas[(ticker,'RAW',benchmark,freq,start,end)]

    return_df = get_log_return_df(ticker,freq,start,end)
    benchmark_return_df = get_log_return_df(benchmark,freq,start,end)

    beta = get_df_beta(return_df,benchmark_return_df)

    equity_betas[(ticker,'RAW',benchmark,freq,start,end)] = beta

    return beta

# Reviewed: 2025-06-27
def get_adj_beta(ticker, benchmark="^GSPC", method = "BLUME", freq='M', start=FIVE_YRS, end=NOW):
    """
    Calculates the adjusted beta of a ticker relative to a benchmark using a specified method.

    Args:
        ticker (str): Ticker symbol of the asset.
        benchmark (str): Benchmark ticker symbol.
        method (str): Method for adjustment (e.g., "BLUME").
        freq (str): Frequency of returns data.
        start (datetime): Start date for calculation.
        end (datetime): End date for calculation.

    Returns:
        float or None: Adjusted beta value, or None if method is unsupported.
    """
    if (ticker,method,benchmark,freq,start,end) in equity_betas: #checks if adj beta has already been calculated for ticker
        return equity_betas[(ticker,method,benchmark,freq,start,end)]

    if method == "BLUME":
        beta = get_raw_beta(ticker, benchmark, freq, start, end)*2/3+1/3

        equity_betas[(ticker,method,benchmark,freq,start,end)] = beta

        return beta
    return

# Reviewed: 2025-06-27
def get_dual_betas(ticker, benchmark="^GSPC", freq='M', start=FIVE_YRS, end=NOW):
    """
    Calculates the upside (beta+) and downside (beta-) betas of a ticker relative to a benchmark.
    """

    if (ticker, 'PLUS', benchmark, freq, start, end) in equity_betas:
        return {
            'PLUS': equity_betas[(ticker, 'PLUS', benchmark, freq, start, end)],
            'MINUS': equity_betas[(ticker, 'MINUS', benchmark, freq, start, end)]
        }

    return_df = get_log_return_df(ticker, freq, start, end)
    benchmark_return_df = get_log_return_df(benchmark, freq, start, end)

    return_df = return_df.loc[benchmark_return_df.index.intersection(return_df.index)]
    benchmark_return_df = benchmark_return_df.loc[return_df.index]

    plus_mask = benchmark_return_df.iloc[:, 0] > 0
    minus_mask = benchmark_return_df.iloc[:, 0] < 0

    plus_return_df = return_df[plus_mask]
    plus_benchmark_return_df = benchmark_return_df[plus_mask]
    minus_return_df = return_df[minus_mask]
    minus_benchmark_return_df = benchmark_return_df[minus_mask]

    dual_betas = {
        "PLUS": get_df_beta(plus_return_df, plus_benchmark_return_df),
        "MINUS": get_df_beta(minus_return_df, minus_benchmark_return_df)
    }

    equity_betas[(ticker, 'PLUS', benchmark, freq, start, end)] = dual_betas['PLUS']
    equity_betas[(ticker, 'MINUS', benchmark, freq, start, end)] = dual_betas['MINUS']

    return dual_betas


# Reviewed: 2025-06-27
def get_betas(ticker, benchmark="^GSPC", freq='M', start=FIVE_YRS, end=NOW):
    """
    Returns a dictionary of raw, adjusted, and upside/downside betas for a given ticker.

    Args:
        ticker (str): Ticker symbol of the asset.
        benchmark (str): Benchmark ticker symbol.
        freq (str): Frequency of returns data.
        start (datetime): Start date for calculation.
        end (datetime): End date for calculation.

    Returns:
        dict: {
            'RAW': float,
            'ADJ': float,
            'PLUS': float,
            'MINUS': float
        }
    """
    return {
        "RAW":get_raw_beta(ticker,benchmark,freq,start,end),
        "ADJ":get_adj_beta(ticker,benchmark,"BLUME",freq,start,end)
    } | get_dual_betas(ticker,benchmark,freq,start,end)

# Reviewed: 2025-06-27
def get_volatility(ticker, days:int = 360):
    """
    Computes the annualized volatility of the ticker's log returns over a rolling window.

    Args:
        ticker (str): Ticker symbol of the asset.
        days (int): Number of trading days for the rolling window (default 360).

    Returns:
        float: Annualized volatility as a decimal.
    """
    return_df = get_log_return_df(ticker)

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    return rolling_volatility.iloc[-1,0]

# Reviewed: 2025-06-27
def get_kurtosis(ticker):
    """
    Calculates the kurtosis of the ticker's log returns.

    Args:
        ticker (str): Ticker symbol of the asset.

    Returns:
        float: Kurtosis value (Fisher=False, so normal distribution has kurtosis 3).
    """
    df = get_log_return_df(ticker)
    series = df.iloc[:, 0].dropna()

    return float(kurtosis(series, fisher=False))

# Reviewed: 2025-06-27
def get_skewness(ticker):
    """
    Calculates the skewness of the ticker's log returns.

    Args:
        ticker (str): Ticker symbol of the asset.

    Returns:
        float: Skewness value.
    """
    df = get_log_return_df(ticker)
    series = df.iloc[:, 0].dropna()

    return float(skew(series))

# Reviewed: 2025-06-27
def get_shannon_entropy(ticker, bins=50):
    """
    Calculates the Shannon entropy of the ticker's log returns histogram.

    Args:
        ticker (str): Ticker symbol of the asset.
        bins (int): Number of histogram bins (default 50).

    Returns:
        float: Shannon entropy value.
    """
    df = get_log_return_df(ticker)
    hist, bin_edges = np.histogram(df['Log Returns'], bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities to avoid log(0)
    entropy = -np.sum(hist * np.log(hist))
    return entropy

# Reviewed: 2025-06-27
def get_df_correlation(df1, df2):
    """
    Calculate the Pearson correlation between the first columns of two DataFrames or Series.
    
    Args:
      df1, df2: pd.DataFrame or pd.Series - input data, only first column used.
    
    Returns:
      float - correlation coefficient.
    """
    df1 = pd.DataFrame(df1) if isinstance(df1, pd.Series) else df1
    df2 = pd.DataFrame(df2) if isinstance(df2, pd.Series) else df2
    return df1.iloc[:, 0].corr(df2.iloc[:, 0])

# Reviewed: 2025-06-27
def get_ticker_correlation(ticker1, ticker2, freq='W', start=FIVE_YRS, end=NOW):
    """
    Compute the correlation of log returns between two tickers over a time period.
    
    Args:
      ticker1, ticker2: str - stock tickers.
      freq: str - frequency of returns (e.g. 'W' for weekly).
      start, end: datetime - date range for returns.
    
    Returns:
      float - correlation coefficient.
    """
    return_df = get_log_return_df([ticker1,ticker2],freq,start,end)
    return get_df_correlation(return_df[ticker1],return_df[ticker2])

# Reviewed: 2025-06-27
def get_portfolio_log_returns(portfolio, freq='D', start=None, end=None):
    """
    Calculate weighted portfolio log returns by summing each holding's weighted log returns.
    Assumes each Holding has a 'ticker' and a 'weight' attribute (weight = fraction of portfolio).
    
    Args:
      portfolio: Portfolio object with holdings list.
      freq: str - frequency of returns.
      start, end: datetime - date range for returns.
    
    Returns:
      pd.Series - portfolio log returns indexed by date.
    """
    import pandas as pd
    
    weighted_returns = []
    for holding in portfolio.holdings:
        returns_df = get_log_return_df(holding.ticker, freq, start, end)
        returns_series = returns_df.iloc[:, 0] if isinstance(returns_df, pd.DataFrame) else returns_df
        returns_series = returns_series.dropna()
        
        weighted = returns_series * holding.weight  # weight is fraction of portfolio
        weighted_returns.append(weighted)
    
    combined = pd.concat(weighted_returns, axis=1, join='inner')
    portfolio_returns = combined.sum(axis=1).dropna()
    
    return portfolio_returns

# Reviewed: 2025-06-27
def get_risk_tier(holding:Holding):
    """
    Calculate a composite risk tier for a holding based on volatility, dual betas, and optional qualitative risk tier.
    
    Args:
      holding: Holding object with attributes 'ticker' and 'qual_risk_tier'.
    
    Returns:
      int - rounded risk tier (1-3 scale).
    """
    dual_betas = get_dual_betas(holding.ticker)
    vol = get_volatility(holding.ticker)

    vol_tier = 1 if vol >= 0.35 else 3 if vol < 0.25 else 2
    dual_betas_tier = 3 if dual_betas["MINUS"] <= 1.1 and dual_betas["PLUS"] >= 0.7 else 2 if dual_betas["MINUS"] <= 0.7 and dual_betas["PLUS"] >= 0.7 else 1
    qual_tier = holding.qual_risk_tier if holding.qual_risk_tier != 0 else (vol_tier + dual_betas_tier) / 2

    return round((vol_tier + dual_betas_tier + qual_tier) / 3)

# Reviewed: 2025-06-27
def get_buy_amounts(holding:Holding, portfolio:Portfolio) -> dict:
    """
    Calculates the recommended minimum, medium, and maximum number of shares to buy for a holding,
    based on the portfolio's total value, the stock's current price, and the holding's risk tier.

    Args:
        holding (Holding): The holding for which to calculate buy amounts.
        portfolio (Portfolio): The portfolio containing the holding.

    Returns:
        dict: Dictionary with keys 'MIN', 'MED', 'MAX' representing respective buy share counts.
    """
    port_value = portfolio.port_value_df.iloc[-1,:].sum()
    stock_price = get_price_df(holding.ticker).iloc[-1,0]
    risk_tier = get_risk_tier(holding)

    if risk_tier == 1:
        return {"MIN":math.ceil(port_value*0.02/stock_price),"MED":round(port_value*0.025/stock_price),"MAX":math.floor(port_value*0.03/stock_price)}
    elif risk_tier == 2:
        return {"MIN":math.ceil(port_value*0.03/stock_price),"MED":round(port_value*0.035/stock_price),"MAX":math.floor(port_value*0.04/stock_price)}
    elif risk_tier == 3:
        return {"MIN":math.ceil(port_value*0.04/stock_price),"MED":round(port_value*0.05/stock_price),"MAX":math.floor(port_value*0.06/stock_price)}

# Reviewed: 2025-06-27    
def get_price_df(tickers, freq='D', start=FIVE_YRS, end=NOW, fx_adj=True):
    """
    Retrieves historical price data for one or more tickers, caches results in `price_dfs`,
    optionally adjusts prices for FX (foreign exchange) rates if ticker is not Canadian.

    Args:
        tickers (str or list): Single ticker symbol or list of ticker symbols.
        freq (str): Frequency of data ('D' for daily, etc.).
        start (datetime): Start date for historical data.
        end (datetime): End date for historical data.
        fx_adj (bool): Whether to adjust prices by FX rates (default True).

    Returns:
        pd.DataFrame: DataFrame of adjusted historical prices indexed by date, columns as tickers.
    """
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

# Reviewed: 2025-06-27
def resample_df(df, freq, start=FIVE_YRS, end=NOW):
    """
    Resamples a DataFrame's time series data to the specified frequency within a date range.
    For daily frequency ('D'), returns data as is, otherwise resamples by taking the last value per period.

    Args:
        df (pd.DataFrame or pd.Series): Time series data indexed by date.
        freq (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
        start (datetime): Start date for slicing data.
        end (datetime): End date for slicing data.

    Returns:
        pd.DataFrame or pd.Series: Resampled time series data.
    """
    df = df.loc[start:end]
    return df if freq == 'D' else df.resample(freq).last().dropna()

# Reviewed: 2025-06-27
def plot_beta_scatter(ticker, benchmark='^GSPC', freq='M', start=FIVE_YRS, end=NOW, title="Portfolio", download_graph=False):
    """
    Plots the scatter plot of portfolio returns vs benchmark returns with 
    best fit lines for overall, positive market returns, and negative market returns.
    
    Parameters:
        ticker (str): Ticker symbol of the portfolio or asset.
        benchmark (str): Benchmark ticker symbol for comparison.
        freq (str): Frequency of returns ('D', 'W', 'M', etc.).
        start (pd.Timestamp): Start date for data.
        end (pd.Timestamp): End date for data.
        title (str): Title for the plot and axis label.
        download_graph (bool): If True, saves the plot as a PNG file.
    """
    return_df = get_log_return_df(ticker,freq,start,end).squeeze()
    market_df = get_log_return_df(benchmark,freq,start,end).squeeze()

    x = market_df
    y = return_df
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y, alpha=0.7, label='Returns', color='#156082')

    plt.plot(x, m*x + b, color='#0E3C51', label=f'Best Fit Line (y = {m:.2f}x + {b:.2f})')

    x_mid = np.mean(x)
    y_mid = m * x_mid + b
    neg_x = x[x < 0]
    neg_y = y[x < 0]
    pos_x = x[x >= 0]
    pos_y = y[x >= 0]

    m_neg, b_neg = np.polyfit(neg_x, neg_y, 1)
    m_pos, b_pos = np.polyfit(pos_x, pos_y, 1)

    plt.plot(neg_x, m_neg * neg_x + b_neg, color='#B84848', label=f'Negative X Best Fit (y = {m_neg:.2f}x + {b_neg:.2f})')
    plt.plot(pos_x, m_pos * pos_x + b_pos, color='#5BB848', label=f'Positive X Best Fit (y = {m_pos:.2f}x + {b_pos:.2f})')

    plt.xlabel('Market')
    plt.ylabel(title)

    padding = 0.01
    plt.grid(True)
    x_range = max(x.max() - x.min(), y.max() - y.min()) + 2 * padding
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    plt.xlim(x_mid - x_range / 2, x_mid + x_range / 2)
    plt.ylim(y_mid - x_range / 2, y_mid + x_range / 2)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Market', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.ylabel(title, fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.text(
        0.02, 0.98,
        f"β: {m:.2f}\nβ⁺: {m_pos:.2f}\nβ⁻: {m_neg:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10, fontfamily='serif', verticalalignment='top',
        bbox=dict(facecolor='#FFFBC6')
    )

    if download_graph:
        plt.savefig(f'{title}_beta_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

# Reviewed: 2025-06-27
def get_df_correlation(df1, df2):
    """
    Computes the correlation coefficient between the first columns of two dataframes or series.
    
    Parameters:
        df1 (pd.DataFrame or pd.Series): First dataset.
        df2 (pd.DataFrame or pd.Series): Second dataset.
        
    Returns:
        float: Correlation coefficient.
    """
    df1 = pd.DataFrame(df1) if isinstance(df1, pd.Series) else df1
    df2 = pd.DataFrame(df2) if isinstance(df2, pd.Series) else df2
    return df1.iloc[:, 0].corr(df2.iloc[:, 0])

# Reviewed: 2025-06-27
def get_ticker_correlation(ticker1, ticker2, freq='W', start=FIVE_YRS, end=NOW):
    """
    Computes the correlation between log returns of two tickers over a specified frequency and date range.
    
    Parameters:
        ticker1 (str): First ticker symbol.
        ticker2 (str): Second ticker symbol.
        freq (str): Frequency of returns ('D', 'W', 'M', etc.).
        start (pd.Timestamp): Start date for data.
        end (pd.Timestamp): End date for data.
        
    Returns:
        float: Correlation coefficient between ticker1 and ticker2 returns.
    """
    return_df = get_log_return_df([ticker1,ticker2],freq,start,end)
    return get_df_correlation(return_df[ticker1],return_df[ticker2])

# Reviewed: 2025-06-27
def get_correlations(holding: Holding, portfolio: Portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """
    Returns a dictionary of correlation values between the given holding and the portfolio.

    Correlations computed:
    - "PORT": correlation of holding with total portfolio returns
    - "SECTORS": dict of correlation with each sector's returns in the portfolio
    - "HOLDINGS": dict of correlation with each individual holding in the portfolio

    All returns are resampled weekly ('W-SUN') and summed before correlation.

    Parameters:
        holding (Holding): The holding for which to compute correlations.
        portfolio (Portfolio): The portfolio containing multiple holdings.
        freq (str): Frequency of returns used to fetch data.
        start, end: Date range for return data.

    Returns:
        dict: Nested dictionary of correlations.
    """
    correlations = {}

    return_df = get_log_return_df(holding.ticker, freq, start, end)
    if isinstance(return_df, pd.DataFrame):
        return_df = return_df.iloc[:, 0]
    return_df = return_df.resample('W-SUN').sum().dropna()

    port_returns = portfolio.get_port_return_df(False, freq, start, end)

    aligned = pd.concat([return_df, port_returns], axis=1, join='inner').dropna()
    correlations["PORT"] = aligned.corr().iloc[0, 1] if not aligned.empty else float('nan')

    correlations["SECTORS"] = {}
    for sector, df in portfolio.get_sector_return_dfs(freq, start, end).items():
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        df = df.resample('W-SUN').sum().dropna()
        aligned_sector = pd.concat([return_df, df], axis=1, join='inner').dropna()
        correlations["SECTORS"][sector] = aligned_sector.corr().iloc[0, 1] if not aligned_sector.empty else float('nan')

    correlations["HOLDINGS"] = {}
    for ticker, df in get_log_return_df([h.ticker for h in portfolio.holdings], freq, start, end).items():
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        df = df.resample('W-SUN').sum().dropna()
        aligned_h = pd.concat([return_df, df], axis=1, join='inner').dropna()
        if not aligned_h.empty:
            correlations["HOLDINGS"][ticker] = aligned_h.corr().iloc[0, 1]

    correlations["HOLDINGS"] = dict(sorted(correlations["HOLDINGS"].items(), key=lambda item: item[1], reverse=True))

    return correlations

# Reviewed: 2025-06-27
def print_risk_report(holding: Holding, portfolio: Portfolio):
    """
    Prints a formatted risk report for the given holding.

    Includes:
    - Beta (total, upside, downside)
    - Volatility (360d and 60d)
    - Skewness and kurtosis
    - Risk tier
    - Suggested buy values and share counts (min, med, max)
    - Correlation to portfolio, sector, and significant holdings

    Parameters:
        holding (Holding): The asset to evaluate.
        portfolio (Portfolio): The portfolio context.
    """
    betas = get_betas(holding.ticker)
    skewness = get_skewness(holding.ticker)
    kurtosis = get_kurtosis(holding.ticker)
    
    corrs = get_correlations(holding, portfolio)

    port_corr = corrs["PORT"]
    sector_corr = corrs["SECTORS"].get(holding.sector, np.nan)
    significant_corrs = ', '.join(
        [f"{h}: {c:.2%}" for h, c in corrs["HOLDINGS"].items() if c >= 0.5]
    )

    beta_raw = betas.get("RAW", None)
    beta_plus = betas.get("PLUS", None)
    beta_minus = betas.get("MINUS", None)
    
    volatility_360d = get_volatility(holding.ticker)
    volatility_60d = get_volatility(holding.ticker, days=60)
    
    risk_tier = get_risk_tier(holding)
    buy_amounts = get_buy_amounts(holding, portfolio)
    price = get_price_df(holding.ticker).iloc[-1, 0]

    buy_values = {
        k: (v * price, v)
        for k, v in buy_amounts.items()
    }

    def fmt(val, pct=False):
        if pd.isna(val):
            return "N/A"
        return f"{val:.2%}" if pct else f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"

    print(f"Beta\n└ {fmt(beta_raw)}")
    print(f"┌Beta+\n└ {fmt(beta_plus)}")
    print(f"┌Beta-\n└ {fmt(beta_minus)}")
    print(f"┌360D Vol\n└ {fmt(volatility_360d, pct=True)}")
    print(f"┌60D Vol\n└ {fmt(volatility_60d, pct=True)}")
    print(f"┌Skewness\n└ {fmt(skewness)}")
    print(f"┌Kurtosis\n└ {fmt(kurtosis)}")
    print(f"┌Risk Tier\n└ {risk_tier}")

    for label in ["MIN", "MED", "MAX"]:
        value, shares = buy_values[label]
        print(f"┌{label.capitalize()} Buy")
        print(f"├ Value: ${value:,.2f}")
        print(f"└ Shares: {shares:.2f}")

    print(f"┌Portfolio Correlation\n└ {fmt(port_corr, pct=True)}")
    print(f"┌Sector Correlation\n└ {fmt(sector_corr, pct=True)}")
    print(f"┌Significant Correlations")
    print(f"└ {significant_corrs if significant_corrs else 'None'}")


# Reviewed: 2025-06-27
def find_lowest_correlated_pair(portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """
    Finds the pair of holdings in the portfolio with the lowest correlation of returns.

    Parameters:
        portfolio (Portfolio): Portfolio containing holdings with tickers.
        freq (str): Frequency of return data (default weekly).
        start, end: Date range for return data.

    Returns:
        tuple: ((ticker1, ticker2), correlation_value)
    """

    # Get all tickers from your portfolio
    tickers = [h.ticker for h in portfolio.holdings]

    # Get return dataframes for all holdings at once
    all_returns = get_log_return_df(tickers, freq, start, end)

    lowest_corr = 1  # Start with highest possible correlation
    lowest_pair = (None, None)

    # Compare each unique pair
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:  # This avoids duplicate pairs and self-comparisons
            if ticker1 != ticker2:
                corr = get_df_correlation(all_returns[ticker1], all_returns[ticker2])

                if corr < lowest_corr:
                    lowest_corr = corr
                    lowest_pair = (ticker1, ticker2)

    return lowest_pair, lowest_corr

# Reviewed: 2025-06-27
def find_highest_correlated_pair(portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """
    Finds the pair of holdings in the portfolio with the highest correlation of returns.

    Parameters:
        portfolio (Portfolio): Portfolio containing holdings with tickers.
        freq (str): Frequency of return data (default weekly).
        start, end: Date range for return data.

    Returns:
        tuple: ((ticker1, ticker2), correlation_value)
    """

    # Get all tickers from your portfolio
    tickers = [h.ticker for h in portfolio.holdings]

    # Get return dataframes for all holdings at once
    all_returns = get_log_return_df(tickers, freq, start, end)

    highest_corr = -1  # Start with lowest possible correlation
    highest_pair = (None, None)

    # Compare each unique pair
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:  # This avoids duplicate pairs and self-comparisons
            if ticker1 != ticker2:
                corr = get_df_correlation(all_returns[ticker1], all_returns[ticker2])

                if corr > highest_corr:
                    highest_corr = corr
                    highest_pair = (ticker1, ticker2)

    return highest_pair, highest_corr