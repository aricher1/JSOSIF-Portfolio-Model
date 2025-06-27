# === functions.py

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
from scipy.stats import kurtosis

from config import FIVE_YRS, NOW, ONE_YR, price_dfs, equity_betas

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
    
    def get_betas(ticker, benchmark="^GSPC", freq='ME', start=FIVE_YRS, end=NOW):
        """Gets a dictionary of raw, adjusted, & + / - betas of a given ticker"""
        return {
            "RAW":get_raw_beta(ticker,benchmark,freq,start,end),
            "ADJ":get_adj_beta(ticker,benchmark,"BLUME",freq,start,end)
        } | get_dual_betas(ticker,benchmark,freq,start,end)
    
    def get_avg_pairwise_corr(self, freq='W', start=FIVE_YRS, end=NOW):
       """Gets the average pairwise correlation for the portfolio's holdings"""
       return_df = get_log_return_df([h.ticker for h in self.holdings], freq, start, end)
       corr_matrix = return_df.corr()
       upper_tri_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()

       return upper_tri_values.mean()
    
    def get_r_squared(self, include_cash:bool=True, freq='W', start=FIVE_YRS, end=NOW):
        """Gets the r squared of the portfolio"""
        price_df = resample_df(self.get_port_value_df(include_cash).sum(axis=1),freq,start,end)
        return_df = np.log(1+price_df.pct_change(fill_method=None).dropna())
        market_df = get_log_return_df(self.benchmark,freq,start,end,True).squeeze()

        return return_df.corr(market_df)**2
    
    def show_corr_matrix(self, freq='W', start=FIVE_YRS, end=NOW, download_graph=False):
        """Shows the portfolios correlation matrix"""
        return_df = get_log_return_df([h.ticker for h in self.holdings], freq, start, end)
        corr_matrix = (return_df*100).corr()
        plt.figure(figsize=(20, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='inferno', fmt=".2f")
        plt.title('Correlation Matrix')

        if download_graph:
            plt.savefig(f'corr_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def get_compliance_report(self):
        """Gets a report for each stock's compliance"""
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
    
    def get_contribution_report(self, historical:bool=False, include_cash:bool=False):
        """Gets a report for each stock's effect on the ports metrics""" #Maybe rename this method?
        report = pd.DataFrame()

        betas = self.get_betas(historical, include_cash)
        vol = self.get_stdev(historical, include_cash, days=360)
        corr = self.get_avg_pairwise_corr()

        #Creates a synthetic portfolio excluding each holding to measure it's contribution
        for h in self.holdings:
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
    
    def get_metrics_report(self, historical:bool=False, download_graphs:bool=False, show_graphs:bool=False):
        """Gets a report combining the portfolio's & individual sector's metrics"""
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
            plot_beta_scatter(self.guid,self.benchmark,'ME',FIVE_YRS,NOW,'Portfolio',download_graphs)

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
            plot_beta_scatter(self.equities_guid,self.benchmark,'ME',FIVE_YRS,NOW,'Equities',download_graphs)

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
                plot_beta_scatter(sector_port.guid,sector_port.benchmark,'ME',FIVE_YRS,NOW,s,download_graphs)

            sector_port.destroy()

        return report
    
    def get_holding_returns(self, start=NOW-relativedelta(days=7), end=NOW):
        """Returns a df of each holdings returns for the timeframe"""
        df = self.port_value_df.copy()
        df = df.loc[start:end]
        df.loc["%Chg"] = df.iloc[-1] / df.iloc[1] - 1
        df.loc["$Chg"] = df.iloc[-2] - df.iloc[1]
        df = df.T.sort_values(by=['%Chg'], ascending=False).T

        #Styles change rows
        df.loc["%Chg"] = df.loc["%Chg"].apply(lambda x: f"{x:.2%}")
        df.loc["$Chg"] = df.loc["$Chg"].apply(lambda x: f"${x:,.2f}")

        return df.loc[df.index[-2:]].T
    
    def get_portfolio_returns(self, start=NOW-relativedelta(days=7), end=NOW):
        """Returns a df of each portfolios returns for the timeframe"""
        df = self.port_value_df.copy()
        df = df.loc[start:end]
        ticker_to_sector = {h.ticker: h.sector for h in self.holdings}
        sector_mapping = pd.Series(ticker_to_sector)
        df_grouped = (
            df.groupby(sector_mapping, axis=1)  # Group by the sector mapping
              .sum()                           # Sum up values row-wise
        )
        df_grouped["Equities"] = df_grouped.sum(axis=1)

        df_grouped.loc["%Chg"] = df_grouped.iloc[-1] / df_grouped.iloc[1] - 1
        df_grouped.loc["$Chg"] = df_grouped.iloc[-2] - df_grouped.iloc[1]
        df_grouped = df_grouped.T.sort_values(by=['%Chg'], ascending=False).T

        #Styles change rows
        df_grouped.loc["%Chg"] = df_grouped.loc["%Chg"].apply(lambda x: f"{x:.2%}")
        df_grouped.loc["$Chg"] = df_grouped.loc["$Chg"].apply(lambda x: f"${x:,.2f}")

        return df_grouped.loc[df_grouped.index[-2:]].T
    
    def get_dividends(self, start=NOW-relativedelta(days=7), end=NOW):
        """Returns a df of every dividend paid for the timeframe"""
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
    
    def get_weekly_update(self, start = NOW-relativedelta(days=7), end=NOW):
        print("-- HOLDING RETURNS --")
        print(self.get_holding_returns(start,end))
        print("")
        print("-- PORTFOLIO RETURNS --")
        print(self.get_portfolio_returns(start,end))
        print("")
        print("-- DIVIDENDS RECORDS --")
        print(self.get_dividends(start,end))

    def simulate_transactions(self, transactions:list[Transaction],cash=0):
        """Simulates proposed transactions and outputs new portfolio metrics & cash balance"""
        #!!!!!!!!!!!!!!!!remove cash param if we can seperate bonds and cash values in the port
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
                    print(f'✅ Bought {shares} shares of {tr.holding.ticker} @ ${price:,.2f} for a total of ${value:,.2f}.')
                else:
                    print(f'❌ Error buying {tr.holding.ticker}: {shares} shares @ ${price:,.2f} (${value:,.2f}) is greater than cash balance (${new_cash:,.2f}).')
                    return
            else:
                #Prevents selling unheld holdings
                if tr.holding.ticker not in [h.ticker for h in new_holdings]:
                    print(f'❌ Error selling {tr.holding.ticker}: Not in portfolio.')
                    return

                position = next((h for h in new_holdings if h.ticker == tr.holding.ticker), None)
                shares = tr.holding.shares if tr.holding.shares > 0 else position.shares
                value = price * shares

                if shares <= position.shares:
                    new_holdings.remove(position)
                    new_cash += value
                    print(f'✅ Sold {shares} shares of {tr.holding.ticker} @ ${price:,.2f} for a total of ${value:,.2f}.')
                else:
                    print(f'❌ Error selling {tr.holding.ticker}: {shares} shares is greater than what is held: {position.shares} shares')
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
    
    def destroy(self):
        price_dfs.pop(self.guid)
        price_dfs.pop(self.equities_guid)
    
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

def resample_df(df, freq, start=FIVE_YRS, end=NOW):
    """Returns a resampled data frame within the date range passed"""
    df = df.loc[start:end]
    return df if freq == 'D' else df.resample(freq).last().dropna()

def plot_beta_scatter(ticker, benchmark='^GSPC', freq='ME', start=FIVE_YRS, end=NOW, title="Portfolio", download_graph=False):
    """Plots the portfolio to benchmark return scatter plot with lines of best fit"""
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

def print_risk_report(holding:Holding, portfolio:Portfolio):
    betas = get_betas(holding.ticker)
    corrs = get_correlations(holding,portfolio)

    beta_raw = betas["RAW"]
    beta_plus = betas["PLUS"]
    beta_minus = betas["MINUS"]
    volatility_360d = get_volatility(holding.ticker)
    port_corr = corrs["PORT"]
    sector_corr = corrs["SECTORS"].get(holding.sector,0)
    significant_corrs = ', '.join([f"{h}: {c:.2%}" for h, c in corrs["HOLDINGS"].items() if c >= 0.5])
    risk_tier = get_risk_tier(holding)
    buy_amounts = get_buy_amounts(holding, portfolio)
    min_buy, med_buy, max_buy = buy_amounts["MIN"], buy_amounts["MED"], buy_amounts["MAX"]
    min_buy_value = min_buy * get_price_df(holding.ticker).iloc[-1,0]
    med_buy_value = med_buy * get_price_df(holding.ticker).iloc[-1,0]
    max_buy_value = max_buy * get_price_df(holding.ticker).iloc[-1,0]


    print("┌Beta")
    print(f"└ {beta_raw:.3f}")
    print("┌Beta+")
    print(f"└ {beta_plus:.3f}")
    print("┌Beta-")
    print(f"└ {beta_minus:.3f}")
    print("┌360D Vol")
    print(f"└ {volatility_360d:.2%}")
    print("┌Port Corr")
    print(f"└ {port_corr:.2%}")
    print("┌Sector Corr")
    print(f"└ {sector_corr:.2%}")
    print("┌Sig. Corrs")
    print(f"└ {significant_corrs}")
    print("┌Risk Tier")
    print(f"└ {risk_tier}")
    print("┌Min Buy")
    print(f"└ {min_buy} shares worth ${min_buy_value:.2f}")
    print("┌Med Buy")
    print(f"└ {med_buy} shares worth ${med_buy_value:.2f}")
    print("┌Max Buy")
    print(f"└ {max_buy} shares worth ${max_buy_value:.2f}")

def find_lowest_correlated_pair(portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """Returns the pair of holdings with the highest correlation in the portfolio"""

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

def find_highest_correlated_pair(portfolio, freq='W', start=FIVE_YRS, end=NOW):
    """Returns the pair of holdings with the highest correlation in the portfolio"""

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