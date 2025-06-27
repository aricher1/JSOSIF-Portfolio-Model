# == riskreport.py

"Logic for generating risk report documents. i.e., execution for risk metrics, graphs etc. for a given ticker"

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from config import TWO_YRS, FIVE_YRS, NOW, ONE_YR
from plotting import show_rolling_volatility_graph, plot_log_returns_histogram, plot_stock_returns, plot_stock_chart
from functions import Portfolio, Holding, print_risk_report, get_correlations
from holdings import HOLDINGS

current_port = Portfolio(HOLDINGS, 80955.00) #$9076 CASH + $71879 BONDS

"""
* ticker = 'INSERT TICKER HERE'
* sector = 'INSERT TICKER'S SECTOR HERE' Sectors:
                                            - CRL: Consumer and Retail
                                            - FIG: Financial Intsitutions Groups and Fixed Income
                                            - FIN: Financials
                                            - HLT: Healthcare
                                            - TMT: Technology, Media, and Telecommunications
                                            - INR: Industirals and Real Estate
* qual_risk = 'INSERT QUALITATIVE RISK TIER HERE' (1: High, 2: Medium, 3: Low)
"""

ticker = 'UNH'
sector = 'HLT'
qual_risk = '2'
print_risk_report(Holding(ticker, sector, qual_risk), current_port)
show_rolling_volatility_graph(ticker, download_graph=False)
show_rolling_volatility_graph(ticker, days=60, download_graph=False)
get_correlations(Holding(ticker, sector), current_port)
plot_log_returns_histogram(ticker)
plot_stock_returns(ticker)
plot_stock_chart(ticker)
current_port.show_corr_matrix()