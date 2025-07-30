"""
risk_report.py

Generates risk metrics, correlation analysis, and visualizations
for portfolio holdings. Supports rolling volatility, return
histograms, and price charts for individual tickers.

Dependencies: pandas, numpy, matplotlib, seaborn, config, functions, holdings

Author: JSOSIF Quant Team  
Verified: 2025-06-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import TWO_YRS, FIVE_YRS, NOW, ONE_YR
from plotting import (
    show_rolling_volatility_graph,
    plot_log_returns_histogram,
    plot_stock_returns,
    plot_stock_chart,
)
from functions import Portfolio, Holding, print_risk_report, get_correlations
from holdings import HOLDINGS

current_port = Portfolio(HOLDINGS, 82846.00) #$12,956 CASH + $69,890 BONDS

"""
* ticker = 'INSERT TICKER HERE'
* sector = 'INSERT TICKER'S SECTOR HERE' 
Sectors (
        CRL: Consumer and Retail
        FIN: Financials
        HLT: Healthcare
        TMT: Technology, Media, and Telecommunications
        INR: Industirals and Natural Resources
        )
* qual_risk = 'INSERT QUALITATIVE RISK TIER HERE' (1: High, 2: Medium, 3: Low)
"""

# ======= User Inputs for Ticker Analysis ======= #
ticker = 'ALL'
sector = 'FIN'
qual_risk = '2'

#print correlations and detialed risk report
print(get_correlations(Holding(ticker, sector), current_port))
print_risk_report(Holding(ticker, sector, qual_risk), current_port)

#plot various risk and performance charts
show_rolling_volatility_graph(ticker, download_graph=False)
show_rolling_volatility_graph(ticker, days=60, download_graph=False)
plot_log_returns_histogram(ticker)
plot_stock_returns(ticker)
plot_stock_chart(ticker)

#display the portfolios correlation matrix and heatmap
current_port.show_corr_matrix()

#print weekly update for PnL
current_port.get_weekly_update()