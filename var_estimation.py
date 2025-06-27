# === var_estimation.py

import numpy as np
import matplotlib.pyplot as plt

from functions import Portfolio, Holding
from config import NOW, FIVE_YRS, ONE_YR

#Current portfolio holdings
HOLDINGS = [
    Holding("LVMHF", "CRL", 5, 2),
    Holding("PEP", "CRL", 34, 3),
    Holding("JWEL.TO", "CRL", 147, 2),
    Holding("GIS", "CRL", 56, 2),
    Holding("COST", "CRL", 9, 3),
    Holding("ATD.TO", "CRL", 111, 3),
    Holding("V", "FIN", 20, 3),
    Holding("SCHW", "FIN", 60, 2),
    Holding("JPM", "FIN", 29, 3),
    Holding("BMO.TO", "FIN", 53, 3),
    Holding("ACN", "TMT", 26, 3),
    Holding("CSCO", "TMT", 85, 2),
    Holding("OTEX.TO", "TMT", 110, 2),
    Holding("DIS", "TMT", 30, 1),
    Holding("PFE", "HLT", 230, 1),
    Holding("VRTX", "HLT", 14, 2),
    Holding("NVO", "HLT", 40, 2),
    Holding("ENB.TO", "INR", 143, 2),
    Holding("CNQ.TO", "INR", 225, 1),
    Holding("J", "INR", 37, 3),
    Holding("MG.TO", "INR", 41, 2),
    Holding("XYL", "INR", 49, 2),
    Holding("CP.TO", "INR", 75, 3),
    Holding("NTR.TO", "INR", 60, 2),
    Holding("AMTM", "INR", 37, 2)
]

current_port = Portfolio(HOLDINGS, 80955.00)

# initialize inputs and calclulate returns
portfolio = current_port
returns = portfolio.log_return_df
weights = portfolio.get_holding_weights()
portfolio_returns = np.dot(returns, list(weights.values()))

# establish confidene level
confidence_level = 0.05

# function
VaR = np.percentile(portfolio_returns, 100 * confidence_level)
print(f"Value at Risk (VaR) at {100*(1-confidence_level)}% confidence: {VaR:.4f}")

# plot
plt.figure(figsize=(10,5))
plt.hist(portfolio_returns, bins=50, edgecolor='black', alpha=0.9)
plt.axvline(VaR, color='red', linestyle='dashed', linewidth=2)
plt.text(VaR, plt.ylim()[1] * 0.9, f'VaR: {VaR:.4f}', color='red', ha='right', fontsize=8)
plt.title('Distribution of Portfolio Returns and Value at Risk (VaR)')
plt.xlabel('Portfolio Returns (Percentage)')
plt.ylabel('Frequency')
plt.legend([f'VaR at {100*(1-confidence_level)}% confidence', 'Returns'], loc='upper left')
plt.grid(True)
plt.show()