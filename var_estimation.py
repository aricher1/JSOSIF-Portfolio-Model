"""
var_estimation.py

Estimates portfolio Value at Risk (VaR) at a specified confidence level
using historical simulation (non-parametric method). Plots the return 
distribution with VaR threshold.

Verified: 2025-06-27
"""

import numpy as np
import matplotlib.pyplot as plt

from functions import Portfolio
from holdings import HOLDINGS
from config import NOW, FIVE_YRS, ONE_YR

#initialize current portfolio
current_port = Portfolio(HOLDINGS, 82846.00) #$12,956 CASH + $69,890 BONDS
returns = current_port.log_return_df
weights = current_port.get_holding_weights()

portfolio_returns = np.dot(returns, list(weights.values())) #compute weighted daily portfolio values

# Reviewed: 2025-06-27
def calculate_var(portfolio_returns, confidence_level=0.05):
    """
    Estimates the Value at Risk (VaR) at a specified confidence level
    using the historical simulation method.

    Parameters:
        portfolio_returns (np.ndarray): Daily portfolio returns.
        confidence_level (float): Confidence level (default = 0.05 for 95% confidence).

    Returns:
        float: Estimated VaR (negative value indicates expected loss).
    """
    return np.percentile(portfolio_returns, 100 * confidence_level)

confidence_level = 0.05 #adjust level if you'd like, 95% and 99% are standard VaR calculations
VaR = calculate_var(portfolio_returns, confidence_level)
print(f"Value at Risk (VaR) at {100 * (1 - confidence_level):.0f}% confidence: {VaR:.4f}")

plt.figure(figsize=(10, 5))
plt.hist(portfolio_returns, bins=50, edgecolor='black', alpha=0.9)
plt.axvline(VaR, color='red', linestyle='dashed', linewidth=2)
plt.text(VaR, plt.ylim()[1] * 0.9, f'VaR: {VaR:.4f}', color='red', ha='right', fontsize=8)
plt.title('Distribution of Portfolio Returns and Value at Risk (VaR)')
plt.xlabel('Portfolio Returns')
plt.ylabel('Frequency')
plt.legend([f'VaR at {100 * (1 - confidence_level):.0f}% confidence', 'Returns'], loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()