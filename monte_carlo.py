"""
monte_carlo.py

Monte Carlo simulation to forecast future portfolio value using log return data
and historical covariance matrix. Visualizes simulated outcomes across a time
horizon to assess return uncertainty.

Author: JSOSIF Quant Team  
Verified: 2025-06-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import NOW, FIVE_YRS, ONE_YR
from functions import Portfolio
from holdings import HOLDINGS 


#portfolio setup
current_port = Portfolio(HOLDINGS, 82846.00) #verify cash balance before proceeding
returns = current_port.log_return_df
weights = current_port.get_holding_weights()
portfolio_value = current_port.port_value_df.iloc[-1, :].sum()

# Reviewed: 2025-06-27
def run_simulation(weights, mean_returns, cov_matrix, portfolio_value, days=252):
  """
  Simulates a single future trajectory of portfolio value over a given number of trading days
  using log-normal returns generated from a multivariate normal distribution.

  Parameters:
    weights (dict): Asset weights in the portfolio (e.g., {'AAPL': 0.2, ...}).
    mean_returns (pd.Series): Expected daily log returns for each asset.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    portfolio_value (float): Initial total portfolio value.
    days (int): Number of trading days to simulate (default is 252).

  Returns:
    dict: Simulated portfolio value at each day (e.g., {'Day 0': 1_000_000, 'Day 1': ..., ...}).
  """
  daily_value = {"Day 0": portfolio_value}
  for day in range(1, days + 1):
      daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
      portfolio_return = np.dot(daily_returns, list(weights.values()))
      portfolio_value *= np.exp(portfolio_return)
      daily_value[f"Day {day}"] = portfolio_value
  return daily_value

# Reviewed: 2025-06-27
def run_simulations(sims, weights, mean_returns, cov_matrix, portfolio_value):
  """
    Runs multiple Monte Carlo simulations of future portfolio value paths.

    Parameters:
      sims (int): Number of independent simulations to run.
      weights (dict): Asset weights in the portfolio.
      mean_returns (pd.Series): Expected daily log returns.
      cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
      portfolio_value (float): Initial portfolio value.

    Returns:
      pd.DataFrame: DataFrame where each row is a simulation and each column is a day.
  """
  return pd.DataFrame([
      run_simulation(weights, mean_returns, cov_matrix, portfolio_value)
      for _ in range(sims)
  ])

#simulation execution
mean_returns = returns.mean()
cov_matrix = returns.cov()
sims = 500  #adjust number of simulations
sims_df = run_simulations(sims, weights, mean_returns, cov_matrix, portfolio_value)

plt.figure(figsize=(10, 6))
for i in range(sims_df.shape[0]): #for-loop to plot individual simulation paths
    plt.plot(sims_df.columns, sims_df.iloc[i, :], color='purple', alpha=0.1)

mean_value = sims_df.mean().iloc[-1]
p05 = sims_df.quantile(0.05, axis=0).iloc[-1]
p95 = sims_df.quantile(0.95, axis=0).iloc[-1]

plt.plot(sims_df.columns, sims_df.mean(), color='blue', linewidth=2, label=f'Mean: ${mean_value:,.2f}')
plt.plot(sims_df.columns, sims_df.quantile(0.05, axis=0), color='gold', linewidth=2, linestyle='--', label=f'5th %ile: ${p05:,.2f}')
plt.plot(sims_df.columns, sims_df.quantile(0.95, axis=0), color='black', linewidth=2, linestyle='--', label=f'95th %ile: ${p95:,.2f}')

plt.title('Monte Carlo Simulation — Portfolio Value (1 Year Horizon)')
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.legend(title=f'{sims} Simulations')
plt.grid(True)
plt.tight_layout()
plt.show()