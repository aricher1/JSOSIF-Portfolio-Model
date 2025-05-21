# === monte_carlo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing import Portfolio, Holding
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
portfolio = current_port
returns = portfolio.log_return_df

# Initialize inputs
returns_mc = returns.mean()
cov_matrix = returns.cov()
weights = portfolio.get_holding_weights()
portfolio_value = portfolio.port_value_df.iloc[-1,:].sum()

# Call the function to generate random returns
np.random.multivariate_normal(returns_mc, cov_matrix)

# Initialize returns
daily_returns = np.random.multivariate_normal(returns_mc, cov_matrix)
print(daily_returns)

# Compute the portfolio return and the new portfolio value
portfolio_return = np.dot(daily_returns, list(weights.values()))
val = portfolio_value * np.exp(portfolio_return)

print(f"{val:,.02f}")

# Create a function to run simulation over a given number of days
def run_simulation(weights, returns_mc, cov_matrix, portfolio_value, days = 252):

  daily_value = {"Day 0": portfolio_value}

  for day in range(1, days + 1):
    daily_returns = np.random.multivariate_normal(returns_mc, cov_matrix)
    portfolio_return = np.dot(daily_returns, list(weights.values()))
    portfolio_value = portfolio_value * np.exp(portfolio_return)
    daily_value[f"Day {day}"] = portfolio_value

  return daily_value

# initialize inputs
returns_mc = returns.mean()
cov_matrix = returns.cov()
weights = portfolio.get_holding_weights()

# call the function
sim = run_simulation(weights, returns_mc, cov_matrix, portfolio_value)
sim
def run_simulations(sims, weights, returns_mc, cov_matrix, portfolio_value):
  simulations = []
  for _ in range(sims):
    sim = run_simulation(weights, returns_mc, cov_matrix, portfolio_value)
    simulations.append(sim)
  return pd.DataFrame(simulations)

# initialize inputs
returns_mc = returns.mean()
cov_matrix = returns.cov()
weights = portfolio.get_holding_weights()
sims = 2

# call the function
sims_df = run_simulations(sims, weights, returns_mc, cov_matrix, portfolio_value)

plt.figure(figsize=(10,6))

for i in range(sims_df.shape[0]):
  plt.plot(sims_df.columns, sims_df.iloc[i, :], color='purple', alpha=0.25)

# Compute the actual mean, 5th percentile, and 95th percentile values
mean_value = sims_df.mean().iloc[-1]
percentile_05 = sims_df.quantile(0.05, axis=0).iloc[-1]
percentile_95 = sims_df.quantile(0.95, axis=0).iloc[-1]

plt.plot(sims_df.columns, sims_df.mean(), color='blue', label=f'Mean Portfolio Value: {mean_value:.2f}', linewidth=2)
plt.plot(sims_df.columns, sims_df.quantile(0.05, axis=0), color='gold', linestyle='solid', linewidth=2, label=f'5th Percentile: {percentile_05:.2f}')
plt.plot(sims_df.columns, sims_df.quantile(0.95, axis=0), color='black', linewidth=2, linestyle='solid', label=f'95th Percentile: {percentile_95:.2f}')
num_sims = sims_df.shape[0]
plt.title('Monte Carlo Simulation of Portfolio Value Over 252 Trading Days')
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.legend(title=f'Number of Simulations: {num_sims}')
plt.grid(True)
plt.show()