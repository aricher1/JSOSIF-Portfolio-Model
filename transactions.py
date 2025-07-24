"""
transactions.py

Module for simulating predefined transactions on an investment portfolio.

This script defines and executes a set of buy and sell transactions on a
portfolio object using the custom Portfolio, Holding, and Transaction classes.
Intended for use within broader portfolio management workflows, such as
report generation or rebalancing simulations.

Example:
    from transactions import simulate_portfolio_transactions

    updated_port = simulate_portfolio_transactions(current_port)

Author: JSOSIF Quant Team
Verified: 2025-06-27
"""

from functions import Portfolio, Transaction, Holding
from holdings import HOLDINGS

current_port = Portfolio(HOLDINGS, 82846.00) #$12,956 CASH + $69,890 BONDS

# Reviewed: 2025-06-27
def simulate_portfolio_transactions(portfolio):
    """
    Simulates a predefined set of portfolio transactions and returns the updated portfolio.

    Args:
        portfolio (Portfolio): The current portfolio object.

    Returns:
        Portfolio: Updated portfolio after transactions.
    """
    buy_transaction = Transaction(
        holding=Holding('UNH', shares=15, sector='HLT'),
        side='BUY'
    )

    sell_transaction = Transaction(
        holding=Holding('PFE', shares=180, sector='HLT'),
        side='SELL'
    )

    transactions = [buy_transaction, sell_transaction]
    return portfolio.simulate_transactions(transactions)

# Reviewed: 2025-06-27
def print_holdings(portfolio):
    """
    Prints the tickers of all holdings in the portfolio.
    """
    tickers = [h.ticker for h in portfolio.holdings]
    print("Current tickers in portfolio:")
    print(tickers)

#transaction simulation
current_port = simulate_portfolio_transactions(current_port)

#sanity check for current holdings
print_holdings(current_port)