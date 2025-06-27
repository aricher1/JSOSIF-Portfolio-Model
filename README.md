# Quantitative Portfolio Analysis Model (JSOSIF)
Quantitative Research and Portfolio Risk Team

Authors: Aidan Richer, Adam Bergen, Max Mullins (2024)

This is a Python-based quantitative modeling framework developed for the **John Simpson Odette Student Investment Fund (JSOSIF)**. The model allows for in-depth portfolio analysis, simulation, risk assessment, and visualization using real market data via `yfinance`. Built for the **John Simpson Odette Student Investment Fund (JSOSIF)** to support internal reporting and research.

---

## Module Overview

### `config.py`
Defines shared date constants and cache dictionaries:
- `NOW`, `FIVE_YRS`, `TWO_YRS`, `ONE_YR`: anchored datetimes
- `price_dfs`, `equity_betas`: global caches for pricing and beta estimates

---

### `functions.py`
Implements core analytics:
- Return and volatility calculations
- Beta estimation (total, upside, downside)
- Correlation analysis (holding-to-portfolio, holding-to-sector)
- Portfolio and holding object definitions
- Risk report generation with summary statistics

---

### `holdings.py`
Stores current JSOSIF equity holdings:
- Each holding includes ticker, sector, quantity, and risk tier
- Sector codes: CRL, FIN, HLT, TMT, INR

---

### `monte_carlo.py`
Runs Monte Carlo simulations on the current portfolio:
- Uses historical means and covariances
- Projects portfolio value paths over a user-defined horizon
- Plots distribution of outcomes

---

### `plotting.py`
Includes visualizations for:
- Rolling volatility
- Price and return histories
- Log return distributions

All plots are formatted consistently and can be saved optionally.

---

### `risk_report.py`
Generates a multi-part risk summary for a given ticker:
- Includes beta decomposition, skew/kurtosis, correlation table
- Plots historical performance and volatility
- Uses portfolio-defined sector and weight references

---

### `transactions.py`
Simulates predefined portfolio transactions:
- Includes buy and sell operations via `Transaction` and `Holding` objects
- Updates portfolio state accordingly
- Useful for rebalancing workflows or scenario testing
- Includes optional holding printout for post-simulation sanity checks

---

### `var_estimation.py`
Calculates historical Value-at-Risk (VaR):
- Based on portfolio’s log return distribution
- Uses percentile threshold (e.g., 5%) to estimate expected loss

---

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `yfinance`, `pytz`, `scipy`, `dateutil`

Install using:

```bash
pip install -r requirements.txt
