"""
holdings.py

Defines the official portfolio holdings for the John Simpson Odette Student Investment Fund.

Each holding is represented by a `Holding` object with attributes:
- ticker: The stock ticker symbol.
- sector: Sector classification code.
- quantity: Number of shares held.
- qualitative risk tier: Risk rating on a scale (1-3).

Sector codes:
- CRL: Consumer and Retail
- FIN: Financials
- HLT: Healthcare
- TMT: Technology, Media, and Telecommunications
- INR: Industrials and Real Estate

The file initializes the holdings list and ensures relevant price data
(for USD/CAD FX rate) is loaded for currency adjustments where applicable.

Dependencies:
- functions.py (for the Holding class and price data utilities)

Author: JSOSIF Quant Team
Verified: 2025-06-27
"""

from functions import Holding, get_price_df

get_price_df("USDCAD=X", fx_adj=False) # Preload FX data for CAD conversion (no FX adjustment)

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
