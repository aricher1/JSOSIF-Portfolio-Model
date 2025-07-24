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
- INR: Industrials and Natural Resources

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
    Holding("LVMHF", "CRL", 5, 2), #Verified: June 2025
    Holding("PEP", "CRL", 34, 3), #Verified: June 2025
    Holding("JWEL.TO", "CRL", 227, 2), #Verified: June 2025
    Holding("GIS", "CRL", 56, 2), #Verified: June 2025
    Holding("COST", "CRL", 9, 3), #Verified: June 2025
    Holding("ATD.TO", "CRL", 111, 3), #Verified: June 2025

    Holding("V", "FIN", 20, 3), #Verified: June 2025
    Holding("SCHW", "FIN", 60, 2), #Verified: June 2025
    Holding("JPM", "FIN", 29, 3), #Verified: June 2025
    Holding("BMO.TO", "FIN", 53, 3), #Verified: June 2025

    Holding("ACN", "TMT", 26, 3), #Verified: June 2025
    Holding("CSCO", "TMT", 85, 2), #Verified: June 2025
    Holding("OTEX.TO", "TMT", 110, 2), #Verified: June 2025
    Holding("DIS", "TMT", 30, 1), #Verified: June 2025

    Holding("PFE", "HLT", 230, 1), #Verified: June 2025
    Holding("VRTX", "HLT", 14, 2), #Verified: June 2025
    Holding("NVO", "HLT", 40, 2), #Verified: June 2025

    Holding("ENB.TO", "INR", 143, 2), #Verified: June 2025
    Holding("CNQ.TO", "INR", 225, 1), #Verified: June 2025
    Holding("J", "INR", 37, 3), #Verified: June 2025
    Holding("MG.TO", "INR", 41, 2), #Verified: June 2025
    Holding("XYL", "INR", 49, 2), #Verified: June 2025
    Holding("CP.TO", "INR", 75, 3), #Verified: June 2025
    Holding("NTR.TO", "INR", 60, 2), #Verified: June 2025
    Holding("AMTM", "INR", 39, 2) #Verified: June 2025
]