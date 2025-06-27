# == holdings.py

from functions import Holding
from functions import get_price_df

# =============================================================================== #
# ===================== OFFICIAL JSOSIF PORTFOLIO HOLDINGS ====================== #
# =============================================================================== #

"""
Sectors:
* CRL: Consumer and Retail
* FIG: Financial Intsitutions Groups and Fixed Income
* FIN: Financials
* HLT: Healthcare
* TMT: Technology, Media, and Telecommunications
* INR: Industirals and Real Estate
"""

get_price_df("USDCAD=X",fx_adj=False)
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