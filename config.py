"""
config.py

Configuration module defining global constants and shared caches
used throughout the financial analysis codebase.

Includes:
- Date constants for consistent time ranges (NOW, ONE_YR, TWO_YRS, FIVE_YRS)
- Shared global dictionaries for caching price data and beta calculations

Dependencies:
- pytz
- datetime
- dateutil.relativedelta

Author: JSOSIF Quant Team  
Verified: 2025-06-27
"""

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

#date constants (timezone-aware UTC)
NOW = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
FIVE_YRS = NOW - relativedelta(years=5)
TWO_YRS = NOW - relativedelta(years=2)
ONE_YR = NOW - relativedelta(years=1)

#shared caches for efficiency
price_dfs = {}     
equity_betas = {}   
