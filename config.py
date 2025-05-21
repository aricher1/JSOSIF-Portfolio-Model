# === config.py

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Date constants
NOW = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
FIVE_YRS = NOW - relativedelta(years=5)
TWO_YRS = NOW - relativedelta(years=2)
ONE_YR = NOW - relativedelta(years=1)

# Shared globals
price_dfs = {}
equity_betas = {}