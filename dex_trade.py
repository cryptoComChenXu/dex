import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from functools import partial
from pathlib import Path
import json


output_dir = Path("capacity_data")
output_dir.mkdir(exist_ok=True, parents=True)


with open("db_config.json", 'r') as f:
    dbconfig = json.load(f)


import dolphindb as ddb 
s = ddb.session()
s.connect(dbconfig["DolphinURL"], dbconfig["DolphinPort"], dbconfig["DolphinAccount"], dbconfig["DolphinPassword"])


q1 = 'trade_table = loadTable("dfs://tick_trade", "trades")'
q2 = 'depth_table = loadTable("dfs://tick_depth", "depths")'
s.run(q1)
s.run(q2)

start_date, end_date = "2022-03-01", "2022-05-25"

with open("symbols.json", 'r') as f:
    symbols = json.load(f)

crossing_pairs = set([
    "USDCUSDT", 
    "DAIUSDT",
    "CROUSDT",
    "CROUSDC",
    "BNBUSDT",
    "BUSDUSDT",
])

wrapped = {
    "BTC": "WBTC",
    "ETH": "WETH",
    "CRO": "WCRO",
}

binance_symbols = set([
    "BUSD",
    "BNB",
])

coins = set([
    "ADA", "ALI", "ALICE", "ATOM", "AVAX", "BIFI", "BNB", "BTC", "C98",
    "CAKE", "CHR", "CRO", "DAI", "DOT", "DOGE", "ELON", "ETH", "LINK",
    "LTC", "MASK", "MATIC", "MMF", "SOL", "SPS", "TONIC", "TRX", "TSUD",
    "UNI", "USDC", "USDT", "VVS", "WBTC", "WCRO", "WETH", "WOO", "XRP",
])

gas_fees = {
    "UNS": 30,
    "MMS": 0.5,
    "VVS": 0.5,
    "PCS": 5,
}

bps_exchanges = set([
    "PCS"
])

delays = {
    "UNS": 15,
    "MMS": 6,
    "VVS": 6,
    "PCS": 3,
}

starts, ends = [], []
start = datetime.strptime(f"{start_date} 00:00:00", "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(f"{end_date} 00:00:00", "%Y-%m-%d %H:%M:%S")
while start < end:
    starts.append(start.strftime("%Y.%m.%dT%H:%M:%S"))
    start += timedelta(days=1)
    ends.append(start.strftime("%Y.%m.%dT%H:%M:%S"))