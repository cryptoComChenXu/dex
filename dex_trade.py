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