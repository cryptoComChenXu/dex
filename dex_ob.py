import os, sys
from re import I
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from functools import partial
from pathlib import Path
import json
from copy import deepcopy


output_dir = Path("capacity_data_ob_PCS")
output_dir.mkdir(exist_ok=True, parents=True)

with open("symbols_ob_PCS.json") as f:
    symbols = json.load(f)

with open("db_config_defi.json", 'r') as f:
    dbconfig = json.load(f)


import dolphindb as ddb 
s = ddb.session()
s.connect(dbconfig["DolphinURL"], dbconfig["DolphinPort"], dbconfig["DolphinAccount"], dbconfig["DolphinPassword"])


q2 = 'depth_table = loadTable("dfs://tick_depth", "depths")'
s.run(q2)

start_date, end_date = "2022-06-29", "2022-07-01"

gas_fees = {
    "UNS": 30,
    "MMS": 0.5,
    "VVS": 0.5,
    "PCS": 5,
    "ORC": 0.00025,
}

bps_exchanges = set([
    "PCS"
])

delays = {
    "UNS": 15,
    "MMS": 6,
    "VVS": 6,
    "PCS": 3,
    "ORC": 1,
}

starts, ends = [], []
start = datetime.strptime(f"{start_date} 00:00:00", "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(f"{end_date} 00:00:00", "%Y-%m-%d %H:%M:%S")
while start < end:
    starts.append(start.strftime("%Y.%m.%dT%H:%M:%S"))
    start += timedelta(days=1)
    ends.append(start.strftime("%Y.%m.%dT%H:%M:%S"))


def get_data_date(sym: str, param: Dict, side: str, delay: int, start: str, end: str, n_levels: int = 20) -> pd.DataFrame:
    ticker, sym_ext = sym.split('.')
    sym_leg1, sym_leg2 = ticker.split('-')
    hedge = param["hedge"]
    hedge_ticker, hedge_ext = hedge.split('.')
    hedge_leg1, hedge_leg2 = hedge_ticker.split('-')

    symbol_adjust = param["symbol_adjust"]
    hedge_adjust = param["hedge_adjust"]
    symbol_reverse = param["symbol_reverse"]
    symbol_adjust_reverse = param["symbol_adjust_reverse"]
    hedge_adjust_reverse = param["hedge_adjust_reverse"]



    oppo_side = 'a' if side == 'b' else 'b'

    t1 = f"t1 = select timestamp, {side}" + f", {side}".join(map(str, range(1, 6))) 
    t1 += f", {side}v"
    t1 += f", {side}v".join(map(str, range(1, 6)))
    t1 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}', {}v1 < {}v2 < {}v3 order by timestamp".format(
        start, end, sym.replace('-', ''), side, side, side)

    if symbol_reverse:
        t1 = "t1 = select timestamp"
        for i in range(1, n_levels + 1):
            t1 += f", 1.0 / {oppo_side}{i} as {side}{i}, {oppo_side}v{i} * {oppo_side}{i} as {side}v{i}"
        t1 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            start, end, sym.replace('-', ''))


    s.run(t1)

    if symbol_adjust != "" and not symbol_adjust_reverse:
        print(f"Use {symbol_adjust} to adjust price for {sym}.")
        sym_adjust_qeury = "sym_adjust = select timestamp, {}1 as adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            side, start, end, symbol_adjust.replace('-', ''))
    elif symbol_adjust!= "":
        print(f"Use {symbol_adjust} to adjust price for {sym}.")
        sym_adjust_qeury = "sym_adjust = select timestamp, 1.0 / {}1 as adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            oppo_side, start, end, symbol_adjust.replace('-', ''))

    if symbol_adjust != "":
        s.run(sym_adjust_qeury)
        s.run("t1 = wj(t1, sym_adjust, 0s:120s, <first(adjust) as adjust>, `timestamp)")


    t2 = f"t2 = select timestamp, {oppo_side}" + f", {oppo_side}".join(map(str, range(1, n_levels + 1))) 
    t2 += f", {oppo_side}v"
    t2 += f", {oppo_side}v".join(map(str, range(1, n_levels + 1)))
    t2 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
        start, end, hedge.replace('-', ''))
    s.run(t2)

    if hedge_adjust != "" and not hedge_adjust_reverse:
        print(f"Use {hedge_adjust} to adjust price for {hedge}.")
        hedge_adjust_qeury = "hedge_adjust = select timestamp, {}1 as hedge_adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            oppo_side, start, end, hedge_adjust.replace('-', ''))
    elif hedge_adjust != "":
        print(f"Use {hedge_adjust} to adjust price for {hedge}.")
        hedge_adjust_qeury = "hedge_adjust = select timestamp, 1.0 / {}1 as hedge_adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            side, start, end, hedge_adjust.replace('-', ''))

    if hedge_adjust != "":
        s.run(hedge_adjust_qeury)
        s.run("t2 = wj(t2, hedge_adjust, 0s:120s, <first(hedge_adjust) as hedge_adjust>, `timestamp)")

    if hedge_adjust == "":
        s.run("trigger = wj(t1, t2, -5s:0s, <last({}1) as trigger_{}1>, `timestamp)".format(oppo_side, oppo_side))
    else:
        s.run("trigger = wj(t1, t2, -5s:0s, <[last({}1) as trigger_{}1, last(hedge_adjust) as trigger_hedge_adjust]>, `timestamp)".format(
            oppo_side, oppo_side))
    
    t3  = "select * from wj(trigger, t2, {}s:{}s, <[".format(delay, delay + 5)
    for level in range(1, n_levels + 1):
        t3 += f"first({oppo_side}{level}) as {oppo_side}{level}, first({oppo_side}v{level}) as {oppo_side}v{level}, "

    if hedge_adjust != "": t3 += "first(hedge_adjust) as hedge_adjust, "

    t3  = t3[:-2] + "]>, `timestamp)"

    df = s.run(t3)

    if not "adjust" in df.columns: df["adjust"] = 1.0
    if not "hedge_adjust" in df.columns: df["hedge_adjust"] = 1.0
    if not "trigger_hedge_adjust" in df.columns: df["trigger_hedge_adjust"] = 1.0


    return df


def get_data(sym: str, param: Dict, side: str, delay: int, starts: List, ends: List) -> pd.DataFrame:
    n = len(starts)
    if not n > 0: return pd.DataFrame()

    res = []
    for i in range(n):
        print(f"Loading data {sym} from {starts[i]}")
        df_tmp = get_data_date(sym, param, side, delay, starts[i], ends[i])
        if not df_tmp.empty:
            res.append(df_tmp)
        else:
            print(f"Empty {side} data on date {starts[i]} to {ends[i]}")

    if len(res) > 0:
        return pd.concat(res)

    return pd.DataFrame()


def ask_match(exch, fee, n_levels, x):
    y = deepcopy(x)
    tot_siz, tot_vol, tot_pnl = 0., 0., 0.
    for i in range(1, 6):
        if not y[f"a{i}"] * y["adjust"] <= y["b1"] * y["hedge_adjust"]:
            break

        for j in range(1, n_levels + 1):
            if not y[f"a{i}"] * y["adjust"] <= y[f"b{j}"] * y["hedge_adjust"]:
                break

            siz = min(y[f"av{i}"], y[f"bv{j}"])
            vol = siz * y[f"a{i}"] * y["adjust"]
            pnl = siz * (y[f"b{j}"] * y["hedge_adjust"] - y[f"a{i}"] * y["adjust"])

            tot_siz += siz
            tot_vol += vol
            tot_pnl += pnl

            y[f"av{i}"] -= siz
            y[f"bv{j}"] -= siz

    if exch in bps_exchanges:
        fee = tot_vol * fee / 1e4

    if tot_pnl < fee:
        return 0, 0, 0

    return tot_vol, tot_pnl, fee


def bid_match(exch, fee, n_levels, x):

    y = deepcopy(x)
    tot_siz, tot_vol, tot_pnl = 0., 0., 0.
    for i in range(1, 6):
        if not y[f"b{i}"] * y["adjust"] >= y["a1"] * y["hedge_adjust"]:
            break

        for j in range(1, n_levels + 1):
            if not y[f"b{i}"] * y["adjust"] >= y[f"a{j}"] * y["hedge_adjust"]:
                break

            siz = min(y[f"bv{i}"], y[f"av{j}"])
            vol = siz * y[f"b{i}"] * y["adjust"]
            pnl = siz * (y[f"b{i}"] * y["adjust"] - y[f"a{j}"] * y["hedge_adjust"])

            tot_siz += siz
            tot_vol += vol
            tot_pnl += pnl

            y[f"bv{i}"] -= siz
            y[f"av{j}"] -= siz

    if exch in bps_exchanges:
        fee = tot_vol * fee / 1e4

    if tot_pnl < fee:
        return 0, 0, 0

    return tot_vol, tot_pnl, fee


pnls = []

for symbol in symbols:
    for sym, param in symbol.items():
        _, exch = sym.split('.')
        gas = gas_fees[exch]
        delay = delays[exch]
        print(f"delay: {delay}, symbol: {sym}")
        df_L = get_data(sym, param, 'a', delay, starts, ends)
        df_S = get_data(sym, param, 'b', delay, starts, ends)

        if not df_L.empty:
            df_L_trigger = df_L.loc[df_L.a1 * df_L.adjust < df_L.trigger_b1 * df_L.trigger_hedge_adjust]
            if not df_L_trigger.empty:
                buy_func = partial(ask_match, exch, gas, 20)
                buys = df_L_trigger.apply(buy_func, axis=1)
                df_L_trigger["notional"] = buys.str[0]
                df_L_trigger["pnl"] = buys.str[1]
                df_L_trigger["fee"] = buys.str[2]
                df_L_trigger["timestamp"] = pd.to_datetime(df_L_trigger["timestamp"])
                df_L_trigger.set_index("timestamp", inplace=True)
                df_L_trade = df_L_trigger.loc[df_L_trigger.pnl > 0]

                pnl_L = df_L_trigger.resample("1D", closed="left", label="left")[["notional", "pnl", "fee"]].sum()
                n_trigger_L = df_L_trigger.resample("1D", closed="left", label="left")["a1"].count()
                pnl_L["trigger"] = n_trigger_L

                if df_L_trade.empty:
                    pnl_L["trade"] = 0
                else:
                    n_trade_L = df_L_trade.resample("1D", closed="left", label="left")["a1"].count()
                    pnl_L["trade"] = n_trade_L

                pnl_L["symbol"] = sym.replace('-', '')
                pnl_L["hedge"] = param["hedge"].replace('-', '')

                pnls.append(pnl_L.reset_index())

        if not df_S.empty:
            df_S_trigger = df_S.loc[df_S.b1 * df_S.adjust > df_S.trigger_a1 * df_S.trigger_hedge_adjust]
            if not df_S_trigger.empty:
                sell_func = partial(bid_match, exch, gas, 20)
                sells = df_S_trigger.apply(sell_func, axis=1)
                df_S_trigger["notional"] = sells.str[0]
                df_S_trigger["pnl"] = sells.str[1]
                df_S_trigger["fee"] = sells.str[2]
                df_S_trigger["timestamp"] = pd.to_datetime(df_S_trigger["timestamp"])
                df_S_trigger.set_index("timestamp", inplace=True)
                df_S_trade = df_S_trigger.loc[df_S_trigger.pnl > 0]
                pnl_S = df_S_trigger.resample("1D", closed="left", label="left")[["notional", "pnl", "fee"]].sum()
                n_trigger_S = df_S_trigger.resample("1D", closed="left", label="left")["a1"].count()
                pnl_S["trigger"] = n_trigger_S

                if df_S_trade.empty:
                    pnl_S["trade"] = 0
                else:
                    n_trade_S = df_S_trade.resample("1D", closed="left", label="left")["a1"].count()
                    pnl_S["trade"] = n_trade_S

                pnl_S["symbol"] = sym.replace('-', '')
                pnl_S["hedge"] = param["hedge"].replace('-', '')

                pnls.append(pnl_S.reset_index())


if len(pnls) > 0:
    df = pd.concat(pnls)
    df_res = df.groupby(["timestamp", "symbol", "hedge"])[["trigger", "trade", "notional", "pnl", "fee"]].sum()
    df_res["margin (bps)"] = df_res["pnl"].div(df_res["notional"]) * 1e4