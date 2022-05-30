import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from functools import partial
from pathlib import Path
import json


output_dir = Path("capacity_data_UNS")
symbol_file = Path("symbols_UNS.json")
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

start_date, end_date = "2022-05-21", "2022-06-01"

with open(symbol_file, 'r') as f:
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


def get_data_date(sym: str, hedge: str, side: str, delay: int, start: str, end: str, n_levels: int = 20) -> pd.DataFrame:
    price_type = 'a' if side == '1' else 'b'
    opposite_side = '-1' if side == '1' else '1'
    opposite_price_type = 'a' if opposite_side == '1' else 'b'
    volume_type = price_type + 'v'
    opposite_volume_type = opposite_price_type + 'v'

    ticker, sym_ext = sym.split('.')
    sym_leg1, sym_leg2 = ticker.split('-')
    hedge_ticker, hedge_ext = hedge.split('.')
    hedge_leg1, hedge_leg2 = hedge_ticker.split('-')

    leg1_equal, leg2_equal = False, False
    if sym_leg1 == hedge_leg1 or ((sym_leg1 in wrapped) and (wrapped[sym_leg1] == hedge_leg1)) \
        or ((hedge_leg1 in wrapped) and (wrapped[hedge_leg1] == sym_leg1)):
        leg1_equal = True
    if sym_leg2 == hedge_leg2 or ((sym_leg2 in wrapped) and (wrapped[sym_leg2] == hedge_leg2)) \
        or ((hedge_leg2 in wrapped) and (wrapped[hedge_leg2] == sym_leg2)):
        leg2_equal = True

    if leg1_equal and leg2_equal:
        print(f"{sym} and {hedge} do not need a middle pair.")

        t1  = "t1 = select timestamp, sum(volume) as volume, wavg(price, volume) as vwap from trade_table"
        t1 += " where timestamp>={}, timestamp<{}, symbol='{}', side={} group by timestamp order by timestamp".format(
            start, end, sym.replace('-', ''), side)

        t2  = "t2 = select timestamp, " + price_type + f", {price_type}".join(map(str, list(range(1, n_levels + 1))))
        t2 += ', '
        t2 += volume_type + f", {volume_type}".join(map(str, list(range(1, n_levels + 1))))
        t2 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}', {}1>0 order by timestamp".format(
            start, end, hedge.replace('-', ''), price_type)

        t3  = "select * from wj(t1, t2, {}s:{}s, <[".format(delay, delay + 1)
        for level in range(1, n_levels + 1):
            t3 += f"first({price_type}{level}) as {price_type}{level}, first({volume_type}{level}) as {volume_type}{level}, "

        t3  = t3[:-2] + "]>, `timestamp)"

        s.run(t1)
        s.run(t2)
        df = s.run(t3)
        df["price"] = 1.0
        return df

    elif not leg1_equal and not leg2_equal:
        if sym_leg1 == hedge_leg2 and sym_leg2 == hedge_leg1:
            print(f"{sym} and {hedge} are inversed.")

            t1  = "t1 = select timestamp, sum(volume) as volume, wavg(price, volume) as vwap from trade_table"
            t1 += " where timestamp>={}, timestamp<{}, symbol='{}', side={} group by timestamp order by timestamp".format(
                start, end, sym.replace('-', ''), opposite_side)

            t2  = "t2 = select timestamp, " + price_type + f", {price_type}".join(map(str, list(range(1, n_levels + 1))))
            t2 += ', '
            t2 += volume_type + f", {volume_type}".join(map(str, list(range(1, n_levels + 1))))
            t2 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}', {}1>0 order by timestamp".format(
                start, end, hedge.replace('-', ''), price_type)

            t3  = "select * from wj(t1, t2, {}s:{}s, <[".format(delay, delay + 1)
            for level in range(1, n_levels + 1):
                t3 += f"first({price_type}{level}) as {price_type}{level}, first({volume_type}{level}) as {volume_type}{level}, "

            t3  = t3[:-2] + "]>, `timestamp)"

            s.run(t1)
            s.run(t2)
            df = s.run(t3)
            df["vwap"] = 1.0 / df["vwap"]
            df["price"] = 1.0
            return df
        else:
            print(f"Couldn't find arb between {sym} and {hedge}.")
            return pd.DataFrame()

    elif leg1_equal:
        print(f"{sym} and {hedge} are leg1_equal: {leg1_equal}.")
        
        cross, reverse_cross = None, None

        if sym_leg2 + hedge_leg2 in crossing_pairs:
            cross = sym_leg2 + hedge_leg2
        elif hedge_leg2 + sym_leg2 in crossing_pairs:
            reverse_cross = hedge_leg2 + sym_leg2

        t1  = "t1 = select timestamp, sum(volume) as volume, wavg(price, volume) as vwap from trade_table"
        t1 += " where timestamp>={}, timestamp<{}, symbol='{}', side={} group by timestamp order by timestamp".format(
            start, end, sym.replace('-', ''), side)

        t2  = "t2 = select timestamp, " + price_type + f", {price_type}".join(map(str, list(range(1, n_levels + 1))))
        t2 += ', '
        t2 += volume_type + f", {volume_type}".join(map(str, list(range(1, n_levels + 1))))
        t2 += " from depth_table where timestamp>={}, timestamp<{}, symbol='{}', {}1>0 order by timestamp".format(
            start, end, hedge.replace('-', ''), price_type)

        t3  = "select * from wj(t1, t2, {}s:{}s, <[".format(delay, delay + 1)
        for level in range(1, n_levels + 1):
            t3 += f"first({price_type}{level}) as {price_type}{level}, first({volume_type}{level}) as {volume_type}{level}, "

        t3  = t3[:-2] + "]>, `timestamp)"

        s.run(t1)

        if cross is None and reverse_cross is None:
            print(f"{sym} and {hedge} do not need a middle pair.")
        else:
            if cross:
                cross_ext = hedge_ext
                if cross.startswith("BNB") or cross.startswith("BUSD"):
                    cross_ext = "BNC"
                print(f"{sym} and {hedge} using crossing pair {cross}.{cross_ext}, side {side}, price type {opposite_price_type}")
                s.run("cross_tbl = select timestamp, {} as price from depth_table where symbol='{}', timestamp>={}, timestamp<{};".format(
                    opposite_price_type+'1', cross + '.' + cross_ext,  start, end))
            else:
                reverse_cross_ext = hedge_ext
                if reverse_cross.startswith("BNB") or reverse_cross.startswith("BUSD"):
                    reverse_cross_ext = "BNC"
                print(f"{sym} and {hedge} using reverse crossing pair {reverse_cross}.{reverse_cross_ext}, side {side} price type 1 / {price_type}")
                s.run("cross_tbl = select timestamp, {} as price from depth_table where symbol='{}', timestamp>={}, timestamp<{};".format(
                    f"1.0 / {price_type}1", reverse_cross + '.' + reverse_cross_ext, start, end))
            s.run("t1 = wj(t1, cross_tbl, 0s:120s, <first(price) as price>, `timestamp)");

        s.run(t2)
        df = s.run(t3)

        return df
    
    else:
        print(f"{sym} and {hedge} are leg2_equal: {leg2_equal}.")
        return pd.DataFrame()

def get_data(sym: str, hedge: str, side: str, delay: int, starts: List, ends: List) -> pd.DataFrame:
    n = len(starts)
    if not n > 0: return pd.DataFrame()

    res = []
    for i in range(n):
        print(f"Loading data {sym} from {starts[i]}")
        df_tmp = get_data_date(sym, hedge, side, delay, starts[i], ends[i])
        if not df_tmp.empty:
            res.append(df_tmp)

    if len(res) > 0:
        return pd.concat(res)

    return pd.DataFrame()

def buy_match(exch, fee, n_levels, x):
    res, pnl = 0, 0
    tot = x["volume"]
    price = x["vwap"] * x["price"]
    for i in range(1, n_levels + 1):
        if not x[f"a{i}"] > 0 or price < x[f"a{i}"] or tot <= 0:
            break

        vol = min(tot, x[f"av{i}"])
        res += vol * x[f"a{i}"]
        pnl += vol * (price - x[f"a{i}"])
        tot -= vol

    if exch in bps_exchanges:
        print(f"bps exchange {exch}")
        fee = res * fee / 1e4

    if pnl < fee:
        return 0, 0, 0

    pnl -= fee 

    return res, pnl, fee

def sell_match(exch, fee, n_levels, x):
    res, pnl = 0, 0
    tot = x["volume"]
    price = x["vwap"] * x["price"]
    for i in range(1, n_levels + 1):
        if not x[f"b{i}"] > 0 or price > x[f"b{i}"] or tot <= 0:
            break

        vol = min(tot, x[f"bv{i}"])
        res += vol * x[f"b{i}"]
        pnl += vol * (x[f"b{i}"] - price)
        tot -= vol

    if exch in bps_exchanges:
        print(f"bps exchange {exch}")
        fee = res * fee / 1e4

    if pnl < fee:
        return 0, 0, 0

    pnl -= fee

    return res, pnl, fee

results = []
trades_L = []
trades_S = []
for sym, hedges in symbols.items():
    _, exch = sym.split('.')
    gas = gas_fees[exch]
    delay = delays[exch]
    print(f"delay: {delay}, symbol: {sym}")
    for hedge in hedges:
        df_L = get_data(sym, hedge, '1', delay, starts, ends)

        if not df_L.empty:
            df_L = df_L.loc[(df_L["vwap"] > 0) & (df_L["volume"] > 0) & (df_L["price"] > 0)]
            if not df_L.empty:
                buy_func = partial(buy_match, exch, gas, 20)
                buys = df_L.apply(buy_func, axis=1)
                df_L["potential_match"] = buys.str[0]
                df_L["potential_pnl"] = buys.str[1]
                df_L["fee"] = buys.str[2]
                df_L["timestamp"] = pd.to_datetime(df_L["timestamp"])
                df_L.set_index("timestamp", inplace=True)
                df_L["symbol"] = sym.replace('-', '')
                df_L["hedge"] = hedge.replace('-', '')
                df_L["delay"] = delay
                trades_L.append(df_L)

        df_S = get_data(sym, hedge, '-1', delay, starts, ends)

        if not df_S.empty:
            df_S = df_S.loc[(df_S["vwap"] > 0) & (df_S["volume"] > 0) & (df_S["price"] > 0)]
            if not df_S.empty:
                sell_func = partial(sell_match, exch, gas, 20)
                sells = df_S.apply(sell_func, axis=1)

                df_S["potential_match"] = sells.str[0]
                df_S["potential_pnl"] = sells.str[1]
                df_S["fee"] = sells.str[2]
                df_S["timestamp"] = pd.to_datetime(df_S["timestamp"])
                df_S.set_index("timestamp", inplace=True)
                df_S["symbol"] = sym.replace('-', '')
                df_S["hedge"] = hedge.replace('-', '')
                df_S["delay"] = delay
                trades_S.append(df_S)

        d = {}
        if len(df_L) > 0:
            d["LongMatch"] = df_L["potential_match"].resample("1D").sum()
            d["LongPnL"] = df_L["potential_pnl"].resample("1D").sum()
            d["LongFee"] = df_L["fee"].resample("1D").sum()

        if len(df_S) > 0:
            d["ShortMatch"] = df_S["potential_match"].resample("1D").sum()
            d["ShortPnL"] = df_S["potential_pnl"].resample("1D").sum()
            d["ShortFee"] = df_S["fee"].resample("1D").sum()

        df = pd.DataFrame(d)

        df["symbol"] = sym.replace('-', '')
        df["hedge"] = hedge.replace('-', '')
        df["delay"] = delay

        results.append(df)

        output_file = output_dir / "{}-{}.csv".format(sym.replace('-', ''), hedge.replace('-', ''))
        df.to_csv(output_file, index=True)

df = pd.concat(results)

if len(trades_L) > 0: 
    df_L_tot = pd.concat(trades_L)
    df_L_tot.to_csv(output_dir / "trades_L.csv")
if len(trades_S) > 0: 
    df_S_tot = pd.concat(trades_S)
    df_S_tot.to_csv(output_dir / "trades_S.csv")


df2 = df.reset_index()

df_sum = pd.DataFrame({
    "LongMatch": df2.groupby(["symbol", "hedge"])["LongMatch"].sum(),
    "ShortMatch": df2.groupby(["symbol", "hedge"])["ShortMatch"].sum(),
    "LongPnL": df2.groupby(["symbol", "hedge"])["LongPnL"].sum(),
    "LongFee": df2.groupby(["symbol", "hedge"])["LongFee"].sum(),
    "ShortPnL": df2.groupby(["symbol", "hedge"])["ShortPnL"].sum(),
    "StartDate": df2.groupby(["symbol", "hedge"])["timestamp"].min(),
    "ShortFee": df2.groupby(["symbol", "hedge"])["ShortFee"].sum(),
})

df_sum["PnL"] = df_sum["LongPnL"] + df_sum["ShortPnL"]
df_sum["Fee"] = df_sum["LongFee"] + df_sum["ShortFee"]
df_sum["NTL"] = df_sum["LongMatch"] + df_sum["ShortMatch"]
df_sum["Margin"] = df_sum["PnL"].div(df_sum["NTL"])

df_sum.to_csv(output_dir / "capacity.csv", index=True)