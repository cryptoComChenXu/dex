import os, sys
from re import I
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from functools import partial
from pathlib import Path
import json
from copy import deepcopy
import re, argparse, pytz, logging, logging.handlers
import dolphindb as ddb 
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError

pd.set_option("display.float_format", "{:.10f}".format)

dollar_gas_fees = defaultdict(float)
dollar_gas_fees["UNS"] = 30
dollar_gas_fees["MMS"] = 0.5
dollar_gas_fees["VVS"] = 0.5
dollar_gas_fees["PCS"] = 2
dollar_gas_fees["ORC"] = 0.00025

bps_gas_fees = defaultdict(float)
bps_gas_fees["PCS"] = 5

blocktimes = {
    "UNS": 15,
    "MMS": 6,
    "VVS": 6,
    "PCS": 3,
    "ORC": 1,
}


def setup_logger(name="Logger", level=logging.INFO):
    print(f"Creating logger {name} with level {level} ...")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    FORMAT = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.FileHandler(f"{name}.log")
    ch.setLevel(level)
    logger.addHandler(ch)
    return logger

def get_thresholds(df_thr, symbol: str, hedge: str) -> Tuple[float, float, float]:
    df = df_thr.loc[(df_thr.sym == symbol) & (df_thr.hedge_sym == hedge)]
    if df.empty: df = df_thr.loc[(df_thr.sym == symbol)]
    if df.empty: df = df_thr.loc[(df_thr.sym == "BNBUSDC.PCS")]
    if len(df) > 1: df = df.iloc[:1]

    min_ntl = df["lv1_ntl"].values[0]
    max_ntl = df["lv9_ntl"].values[0]
    entry = df["lv2_entry_bips"].values[0]
    return min_ntl, max_ntl, entry




def get_data_date(sym: str, param: Dict, side: str, start: str, end: str, delay: int = 5, n_levels: int = 20, logger=logging.getLogger("DEX")) -> pd.DataFrame:

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
        sym_adjust_qeury = "adjust = select timestamp, {}1 as adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            side, start, end, symbol_adjust.replace('-', ''))
    elif symbol_adjust!= "":
        print(f"Use {symbol_adjust} to adjust price for {sym}.")
        sym_adjust_qeury = "adjust = select timestamp, 1.0 / {}1 as adjust from depth_table where timestamp>={}, timestamp<{}, symbol='{}' order by timestamp".format(
            oppo_side, start, end, symbol_adjust.replace('-', ''))

    if symbol_adjust != "":
        s.run(sym_adjust_qeury)
        #s.run("t1 = wj(t1, sym_adjust, 0s:120s, <first(adjust) as adjust>, `timestamp)")
        s.run("t1 = aj(t1, adjust, `timestamp)")


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
        # s.run("t2 = wj(t2, hedge_adjust, 0s:120s, <first(hedge_adjust) as hedge_adjust>, `timestamp)")
        s.run("t2 = aj(t2, hedge_adjust, `timestamp)")

    t3  = f"select * from wj(t1, t2, 0s:{delay}s, <["
    for level in range(1, n_levels + 1):
        t3 += f"first({oppo_side}{level}) as {oppo_side}{level}, first({oppo_side}v{level}) as {oppo_side}v{level}, "

    if hedge_adjust != "": t3 += "first(hedge_adjust) as hedge_adjust, "

    t3  = t3[:-2] + "]>, `timestamp)"

    df = s.run(t3)

    if not "adjust" in df.columns: df["adjust"] = 1.0
    if not "hedge_adjust" in df.columns: df["hedge_adjust"] = 1.0

    return df


def ask_match(maxd, n_levels, x):
    y = deepcopy(x)
    tot_vol, tot_pnl = 0., 0.
    for i in range(1, 6):
        if not y[f"a{i}"] * y["adjust"] <= y["b1"] * y["hedge_adjust"]:
            break

        for j in range(1, n_levels + 1):
            if not y[f"a{i}"] * y["adjust"] <= y[f"b{j}"] * y["hedge_adjust"]:
                break

            siz = min(y[f"av{i}"], y[f"bv{j}"])
            vol = siz * (y[f"b{j}"] * y["hedge_adjust"] + y[f"a{i}"] * y["adjust"])
            pnl = siz * (y[f"b{j}"] * y["hedge_adjust"] - y[f"a{i}"] * y["adjust"])

            if tot_vol + vol <= maxd:
                tot_vol += vol
                tot_pnl += pnl

                y[f"av{i}"] -= siz
                y[f"bv{j}"] -= siz
            else:
                vol = maxd - tot_vol
                siz = vol / (y[f"b{j}"] * y["hedge_adjust"] + y[f"a{i}"] * y["adjust"])
                pnl = siz * (y[f"b{j}"] * y["hedge_adjust"] - y[f"a{i}"] * y["adjust"])
                tot_vol = maxd
                tot_pnl += pnl

                y[f"av{i}"] -= siz
                y[f"bv{j}"] -= siz

                break

    return tot_vol, tot_pnl

def bid_match(maxd, n_levels, x):

    y = deepcopy(x)
    tot_vol, tot_pnl = 0., 0.
    for i in range(1, 6):
        if not y[f"b{i}"] * y["adjust"] >= y["a1"] * y["hedge_adjust"]:
            break

        for j in range(1, n_levels + 1):
            if not y[f"b{i}"] * y["adjust"] >= y[f"a{j}"] * y["hedge_adjust"]:
                break

            siz = min(y[f"bv{i}"], y[f"av{j}"])
            vol = siz * (y[f"b{i}"] * y["adjust"] + y[f"a{j}"] * y["hedge_adjust"])
            pnl = siz * (y[f"b{i}"] * y["adjust"] - y[f"a{j}"] * y["hedge_adjust"])

            if tot_vol + vol <= maxd:
                tot_vol += vol
                tot_pnl += pnl

                y[f"bv{i}"] -= siz
                y[f"av{j}"] -= siz
            else:
                vol = maxd - tot_vol
                siz = vol / (y[f"b{i}"] * y["adjust"] + y[f"a{j}"] * y["hedge_adjust"])
                pnl = siz * (y[f"b{i}"] * y["adjust"] - y[f"a{j}"] * y["hedge_adjust"])
                tot_vol = maxd
                tot_pnl += pnl

                y[f"bv{i}"] -= siz
                y[f"av{j}"] -= siz

                break

    return tot_vol, tot_pnl


def calc_dates(start, end):
    starts, ends = [], []
    start = datetime.strptime(f"{start} 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(f"{end} 00:00:00", "%Y-%m-%d %H:%M:%S")
    while start < end:
        starts.append(start.strftime("%Y.%m.%dT%H:%M:%S"))
        start += timedelta(days=1)
        ends.append(start.strftime("%Y.%m.%dT%H:%M:%S"))

    return starts, ends


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DEX", description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rootdir", nargs="?", type=str, default=".", dest="rootdir", help="")
    parser.add_argument("--s3_bucket", nargs="?", type=str, default="xdev-sg-trading-desk-quant-data", dest="s3_bucket", help="")
    parser.add_argument("--s3_folder", nargs="?", type=str, default="sim_dex_ob", dest="s3_folder", help="")
    parser.add_argument("--symbolfilename", nargs="?", type=str, default="symbols_ob_PCS.json", dest="symbolfilename", help="")
    parser.add_argument("--start", nargs="?", type=str, default="2022-07-03", dest="start", help="")
    parser.add_argument("--end", nargs="?", type=str, default="2022-07-05", dest="end", help="")
    parser.add_argument("--blocktimedelay", nargs="?", type=int, default=1, dest="blocktimedelay", help="")

    args = parser.parse_args()

    logger = setup_logger("DEX")
    rootdir = Path(args.rootdir)

    with open(rootdir / args.symbolfilename) as f:
        symbols = json.load(f)

    with open(rootdir / "db_config_defi.json", 'r') as f:
        dbconfig = json.load(f)

    try:
        df_thresholds = pd.read_csv(rootdir / "thresholds.csv")
    except Exception as e:
        logger.error(f"Couldn't load the threshold file!")
        sys.exit(1)


    s = ddb.session()
    s.connect(dbconfig["DolphinURL"], dbconfig["DolphinPort"], dbconfig["DolphinAccount"], dbconfig["DolphinPassword"])


    q2 = 'depth_table = loadTable("dfs://tick_depth", "depths")'
    s.run(q2)

    start_date, end_date = args.start, args.end
    if start_date == "":
        now = datetime.now()
        start_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

    starts, ends = calc_dates(start_date, end_date)

    trades = []

    for symbol in symbols:
        for sym, param in symbol.items():
            _, dex = sym.split('.')

            min_ntl, max_ntl, entry = get_thresholds(df_thresholds, sym.replace('-', ''), param["hedge"].replace('-', ''))
            
            data_L, data_S = [], []
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                logger.info(f"Getting long {sym} data from {start}, {end} ...")
                df_L = get_data_date(sym, param, 'a', start, end, 5, 20, logger)
                logger.info(f"Getting short {sym} data from {start}, {end} ...")
                df_S = get_data_date(sym, param, 'b', start, end, 5, 20, logger)

                if not df_L.empty: data_L.append(df_L)
                if not df_S.empty: data_S.append(df_S)

            if len(data_L) > 0:
                df_L = pd.concat(data_L)
                match_func = partial(ask_match, max_ntl, 20)
                match_L = df_L.apply(match_func, axis=1)
                df_L["trigger_ntl"] = match_L.str[0]
                df_L["trigger_pnl"] = match_L.str[1]
                df_L["trigger_bps"] = df_L["trigger_pnl"].div(df_L["trigger_ntl"]) * 1e4
                df_L_trigger = df_L.loc[(df_L.trigger_ntl >= min_ntl) & (df_L.trigger_bps >= entry)]

                if not df_L_trigger.empty:
                    df_L_trigger.rename(columns={"timestamp": "triggertime"}, inplace=True)
                    df_L_trigger["timestamp"] = df_L_trigger["triggertime"] + pd.Timedelta(seconds=args.blocktimedelay * blocktimes[dex])

                    df_L_trade = pd.merge_asof(df_L_trigger, df_L, on="timestamp", direction="forward", suffixes=("_trigger", ""))
                    trade_L = df_L_trade.apply(match_func, axis=1)
                    df_L_trade["trade_ntl"]  = trade_L.str[0]
                    df_L_trade["trade_pnl"]  = trade_L.str[1]
                    df_L_trade["gross_margin"] = df_L_trade["trade_pnl"].div(df_L_trade["trade_ntl"]) * 1e4
                    df_L_trade["fees"] = df_L_trade["trade_ntl"] * bps_gas_fees[dex] / 1e4 + dollar_gas_fees[dex]
                    df_L_trade["net_margin"] = df_L_trade.apply(lambda x: (x["trade_pnl"] - x["fees"]) / x["trade_ntl"] if x["trade_ntl"] > 0 else np.nan, axis=1) 

                    df_L_trade["symbol"] = sym.replace('-', '')
                    df_L_trade["hedge"] = param["hedge"].replace('-', '')

                    trades.append(df_L_trade[["timestamp", "symbol", "hedge", "trade_ntl", "trade_pnl", "fees", "net_margin", "triggertime", "a1", "av1", "b1", "bv1"]])
                else:
                    logger.error("No L trigger for symbol {}, hedge {}.".format(sym, param["hedge"]))
            else:
                logger.error("No L data for symbol {}, hedge {}.".format(sym, param["hedge"]))


            if len(data_S) > 0:
                df_S = pd.concat(data_S)
                match_func = partial(bid_match, max_ntl, 20)
                match_S = df_S.apply(match_func, axis=1)
                df_S["trigger_ntl"] = match_S.str[0]
                df_S["trigger_pnl"] = match_S.str[1]
                df_S["trigger_bps"] = df_S["trigger_pnl"].div(df_S["trigger_ntl"]) * 1e4
                df_S_trigger = df_S.loc[(df_S.trigger_ntl >= min_ntl) & (df_S.trigger_bps >= entry)]

                if not df_S_trigger.empty:
                    df_S_trigger.rename(columns={"timestamp": "triggertime"}, inplace=True)
                    df_S_trigger["timestamp"] = df_S_trigger["triggertime"] + pd.Timedelta(seconds=args.blocktimedelay * blocktimes[dex])

                    df_S_trade = pd.merge_asof(df_S_trigger, df_S, on="timestamp", direction="forward", suffixes=("_trigger", ""))
                    trade_S = df_S_trade.apply(match_func, axis=1)
                    df_S_trade["trade_ntl"]  = trade_S.str[0]
                    df_S_trade["trade_pnl"]  = trade_S.str[1]
                    df_S_trade["gross_margin"] = df_S_trade["trade_pnl"].div(df_S_trade["trade_ntl"]) * 1e4
                    df_S_trade["fees"] = df_S_trade["trade_ntl"] * bps_gas_fees[dex] / 1e4 + dollar_gas_fees[dex]
                    df_S_trade["net_margin"] = df_S_trade.apply(lambda x: (x["trade_pnl"] - x["fees"]) / x["trade_ntl"] if x["trade_ntl"] > 0 else np.nan, axis=1) 

                    df_S_trade["symbol"] = sym.replace('-', '')
                    df_S_trade["hedge"] = param["hedge"].replace('-', '')

                    trades.append(df_S_trade[["timestamp", "symbol", "hedge", "trade_ntl", "trade_pnl", "fees", "net_margin", "triggertime", "a1", "av1", "b1", "bv1"]])
                else:
                    logger.error("No S trigger for symbol {}, hedge {}.".format(sym, param["hedge"]))
            else:
                logger.error("No S data for symbol {}, hedge {}.".format(sym, param["hedge"]))

    if len(trades) > 0:
        df = pd.concat(trades)

        symbolfilename, _ = args.symbolfilename.split('.')
        tradefilename = "Sim_trades_{}_{}-{}.csv".format(
            symbolfilename, start_date.replace('-', ''), end_date.replace('-', ''))
        df.to_csv(rootdir / tradefilename, index=False)

        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        df_res = df.groupby(["date", "symbol", "hedge"]).agg(
            trigger=pd.NamedAgg(column="trade_ntl", aggfunc=len),
            trade=pd.NamedAgg(column="trade_ntl", aggfunc=len),
            notional=pd.NamedAgg(column="trade_ntl", aggfunc="sum"),
            gross_pnl=pd.NamedAgg(column="trade_pnl", aggfunc="sum"),
            fees=pd.NamedAgg(column="fees", aggfunc="sum"),
            winTrades=pd.NamedAgg(column="net_margin", aggfunc=lambda x: (x > 0).sum())
        )


        df_res["gross_margin"] = df_res["gross_pnl"].div(df_res["notional"]) * 1e4
        df_res["net_margin"] = (df_res["gross_pnl"] - df_res["fees"]).div(df_res["notional"]) * 1e4
        df_res["winRatio"] = df_res["winTrades"].div(df_res["trade"]) * 100.


        summaryfilename = "Sim_summary_{}_{}-{}.csv".format(
            symbolfilename, start_date.replace('-', ''), end_date.replace('-', ''))
        df_res.to_csv(rootdir / summaryfilename, index=True)


        s3_client = boto3.client("s3")
        key = args.s3_folder + '/' + tradefilename
        try:
            response = s3_client.upload_file(str(rootdir / tradefilename), args.s3_bucket, key)
        except ClientError as e:
            logger.error(f"Couldn't upload {tradefilename}, {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(e)
            sys.exit(1)

        key = args.s3_folder + '/' + summaryfilename
        try:
            response = s3_client.upload_file(str(rootdir / summaryfilename), args.s3_bucket, key)
        except ClientError as e:
            logger.error(f"Couldn't upload {summaryfilename}, {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(e)
            sys.exit(1)

        sys.exit(0)