"""訊號層：策略 A / B / C 的進場訊號與初始停損（STRATEGY.md 第 4~6 節）。

時序規範（不可違反）：
- 訊號一律以 T 日收盤資料計算，欄位值代表「T 日盤後是否成立」。
- 所有「前 N 日極值」一律 shift(1)，不含當日。
- 回測引擎於 T+1 開盤成交。

輸出欄位（附加在個股 DataFrame 上）：
- sig_a / sig_b / sig_c : bool，當日訊號
- trig_a : T1_BO20 / T2_BO10 / T3_RECLAIM（歸因標籤）
- stop_a / stop_b / stop_c : 初始停損價
- rank_score : 候選排序分數（成交量 / VolMA20）
- 以及回測引擎需要的指標欄（ma20, ma60, atr, st_line, st_dir, liquid ...）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from . import indicators as ind


def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """計算所有指標欄位。輸入個股日線（index=date, 含 open/high/low/close/volume/amount）。"""
    out = df.copy()
    c = out["close"]

    out["ma20"] = ind.sma(c, 20)
    out["ma60"] = ind.sma(c, 60)
    out["vol_ma5"] = ind.sma(out["volume"], 5)
    out["vol_ma20"] = ind.sma(out["volume"], 20)
    out["atr"] = ind.atr(out, cfg.atr_period)
    out["rsi"] = ind.rsi(c, cfg.rsi_period)
    out["kd_k"], out["kd_d"] = ind.stochastic_kd(out, cfg.kd_period)
    out["st_line"], out["st_dir"] = ind.supertrend(out, cfg.supertrend_period, cfg.supertrend_mult)

    out["close_pos"] = ind.close_position(out)
    out["upper_shadow"] = ind.upper_shadow_ratio(out)
    out["chg"] = c.pct_change()

    # 前 N 日極值（不含當日）
    out["prev_high20"] = out["high"].rolling(20, min_periods=20).max().shift(1)
    out["prev_high10"] = out["high"].rolling(10, min_periods=10).max().shift(1)
    out["low60"] = out["low"].rolling(60, min_periods=60).min()
    out["high60"] = out["high"].rolling(60, min_periods=60).max()

    # 布林帶寬與其歷史百分位（策略 C）
    bbw = ind.bollinger_bandwidth(c, cfg.strat_c.bbw_window)
    out["bbw"] = bbw
    out["bbw_pctile"] = ind.rolling_percentile_rank(bbw, cfg.strat_c.bbw_pctile_window)

    # 流動性（中位數，避免單日爆量誤判）
    lq = cfg.liquidity
    med_vol = out["volume"].rolling(lq.lookback, min_periods=lq.lookback).median()
    med_amt = out["amount"].rolling(lq.lookback, min_periods=lq.lookback).median()
    out["liquid"] = (
        (med_vol >= lq.vol_min)
        & (med_amt >= lq.amt_min)
        & (c >= lq.min_price)
    )
    return out


def _signal_a(out: pd.DataFrame, cfg: Config) -> None:
    a = cfg.strat_a
    c = out["close"]

    trend = (
        (c > out["ma20"])
        & (out["ma20"] > out["ma60"])
        & (out["ma20"] > out["ma20"].shift(a.ma_slope_days))
    )
    quality = (
        (out["close_pos"] >= a.close_pos_min)
        & (out["upper_shadow"] <= a.upper_shadow_max)
        & (out["chg"] > 0)
        & (out["chg"] <= a.chg_max)
        & (c <= out["ma20"] * a.ext_max)
    )
    vol_bo = out["volume"] >= a.vol_mult_breakout * out["vol_ma20"]
    vol_rc = out["volume"] >= a.vol_mult_reclaim * out["vol_ma20"]

    t1 = (c > out["prev_high20"]) & vol_bo
    t2 = (c > out["prev_high10"]) & (out["close_pos"] >= a.close_pos_t2) & vol_bo & ~t1 & a.enable_t2
    t3 = (
        (c.shift(1) < out["ma20"].shift(1))
        & (c > out["ma20"])
        & (out["close_pos"] >= a.close_pos_t3)
        & vol_rc
        & ~t1 & ~t2
        & a.enable_t3
    )

    base = trend & quality & out["liquid"]
    out["sig_a"] = base & (t1 | t2 | t3)
    trig = np.select(
        [base & t1, base & t2, base & t3],
        ["T1_BO20", "T2_BO10", "T3_RECLAIM"],
        default="",
    )
    out["trig_a"] = trig
    # 初始停損：結構低點與波動停損取較遠者
    out["stop_a"] = np.minimum(out["low"], c - a.atr_stop_mult * out["atr"])


def _signal_b(out: pd.DataFrame, cfg: Config) -> None:
    b = cfg.strat_b
    c = out["close"]

    rng = (out["high60"] - out["low60"]).replace(0.0, np.nan)
    range_pos = (c - out["low60"]) / rng
    low_zone = range_pos <= b.range_pos_max

    # 支撐線 = 前 (offset+window)~offset 根 K 的最低點
    support = out["low"].shift(b.support_offset).rolling(b.support_window, min_periods=b.support_window).min()
    out["support_b"] = support

    # 最近 N 根中至少 M 根最低價跌破「今日的」支撐線
    undercuts = sum(
        (out["low"].shift(k) < support).astype(int) for k in range(b.undercut_lookback)
    )
    washed = undercuts >= b.undercut_min

    reclaim = (c > support) & (c > out["high"].shift(1)) & (out["close_pos"] >= b.close_pos_min)
    vol_sig = out["volume"] >= b.vol_mult * out["vol_ma5"].shift(1)
    knife_guard = (out["ma60"] / out["ma60"].shift(b.ma60_slope_days) - 1.0) > b.ma60_slope_min

    out["sig_b"] = low_zone & washed & reclaim & vol_sig & knife_guard & out["liquid"] & b.enabled
    spring_low = out["low"].rolling(b.undercut_lookback, min_periods=b.undercut_lookback).min()
    out["stop_b"] = spring_low * b.stop_buffer


def _signal_c(out: pd.DataFrame, cfg: Config) -> None:
    cc = cfg.strat_c
    c = out["close"]

    base = (
        (c > out["ma20"])
        & (out["ma20"] > out["ma60"])
        & (out["ma20"] > out["ma20"].shift(cc.ma_slope_days))
    )
    # 壓縮：帶寬百分位優先，資料不足退回固定振幅門檻
    high20 = out["high"].rolling(20, min_periods=20).max()
    low20 = out["low"].rolling(20, min_periods=20).min()
    range_compress = (high20 - low20) / c < cc.range_compress_fallback
    squeeze = out["bbw_pctile"] <= cc.bbw_pctile_max
    compress = squeeze.where(out["bbw_pctile"].notna(), range_compress).astype(bool)

    dryup = out["volume"] < out["vol_ma20"]
    out["watch_c"] = base & compress & dryup & out["liquid"]

    # 進場觸發：近 N 日曾在觀察名單（不含當日）＋突破 20 日高＋量能＋K 棒品質
    a = cfg.strat_a
    was_watched = (
        out["watch_c"].shift(1).rolling(cc.watch_valid_days, min_periods=1).max().fillna(0.0) > 0
    )
    quality = (
        (out["close_pos"] >= a.close_pos_min)
        & (out["upper_shadow"] <= a.upper_shadow_max)
        & (out["chg"] > 0)
        & (out["chg"] <= a.chg_max)
    )
    breakout = (c > out["prev_high20"]) & (out["volume"] >= cc.vol_mult_breakout * out["vol_ma20"])
    out["sig_c"] = was_watched & breakout & quality & out["liquid"]
    out["stop_c"] = np.minimum(out["low"], c - cc.atr_stop_mult * out["atr"])


def compute_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """features + 三策略訊號。資料不足 min_history 時回傳空訊號（但保留指標欄）。"""
    out = compute_features(df, cfg)
    _signal_a(out, cfg)
    _signal_b(out, cfg)
    _signal_c(out, cfg)

    # C 與 A 同日觸發 → 記為 C（歸因分開）
    out.loc[out["sig_c"], "sig_a"] = False

    # 暖機期一律無訊號
    warm = np.arange(len(out)) < cfg.liquidity.min_history
    for col in ("sig_a", "sig_b", "sig_c", "watch_c"):
        out.loc[warm, col] = False

    out["rank_score"] = (out["volume"] / out["vol_ma20"]).fillna(0.0)
    for col in ("sig_a", "sig_b", "sig_c", "watch_c", "liquid"):
        out[col] = out[col].fillna(False).astype(bool)
    return out
