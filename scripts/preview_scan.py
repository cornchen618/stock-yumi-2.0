"""盤中預掃（排程於交易日 13:05）。

用 Yahoo 盤中資料組出「今日進行中的日 K」，成交量以台股盤中累積量曲線推估全日量，
接上歷史資料跑正式訊號邏輯，提前發出「今日收盤可能觸發」的候選清單。

重要：這是**預估**——收盤前價格與量能都可能變化，正式訊號以 17:40 盤後掃描為準。
量能推估曲線（台股 09:00-13:30，含收盤集合競價約占全日 10%）：
  10:00≈35%、11:00≈52%、12:00≈65%、13:00≈80%、13:25≈88%、收盤=100%
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.config import Config
from qts.data import load_benchmark, load_names, load_ohlcv
from qts.notify import send
from qts.scanner import format_scan_discord, scan

# 盤中累積量占比曲線（分鐘 → 累積比例），線性內插
_CURVE = [(0, 0.08), (30, 0.22), (60, 0.35), (120, 0.52), (180, 0.65),
          (240, 0.80), (265, 0.88), (270, 1.00)]


def vol_progress(now: datetime) -> float:
    minutes = (now.hour - 9) * 60 + now.minute
    minutes = max(1, min(270, minutes))
    xs, ys = zip(*_CURVE)
    return float(np.interp(minutes, xs, ys))


def fetch_today_bars(symbols: list[str], sfx: dict[str, str]) -> pd.DataFrame:
    """今日進行中的日 K（open/high/low/close/volume），index=symbol。"""
    frames = []
    tickers = {f"{s}.{sfx.get(s, 'TW')}": s for s in symbols}
    keys = list(tickers)
    for i in range(0, len(keys), 150):
        chunk = keys[i:i + 150]
        raw = yf.download(chunk, period="2d", interval="1d", progress=False,
                          auto_adjust=False, group_by="ticker", threads=True)
        if raw is None or raw.empty:
            continue
        today = pd.Timestamp(datetime.now().date())
        for tkr in chunk:
            try:
                sub = raw[tkr] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.dropna(subset=["Close"])
            if not len(sub) or pd.Timestamp(sub.index[-1].date()) != today:
                continue
            r = sub.iloc[-1]
            if r["Volume"] <= 0:
                continue
            frames.append({"symbol": tickers[tkr], "open": r["Open"], "high": r["High"],
                           "low": r["Low"], "close": r["Close"], "volume": r["Volume"]})
    return pd.DataFrame(frames).set_index("symbol") if frames else pd.DataFrame()


def main() -> None:
    now = datetime.now()
    if now.weekday() >= 5:
        return
    equity = 1_000_000.0
    sp = ROOT / "settings.json"
    if sp.exists():
        equity = float(json.loads(sp.read_text(encoding="utf-8")).get("equity", equity))

    data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")
    names = load_names(ROOT / "data" / "universe.csv")
    uni = pd.read_csv(ROOT / "data" / "universe.csv", dtype=str)
    sfx = dict(zip(uni["code"], uni["suffix"]))

    # 流動性子集（用最近一日快照，減少盤中抓取量）
    last_day = max(df.index.max() for df in data.values())
    liquid = []
    for s, df in data.items():
        t = df.tail(20)
        if len(t) >= 20 and t["volume"].median() >= 3e5 and t["amount"].median() >= 2e7:
            liquid.append(s)

    today_bars = fetch_today_bars(liquid, sfx)
    if len(today_bars) < len(liquid) * 0.3:
        send(":satellite: 盤中預掃：今日盤中資料不足（可能休市），跳過。")
        return

    prog = vol_progress(now)
    today = pd.Timestamp(now.date())
    data2 = {}
    for s in today_bars.index:
        hist = data[s]
        if hist.index.max() >= today:
            hist = hist[hist.index < today]
        b = today_bars.loc[s]
        est_vol = float(b["volume"]) / prog
        row = pd.DataFrame({
            "open": [b["open"]], "high": [b["high"]], "low": [b["low"]],
            "close": [b["close"]], "volume": [est_vol],
            "amount": [b["close"] * est_vol], "raw_close": [b["close"]],
        }, index=[today])
        data2[s] = pd.concat([hist, row])

    # 大盤今日 bar（濾網用）
    twii = yf.download("^TWII", period="2d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.get_level_values(0)
    bench2 = benchmark[benchmark.index < today]
    if len(twii) and pd.Timestamp(twii.index[-1].date()) == today:
        r = twii.iloc[-1]
        bench2 = pd.concat([bench2, pd.DataFrame(
            {"open": [r["Open"]], "high": [r["High"]], "low": [r["Low"]],
             "close": [r["Close"]], "volume": [r["Volume"]]}, index=[today])])

    res = scan(data2, bench2, Config(), equity, names, asof=today)
    intro = (f":satellite: **盤中預掃 {now:%H:%M}**（量能以進度 {prog * 100:.0f}% 推估全日量）\n"
             f"_預估性質：收盤前價量可能變化，正式訊號以 17:40 盤後掃描為準_")
    send(intro)
    send(format_scan_discord(res))
    print(f"preview sent: {len(res.candidates)} candidates from {len(data2)} symbols")


if __name__ == "__main__":
    main()
