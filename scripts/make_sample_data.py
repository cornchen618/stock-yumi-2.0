"""合成資料產生器 — 僅用於驗證程式管線可執行。

合成資料上的回測績效【沒有任何意義】，不代表策略在真實市場的表現。
真實驗證請依 STRATEGY.md 第 2 節提供實際台股資料。

輸出：
  data/sample/ohlcv.parquet   （100 檔合成個股日線，含 symbol 欄）
  data/sample/benchmark.csv   （合成大盤指數）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 20260702


def gen_market(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """市場因子日報酬：慢速多空循環 + 噪音。"""
    n = len(dates)
    t = np.arange(n)
    drift = 0.0006 + 0.0009 * np.sin(2 * np.pi * t / 500.0)   # 多空循環
    vol = 0.010 + 0.004 * (np.sin(2 * np.pi * t / 750.0 + 1.0) > 0)
    r = drift + vol * rng.standard_normal(n)
    return pd.Series(r, index=dates)


def gen_benchmark(dates: pd.DatetimeIndex, mkt_ret: pd.Series) -> pd.DataFrame:
    close = 15000.0 * (1.0 + mkt_ret).cumprod()
    open_ = close.shift(1).fillna(15000.0)
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return pd.DataFrame({
        "date": dates, "open": open_.round(2), "high": high.round(2),
        "low": low.round(2), "close": close.round(2),
        "volume": 3_000_000_000,
    })


def gen_stock(sym: str, dates: pd.DatetimeIndex, mkt_ret: pd.Series,
              rng: np.random.Generator, illiquid: bool) -> pd.DataFrame:
    n = len(dates)
    beta = rng.uniform(0.6, 1.5)
    idio = rng.uniform(0.012, 0.030)
    jumps = (rng.random(n) < 0.01) * rng.normal(0.0, 0.05, n)   # 偶發跳動
    r = beta * mkt_ret.to_numpy() + idio * rng.standard_normal(n) + jumps
    r = np.clip(r, -0.099, 0.099)                               # 台股漲跌停

    p0 = rng.uniform(15, 300)
    close = p0 * np.cumprod(1.0 + r)
    prev_close = np.concatenate([[p0], close[:-1]])
    gap = np.clip(rng.normal(0.0, 0.004, n), -0.05, 0.05)
    open_ = np.clip(prev_close * (1.0 + gap), prev_close * 0.901, prev_close * 1.099)
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = np.minimum(body_hi * (1.0 + np.abs(rng.normal(0, 0.006, n))), prev_close * 1.10)
    low = np.maximum(body_lo * (1.0 - np.abs(rng.normal(0, 0.006, n))), prev_close * 0.90)
    high = np.maximum(high, body_hi)
    low = np.minimum(low, body_lo)

    base_vol = rng.uniform(0.15, 8.0) * 1e6 * (0.05 if illiquid else 1.0)
    volume = (base_vol * np.exp(0.6 * rng.standard_normal(n)) * (1.0 + 8.0 * np.abs(r))).astype(np.int64)
    volume = np.maximum(volume, 1000) // 1000 * 1000

    return pd.DataFrame({
        "date": dates, "symbol": sym,
        "open": np.round(open_, 2), "high": np.round(high, 2),
        "low": np.round(low, 2), "close": np.round(close, 2),
        "volume": volume, "amount": np.round(close * volume, 0),
    })


def generate(n_stocks: int = 100, start: str = "2021-01-04", end: str = "2026-06-30",
             seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    mkt = gen_market(dates, rng)
    bench = gen_benchmark(dates, mkt)
    stocks = []
    for k in range(n_stocks):
        sym = f"{1000 + k}"
        illiquid = k % 5 == 4   # 20% 低流動性，驗證流動性門檻
        stocks.append(gen_stock(sym, dates, mkt, rng, illiquid))
    return pd.concat(stocks, ignore_index=True), bench


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv, bench = generate()
    ohlcv.to_parquet(out_dir / "ohlcv.parquet", index=False)
    bench.to_csv(out_dir / "benchmark.csv", index=False)
    print(f"已產生合成資料（僅供管線驗證，績效無意義）：")
    print(f"  {out_dir / 'ohlcv.parquet'}  （{ohlcv['symbol'].nunique()} 檔 × {len(bench)} 日）")
    print(f"  {out_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
