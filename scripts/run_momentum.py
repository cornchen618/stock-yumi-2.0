"""橫斷面動能組合回測。

主配置（預先登記）：12-1 動能、top 20、band 40、月調倉。
敏感度檢查（一併報告、不挑最好的）：--lookback 126（6-1）、--top 10。

用法：
  python scripts/run_momentum.py --data data/ohlcv.parquet --benchmark data/benchmark/taiex.csv --start 2019-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qts import metrics as M
from qts.data import load_benchmark, load_ohlcv
from qts.momentum import MomentumBacktester, MomentumConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--band", type=int, default=None, help="預設 top×2")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = MomentumConfig(
        lookback=args.lookback, skip=args.skip, top_n=args.top,
        band_rank=args.band if args.band else args.top * 2,
    )
    print(f"載入資料 {args.data} ...")
    data = load_ohlcv(args.data)
    benchmark = load_benchmark(args.benchmark)
    print(f"{len(data)} 檔；配置 lookback={cfg.lookback} skip={cfg.skip} top={cfg.top_n} band={cfg.band_rank}")

    res = MomentumBacktester(cfg).run(data, benchmark, start=args.start, end=args.end)

    eq = M.equity_stats(res.equity, res.benchmark)
    tr = M.trade_stats(res.trades)
    yearly = (res.equity.resample("YE").last() / res.equity.resample("YE").first() - 1) * 100
    b_yearly = (res.benchmark.resample("YE").last() / res.benchmark.resample("YE").first() - 1) * 100

    print("=" * 64)
    print(f"區間 {eq['start']} ~ {eq['end']}   年化換手 {res.annual_turnover:.1f}x   平均曝險 {res.exposure.mean() * 100:.0f}%")
    print(f"總報酬 {eq['total_return'] * 100:+.1f}%   大盤 {eq.get('benchmark_return', np.nan) * 100:+.1f}%")
    print(f"CAGR {eq['cagr'] * 100:+.2f}%   Sharpe {eq['sharpe']:.2f}（大盤 {eq.get('benchmark_sharpe', np.nan):.2f}）   MDD {eq['mdd'] * 100:.1f}%   Calmar {eq['calmar']:.2f}")
    if tr["n_trades"]:
        print(f"交易 {tr['n_trades']} 筆   勝率 {tr['win_rate'] * 100:.1f}%   PF {tr['profit_factor']:.2f}   平均持有 {tr['avg_hold_days']:.0f} 日")
    print("-" * 64)
    print("年度報酬（% ｜ 策略 vs 大盤）：")
    cmp = pd.DataFrame({"strategy": yearly.round(1), "benchmark": b_yearly.round(1)})
    cmp.index = cmp.index.year
    print(cmp.to_string())
    print("=" * 64)

    out = Path(args.out or f"output/momentum_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "equity.csv", header=True)
    if not res.trades.empty:
        res.trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 6))
    (res.equity / res.equity.iloc[0]).plot(ax=ax, label=f"Momentum {cfg.lookback}/{cfg.skip} top{cfg.top_n}")
    (res.benchmark / res.benchmark.iloc[0]).plot(ax=ax, label="TAIEX", alpha=0.7)
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Cross-sectional Momentum (monthly rebalance)")
    fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=120); plt.close(fig)
    print(f"輸出：{out.resolve()}")


if __name__ == "__main__":
    main()
