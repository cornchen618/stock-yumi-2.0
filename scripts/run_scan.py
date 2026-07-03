"""每日盤後掃描 CLI（核心邏輯在 qts/scanner.py，與 eod_task 共用）。

用法：
  python scripts/run_scan.py --data data/ohlcv.parquet --benchmark data/benchmark/taiex.csv --equity 1000000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import Config
from qts.data import load_benchmark, load_names, load_ohlcv
from qts.scanner import STRAT_ZH, TRIGGER_ZH, scan


def main() -> None:
    p = argparse.ArgumentParser(description="台股波段量化系統 — 每日盤後掃描")
    p.add_argument("--data", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--equity", type=float, required=True, help="目前帳戶權益（元）")
    p.add_argument("--date", default=None, help="掃描基準日 YYYY-MM-DD（預設最新交易日）")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="輸出 CSV 路徑")
    args = p.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()
    data = load_ohlcv(args.data)
    benchmark = load_benchmark(args.benchmark)
    names = load_names()

    res = scan(data, benchmark, cfg, args.equity, names,
               asof=pd.Timestamp(args.date) if args.date else None)

    print(f"訊號基準日：{res.asof.date()} 收盤（表中價格 = 該日收盤市價）")
    print(f"市場濾網：{'多頭（三策略皆可進場）' if res.bull else '空頭（A/C 停止、B 風險減半）'}")
    print("=" * 76)
    if len(res.candidates):
        show = res.candidates.copy()
        show["strategy"] = show["strategy"].map(STRAT_ZH)
        show["trigger"] = show["trigger"].map(lambda t: TRIGGER_ZH.get(str(t), str(t)))
        print("【明日進場候選】（次一交易日開盤執行；開盤漲幅 >= 9.5% 放棄）")
        print(show.to_string(index=False))
        out = args.out or f"output/scan_{res.asof:%Y%m%d}.csv"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        res.candidates.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已輸出：{Path(out).resolve()}")
    else:
        print("【明日進場候選】無")
    if len(res.watch):
        print("-" * 76)
        print(f"【策略 C 觀察名單】（波動壓縮中，等待突破）共 {len(res.watch)} 檔")
        print(res.watch.to_string(index=False))


if __name__ == "__main__":
    main()
