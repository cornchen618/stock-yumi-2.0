"""回測主程式。

用法：
  python scripts/run_backtest.py --data data/ohlcv --benchmark data/benchmark/taiex.csv
  python scripts/run_backtest.py --data data/sample/ohlcv.parquet --benchmark data/sample/benchmark.csv --start 2022-01-01
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.backtest import Backtester
from qts.config import Config
from qts.data import load_benchmark, load_ohlcv
from qts.report import save_report, summarize


def main() -> None:
    p = argparse.ArgumentParser(description="台股波段量化系統 — 回測")
    p.add_argument("--data", required=True, help="OHLCV 路徑（目錄或單一 CSV/Parquet）")
    p.add_argument("--benchmark", required=True, help="大盤指數 CSV/Parquet")
    p.add_argument("--start", default=None, help="回測起日 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="回測迄日 YYYY-MM-DD")
    p.add_argument("--config", default=None, help="JSON 參數覆寫檔")
    p.add_argument("--out", default=None, help="輸出目錄（預設 output/backtest_<時間戳>）")
    args = p.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()

    t0 = time.time()
    print(f"[1/3] 載入資料：{args.data}")
    data = load_ohlcv(args.data)
    benchmark = load_benchmark(args.benchmark)
    print(f"      {len(data)} 檔股票，{time.time() - t0:.1f}s")

    print("[2/3] 計算訊號並執行回測 ...")
    t1 = time.time()
    result = Backtester(cfg).run(data, benchmark, start=args.start, end=args.end)
    print(f"      完成，{time.time() - t1:.1f}s")

    print("[3/3] 產出報告")
    out_dir = args.out or f"output/backtest_{datetime.now():%Y%m%d_%H%M%S}"
    save_report(result, out_dir)
    print(summarize(result))
    print(f"\n報告已輸出：{Path(out_dir).resolve()}（summary.txt / trades.csv / equity.csv / equity_curve.png）")


if __name__ == "__main__":
    main()
