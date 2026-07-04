"""動能組合 MA200 曝險調節 — 預登記檢定（規則先寫死再看數據）。

規則：月調倉訊號日，TAIEX < MA200 → 該次新買部位目標金額減半（既有持股與賣出規則不變）。
採用標準（執行前登記）：
  MDD 改善 ≥ 5 個百分點，且 Sharpe 下降 ≤ 0.05。
標註：此為文獻標準的 momentum crash 對策（時間序列動能疊加），
但在本資料集只做這一次檢定，不掃參數。

用法：python scripts/exposure_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_benchmark, load_ohlcv
from qts.metrics import equity_stats, trade_stats
from qts.momentum import MomentumBacktester, MomentumConfig


def main() -> None:
    print("載入資料 ...")
    data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")

    rows = []
    for tag, cfg in (
        ("基準（永遠滿倉）", MomentumConfig()),
        ("MA200 調節（熊市新倉減半）", MomentumConfig(exposure_ma=200, exposure_bear_mult=0.5)),
    ):
        res = MomentumBacktester(cfg).run(data, benchmark, start="2019-01-01")
        eq = equity_stats(res.equity, res.benchmark)
        tr = trade_stats(res.trades)
        rows.append({"配置": tag, "總報酬%": round(eq["total_return"] * 100, 1),
                     "CAGR%": round(eq["cagr"] * 100, 2), "Sharpe": round(eq["sharpe"], 2),
                     "MDD%": round(eq["mdd"] * 100, 1), "PF": round(tr.get("profit_factor", np.nan), 2),
                     "曝險%": round(res.exposure.mean() * 100)})
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))
    base, var = rows[0], rows[1]
    mdd_gain = var["MDD%"] - base["MDD%"]          # MDD 為負值，變大（趨近0）= 改善
    sharpe_drop = base["Sharpe"] - var["Sharpe"]
    ok = mdd_gain >= 5.0 and sharpe_drop <= 0.05
    print(f"MDD 改善 {mdd_gain:+.1f}pp（門檻 ≥ +5）｜Sharpe 變化 {-sharpe_drop:+.2f}（門檻 ≥ −0.05）")
    print(f"→ 判定：{'PASS，採用' if ok else 'FAIL，不採用（維持滿倉版）'}")


if __name__ == "__main__":
    main()
