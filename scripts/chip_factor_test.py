"""籌碼因子預登記檢定（登記於 2026-07-03 抓取資料之前，規則不得事後修改）。

因子：chip_score = (外資 + 投信) 近 20 日累積淨買超（股）÷ 近 20 日均量
檢定一（IC）：每月最後交易日，橫斷面 Spearman 相關（因子 vs 未來 21 日報酬）
             通過標準：mean IC > 0 且 t-stat > 2
檢定二（雙重排序）：動能前 40 名中取 chip_score 前 20 vs 同股池純動能 top 20
             通過標準：Sharpe 提升 且 MDD 不惡化
兩者皆過才把籌碼因子納入動能組合。

用法：python scripts/chip_factor_test.py
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

START = "2019-04-01"   # 籌碼資料自 2019-01 起，留 20 日暖機＋緩衝


def build_chip_score(price_data: dict[str, pd.DataFrame], cal: pd.DatetimeIndex) -> pd.DataFrame:
    ch = pd.read_parquet(ROOT / "data" / "chips_institutional.parquet")
    ch["date"] = pd.to_datetime(ch["date"])
    ch = ch[ch["name"].isin(["Foreign_Investor", "Investment_Trust"])]
    ch["net"] = ch["buy"] - ch["sell"]
    net = ch.groupby(["date", "stock_id"])["net"].sum().unstack()

    syms = [s for s in net.columns if s in price_data]
    net = net[syms].reindex(cal).fillna(0.0)
    vol = pd.DataFrame({s: price_data[s]["volume"] for s in syms}).reindex(cal)

    net_sum20 = net.rolling(20, min_periods=15).sum()
    vol_ma20 = vol.rolling(20, min_periods=15).mean()
    score = net_sum20 / vol_ma20.replace(0.0, np.nan)
    # 籌碼資料起始前的期間視為無資料
    score[score.index < pd.Timestamp("2019-02-01")] = np.nan
    return score


def ic_test(score: pd.DataFrame, close: pd.DataFrame, cal: pd.DatetimeIndex) -> dict:
    fwd = close.shift(-21) / close - 1.0
    month_last = pd.Series(np.arange(len(cal)), index=cal).groupby(cal.to_period("M")).last()
    ics = []
    for i in month_last:
        d = cal[i]
        if d < pd.Timestamp(START) or i + 21 >= len(cal):
            continue
        s = score.loc[d]
        f = fwd.loc[d].reindex(s.index)
        both = pd.concat([s, f], axis=1, keys=["s", "f"]).dropna()
        if len(both) < 50:
            continue
        # Spearman = 排名後的 Pearson（免 scipy 依賴）
        ics.append({"date": d, "ic": both["s"].rank().corr(both["f"].rank()), "n": len(both)})
    icdf = pd.DataFrame(ics)
    mean_ic = icdf["ic"].mean()
    t = mean_ic / icdf["ic"].std(ddof=1) * np.sqrt(len(icdf))
    return {"months": len(icdf), "mean_ic": mean_ic, "t_stat": t,
            "pct_positive": (icdf["ic"] > 0).mean(), "icdf": icdf}


def main() -> None:
    print("載入資料 ...")
    all_data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")
    cal = benchmark.index

    score = build_chip_score(all_data, cal)
    chip_syms = list(score.columns)
    data = {s: all_data[s] for s in chip_syms}
    close = pd.DataFrame({s: data[s]["close"] for s in chip_syms}).reindex(cal)
    print(f"籌碼股池 {len(chip_syms)} 檔")

    print("=" * 62)
    print("【檢定一：月頻橫斷面 IC（因子 vs 未來 21 日報酬）】")
    ic = ic_test(score, close, cal)
    ic_pass = ic["mean_ic"] > 0 and ic["t_stat"] > 2.0
    print(f"  樣本月數     {ic['months']}")
    print(f"  平均 IC      {ic['mean_ic']:+.4f}")
    print(f"  t 統計量     {ic['t_stat']:+.2f}   （通過門檻 > 2）")
    print(f"  IC>0 月份比  {ic['pct_positive'] * 100:.0f}%")
    print(f"  逐年平均 IC：")
    ydf = ic["icdf"].copy(); ydf["y"] = ydf["date"].dt.year
    print("  " + ydf.groupby("y")["ic"].mean().round(3).to_string().replace("\n", "\n  "))
    print(f"  → 檢定一：{'PASS' if ic_pass else 'FAIL'}")

    print("=" * 62)
    print("【檢定二：雙重排序疊加（同股池公平比較，{}~）】".format(START))
    base_cfg = MomentumConfig()
    ov_cfg = MomentumConfig(overlay_pool=40)
    rows = []
    for tag, cfg, ov in (("純動能 top20", base_cfg, None), ("動能40→籌碼20", ov_cfg, score)):
        res = MomentumBacktester(cfg).run(data, benchmark, start=START, overlay=ov)
        eq = equity_stats(res.equity, res.benchmark)
        tr = trade_stats(res.trades)
        rows.append({
            "配置": tag, "總報酬%": round(eq["total_return"] * 100, 1),
            "CAGR%": round(eq["cagr"] * 100, 2), "Sharpe": round(eq["sharpe"], 2),
            "MDD%": round(eq["mdd"] * 100, 1), "PF": round(tr.get("profit_factor", np.nan), 2),
            "交易數": tr.get("n_trades", 0), "換手": round(res.annual_turnover, 1),
        })
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))
    base, over = rows[0], rows[1]
    ds_pass = over["Sharpe"] > base["Sharpe"] and over["MDD%"] >= base["MDD%"] - 0.01
    print(f"  → 檢定二（Sharpe 升且 MDD 不惡化）：{'PASS' if ds_pass else 'FAIL'}")

    print("=" * 62)
    verdict = ic_pass and ds_pass
    print(f"【總判定】兩項{'皆過 → 採用籌碼疊加' if verdict else '未全過 → 不採用，動能組合維持原版'}")


if __name__ == "__main__":
    main()
