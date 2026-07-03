"""動能組合月度換股清單（MOMENTUM.md 規格的實盤執行工具）。

在每月最後交易日收盤後執行，輸出：
  - 續抱清單（動能 > 0 且排名 ≤ band）
  - 賣出清單（跌出 band 或動能轉負）
  - 買進清單（補滿 top N，含建議股數）
隔一個交易日開盤執行；開盤漲幅 ≥ 9.5% 的買單放棄。

持股狀態存在 holdings.csv（symbol,shares），本腳本會在確認後更新它？——不會，
腳本只產生清單；實際成交後請自行更新 holdings.csv（成交價格與股數以券商回報為準）。

用法：
  python scripts/scan_momentum.py --data data/ohlcv.parquet --equity 1000000
  python scripts/scan_momentum.py --data data/ohlcv.parquet --equity 1000000 --holdings holdings.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.data import load_names, load_ohlcv
from qts.momentum import MomentumConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--equity", type=float, required=True, help="目前帳戶權益（元）")
    p.add_argument("--holdings", default="holdings.csv", help="持股 CSV（symbol,shares）；不存在視為空手")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--band", type=int, default=40)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    args = p.parse_args()

    cfg = MomentumConfig(lookback=args.lookback, skip=args.skip, top_n=args.top, band_rank=args.band)
    data = load_ohlcv(args.data)
    names = load_names(Path(args.data).parent / "universe.csv" if Path(args.data).is_file() else "data/universe.csv")

    close = pd.DataFrame({s: d["close"] for s, d in data.items()})
    raw_close = pd.DataFrame({s: d["raw_close"] for s, d in data.items()}).reindex(close.index)
    volume = pd.DataFrame({s: d["volume"] for s, d in data.items()}).reindex(close.index)
    amount = pd.DataFrame({s: d["amount"] for s, d in data.items()}).reindex(close.index)

    asof = close.index.max()
    mom = close.shift(cfg.skip) / close.shift(cfg.lookback) - 1.0
    med_vol = volume.rolling(cfg.liq_window, min_periods=cfg.liq_window).median()
    med_amt = amount.rolling(cfg.liq_window, min_periods=cfg.liq_window).median()
    eligible = (
        (med_vol.loc[asof] >= cfg.liq_vol_min)
        & (med_amt.loc[asof] >= cfg.liq_amt_min)
        & (close.loc[asof] >= cfg.min_price)
        & mom.loc[asof].notna()
    )
    scores = mom.loc[asof].where(eligible).dropna().sort_values(ascending=False)
    rank_map = {s: r + 1 for r, s in enumerate(scores.index)}

    held: dict[str, int] = {}
    hp = Path(args.holdings)
    if hp.exists():
        h = pd.read_csv(hp, dtype={"symbol": str})
        held = dict(zip(h["symbol"], h["shares"]))

    keep = [s for s in held if rank_map.get(s, 10**9) <= cfg.band_rank and scores.get(s, -1.0) > 0.0]
    sells = [s for s in held if s not in keep]
    slots = cfg.top_n - len(keep)
    buys = [s for s in scores.index if s not in held and scores[s] > 0.0][:max(slots, 0)]

    def nm(s: str) -> str:
        return names.get(s, "")

    month_end = asof == close.index[close.index.to_period("M") == asof.to_period("M")].max()
    print(f"訊號基準日：{asof.date()} 收盤（{'是' if month_end else '⚠ 不是'}該月最後一個資料日）")
    print(f"下表價格 = {asof:%m/%d} 收盤市價；實際成交價以次一交易日開盤為準")
    print(f"合格股池 {int(eligible.sum())} 檔｜動能>0 共 {(scores > 0).sum()} 檔")
    print("=" * 66)
    if held:
        print(f"【續抱 {len(keep)} 檔】")
        for s in keep:
            print(f"  {s} {nm(s)}  排名 {rank_map[s]:>3}  動能 {scores[s] * 100:+.1f}%  持有 {held[s]} 股")
        print(f"【賣出 {len(sells)} 檔】（次一交易日開盤市價賣出）")
        for s in sells:
            why = "動能轉負" if scores.get(s, -1.0) <= 0 else ("跌出band" if s in rank_map else "失去資格")
            print(f"  {s} {nm(s)}  {why}  持有 {held[s]} 股")
    target_value = args.equity / cfg.top_n
    rows = []
    for s in buys:
        px = float(raw_close.loc[asof, s])
        rows.append({"symbol": s, "name": nm(s), "rank": rank_map[s],
                     "momentum%": round(scores[s] * 100, 1),
                     f"close_{asof:%m%d}": px, "suggest_shares": int(target_value // px),
                     "suggest_notional": int(target_value // px * px)})
    print(f"【買進 {len(buys)} 檔】（次一交易日開盤；每檔目標 {target_value:,.0f} 元；開盤漲幅≥9.5% 放棄）")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    out = Path(f"output/momentum_rebalance_{asof:%Y%m%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"symbol": s, "action": "KEEP"} for s in keep]
        + [{"symbol": s, "action": "SELL"} for s in sells]
        + [{**r, "action": "BUY"} for r in rows]
    ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已輸出：{out.resolve()}")
    print("成交後請手動更新 holdings.csv（symbol,shares）")


if __name__ == "__main__":
    main()
