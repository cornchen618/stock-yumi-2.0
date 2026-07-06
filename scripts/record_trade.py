"""成交入帳一條龍：記帳 → Discord 即時確認卡 → 儀表板重生。

用法：
  python scripts/record_trade.py BUY 6239 150 356
  python scripts/record_trade.py SELL 6239 150 360 --fee 25
  python scripts/record_trade.py BUY 2330 1000 1150 --date 2026-07-05
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_names
from qts.notify import C_BLUE, C_GREEN, send_embed

COMM = 0.001425 * 0.6
TAX = 0.003


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["BUY", "SELL", "buy", "sell"])
    p.add_argument("symbol")
    p.add_argument("shares", type=int)
    p.add_argument("price", type=float)
    p.add_argument("--date", default=f"{datetime.now():%Y-%m-%d}")
    p.add_argument("--fee", type=float, default=None)
    args = p.parse_args()
    action = args.action.upper()
    sym = str(args.symbol)

    names = load_names(ROOT / "data" / "universe.csv")
    name = names.get(sym, "")
    notional = args.shares * args.price
    fee = args.fee if args.fee is not None else round(
        max(1, notional * (COMM if action == "BUY" else COMM)) + (notional * TAX if action == "SELL" else 0), 2)

    # holdings.csv
    hp = ROOT / "holdings.csv"
    h = pd.read_csv(hp, dtype={"symbol": str}) if hp.exists() else pd.DataFrame(columns=["symbol", "shares"])
    h["shares"] = h.get("shares", pd.Series(dtype=int)).astype(int) if len(h) else h.get("shares")
    held = int(h.loc[h["symbol"] == sym, "shares"].sum()) if len(h) else 0
    if action == "SELL" and args.shares > held:
        raise SystemExit(f"錯誤：帳上僅持有 {sym} {held} 股，無法賣出 {args.shares} 股。請確認後重試。")
    new_shares = held + args.shares if action == "BUY" else held - args.shares
    h = h[h["symbol"] != sym]
    if new_shares > 0:
        h = pd.concat([h, pd.DataFrame([{"symbol": sym, "shares": new_shares}])], ignore_index=True)
    h.to_csv(hp, index=False, encoding="utf-8-sig")

    # transactions.csv
    tp = ROOT / "transactions.csv"
    t = pd.read_csv(tp, dtype={"symbol": str}) if tp.exists() else pd.DataFrame(
        columns=["date", "symbol", "action", "shares", "price", "fee"])
    t = pd.concat([t, pd.DataFrame([{
        "date": args.date, "symbol": sym, "action": action,
        "shares": args.shares, "price": args.price, "fee": args.fee if args.fee is not None else "",
    }])], ignore_index=True)
    t["date"] = pd.to_datetime(t["date"], format="mixed").dt.strftime("%Y-%m-%d")
    t.to_csv(tp, index=False, encoding="utf-8-sig")

    # 最新收盤（估值參考）
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date", "symbol", "raw_close"])
    g = px[px["symbol"] == sym].sort_values("date").tail(1)
    last_close, last_day = (float(g["raw_close"].iloc[0]), pd.Timestamp(g["date"].iloc[0]).date()) if len(g) else (None, None)

    # Discord 即時確認
    zh = "買進" if action == "BUY" else "賣出"
    lines = [f"**{sym} {name}**　{zh} {args.shares:,} 股 @ {args.price:g} 元",
             f"成交金額 {notional:,.0f} 元｜{'估' if args.fee is None else ''}手續費{'＋稅' if action == 'SELL' else ''} {fee:,.0f} 元",
             f"持股變動：{held:,} → **{new_shares:,} 股**"]
    if last_close is not None and new_shares > 0:
        lines.append(f"參考估值：最新收盤 {last_close:g}（{last_day}）｜此部位未實現 {new_shares * (last_close - args.price):+,.0f} 元（對本筆成本）")
    lines.append("監控將於下一輪輪詢（≤5 分鐘，盤中）或明日 08:55 納入此持股")
    send_embed(f"{'🟥' if action == 'BUY' else '🟩'} 成交入帳確認 {args.date}", "\n".join(lines),
               color=C_GREEN if action == "BUY" else C_BLUE,
               footer="已寫入 holdings.csv / transactions.csv；儀表板重生中")
    print("\n".join(lines))

    # 儀表板重生
    subprocess.run([sys.executable, "scripts/make_dashboard.py"], cwd=ROOT, capture_output=True, timeout=600)
    print("dashboard updated")


if __name__ == "__main__":
    main()
