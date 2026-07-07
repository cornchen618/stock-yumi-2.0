"""每日戰報 PNG（盤後推播；Discord 內直接顯示，取代 HTML 附件）。

一張圖呈現：大盤狀態｜持股逐檔（題材/產業、均價、現價、當日%、報酬%）｜帳戶摘要。
資料：transactions.csv（平均成本法，含費用與股利）＋ data/ohlcv.parquet 最新兩日收盤。

用法：python scripts/daily_report.py [--no-discord]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_industry, load_names, load_themes
from qts.render import table_png

COMM = 0.001425 * 0.6
TAX = 0.003


def build_report_png() -> bytes:
    names = load_names(ROOT / "data" / "universe.csv")
    industry = load_industry(ROOT / "data" / "universe.csv")
    themes = load_themes(ROOT / "themes.csv")

    # 大盤
    bench = pd.read_csv(ROOT / "data" / "benchmark" / "taiex.csv", parse_dates=["date"]).sort_values("date")
    tx_close, tx_prev = float(bench["close"].iloc[-1]), float(bench["close"].iloc[-2])
    tx_chg = (tx_close / tx_prev - 1) * 100
    ma60 = bench["close"].rolling(60).mean().iloc[-1]
    asof = bench["date"].iloc[-1].date()

    # 持倉（平均成本法，與儀表板同邏輯）
    pos: dict[str, dict] = {}
    realized = dividends = fees_total = 0.0
    tp = ROOT / "transactions.csv"
    if tp.exists():
        tx = pd.read_csv(tp, dtype={"symbol": str})
        tx["date"] = pd.to_datetime(tx["date"], format="mixed")
        for r in tx.sort_values("date").itertuples():
            amt = r.shares * r.price
            if r.action == "DIV_CASH":
                realized += amt
                dividends += amt
                continue
            if r.action == "DIV_STOCK":
                if r.symbol in pos:
                    pos[r.symbol]["sh"] += int(r.shares)
                continue
            fee = float(r.fee) if pd.notna(r.fee) and str(r.fee) != "" else (
                amt * COMM if r.action == "BUY" else amt * (COMM + TAX))
            fees_total += fee
            if r.action == "BUY":
                p = pos.setdefault(r.symbol, {"sh": 0, "cost": 0.0})
                p["sh"] += int(r.shares)
                p["cost"] += amt + fee
            else:  # SELL
                p = pos.get(r.symbol)
                if not p or p["sh"] < r.shares:
                    continue
                avg = p["cost"] / p["sh"]
                realized += (r.price * r.shares - fee) - avg * r.shares
                p["cost"] -= avg * r.shares
                p["sh"] -= int(r.shares)
                if p["sh"] == 0:
                    del pos[r.symbol]

    # 最新/前日收盤
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date", "symbol", "raw_close"])
    px = px[px["symbol"].isin(pos)] if pos else px.iloc[0:0]
    rows, mkt_val, unreal, day_chg_val = [], 0.0, 0.0, 0.0
    for s, p in pos.items():
        g = px[px["symbol"] == s].sort_values("date").tail(2)
        cur = float(g["raw_close"].iloc[-1]) if len(g) else float("nan")
        prev = float(g["raw_close"].iloc[-2]) if len(g) > 1 else cur
        avg = p["cost"] / p["sh"]
        val = p["sh"] * cur
        u = val - p["cost"]
        mkt_val += val
        unreal += u
        day_chg_val += p["sh"] * (cur - prev)
        day_pct = (cur / prev - 1) * 100 if prev else 0.0
        ret_pct = (cur / avg - 1) * 100
        rows.append([
            f"{s} {names.get(s, '')}",
            themes.get(s) or industry.get(s, "—") or "—",
            f"{p['sh']:,}",
            f"{avg:.1f}", f"{cur:g}",
            f"{day_pct:+.1f}%",
            f"{ret_pct:+.1f}%",
        ])
    if not rows:
        rows.append(["（空手）", "—", "—", "—", "—", "—", "—"])

    regime = "多頭（>60日線）" if tx_close > ma60 else "空頭（<60日線）"
    title = (f"每日戰報 {asof}　大盤 {tx_close:,.0f}（{tx_chg:+.1f}%）｜{regime}")
    footer = (f"帳戶：持股市值 {mkt_val:,.0f}｜未實現 {unreal:+,.0f}（{(unreal / mkt_val * 100) if mkt_val else 0:+.1f}%）"
              f"｜當日變動 {day_chg_val:+,.0f}｜已實現(含股利) {realized:+,.0f}｜股利 {dividends:,.0f}｜費稅 {fees_total:,.0f}")
    return table_png(title,
                     ["標的", "題材/產業", "股數", "均價(含費)", "現價", "當日", "報酬%"],
                     rows, footer=footer)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-discord", action="store_true")
    args = p.parse_args()
    png = build_report_png()
    out = ROOT / "output" / "daily_report.png"
    out.write_bytes(png)
    print(f"saved {out}（{len(png) // 1024} KB）")
    if not args.no_discord:
        from qts.notify import send_png
        send_png(png, filename="daily_report.png", content="🗒️ **每日戰報**")


if __name__ == "__main__":
    main()
