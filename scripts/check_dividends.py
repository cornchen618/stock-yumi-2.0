"""持股除權息管理（eod_task 每日呼叫；無持股時靜默結束）。

功能：
  1. 預告：持股在未來 7 天內除權息 → Discord 黃色卡片提醒（可選擇參加或前一日賣出棄權）
  2. 入帳：除權息日已到且尚未記帳 →
       現金股利：transactions.csv 追加 DIV_CASH（shares=持股數, price=每股股利）
       股票股利：追加 DIV_STOCK（shares=配得股數）並同步增加 holdings.csv 股數
     防呆：只對「transactions.csv 中除息日前已有買進紀錄」的持股自動入帳；
           無法確認持有時點者僅提醒、不入帳。
  3. 監控防誤報：寫出 data/dividends_upcoming.csv 供盤中監控調整除權息日的昨收基準。

資料源：FinMind TaiwanStockDividend（現金=盈餘+公積配息/股；股票股利金額/10=配股率）。
稅務註記：股利所得課稅與二代健保補充保費（單筆 >2 萬）不入帳，僅在訊息中提醒。

用法：python scripts/check_dividends.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_names
from qts.notify import C_BLUE, C_YELLOW, send_embed

API = "https://api.finmindtrade.com/api/v4/data"
NOTICE_DAYS = 7


def fetch_dividends(symbol: str, token: str) -> list[dict]:
    """回傳該股一年內的除權息事件：[{kind, ex_date, per_share, pay_date}]。"""
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        r = requests.get(API, params={"dataset": "TaiwanStockDividend", "data_id": symbol,
                                      "start_date": start, "token": token}, timeout=30).json()
    except Exception as e:  # noqa: BLE001
        print(f"  {symbol}: 股利查詢失敗 {e}")
        return []
    events = []
    for row in r.get("data", []):
        cash = float(row.get("CashEarningsDistribution") or 0) + float(row.get("CashStatutorySurplus") or 0)
        stock = float(row.get("StockEarningsDistribution") or 0) + float(row.get("StockStatutorySurplus") or 0)
        if cash > 0 and row.get("CashExDividendTradingDate"):
            events.append({"kind": "CASH", "ex_date": row["CashExDividendTradingDate"],
                           "per_share": round(cash, 4), "pay_date": row.get("CashDividendPaymentDate", "")})
        if stock > 0 and row.get("StockExDividendTradingDate"):
            events.append({"kind": "STOCK", "ex_date": row["StockExDividendTradingDate"],
                           "per_share": round(stock, 4), "pay_date": ""})  # 面額10元:每股配 stock/10 股
    return events


def main() -> None:
    hp = ROOT / "holdings.csv"
    if not hp.exists():
        return
    holdings = pd.read_csv(hp, dtype={"symbol": str})
    if not len(holdings):
        return
    token = os.environ.get("FINMIND_TOKEN", "")
    names = load_names(ROOT / "data" / "universe.csv")
    today = datetime.now().date()

    tx_path = ROOT / "transactions.csv"
    tx = pd.read_csv(tx_path, dtype={"symbol": str}, parse_dates=["date"]) if tx_path.exists() else pd.DataFrame(
        columns=["date", "symbol", "action", "shares", "price", "fee"])

    upcoming_rows, notices, booked, warns = [], [], [], []
    for h in holdings.itertuples():
        sym, shares_held = h.symbol, int(h.shares)
        events = fetch_dividends(sym, token)
        time.sleep(1.0)
        first_buy = tx[(tx["symbol"] == sym) & (tx["action"] == "BUY")]["date"].min()

        for ev in events:
            ex_date = pd.Timestamp(ev["ex_date"]).date()
            label = f"{sym} {names.get(sym, '')}"
            kind_zh = "除息" if ev["kind"] == "CASH" else "除權"

            # 1) 未來 7 天預告
            if today < ex_date <= today + timedelta(days=NOTICE_DAYS):
                extra = f"每股配息 {ev['per_share']} 元" if ev["kind"] == "CASH" else f"每股配股 {ev['per_share'] / 10:.4f} 股"
                pay = f"（發放日 {ev['pay_date']}）" if ev["pay_date"] else ""
                notices.append(f"**{label}** 將於 {ex_date:%m/%d} {kind_zh}：{extra}{pay}\n"
                               f"　參加＝續抱不動作；棄權＝{ex_date - timedelta(days=1):%m/%d} 前賣出")
                upcoming_rows.append({"symbol": sym, "ex_date": str(ex_date), "kind": ev["kind"],
                                      "per_share": ev["per_share"]})

            # 2) 已到期 → 自動入帳（限能確認除息日前已持有者）
            elif ex_date <= today:
                if pd.isna(first_buy) or first_buy.date() >= ex_date:
                    if (today - ex_date).days <= NOTICE_DAYS and not pd.isna(first_buy):
                        warns.append(f"{label} {ex_date:%m/%d} {kind_zh}：無法確認除息日前已持有，未自動入帳，請人工確認")
                    continue
                action = "DIV_CASH" if ev["kind"] == "CASH" else "DIV_STOCK"
                dup = tx[(tx["symbol"] == sym) & (tx["action"] == action)
                         & (tx["date"] == pd.Timestamp(ex_date))]
                if len(dup):
                    continue
                if ev["kind"] == "CASH":
                    amount = shares_held * ev["per_share"]
                    new_row = {"date": str(ex_date), "symbol": sym, "action": "DIV_CASH",
                               "shares": shares_held, "price": ev["per_share"], "fee": 0}
                    health = "（單筆 >2 萬將扣 2.11% 二代健保補充保費，實收以入帳為準）" if amount > 20000 else ""
                    booked.append(f"**{label}** {ex_date:%m/%d} 除息入帳：{shares_held} 股 × {ev['per_share']} 元 = "
                                  f"**+{amount:,.0f} 元**{health}" + (f"｜現金 {ev['pay_date']} 發放" if ev["pay_date"] else ""))
                else:
                    add_shares = int(shares_held * ev["per_share"] / 10)
                    if add_shares <= 0:
                        continue
                    new_row = {"date": str(ex_date), "symbol": sym, "action": "DIV_STOCK",
                               "shares": add_shares, "price": 0, "fee": 0}
                    holdings.loc[holdings["symbol"] == sym, "shares"] = shares_held + add_shares
                    booked.append(f"**{label}** {ex_date:%m/%d} 除權配股：+{add_shares} 股"
                                  f"（{shares_held} → {shares_held + add_shares} 股，holdings.csv 已更新）")
                tx = pd.concat([tx, pd.DataFrame([new_row])], ignore_index=True)

    # 寫回（日期統一為 YYYY-MM-DD，避免混合格式）
    if booked:
        tx["date"] = pd.to_datetime(tx["date"], format="mixed").dt.strftime("%Y-%m-%d")
        tx.to_csv(tx_path, index=False, encoding="utf-8-sig")
        holdings.to_csv(hp, index=False, encoding="utf-8-sig")
    up_path = ROOT / "data" / "dividends_upcoming.csv"
    pd.DataFrame(upcoming_rows, columns=["symbol", "ex_date", "kind", "per_share"]).to_csv(up_path, index=False)

    # Discord
    if notices:
        send_embed("📢 持股除權息預告（7 日內）", "\n\n".join(notices)[:3900], color=C_YELLOW,
                   footer="除權息日開盤價會下調，盤中警示已自動校正，不會誤報下跌")
    if booked:
        send_embed("💵 除權息已自動入帳", "\n\n".join(booked)[:3900], color=C_BLUE,
                   footer="已寫入 transactions.csv；儀表板損益已含股利")
    if warns:
        send_embed("⚠️ 除權息需人工確認", "\n".join(warns)[:3900], color=C_YELLOW)
    print(f"dividends: 預告 {len(notices)}、入帳 {len(booked)}、待確認 {len(warns)}")


if __name__ == "__main__":
    main()
