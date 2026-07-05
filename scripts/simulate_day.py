"""完整流程模擬（dry-run，可視化版）：以最近一個真實交易日重演整天的訊息序列。

情境：假設前一晚收到動能組合換股清單（100 萬、20 檔），隔天照流程執行。
輸出格式與正式排程相同：embed 彩色卡片＋深色表格 PNG，全部標註【模擬】。

用法：python scripts/simulate_day.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_names
from qts.notify import C_BLUE, C_GRAY, C_GREEN, C_YELLOW, send_embed, send_png
from qts.render import RED, GREEN, rebalance_table_png, table_png

SLIP = 0.001
EQUITY = 1_000_000.0
TOP_N = 20

NAMES = load_names(ROOT / "data" / "universe.csv")


def label(sym: str) -> str:
    n = NAMES.get(sym, "")
    return f"{sym} {n}" if n else sym


def main() -> None:
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet",
                         columns=["date", "symbol", "open", "high", "low", "close", "raw_close",
                                  "volume", "amount"])
    px["date"] = pd.to_datetime(px["date"])
    days = sorted(px["date"].unique())
    t_day, prev_day = days[-1], days[-2]          # 執行日 = 最近交易日；訊號日 = 前一日

    # 以訊號日收盤重算換股清單（與 scan_momentum 同規格）
    hist = px[px["date"] <= prev_day]
    close_p = hist.pivot_table(index="date", columns="symbol", values="close")
    vol_p = hist.pivot_table(index="date", columns="symbol", values="volume")
    amt_p = hist.pivot_table(index="date", columns="symbol", values="amount")
    mom = (close_p.shift(21) / close_p.shift(252) - 1.0).iloc[-1]
    elig = (
        (vol_p.rolling(20).median().iloc[-1] >= 500_000)
        & (amt_p.rolling(20).median().iloc[-1] >= 30_000_000)
        & (close_p.iloc[-1] >= 10.0)
        & mom.notna()
    )
    scores = mom.where(elig).dropna().sort_values(ascending=False)
    top = scores[scores > 0].head(TOP_N)
    close_col = f"close_{pd.Timestamp(prev_day):%m%d}"
    buys = pd.DataFrame({
        "symbol": top.index.astype(str),
        "name": [NAMES.get(s, "") for s in top.index],
        "action": "BUY",
        "rank": range(1, len(top) + 1),
        "momentum%": (top.values * 100).round(1),
        close_col: close_p.iloc[-1][top.index].round(2).values,
    })
    buys["suggest_shares"] = (EQUITY / TOP_N // buys[close_col]).astype(int)

    t = px[px["date"] == t_day].set_index("symbol")
    p = px[px["date"] == prev_day].set_index("symbol")

    send_embed("🧪【模擬】完整流程演練（可視化版）",
               f"情境：{pd.Timestamp(prev_day):%m/%d} 晚收到換股清單（權益 100 萬、20 檔），"
               f"以 {pd.Timestamp(t_day):%m/%d} **真實行情**重演全天訊息。\n"
               "以下每則的格式＝正式排程的樣子，僅標題多了【模擬】。", color=C_BLUE)
    time.sleep(1)

    # ---- T-1 17:40 換股清單（表格圖）----
    send_png(rebalance_table_png(buys, f"{pd.Timestamp(prev_day):%m/%d}", EQUITY),
             filename="rebalance.png",
             content="💰【模擬｜T-1 17:40】動能組合月度換股清單")
    time.sleep(1)

    # ---- T 08:55 盤前摘要 ----
    send_embed(f"🌅【模擬｜T 08:55】盤前摘要 {pd.Timestamp(t_day):%Y-%m-%d}",
               fields=[("持股", "0 檔（空手）", True),
                       ("今日待辦", f"換股執行日：買進 {len(buys)} 檔", True),
                       ("執行規則", "開盤市價買進；開盤漲幅 ≥ 9.5% 放棄；系統不下單，委託由你執行", False)],
               color=C_YELLOW)
    time.sleep(1)

    # ---- T 09:05 開盤回報（表格圖）----
    holdings: dict[str, dict] = {}
    rows, colors = [], {}
    for r in buys.itertuples():
        s = r.symbol
        if s not in t.index or s not in p.index:
            rows.append([label(s), "—", "—", "—", "無行情"])
            continue
        o, adj_pc = float(t.loc[s, "open"]), float(p.loc[s, "close"])
        gap = o / adj_pc - 1.0
        if gap >= 0.095:
            rows.append([label(s), f"{o:g}", f"{gap * 100:+.1f}%", "—", "⛔漲停放棄"])
            continue
        fill = o * (1 + SLIP)
        shares = int(EQUITY / TOP_N // fill)
        holdings[s] = {"shares": shares, "fill": fill, "prev_close": adj_pc}
        rows.append([label(s), f"{o:g}", f"{gap * 100:+.1f}%", f"{shares:,}", "✅成交"])
    png = table_png(f"【模擬｜T 09:05】開盤回報　成交 {len(holdings)}/{len(buys)} 檔（含 0.1% 滑價）",
                    ["標的", "開盤", "開盤漲幅", "股數", "狀態"], rows)
    send_png(png, filename="open_report.png", content="🔔【模擬｜T 09:05】開盤回報")
    time.sleep(1)

    # ---- 盤中警示（格式示範）----
    alerts = []
    for s, h in holdings.items():
        lo = float(t.loc[s, "low"])
        chg = lo / h["prev_close"] - 1.0
        if chg <= -0.08:
            alerts.append((s, lo, chg, h))
    if alerts:
        for s, lo, chg, h in alerts:
            send_embed(f"⚠️【模擬｜盤中】{label(s)} 觸發警示",
                       f"最低 {lo:g}，當日 {chg * 100:+.1f}%（持有 {h['shares']} 股）\n資訊警示；依紀律月調倉才動作。",
                       color=0xE05555)
    else:
        send_embed("⚠️【模擬｜盤中】持股警示（本日未觸發，此為格式示範）",
                   "觸發條件：持股單日 ≤ −8% 或大盤 ≤ −3%（每檔每日最多一則）", color=C_GRAY)
    time.sleep(1)

    # ---- 10:30 心跳 ----
    est = sum(h["shares"] * (float(t.loc[s, "close"]) - h["prev_close"]) for s, h in holdings.items())
    send_embed("⏳【模擬｜T 10:30】監控心跳",
               f"持股 {len(holdings)} 檔｜估當日損益 **{est:+,.0f}** 元\n（11:30、12:30 亦各一則；心跳消失＝系統異常）",
               color=C_GRAY)
    time.sleep(1)

    # ---- 13:35 收盤摘要（表格圖）----
    rows = []
    total_day = total_pos = 0.0
    for s, h in holdings.items():
        c = float(t.loc[s, "close"])
        day_pnl = h["shares"] * (c - h["prev_close"])
        pos_pnl = h["shares"] * (c - h["fill"])
        total_day += day_pnl
        total_pos += pos_pnl
        rows.append([label(s), f"{c:g}", f"{(c / h['prev_close'] - 1) * 100:+.1f}%", f"{pos_pnl:+,.0f}"])
    png = table_png(
        f"【模擬｜T 13:35】收盤摘要　當日 {total_day:+,.0f} 元｜對成本 {total_pos:+,.0f} 元（未含手續費）",
        ["標的", "收盤", "當日漲跌", "持倉損益"], rows, col_colors={3: RED if total_pos >= 0 else GREEN})
    send_png(png, filename="close_report.png", content="🏁【模擬｜T 13:35】收盤摘要")
    time.sleep(1)

    # ---- 17:40 盤後 ----
    send_embed("⚙️【模擬｜T 17:40】盤後流程",
               "1️⃣ 價格資料更新（1,953 檔）\n2️⃣ 波段掃描表格圖（紙上觀察）\n"
               "3️⃣ 八問決策簡報卡片\n4️⃣ 儀表板重生\n——【模擬】演練結束——\n"
               "之後每個交易日收到的就是這一整套；換股清單只在月底出現。", color=C_GREEN)
    print(f"simulation sent: {len(holdings)}/{len(buys)} filled, day pnl {total_day:+,.0f}")


if __name__ == "__main__":
    main()
