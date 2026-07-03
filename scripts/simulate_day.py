"""完整流程模擬（dry-run）：以最近一個真實交易日重演整天的訊息序列。

情境：假設前一晚收到動能組合換股清單（100 萬、20 檔），隔天照流程執行。
所有訊息前綴【模擬】並使用真實行情資料：
  T-1 17:40 月底換股清單 → T 08:55 盤前摘要 → 09:05 開盤回報（漲停放棄檢查）
  → 10:30 心跳 → 盤中警示（如有）→ 13:35 收盤摘要 → 17:40 盤後報告

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
from qts.notify import send

NAMES = load_names(ROOT / "data" / "universe.csv")


def label(sym: str) -> str:
    n = NAMES.get(sym, "")
    return f"{sym} {n}" if n else sym

SLIP = 0.001
EQUITY = 1_000_000.0
TOP_N = 20


def main() -> None:
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet",
                         columns=["date", "symbol", "open", "high", "low", "close", "raw_close",
                                  "volume", "amount"])
    px["date"] = pd.to_datetime(px["date"])
    days = sorted(px["date"].unique())
    t_day, prev_day = days[-1], days[-2]          # 執行日 = 最近交易日；訊號日 = 前一日

    # 以訊號日（prev_day）收盤重算換股清單 — 與 scan_momentum.py 同規格（12-1、top20、流動性門檻）
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
    buys = pd.DataFrame({
        "symbol": top.index.astype(str),
        "rank": range(1, len(top) + 1),
        "momentum%": (top.values * 100).round(1),
        "close": close_p.iloc[-1][top.index].round(2).values,
    })
    buys["suggest_shares"] = (EQUITY / TOP_N // buys["close"]).astype(int)

    t = px[px["date"] == t_day].set_index("symbol")
    p = px[px["date"] == prev_day].set_index("symbol")

    send(f":test_tube: ——【模擬】完整流程演練開始——\n"
         f"情境：假設 {pd.Timestamp(prev_day):%m/%d} 晚上收到換股清單（權益 100 萬、20 檔動能組合），"
         f"以 {pd.Timestamp(t_day):%m/%d} 的**真實行情**重演一整天的訊息。實際排程的發送時間標在每則開頭。")
    time.sleep(1)

    # ---- T-1 17:40 換股清單 ----
    lines = [f":moneybag:【模擬｜T-1 17:40】**動能組合月度換股清單（明日開盤執行）**",
             f"目前空手 → 買進 {len(buys)} 檔，每檔目標 {EQUITY / TOP_N:,.0f} 元："]
    for _, r in buys.iterrows():
        lines.append(f"  {label(r['symbol'])}  排名{int(r['rank'])}  動能{r['momentum%']:+.0f}%  "
                     f"收盤 {r['close']}  建議 {int(r['suggest_shares'])} 股")
    lines.append(":point_right: 明早 09:00 開盤市價買進；開盤漲幅 ≥ 9.5% 放棄；成交後更新 holdings.csv")
    send("\n".join(lines))
    time.sleep(1)

    # ---- T 08:55 盤前摘要 ----
    send(f":sunrise:【模擬｜T 08:55】**盤前摘要 {pd.Timestamp(t_day):%Y-%m-%d}**\n"
         f"持股 0 檔（空手）\n"
         f":arrows_counterclockwise: 今日為換股執行日：買進 {len(buys)} 檔（清單見昨晚訊息）\n"
         f"_規則：盤中警示僅供參考；系統不下單，委託由你在券商執行。_")
    time.sleep(1)

    # ---- T 09:05 開盤回報 + 模擬成交 ----
    holdings: dict[str, dict] = {}
    rep = [":bell:【模擬｜T 09:05】**開盤回報（換股執行）**"]
    for _, r in buys.iterrows():
        s = r["symbol"]
        if s not in t.index or s not in p.index:
            rep.append(f"  {s}：無行情，跳過")
            continue
        o, pc_raw = float(t.loc[s, "open"]), float(p.loc[s, "raw_close"])
        adj_pc = float(p.loc[s, "close"])
        gap = o / adj_pc - 1.0
        if gap >= 0.095:
            rep.append(f"  {label(s)}：開盤 {o:.2f}（{gap * 100:+.1f}%）→ :no_entry: **漲幅≥9.5%，放棄此買單**")
            continue
        fill = o * (1 + SLIP)
        shares = int(EQUITY / TOP_N // fill)
        holdings[s] = {"shares": shares, "fill": fill, "prev_close": adj_pc}
        rep.append(f"  {label(s)}：開盤 {o:.2f}（{gap * 100:+.1f}%）→ 買進 {shares} 股 @ 約 {fill:.2f}")
    rep.append(f"共成交 {len(holdings)} 檔（模擬含 0.1% 滑價）")
    send("\n".join(rep))
    time.sleep(1)

    # ---- 盤中警示（用當日最低價檢查 −8%）----
    alerts = []
    for s, h in holdings.items():
        lo = float(t.loc[s, "low"])
        chg = lo / h["prev_close"] - 1.0
        if chg <= -0.08:
            alerts.append(f":warning:【模擬｜盤中】**{s}** 最低 {lo:.2f}，當日 {chg * 100:+.1f}%"
                          f"（持有 {h['shares']} 股）\n_資訊警示；依紀律月調倉才動作。_")
    if alerts:
        send("\n".join(alerts))
    else:
        send(":warning:【模擬｜盤中】本日無持股觸發 −8% 警示（此為警示訊息的格式範例位置：\n"
             "「**2330** 現價 xxx，當日 −8.2%（持有 91 股，估 −4,500 元）」）")
    time.sleep(1)

    # ---- 10:30 心跳（以當日盤中價估計 → 模擬用收盤代替）----
    est = sum(h["shares"] * (float(t.loc[s, "close"]) - h["prev_close"]) for s, h in holdings.items())
    send(f":hourglass:【模擬｜T 10:30】監控正常｜持股 {len(holdings)} 檔｜估當日損益 {est:+,.0f} 元"
         f"\n（11:30、12:30 亦各有一則；心跳消失 = 系統異常）")
    time.sleep(1)

    # ---- 13:35 收盤摘要 ----
    lines = [f":checkered_flag:【模擬｜T 13:35】**收盤摘要 {pd.Timestamp(t_day):%Y-%m-%d}**"]
    total_day = 0.0
    total_pos = 0.0
    for s, h in holdings.items():
        c = float(t.loc[s, "close"])
        day_pnl = h["shares"] * (c - h["prev_close"])
        pos_pnl = h["shares"] * (c - h["fill"])
        total_day += day_pnl
        total_pos += pos_pnl
        lines.append(f"  {label(s)}：{c:.2f}（{(c / h['prev_close'] - 1) * 100:+.1f}%）持倉損益 {pos_pnl:+,.0f}")
    lines.append(f"估當日組合損益 {total_day:+,.0f} 元｜對成本損益 {total_pos:+,.0f} 元（未含手續費）")
    send("\n".join(lines))
    time.sleep(1)

    # ---- 17:40 盤後報告 ----
    send(f":gear:【模擬｜T 17:40】**盤後更新**\n"
         f":white_check_mark: 價格資料已更新（1,953 檔）\n"
         f":clipboard: 波段掃描（紙上觀察 — KPI Gate 未過，勿實際下單）：今日候選 N 檔（詳表另發）\n"
         f":calendar: 非月底，動能組合無動作。\n"
         f"——【模擬】演練結束——\n"
         f"之後每個交易日你收到的就是這一整套；差別只在：真實持股來自 holdings.csv、"
         f"換股清單只在月底出現。")


if __name__ == "__main__":
    main()
