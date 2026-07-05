"""盤中監控（排程於交易日 08:55 啟動，13:35 自動結束）。

輸出規則（訊號雜訊比優先，事件驅動而非灌水）：
  08:55 盤前摘要   持股清單＋今日待辦（換股日附買賣清單與放棄規則）
  09:05 開盤回報   換股日：逐筆回報開盤價、標記「開盤漲幅≥9.5% → 放棄」的買單
  盤中事件警示     持股單日 ≤ −8% 或觸及跌停（每檔每日一次）
                   TAIEX 單日 ≤ −3%（每日一次）
  10:30/11:30/12:30 心跳一行（持股數、估當日損益）— 沉默≠正常，心跳確認系統活著
  13:35 收盤摘要   每檔持股當日漲跌、組合估值變化

注意：報價來自 Yahoo，可能延遲 5~15 分鐘；警示僅供參考，月調倉紀律不因盤中波動改變。
系統不下單。所有實際委託由使用者於券商執行。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_names
from qts.notify import C_BLUE, C_GRAY, C_RED, C_YELLOW, send_embed

NAMES = load_names(ROOT / "data" / "universe.csv")


def label(sym: str) -> str:
    n = NAMES.get(sym, "")
    return f"{sym} {n}" if n else sym

POLL_SEC = 300
END_TIME = "13:35"
ALERT_POS_DROP = -0.08
ALERT_TAIEX_DROP = -0.03


def _now() -> datetime:
    return datetime.now()


def _suffix_map() -> dict[str, str]:
    uni = pd.read_csv(ROOT / "data" / "universe.csv", dtype=str)
    return dict(zip(uni["code"], uni["suffix"]))


def load_state() -> tuple[dict[str, int], pd.DataFrame | None, dict[str, float]]:
    """回傳 (持股, 今日換股清單或 None, 昨收價)。"""
    holdings: dict[str, int] = {}
    hp = ROOT / "holdings.csv"
    if hp.exists():
        h = pd.read_csv(hp, dtype={"symbol": str})
        holdings = dict(zip(h["symbol"], h["shares"].astype(int)))

    rebalance = None
    files = sorted((ROOT / "output").glob("momentum_rebalance_*.csv"))
    if files:
        latest = files[-1]
        file_date = latest.stem.split("_")[-1]
        # 換股清單基準日 = 上一交易日 → 今天是執行日
        if (pd.Timestamp(file_date) + pd.tseries.offsets.BDay(1)).date() >= _now().date():
            rebalance = pd.read_csv(latest, dtype={"symbol": str})

    prev_close: dict[str, float] = {}
    watch = set(holdings)
    if rebalance is not None:
        watch |= set(rebalance["symbol"])
    if watch:
        px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["symbol", "date", "raw_close"])
        px = px[px["symbol"].isin(watch)]
        last = px.sort_values("date").groupby("symbol").tail(1)
        prev_close = dict(zip(last["symbol"], last["raw_close"]))

    # 今日除權息 → 校正昨收基準（配息/配股造成的開低不是下跌，避免 -8% 誤報）
    div_f = ROOT / "data" / "dividends_upcoming.csv"
    if div_f.exists():
        try:
            dv = pd.read_csv(div_f, dtype={"symbol": str})
            today_str = f"{_now():%Y-%m-%d}"
            for r in dv[dv["ex_date"] == today_str].itertuples():
                if r.symbol in prev_close:
                    pc = prev_close[r.symbol]
                    if r.kind == "CASH":
                        prev_close[r.symbol] = pc - float(r.per_share)
                    else:  # STOCK：每股配 per_share/10 股 → 除權參考價 = 昨收/(1+配股率)
                        prev_close[r.symbol] = pc / (1.0 + float(r.per_share) / 10.0)
                    print(f"[div] {r.symbol} 今日{('除息' if r.kind == 'CASH' else '除權')}，"
                          f"警示基準 {pc:g} → {prev_close[r.symbol]:.2f}")
        except Exception as e:  # noqa: BLE001 - 校正失敗不影響監控主功能
            print(f"[div] 除權息校正失敗：{e}")
    return holdings, rebalance, prev_close


def get_quotes(symbols: list[str], sfx: dict[str, str]) -> dict[str, float]:
    """最新成交價（Yahoo 盤中，可能延遲）。"""
    if not symbols:
        return {}
    tickers = {f"{s}.{sfx.get(s, 'TW')}": s for s in symbols}
    try:
        raw = yf.download(list(tickers), period="1d", interval="5m", progress=False, auto_adjust=False)
    except Exception as e:  # noqa: BLE001
        print(f"quote error: {e}")
        return {}
    out: dict[str, float] = {}
    if raw is None or raw.empty:
        return out
    closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(list(tickers)[0])
    for tkr, sym in tickers.items():
        if tkr in closes.columns:
            s = closes[tkr].dropna()
            if len(s):
                out[sym] = float(s.iloc[-1])
    return out


def main() -> None:
    sfx = _suffix_map()
    holdings, rebalance, prev_close = load_state()

    # ---- 盤前摘要 ----
    fields = []
    if holdings:
        fields.append(("持股", "、".join(f"{label(s)}({n}股)" for s, n in holdings.items())[:1000], False))
    else:
        fields.append(("持股", "空手（holdings.csv 無持股）", True))
    if rebalance is not None:
        buys = rebalance[rebalance["action"] == "BUY"]
        sells = rebalance[rebalance["action"] == "SELL"]
        fields.append(("今日待辦", f"🔄 換股執行日：賣 {len(sells)} 檔 → 買 {len(buys)} 檔", True))
        if len(sells):
            fields.append(("賣出（開盤市價）", "、".join(label(s) for s in sells["symbol"])[:1000], False))
        if len(buys):
            buy_lines = [f"{label(r['symbol'])}　{int(r['suggest_shares'])} 股（約 {int(r['suggest_notional']):,} 元）"
                         for _, r in buys.iterrows()]
            fields.append(("買進（開盤市價；漲幅≥9.5% 放棄）", "\n".join(buy_lines)[:1000], False))
    else:
        fields.append(("今日待辦", "無換股動作，僅監控", True))
    send_embed(f"🌅 盤前摘要 {_now():%Y-%m-%d}", fields=fields, color=C_YELLOW if rebalance is not None else C_BLUE,
               footer="盤中警示僅供參考，月調倉紀律不因盤中波動改變。系統不下單。")

    watch = sorted(set(holdings) | (set(rebalance["symbol"]) if rebalance is not None else set()))
    alerted: set[str] = set()
    heartbeats_due = {"10:30", "11:30", "12:30"}
    open_reported = False
    no_data_polls = 0

    while _now().strftime("%H:%M") < END_TIME:
        quotes = get_quotes(watch, sfx)
        taiex = get_quotes_taiex()

        if not quotes and not taiex:
            no_data_polls += 1
            if no_data_polls == 5 and _now().strftime("%H:%M") > "09:20":
                send_embed("💤 監控結束", "09:20 仍無任何報價——今日可能休市或資料源異常。", color=C_GRAY)
                return
        else:
            no_data_polls = 0

        # 開盤回報（換股日，09:05 後第一次有報價）
        if rebalance is not None and not open_reported and quotes and _now().strftime("%H:%M") >= "09:05":
            rep = []
            for _, r in rebalance[rebalance["action"] == "BUY"].iterrows():
                s = r["symbol"]
                q, pc = quotes.get(s), prev_close.get(s)
                if q is None or pc is None:
                    rep.append(f"{label(s)}：無報價")
                    continue
                gap = q / pc - 1.0
                flag = " → ⛔ 漲幅≥9.5%，**放棄此買單**" if gap >= 0.095 else ""
                rep.append(f"{label(s)}：現價 {q:.2f}（{gap * 100:+.1f}%）{flag}")
            send_embed("🔔 開盤回報（換股執行）", "\n".join(rep)[:3900], color=C_YELLOW)
            open_reported = True

        # 持股警示
        day_pnl = 0.0
        for s, n in holdings.items():
            q, pc = quotes.get(s), prev_close.get(s)
            if q is None or pc is None:
                continue
            chg = q / pc - 1.0
            day_pnl += n * (q - pc)
            if chg <= ALERT_POS_DROP and s not in alerted:
                send_embed(f"⚠️ {label(s)} 觸發警示",
                           f"現價 {q:.2f}，當日 **{chg * 100:+.1f}%**（持有 {n} 股，估 {n * (q - pc):+,.0f} 元）",
                           color=C_RED, footer="資訊警示；依紀律月調倉才動作")
                alerted.add(s)
        if taiex is not None and "TAIEX" not in alerted and taiex <= ALERT_TAIEX_DROP:
            send_embed("📉 大盤警示", f"當日 **{taiex * 100:+.1f}%**，注意風險（資訊警示）", color=C_RED)
            alerted.add("TAIEX")

        # 心跳
        hhmm = _now().strftime("%H:%M")
        due = {t for t in heartbeats_due if t <= hhmm}
        if due:
            heartbeats_due -= due
            send_embed(f"⏳ {hhmm} 監控正常",
                       f"持股 {len(holdings)} 檔｜估當日損益 **{day_pnl:+,.0f}** 元", color=C_GRAY)

        time.sleep(POLL_SEC)

    # ---- 收盤摘要 ----
    quotes = get_quotes(watch, sfx)
    lines = []
    total = 0.0
    for s, n in holdings.items():
        q, pc = quotes.get(s), prev_close.get(s)
        if q is None or pc is None:
            lines.append(f"{label(s)}：無報價")
            continue
        total += n * (q - pc)
        lines.append(f"{label(s)}：{q:.2f}（{(q / pc - 1) * 100:+.1f}%）  {n * (q - pc):+,.0f} 元")
    desc = ("\n".join(lines)[:3800] + f"\n\n估當日組合損益：**{total:+,.0f}** 元") if lines else "空手，無持股損益。"
    send_embed(f"🏁 收盤摘要 {_now():%Y-%m-%d}", desc, color=C_BLUE,
               footer="盤後 17:40 將自動更新資料並發送掃描報告")


def get_quotes_taiex() -> float | None:
    """TAIEX 當日漲跌幅。"""
    try:
        raw = yf.download("^TWII", period="2d", interval="1d", progress=False, auto_adjust=False)
        if raw is None or len(raw) < 2:
            return None
        c = raw["Close"].to_numpy().ravel()
        return float(c[-1] / c[-2] - 1.0)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
