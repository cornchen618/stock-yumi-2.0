"""盤中監控（排程於交易日 08:55 啟動，13:35 自動結束）。

輸出規則（訊號雜訊比優先，事件驅動而非灌水）：
  08:55 盤前       先推波段候選表（PNG）＋盤前摘要（持股含產業、強勢族群、今日待辦）
  09:05 開盤回報   換股日：逐筆回報開盤價、標記「開盤漲幅≥9.5% → 放棄」的買單
  09:30~13:00 即時報價  每 30 分鐘推持股現價與漲跌%（追蹤價格而非損益，兼作系統心跳）
  急跌警示         持股單日 ≤ −8% 或大盤 ≤ −3%（價格導向，每檔每日一次）
  13:35 收盤摘要   每檔當日漲跌%；當日組合變動僅列一行

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
from qts.market import sector_strength
from qts.notify import C_BLUE, C_GRAY, C_GREEN, C_RED, C_YELLOW, send_embed, send_png
from qts.render import scan_table_png
from qts.scanner import STRAT_ZH, TRIGGER_ZH

NAMES = load_names(ROOT / "data" / "universe.csv")


def _dot(chg_pct: float) -> str:
    """台股慣例：紅漲綠跌。"""
    return "🔴" if chg_pct > 0 else ("🟢" if chg_pct < 0 else "⚪")


def _load_industry() -> dict[str, str]:
    uni = pd.read_csv(ROOT / "data" / "universe.csv", dtype=str)
    if "industry" in uni.columns:
        return dict(zip(uni["code"], uni["industry"].fillna("未分類")))
    return {}


INDUSTRY = _load_industry()


def label(sym: str) -> str:
    n = NAMES.get(sym, "")
    return f"{sym} {n}" if n else sym


def strong_sectors_field(top_n: int = 5) -> tuple[str, str] | None:
    """近期強勢族群（依最新收盤資料；盤前 08:55 當日尚未開盤，故為前一交易日基準）。

    回傳 (欄位標題, 內容) 或 None（資料不足時略過，不阻斷盤前摘要）。
    """
    try:
        px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date", "symbol", "close"])
        px["date"] = pd.to_datetime(px["date"])
        recent = sorted(px["date"].unique())[-130:]           # 只取近 130 交易日，加速 pivot
        px = px[px["date"].isin(recent)]
        panel = px.pivot_table(index="date", columns="symbol", values="close")
        asof = panel.index.max()
        sec = sector_strength(panel, INDUSTRY).head(top_n)
        if not len(sec):
            return None
        held_sectors = {INDUSTRY.get(s) for s in
                        (pd.read_csv(ROOT / "holdings.csv", dtype={"symbol": str})["symbol"]
                         if (ROOT / "holdings.csv").exists() else [])}
        lines = []
        for r in sec.itertuples():
            leaders = "、".join(label(s) for s in r.leaders[:3])
            star = " ⭐持股所在" if r.sector in held_sectors else ""
            lines.append(f"{r.sector}：20日 {r.ret20 * 100:+.1f}%｜{r.breadth60 * 100:.0f}%站上季線{star}\n　領頭：{leaders}")
        return (f"近期強勢族群（依 {asof:%m/%d} 收盤，前 {top_n} 名）", "\n".join(lines)[:1024])
    except Exception as e:  # noqa: BLE001 - 族群計算失敗不阻斷盤前摘要
        print(f"strong_sectors error: {e}")
        return None


def push_latest_scan() -> int:
    """盤前推最新一份波段候選表（來自前一交易日 17:40 盤後掃描）。回傳候選檔數。"""
    files = sorted((ROOT / "output").glob("scan_*.csv"))
    if not files:
        return 0
    day = files[-1].stem.split("_")[-1]
    d = pd.read_csv(files[-1], dtype={"symbol": str})
    if not len(d):
        return 0
    try:
        png = scan_table_png(d, f"{day[:4]}-{day[4:6]}-{day[6:]}", STRAT_ZH, TRIGGER_ZH)
        send_png(png, filename="scan.png",
                 content="📋 **今日波段候選**（前一交易日收盤掃出；紙上觀察，未過上線門檻）")
    except Exception as e:  # noqa: BLE001 - 表格渲染失敗不阻斷盤前
        print(f"push_latest_scan error: {e}")
    return len(d)

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

    # ---- 盤前：先推波段候選表（你指定盤前先看的內容）----
    push_latest_scan()

    # ---- 盤前摘要（持股含產業＋強勢族群＋今日待辦）----
    fields = []
    if holdings:
        hold_lines = [f"{label(s)}｜{INDUSTRY.get(s, '未分類')}｜{n} 股" for s, n in holdings.items()]
        fields.append((f"持股（{len(holdings)} 檔）", "\n".join(hold_lines)[:1024], False))
    else:
        fields.append(("持股", "空手（holdings.csv 無持股）", True))
    # 當天強勢族群（依最新收盤；⭐標記你的持股所在族群）
    sec_field = strong_sectors_field()
    if sec_field is not None:
        fields.append((sec_field[0], sec_field[1], False))
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
               footer="強勢族群為最新收盤基準（盤前尚未開盤）；警示僅供參考，系統不下單。")

    watch = sorted(set(holdings) | (set(rebalance["symbol"]) if rebalance is not None else set()))
    alerted: set[str] = set()
    # 即時報價推送時點：09:30 起每 30 分鐘一次，至 13:00
    snapshot_due = {t for h in range(9, 14) for m in (0, 30)
                    if "09:30" <= (t := f"{h:02d}:{m:02d}") <= "13:00"}
    open_reported = False
    no_data_polls = 0

    while _now().strftime("%H:%M") < END_TIME:
        # 持股熱重載：盤中經 record_trade.py 入帳後，下一輪輪詢即納入監控
        try:
            hp = ROOT / "holdings.csv"
            new_h: dict[str, int] = {}
            if hp.exists():
                hdf = pd.read_csv(hp, dtype={"symbol": str})
                new_h = dict(zip(hdf["symbol"], hdf["shares"].astype(int)))
            if new_h != holdings:
                added = set(new_h) - set(prev_close)
                if added:
                    pxx = pd.read_parquet(ROOT / "data" / "ohlcv.parquet",
                                          columns=["symbol", "date", "raw_close"])
                    pxx = pxx[pxx["symbol"].isin(added)]
                    lastx = pxx.sort_values("date").groupby("symbol").tail(1)
                    prev_close.update(dict(zip(lastx["symbol"], lastx["raw_close"])))
                holdings = new_h
                watch = sorted(set(holdings) | (set(rebalance["symbol"]) if rebalance is not None else set()))
                send_embed("🔄 持股名單已同步",
                           "監控中：" + ("、".join(f"{label(s)}({n}股)" for s, n in holdings.items()) or "空手"),
                           color=C_BLUE)
        except Exception as e:  # noqa: BLE001 - 熱重載失敗不影響既有監控
            print(f"holdings reload error: {e}")

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

        # 急跌警示（價格導向，不報虧損金額；每檔每日一次）
        for s in holdings:
            q, pc = quotes.get(s), prev_close.get(s)
            if q is None or pc is None:
                continue
            chg = q / pc - 1.0
            if chg <= ALERT_POS_DROP and s not in alerted:
                send_embed(f"⚠️ {label(s)} 急跌 {chg * 100:+.1f}%",
                           f"現價 {q:g}（當日跌破 −8%）", color=C_RED,
                           footer="資訊警示；依紀律月調倉才動作")
                alerted.add(s)
        if taiex is not None and "TAIEX" not in alerted and taiex <= ALERT_TAIEX_DROP:
            send_embed("📉 大盤急跌", f"加權指數當日 **{taiex * 100:+.1f}%**（資訊警示）", color=C_RED)
            alerted.add("TAIEX")

        # 持股即時報價（每 30 分鐘；兼作系統心跳，追蹤價格而非損益）
        hhmm = _now().strftime("%H:%M")
        due = {t for t in snapshot_due if t <= hhmm}
        if due:
            snapshot_due -= due
            lines = []
            for s in holdings:
                q, pc = quotes.get(s), prev_close.get(s)
                if q is None or pc is None:
                    lines.append(f"　{label(s)}：無報價")
                    continue
                chg = (q / pc - 1.0) * 100
                lines.append(f"{_dot(chg)} {label(s)}　現價 {q:g}　{chg:+.1f}%")
            if taiex is not None:
                lines.append(f"{_dot(taiex * 100)} 加權指數　{taiex * 100:+.1f}%")
            body = "\n".join(lines) if lines else "空手，僅追蹤大盤。"
            send_embed(f"📈 持股即時報價 {hhmm}", body, color=C_GRAY,
                       footer="Yahoo 報價延遲約 15 分鐘｜紅漲綠跌")

        time.sleep(POLL_SEC)

    # ---- 收盤摘要（價格導向；當日組合變動只列一行，非逐檔報虧損）----
    quotes = get_quotes(watch, sfx)
    lines = []
    total = 0.0
    for s, n in holdings.items():
        q, pc = quotes.get(s), prev_close.get(s)
        if q is None or pc is None:
            lines.append(f"　{label(s)}：無報價")
            continue
        chg = (q / pc - 1.0) * 100
        total += n * (q - pc)
        lines.append(f"{_dot(chg)} {label(s)}　收 {q:g}　{chg:+.1f}%")
    desc = "\n".join(lines)[:3800] if lines else "空手，無持股。"
    if holdings:
        desc += f"\n\n當日組合市值變動（收盤 vs 昨收）：**{total:+,.0f}** 元"
    send_embed(f"🏁 收盤摘要 {_now():%Y-%m-%d}", desc, color=C_BLUE,
               footer="盤後 17:40 將自動更新資料並發送掃描報告與八問簡報")


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
