"""波段掃描核心（run_scan CLI 與 eod_task 共用）＋ Discord 格式化。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .regime import compute_regime
from .signals import compute_signals

TRIGGER_ZH = {
    "T1_BO20": "破20日高",
    "T2_BO10": "破10日高",
    "T3_RECLAIM": "站回月線",
    "B": "破底翻",
    "C": "蓄勢突破",
}
STRAT_ZH = {"A": "A 順勢突破", "B": "B 破底翻", "C": "C 蓄勢突破"}


@dataclass
class ScanResult:
    asof: pd.Timestamp
    bull: bool
    candidates: pd.DataFrame   # symbol,name,strategy,trigger,close,init_stop,suggest_shares,...
    watch: pd.DataFrame        # symbol,name,close


def scan(
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    cfg: Config,
    equity: float,
    names: dict[str, str] | None = None,
    asof: pd.Timestamp | None = None,
) -> ScanResult:
    names = names or {}
    pf = cfg.portfolio
    regime = compute_regime(benchmark, cfg)
    if asof is None:
        asof = max(df.index.max() for df in data.values())
    ridx = regime.index[regime.index <= asof]
    if len(ridx) == 0:
        raise ValueError(f"指數資料早於掃描日 {asof.date()}，請更新指數資料")
    bull = bool(regime.loc[ridx[-1]])

    rows, watch = [], []
    for sym, sdf in data.items():
        if asof not in sdf.index:
            continue
        sig = compute_signals(sdf.loc[:asof], cfg)
        r = sig.iloc[-1]
        if bool(r["watch_c"]):
            watch.append({"symbol": sym, "name": names.get(sym, ""), "close": float(r["raw_close"])})
        for strat, sig_col, stop_col in (("A", "sig_a", "stop_a"), ("B", "sig_b", "stop_b"), ("C", "sig_c", "stop_c")):
            if not bool(r[sig_col]):
                continue
            if strat in ("A", "C") and not bull:
                continue
            scale = 1.0 if (strat != "B" or bull) else pf.bear_b_risk_scale
            stop = float(r[stop_col])
            close = float(r["raw_close"])
            rps = close - stop
            if rps <= 0:
                continue
            shares = int(equity * pf.risk_per_trade * scale / rps // pf.lot_size) * pf.lot_size
            cap = int(equity * pf.max_position_pct / close // pf.lot_size) * pf.lot_size
            shares = min(shares, cap)
            rows.append({
                "symbol": sym,
                "name": names.get(sym, ""),
                "strategy": strat,
                "trigger": r["trig_a"] if strat == "A" else strat,
                "close": close,
                "init_stop": round(stop, 2),
                "target_partial": round(close + pf.partial_take_r * rps, 2),
                "risk_per_share": round(rps, 2),
                "suggest_shares": shares,
                "suggest_notional": round(shares * close),
                "max_loss": round(rps * shares),   # 停損打到時的實際虧損金額
                "rank_score": round(float(r["rank_score"]), 2),
            })

    cand = pd.DataFrame(rows)
    if len(cand):
        cand = cand.sort_values(["strategy", "rank_score"], ascending=[True, False]).reset_index(drop=True)
    return ScanResult(asof=asof, bull=bull, candidates=cand, watch=pd.DataFrame(watch))


# ---------- Discord 格式化（等寬區塊、中文欄位、行動裝置友善） ----------

def _w(s: str) -> int:
    """顯示寬度：CJK 字元算 2 格。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))


def pad(s: str, width: int) -> str:
    s = str(s)
    return s + " " * max(0, width - _w(s))


def format_scan_discord(res: ScanResult) -> str:
    c = res.candidates
    n_a = int((c["strategy"] == "A").sum()) if len(c) else 0
    n_b = int((c["strategy"] == "B").sum()) if len(c) else 0
    n_c = int((c["strategy"] == "C").sum()) if len(c) else 0
    head = (
        f":clipboard: **波段掃描 {res.asof:%m/%d}**（紙上觀察—未過上線門檻，勿實單）\n"
        f"市場濾網：{'多頭 ✅（三策略皆可）' if res.bull else '空頭 ⛔（A/C 停用、B 風險減半）'}"
        f"｜候選 A:{n_a} B:{n_b} C:{n_c}｜壓縮觀察 {len(res.watch)} 檔"
    )
    if not len(c):
        return head + "\n今日無任何進場候選。"

    blocks = [head]
    for strat in ("A", "B", "C"):
        g = c[c["strategy"] == strat]
        if not len(g):
            continue
        lines = [f"【{STRAT_ZH[strat]}】{len(g)} 檔"]
        for _, r in g.iterrows():
            shares = int(r["suggest_shares"])
            size = f"{shares // 1000}張" if shares >= 1000 else "資金不足"
            lines.append(
                pad(r["symbol"], 6) + pad(r["name"], 11)
                + pad(TRIGGER_ZH.get(str(r["trigger"]), str(r["trigger"])), 10)
                + pad(f"收{r['close']:g}", 9)
                + pad(f"損{r['init_stop']:g}", 9)
                + pad(f"利{r['target_partial']:g}", 9)
                + size
            )
        blocks.append("```\n" + "\n".join(lines) + "\n```")
    blocks.append(
        "_欄位說明：收=掃描日收盤價｜損=初始停損（跌破次日出場）｜"
        "利=+2R 分批停利價（到價先出一半、剩餘停損上移到成本，之後吊燈式移動停損讓獲利奔跑）｜"
        "張=以 1% 風險與 15% 部位上限算出的建議張數，「資金不足」= 該股一張就超過風險額度_"
    )
    return "\n".join(blocks)


def format_rebalance_discord(reb: pd.DataFrame, asof_label: str, equity: float, top_n: int = 20) -> str:
    """動能換股清單（scan_momentum 產出的 CSV）→ Discord 格式。"""
    head = (
        f":moneybag: **動能組合月度換股清單**（訊號日 {asof_label} 收盤｜明日開盤執行）\n"
        f"每檔目標 {equity / top_n:,.0f} 元｜開盤漲幅 ≥ 9.5% 的買單放棄"
    )
    blocks = [head]
    for action, title in (("KEEP", "續抱"), ("SELL", "賣出（開盤市價）"), ("BUY", "買進（開盤市價）")):
        g = reb[reb["action"] == action]
        if not len(g):
            continue
        lines = [f"【{title}】{len(g)} 檔"]
        for _, r in g.iterrows():
            if action == "BUY":
                close_col = [x for x in reb.columns if x.startswith("close_")]
                px = r[close_col[0]] if close_col else ""
                lines.append(
                    pad(r["symbol"], 6) + pad(r.get("name", ""), 11)
                    + pad(f"#{int(r['rank'])}", 5)
                    + pad(f"動能{r['momentum%']:+.0f}%", 11)
                    + pad(f"收{px:g}", 10)
                    + f"{int(r['suggest_shares'])}股"
                )
            else:
                lines.append(pad(r["symbol"], 6) + pad(r.get("name", ""), 11))
        blocks.append("```\n" + "\n".join(lines) + "\n```")
    blocks.append("_成交後請更新 holdings.csv（symbol,shares）；#=動能排名（12個月報酬、跳過最近1個月）_")
    return "\n".join(blocks)
