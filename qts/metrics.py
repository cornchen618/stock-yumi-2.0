"""評估層：績效指標（STRATEGY.md 第 9 節 KPI Gate 對應）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_stats(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """由日權益曲線計算組合層指標。"""
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    n_days = len(eq)
    years = n_days / TRADING_DAYS

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_vol = ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() / ret.std(ddof=0) * np.sqrt(TRADING_DAYS)) if ret.std(ddof=0) > 0 else np.nan
    downside = ret[ret < 0]
    sortino = (
        ret.mean() / downside.std(ddof=0) * np.sqrt(TRADING_DAYS)
        if len(downside) > 1 and downside.std(ddof=0) > 0
        else np.nan
    )

    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = dd.min()
    mdd_end = dd.idxmin()
    mdd_start = eq.loc[:mdd_end].idxmax()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    out = {
        "start": eq.index[0].date(),
        "end": eq.index[-1].date(),
        "days": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "mdd_start": mdd_start.date(),
        "mdd_end": mdd_end.date(),
        "calmar": calmar,
    }
    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.dropna()
        out["benchmark_return"] = b.iloc[-1] / b.iloc[0] - 1.0
        bret = b.pct_change().dropna()
        out["benchmark_sharpe"] = (
            bret.mean() / bret.std(ddof=0) * np.sqrt(TRADING_DAYS) if bret.std(ddof=0) > 0 else np.nan
        )
    return out


def trade_stats(trades: pd.DataFrame) -> dict:
    """由交易明細計算交易層指標。分批出場的每一段各記一筆。"""
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.inf,
        "expectancy_r": trades["r_multiple"].mean(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "avg_hold_days": trades["hold_days"].mean(),
        "total_pnl": pnl.sum(),
    }


def attribution(trades: pd.DataFrame) -> pd.DataFrame:
    """逐策略／逐觸發標籤歸因表。"""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["strategy", "trigger"])
    tbl = g.agg(
        n=("pnl", "size"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
        avg_hold=("hold_days", "mean"),
    )
    pf = g["pnl"].apply(lambda s: s[s > 0].sum() / -s[s <= 0].sum() if (s <= 0).any() and s[s <= 0].sum() < 0 else np.inf)
    tbl["profit_factor"] = pf
    return tbl.sort_values("total_pnl", ascending=False)


def exit_reason_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("exit_reason").agg(
        n=("pnl", "size"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    ).sort_values("n", ascending=False)


def kpi_gate(eq_stats: dict, tr_stats: dict) -> pd.DataFrame:
    """STRATEGY.md 第 9 節上線門檻檢核。"""
    checks = [
        ("Profit Factor >= 1.5", tr_stats.get("profit_factor", np.nan), lambda v: v >= 1.5),
        ("期望值 >= +0.15R/筆", tr_stats.get("expectancy_r", np.nan), lambda v: v >= 0.15),
        ("MDD <= 20%", eq_stats.get("mdd", np.nan), lambda v: v >= -0.20),
        ("Sharpe >= 1.0", eq_stats.get("sharpe", np.nan), lambda v: v >= 1.0),
        ("交易筆數 >= 100", tr_stats.get("n_trades", 0), lambda v: v >= 100),
    ]
    rows = []
    for name, val, fn in checks:
        ok = bool(fn(val)) if val is not None and not (isinstance(val, float) and np.isnan(val)) else False
        rows.append({"KPI": name, "值": val, "通過": "PASS" if ok else "FAIL"})
    return pd.DataFrame(rows)
