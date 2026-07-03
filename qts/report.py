"""報告輸出：文字摘要、權益曲線圖、交易明細 CSV。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from . import metrics as M


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%" if pd.notna(v) else "N/A"


def summarize(result: BacktestResult) -> str:
    eq = M.equity_stats(result.equity, result.benchmark)
    tr = M.trade_stats(result.trades)
    lines = [
        "=" * 64,
        f"回測區間        {eq['start']} ~ {eq['end']}（{eq['days']} 個交易日）",
        "=" * 64,
        f"總報酬          {_fmt_pct(eq['total_return'])}    大盤同期 {_fmt_pct(eq.get('benchmark_return', float('nan')))}",
        f"CAGR            {_fmt_pct(eq['cagr'])}",
        f"年化波動        {_fmt_pct(eq['ann_vol'])}",
        f"Sharpe          {eq['sharpe']:.2f}    大盤 Sharpe {eq.get('benchmark_sharpe', float('nan')):.2f}",
        f"Sortino         {eq['sortino']:.2f}",
        f"最大回撤        {_fmt_pct(eq['mdd'])}（{eq['mdd_start']} ~ {eq['mdd_end']}）",
        f"Calmar          {eq['calmar']:.2f}",
        f"平均曝險        {_fmt_pct(result.exposure.mean())}",
        "-" * 64,
    ]
    if tr["n_trades"] == 0:
        lines.append("無任何交易（檢查資料區間 / 流動性門檻 / 濾網）")
    else:
        lines += [
            f"交易筆數        {tr['n_trades']}（含分批出場各記一筆）",
            f"勝率            {_fmt_pct(tr['win_rate'])}",
            f"Profit Factor   {tr['profit_factor']:.2f}",
            f"期望值          {tr['expectancy_r']:+.3f} R / 筆",
            f"平均獲利/虧損   {tr['avg_win']:,.0f} / {tr['avg_loss']:,.0f}",
            f"平均持有        {tr['avg_hold_days']:.1f} 日",
            f"總損益          {tr['total_pnl']:,.0f}",
        ]
    lines.append("-" * 64)
    lines.append("【策略／觸發歸因】")
    attr = M.attribution(result.trades)
    lines.append(attr.to_string() if not attr.empty else "（無交易）")
    lines.append("-" * 64)
    lines.append("【出場原因分布】")
    er = M.exit_reason_table(result.trades)
    lines.append(er.to_string() if not er.empty else "（無交易）")
    lines.append("-" * 64)
    lines.append("【上線門檻 KPI Gate（STRATEGY.md 第 9 節）】")
    lines.append(M.kpi_gate(eq, tr).to_string(index=False))
    if result.skip_counts:
        lines.append("-" * 64)
        lines.append("【被略過的訊號統計】")
        for k, v in sorted(result.skip_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<18} {v}")
    lines.append("=" * 64)
    return "\n".join(lines)


def save_report(result: BacktestResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    text = summarize(result)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    result.equity.to_csv(out / "equity.csv", header=True)
    if not result.trades.empty:
        result.trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    eq_norm = result.equity / result.equity.iloc[0]
    bm_norm = result.benchmark / result.benchmark.iloc[0]
    axes[0].plot(eq_norm.index, eq_norm.values, label="Strategy", linewidth=1.4)
    axes[0].plot(bm_norm.index, bm_norm.values, label="Benchmark", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Equity Curve (normalized)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    dd = result.equity / result.equity.cummax() - 1.0
    axes[1].fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=120)
    plt.close(fig)
    return out
