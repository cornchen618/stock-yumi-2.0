"""系統自我檢查：指標正確性、無前視偏差、回測引擎帳務一致性。

全部通過才輸出 ALL PASS。任何 assert 失敗代表系統不可信，禁止上線。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts import indicators as ind
from qts.backtest import Backtester
from qts.config import Config
from qts.signals import compute_signals
from scripts.make_sample_data import generate


def _mkdf(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1e6), "amount": close * 1e6,
    }, index=idx)


def test_indicators() -> None:
    # SMA 精確值
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ind.sma(s, 3).iloc[-1] == 4.0

    # RSI：純上漲 → 100；區間 [0, 100]
    up = _mkdf(np.linspace(100, 200, 60))
    r = ind.rsi(up["close"], 14)
    assert abs(r.iloc[-1] - 100.0) < 1e-6, f"純上漲 RSI 應為 100，得 {r.iloc[-1]}"
    rng = np.random.default_rng(1)
    noisy = _mkdf(100 * np.cumprod(1 + 0.02 * rng.standard_normal(300)))
    rn = ind.rsi(noisy["close"], 14).dropna()
    assert ((rn >= 0) & (rn <= 100)).all()

    # ATR 恆正
    a = ind.atr(noisy, 14).dropna()
    assert (a > 0).all()

    # KD 在 [0, 100]
    k, d = ind.stochastic_kd(noisy, 9)
    kd = pd.concat([k, d], axis=1).dropna()
    assert ((kd >= -1e-9) & (kd <= 100 + 1e-9)).all().all()

    # SuperTrend：持續上漲末端應為多方
    st_line, st_dir = ind.supertrend(up, 10, 3.0)
    assert st_dir.iloc[-1] == 1
    assert (st_line.dropna().iloc[-5:] < up["close"].iloc[-5:]).all()

    # 收盤位置：一字線 = 0.5
    doji = _mkdf(np.full(30, 50.0))
    doji["high"] = doji["low"] = doji["open"] = doji["close"]
    assert (ind.close_position(doji) == 0.5).all()
    print("  [PASS] indicators：SMA / RSI / ATR / KD / SuperTrend / close_position")


def test_no_lookahead() -> None:
    """截斷尾部資料不得改變任何歷史訊號值（全因果性檢查）。"""
    cfg = Config()
    rng = np.random.default_rng(7)
    n = 500
    close = 100 * np.cumprod(1 + np.clip(0.0005 + 0.02 * rng.standard_normal(n), -0.099, 0.099))
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * (1 + np.clip(rng.normal(0, 0.004, n), -0.05, 0.05))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = np.abs(rng.normal(2e6, 8e5, n)) + 6e5
    df = pd.DataFrame({
        "open": open_, "high": hi, "low": lo, "close": close,
        "volume": vol, "amount": close * vol,
    }, index=pd.bdate_range("2023-01-02", periods=n))

    full = compute_signals(df, cfg)
    trunc = compute_signals(df.iloc[:-30], cfg)
    cols = ["sig_a", "sig_b", "sig_c", "watch_c", "trig_a", "stop_a", "stop_b", "stop_c", "rank_score"]
    a = full.iloc[:-30][cols]
    b = trunc[cols]
    for col in cols:
        sa, sb = a[col], b[col]
        if sa.dtype == object:
            ok = (sa.fillna("") == sb.fillna("")).all()
        else:
            ok = np.allclose(sa.astype(float).fillna(-9e9), sb.astype(float).fillna(-9e9))
        assert ok, f"前視偏差：截斷資料後 {col} 的歷史值改變"
    print("  [PASS] no-lookahead：截斷 30 日後所有歷史訊號不變")


def test_backtest_accounting() -> None:
    """帳務恆等式：期末權益 − 期初 = Σ已實現損益 + Σ未平倉(市值−成本−進場費攤提)。"""
    cfg = Config()
    ohlcv, bench = generate(n_stocks=40, start="2021-01-04", end="2024-12-31", seed=99)
    data = {s: g.set_index("date").drop(columns="symbol") for s, g in ohlcv.groupby("symbol")}
    benchmark = bench.set_index(pd.to_datetime(bench["date"])).drop(columns="date")

    res = Backtester(cfg).run(data, benchmark)
    assert len(res.trades) > 0, "合成資料應產生交易（訊號→執行鏈路斷裂？）"
    assert res.equity.notna().all() and (res.equity > 0).all(), "權益曲線含 NaN 或負值"

    realized = res.trades["pnl"].sum()
    unrealized = res.final_positions["unrealized"].sum() if not res.final_positions.empty else 0.0
    diff = res.equity.iloc[-1] - cfg.portfolio.initial_equity
    gap = abs(diff - (realized + unrealized))
    assert gap < 1.0, f"帳務不一致：權益變動 {diff:,.2f} != 已實現 {realized:,.2f} + 未實現 {unrealized:,.2f}（差 {gap:,.2f}）"

    # 交易紀錄結構完整性
    t = res.trades
    assert (t["exit_date"] >= t["entry_date"]).all(), "出場日早於進場日"
    assert (t["shares"] > 0).all() and (t["shares"] % cfg.portfolio.lot_size == 0).all(), "非整張交易"
    assert (t["entry_price"] > t["init_stop"]).all(), "初始停損高於進場價"
    stop_trades = t[t["exit_reason"] == "STOP_INIT"]
    if len(stop_trades):
        assert (stop_trades["r_multiple"] <= 0.05).all(), "初始停損出場的 R 應約 <= 0（滑價與跳空除外）"
    print(f"  [PASS] backtest：{len(t)} 筆交易，帳務恆等式通過（差額 {gap:.4f} 元）")


def main() -> None:
    print("qts selftest")
    print("-" * 48)
    test_indicators()
    test_no_lookahead()
    test_backtest_accounting()
    print("-" * 48)
    print("ALL PASS")


if __name__ == "__main__":
    main()
