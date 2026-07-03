"""指標層：向量化技術指標。

所有函式輸入 pandas Series/DataFrame（index=date），輸出對齊的 Series。
凡是「前 N 日極值」類指標一律由呼叫端 shift(1)，本模組不隱含位移。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR（ewm alpha=1/n）。"""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)          # 期間內無下跌 → RSI = 100
    out[avg_gain.isna() | avg_loss.isna()] = np.nan  # 暖機期維持 NaN
    return out


def stochastic_kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """台股慣用 KD：K = 前K*2/3 + RSV/3（即 ewm alpha=1/3），D 同理。"""
    llv = df["low"].rolling(n, min_periods=n).min()
    hhv = df["high"].rolling(n, min_periods=n).max()
    rng = (hhv - llv).replace(0.0, np.nan)
    rsv = (df["close"] - llv) / rng * 100.0
    k = rsv.ewm(alpha=1.0 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_smooth, adjust=False).mean()
    return k, d


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """SuperTrend。回傳 (st_line, direction)；direction: 1 多方 / -1 空方 / 0 未定。"""
    atr_ = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    upper = hl2 + mult * atr_
    lower = hl2 - mult * atr_

    n = len(df)
    f_upper = np.full(n, np.nan)
    f_lower = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int64)
    line = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if i == 0 or np.isnan(f_upper[i - 1]):
            f_upper[i], f_lower[i] = upper[i], lower[i]
            direction[i] = 1 if close[i] > f_upper[i] else -1
        else:
            f_upper[i] = upper[i] if (upper[i] < f_upper[i - 1] or close[i - 1] > f_upper[i - 1]) else f_upper[i - 1]
            f_lower[i] = lower[i] if (lower[i] > f_lower[i - 1] or close[i - 1] < f_lower[i - 1]) else f_lower[i - 1]
            if close[i] > f_upper[i]:
                direction[i] = 1
            elif close[i] < f_lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1] if direction[i - 1] != 0 else -1
        line[i] = f_lower[i] if direction[i] == 1 else f_upper[i]

    idx = df.index
    return pd.Series(line, index=idx), pd.Series(direction, index=idx)


def close_position(df: pd.DataFrame) -> pd.Series:
    """收盤位置 = (C - L) / (H - L)；一字線（H==L）視為 0.5。"""
    rng = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / rng.replace(0.0, np.nan)
    return pos.fillna(0.5)


def upper_shadow_ratio(df: pd.DataFrame) -> pd.Series:
    """上影線 / 全日振幅；一字線視為 0。"""
    rng = df["high"] - df["low"]
    body_top = df[["open", "close"]].max(axis=1)
    ratio = (df["high"] - body_top) / rng.replace(0.0, np.nan)
    return ratio.fillna(0.0)


def bollinger_bandwidth(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """布林帶寬 = (upper - lower) / mid = 2k * std / mid。"""
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return (2.0 * k * std) / mid


def rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """s 當前值在自身過去 window 日（含當日）的百分位（0~1）。"""
    def _rank(x: np.ndarray) -> float:
        return float((x <= x[-1]).mean())
    return s.rolling(window, min_periods=window).apply(_rank, raw=True)
