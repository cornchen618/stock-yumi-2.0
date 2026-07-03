"""市場濾網（STRATEGY.md 第 7.1 節）：大盤收盤 vs MA60。

bull=True  → 策略 A/C 可新進場、B 正常風險
bull=False → 策略 A/C 停止新進場、B 風險縮放（cfg.portfolio.bear_b_risk_scale）
"""
from __future__ import annotations

import pandas as pd

from .config import Config


def compute_regime(benchmark: pd.DataFrame, cfg: Config) -> pd.Series:
    """回傳 index=date 的布林 Series：True=多頭（收盤 > MA60）。"""
    ma = benchmark["close"].rolling(cfg.portfolio.regime_ma, min_periods=cfg.portfolio.regime_ma).mean()
    bull = benchmark["close"] > ma
    rising = cfg.portfolio.regime_ma_rising_days
    if rising > 0:
        bull &= ma > ma.shift(rising)
    return bull.fillna(False)
