"""全系統參數（STRATEGY.md 第 11 節）。

所有可調參數集中於此，可用 JSON 檔覆寫：Config.from_json(path)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class LiquidityConfig:
    """共通前置條件（STRATEGY.md 第 3 節）。"""
    vol_min: float = 500_000          # 近 N 日中位數成交量下限（股）
    amt_min: float = 30_000_000       # 近 N 日中位數成交金額下限（元）
    lookback: int = 20                # 中位數視窗
    min_price: float = 10.0           # 收盤價下限
    min_history: int = 260            # 指標暖機所需最少 K 棒數


@dataclass
class StrategyAConfig:
    """策略 A：順勢突破（STRATEGY.md 第 4 節）。"""
    enable_t2: bool = True            # T2_BO10 觸發開關
    enable_t3: bool = True            # T3_RECLAIM 觸發開關
    ma_fast: int = 20
    ma_slow: int = 60
    ma_slope_days: int = 3            # MA20 上彎判定：與 N 日前比較
    breakout_high: int = 20           # T1 突破視窗
    breakout_high_short: int = 10     # T2 突破視窗
    vol_mult_breakout: float = 1.5    # T1/T2 量能倍數（× VolMA20）
    vol_mult_reclaim: float = 1.2     # T3 量能倍數
    close_pos_min: float = 0.50       # K 棒品質：收盤位置下限
    close_pos_t2: float = 0.55
    close_pos_t3: float = 0.65
    upper_shadow_max: float = 0.50    # 上影線佔全日振幅上限
    chg_max: float = 0.095            # 當日漲幅上限（漲停鎖死不追）
    ext_max: float = 1.15             # 收盤 / MA20 乖離上限
    atr_stop_mult: float = 1.5        # 初始停損 ATR 倍數
    max_hold_days: int = 60
    ma_exit_days: int = 2             # 連續 N 日收盤 < MA20 出場


@dataclass
class StrategyBConfig:
    """策略 B：破底翻 spring（STRATEGY.md 第 5 節）。"""
    enabled: bool = True
    range_window: int = 60            # 低檔判定視窗
    range_pos_max: float = 0.35       # 60 日區間位置上限
    support_offset: int = 5           # 支撐線：前 offset+window 到前 offset 根
    support_window: int = 20
    undercut_lookback: int = 4        # 檢查最近 N 根跌破
    undercut_min: int = 2             # 至少 N 根跌破支撐
    close_pos_min: float = 0.60
    vol_mult: float = 1.2             # 收復量能 × 昨日 VolMA5
    ma60_slope_days: int = 20         # 接刀防護：MA60 斜率視窗
    ma60_slope_min: float = -0.10     # MA60 N 日變化率下限
    stop_buffer: float = 0.99         # 停損 = spring 低點 × buffer
    time_stop_days: int = 5           # N 日未達 +1R 出場
    time_stop_r: float = 1.0
    max_hold_days: int = 20


@dataclass
class StrategyCConfig:
    """策略 C：波段蓄勢 → 突破（STRATEGY.md 第 6 節）。"""
    ma_fast: int = 20
    ma_slow: int = 60
    ma_slope_days: int = 5
    bbw_window: int = 20              # 布林帶寬視窗
    bbw_pctile_window: int = 252      # 百分位分母
    bbw_pctile_max: float = 0.30      # 壓縮門檻：帶寬 ≤ 自身 30 百分位
    range_compress_fallback: float = 0.15   # 資料不足時退回 (H20-L20)/C 門檻
    watch_valid_days: int = 10        # 觀察名單有效期
    breakout_high: int = 20
    vol_mult_breakout: float = 1.5
    atr_stop_mult: float = 1.5
    max_hold_days: int = 60
    ma_exit_days: int = 2


@dataclass
class PortfolioConfig:
    """組合層與風控（STRATEGY.md 第 7 節）。"""
    initial_equity: float = 1_000_000.0
    risk_per_trade: float = 0.01      # 單筆風險 = 權益 × 1%
    max_position_pct: float = 0.15    # 單一部位市值上限
    max_positions: int = 10
    daily_new_risk_cap: float = 0.04  # 單日新增初始風險上限（權益比）
    cooldown_days: int = 3            # 出場後冷卻期
    regime_ma: int = 60               # 市場濾網 MA
    regime_ma_rising_days: int = 0    # >0 時額外要求濾網 MA 較 N 日前上彎
    bear_b_risk_scale: float = 0.5    # 空頭時策略 B 風險縮放
    partial_take_r: float = 2.0       # +NR 分批停利
    partial_frac: float = 0.5         # 分批比例
    chandelier_mult: float = 2.5      # 吊燈出場 ATR 倍數
    trail_activate_r: float = 1.0     # 獲利 ≥ NR 後啟動移動停損
    lot_size: int = 1000              # 整股單位


@dataclass
class ExecutionConfig:
    """執行成本模型（STRATEGY.md 第 8 節）。"""
    commission_rate: float = 0.001425
    commission_discount: float = 0.6
    commission_min: float = 20.0      # 單筆最低手續費（元）
    tax_sell: float = 0.003           # 證交稅（賣出）
    slippage: float = 0.001           # 單邊滑價
    limit_up_gap: float = 0.095       # T+1 開盤漲幅 ≥ 此值不追
    limit_pct: float = 0.10           # 台股漲跌停幅度


@dataclass
class Config:
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    strat_a: StrategyAConfig = field(default_factory=StrategyAConfig)
    strat_b: StrategyBConfig = field(default_factory=StrategyBConfig)
    strat_c: StrategyCConfig = field(default_factory=StrategyCConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    atr_period: int = 14
    rsi_period: int = 14
    kd_period: int = 9
    supertrend_period: int = 10
    supertrend_mult: float = 3.0

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """從 JSON 覆寫預設參數。JSON 結構為兩層：{"strat_a": {"vol_mult_breakout": 2.0}, ...}"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for f in fields(cfg):
            if f.name not in raw:
                continue
            section = getattr(cfg, f.name)
            override = raw[f.name]
            if hasattr(section, "__dataclass_fields__") and isinstance(override, dict):
                for k, v in override.items():
                    if k not in section.__dataclass_fields__:
                        raise KeyError(f"未知參數 {f.name}.{k}")
                    setattr(section, k, v)
            else:
                setattr(cfg, f.name, override)
        return cfg
