"""組合層事件式回測引擎（STRATEGY.md 第 7、8 節）。

時序模型（每個交易日 T）：
  1. 執行前一日排入的出場單（T 開盤價 − 滑價）
  2. 執行前一日排入的進場單（T 開盤價 + 滑價；開盤漲幅 ≥ 9.5% 放棄）
  3. 盤中管理：初始/移動停損（保守假設：停損優先於停利）、+2R 分批停利
  4. 收盤更新：移動停損上移、訊號出場/時間停損排入次日、下市強制出清
  5. 收盤產生新訊號 → 市場濾網 → 排序 → 排入次日進場
  6. 權益結算（cash + Σ 持股 × 收盤價）

成本模型：手續費（買賣各一次、最低 20 元）、證交稅（賣出）、單邊滑價。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .regime import compute_regime
from .signals import compute_signals


@dataclass
class Position:
    symbol: str
    strategy: str            # "A" / "B" / "C"
    trigger: str
    entry_i: int
    entry_date: pd.Timestamp
    entry_price: float       # 含滑價
    shares: int
    init_stop: float
    stop: float
    r_value: float           # entry_price - init_stop
    entry_fee_ps: float      # 進場手續費 / 股（供部分出場攤提）
    half_taken: bool = False
    bars_held: int = 0
    hh_close: float = 0.0
    hh_high: float = 0.0
    below_ma_count: int = 0
    last_close: float = 0.0


@dataclass
class PendingEntry:
    symbol: str
    strategy: str
    trigger: str
    stop: float
    risk_scale: float
    rank: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    benchmark: pd.Series
    regime: pd.Series
    skip_counts: dict[str, int]
    exposure: pd.Series
    cfg: Config
    final_positions: pd.DataFrame = field(default_factory=pd.DataFrame)


# 引擎內部使用的欄位（reindex 至主日曆後轉 numpy）
_ARRAY_COLS = [
    "open", "high", "low", "close", "volume", "ma20", "atr", "st_line", "st_dir",
    "stop_a", "stop_b", "stop_c", "rank_score",
]


class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---------- 資料準備 ----------

    def _prepare(
        self, data: dict[str, pd.DataFrame], cal: pd.DatetimeIndex
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, list[PendingEntry]], dict[str, int]]:
        arrays: dict[str, dict[str, np.ndarray]] = {}
        candidates_by_i: dict[int, list[PendingEntry]] = {}
        last_valid_i: dict[str, int] = {}

        for sym, sdf in data.items():
            sig = compute_signals(sdf, self.cfg)
            sig = sig.reindex(cal)

            arr = {col: sig[col].to_numpy(dtype=float) for col in _ARRAY_COLS}
            close_ffill = sig["close"].ffill().to_numpy(dtype=float)
            arr["close_ffill"] = close_ffill
            arr["prev_close_traded"] = np.concatenate([[np.nan], close_ffill[:-1]])
            arrays[sym] = arr

            valid = np.nonzero(~np.isnan(arr["close"]))[0]
            if len(valid) == 0:
                continue
            last_valid_i[sym] = int(valid[-1])

            trig_a = sig["trig_a"].fillna("").to_numpy(dtype=object)
            for strat, sig_col, stop_col in (
                ("A", "sig_a", "stop_a"),
                ("B", "sig_b", "stop_b"),
                ("C", "sig_c", "stop_c"),
            ):
                mask = sig[sig_col].fillna(False).to_numpy(dtype=bool)
                for i in np.nonzero(mask)[0]:
                    stop = arr[stop_col][i]
                    if np.isnan(stop):
                        continue
                    trig = trig_a[i] if strat == "A" else strat
                    candidates_by_i.setdefault(int(i), []).append(
                        PendingEntry(sym, strat, str(trig), float(stop), 1.0, float(arr["rank_score"][i]))
                    )
        return arrays, candidates_by_i, last_valid_i

    # ---------- 成本 ----------

    def _buy_fee(self, notional: float) -> float:
        ex = self.cfg.execution
        return max(ex.commission_min, notional * ex.commission_rate * ex.commission_discount)

    def _sell_fee(self, notional: float) -> float:
        ex = self.cfg.execution
        return max(ex.commission_min, notional * ex.commission_rate * ex.commission_discount) + notional * ex.tax_sell

    # ---------- 主迴圈 ----------

    def run(
        self,
        data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        pf, ex = cfg.portfolio, cfg.execution

        cal = benchmark.index
        if start:
            cal = cal[cal >= pd.Timestamp(start)]
        if end:
            cal = cal[cal <= pd.Timestamp(end)]
        if len(cal) < 2:
            raise ValueError("回測區間不足 2 個交易日")

        bench = benchmark.reindex(benchmark.index)  # 完整指數算濾網（含區間前暖機）
        regime_full = compute_regime(bench, cfg)
        regime = regime_full.reindex(cal).fillna(False)

        arrays, candidates_by_i, last_valid_i = self._prepare(data, cal)

        cash = pf.initial_equity
        last_equity = pf.initial_equity
        positions: dict[str, Position] = {}
        pending_entries: list[PendingEntry] = []
        pending_exits: list[tuple[str, str]] = []  # (symbol, reason)
        last_exit_i: dict[str, int] = {}
        trades: list[dict] = []
        skip_counts: dict[str, int] = {}
        equity_arr = np.zeros(len(cal))
        exposure_arr = np.zeros(len(cal))

        def _skip(reason: str) -> None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1

        def _record(pos: Position, i: int, shares: int, price: float, reason: str) -> float:
            """賣出 shares 股，回傳入帳現金；同時寫入交易紀錄。"""
            notional = shares * price
            fee = self._sell_fee(notional)
            pnl = shares * (price - pos.entry_price) - shares * pos.entry_fee_ps - fee
            trades.append({
                "symbol": pos.symbol,
                "strategy": pos.strategy,
                "trigger": pos.trigger,
                "entry_date": pos.entry_date,
                "entry_price": round(pos.entry_price, 4),
                "exit_date": cal[i],
                "exit_price": round(price, 4),
                "shares": shares,
                "init_stop": round(pos.init_stop, 4),
                "r_value": round(pos.r_value, 4),
                "r_multiple": round((price - pos.entry_price) / pos.r_value, 3) if pos.r_value > 0 else np.nan,
                "hold_days": pos.bars_held,
                "pnl": round(pnl, 2),
                "pnl_pct": round(price / pos.entry_price - 1.0, 5),
                "exit_reason": reason,
            })
            return notional - fee

        for i in range(len(cal)):
            today_new_risk = 0.0

            # ---- 1. 執行排入的出場單（開盤） ----
            still_pending: list[tuple[str, str]] = []
            for sym, reason in pending_exits:
                pos = positions.get(sym)
                if pos is None:
                    continue
                o = arrays[sym]["open"][i] if i < len(cal) else np.nan
                if np.isnan(o):
                    still_pending.append((sym, reason))  # 停牌，順延
                    continue
                price = o * (1.0 - ex.slippage)
                cash += _record(pos, i, pos.shares, price, reason)
                del positions[sym]
                last_exit_i[sym] = i
            pending_exits = still_pending

            # ---- 2. 執行排入的進場單（開盤） ----
            for pe in sorted(pending_entries, key=lambda x: -x.rank):
                if len(positions) >= pf.max_positions:
                    _skip("max_positions")
                    continue
                if pe.symbol in positions:
                    _skip("already_held")
                    continue
                arr = arrays[pe.symbol]
                o = arr["open"][i]
                if np.isnan(o):
                    _skip("no_data")
                    continue
                prev_c = arr["prev_close_traded"][i]
                if not np.isnan(prev_c) and (o / prev_c - 1.0) >= ex.limit_up_gap:
                    _skip("limit_up_gap")
                    continue
                fill = o * (1.0 + ex.slippage)
                if fill <= pe.stop:
                    _skip("gap_below_stop")
                    continue

                risk_amt = last_equity * pf.risk_per_trade * pe.risk_scale
                rps = fill - pe.stop
                lots = int(risk_amt / rps // pf.lot_size)
                shares = lots * pf.lot_size
                cap_shares = int(last_equity * pf.max_position_pct / fill // pf.lot_size) * pf.lot_size
                shares = min(shares, cap_shares)
                comm_eff = ex.commission_rate * ex.commission_discount
                affordable = int(cash / (fill * (1.0 + comm_eff)) // pf.lot_size) * pf.lot_size
                shares = min(shares, affordable)
                if shares < pf.lot_size:
                    _skip("size_zero")
                    continue
                new_risk = rps * shares
                if today_new_risk + new_risk > last_equity * pf.daily_new_risk_cap:
                    _skip("daily_risk_cap")
                    continue

                notional = shares * fill
                fee = self._buy_fee(notional)
                cash -= notional + fee
                positions[pe.symbol] = Position(
                    symbol=pe.symbol, strategy=pe.strategy, trigger=pe.trigger,
                    entry_i=i, entry_date=cal[i], entry_price=fill, shares=shares,
                    init_stop=pe.stop, stop=pe.stop, r_value=rps,
                    entry_fee_ps=fee / shares, hh_close=fill, hh_high=fill,
                    last_close=fill,
                )
                today_new_risk += new_risk
            pending_entries = []

            # ---- 3. 盤中管理（停損優先於停利：保守假設） ----
            for sym in list(positions.keys()):
                pos = positions[sym]
                arr = arrays[sym]
                o, h, l = arr["open"][i], arr["high"][i], arr["low"][i]
                if np.isnan(l):
                    continue  # 停牌
                if l <= pos.stop:
                    raw = o if o < pos.stop else pos.stop
                    price = raw * (1.0 - ex.slippage)
                    reason = "STOP_INIT" if pos.stop <= pos.init_stop else "STOP_TRAIL"
                    cash += _record(pos, i, pos.shares, price, reason)
                    del positions[sym]
                    last_exit_i[sym] = i
                    continue
                target = pos.entry_price + pf.partial_take_r * pos.r_value
                if not pos.half_taken and h >= target:
                    pos.half_taken = True
                    pos.stop = max(pos.stop, pos.entry_price)  # 停損上移至損益兩平
                    sell = int(pos.shares * pf.partial_frac // pf.lot_size) * pf.lot_size
                    if sell >= pf.lot_size and pos.shares - sell >= pf.lot_size:
                        raw = max(o, target)  # 跳空高於目標則以開盤價成交（有利）
                        price = raw * (1.0 - ex.slippage)
                        cash += _record(pos, i, sell, price, "PARTIAL_2R")
                        pos.shares -= sell

            # ---- 4. 收盤更新與出場訊號 ----
            for sym in list(positions.keys()):
                pos = positions[sym]
                arr = arrays[sym]
                c = arr["close"][i]
                if np.isnan(c):
                    continue  # 停牌：不更新、不出訊號
                pos.bars_held += 1
                pos.last_close = c
                pos.hh_close = max(pos.hh_close, c)
                pos.hh_high = max(pos.hh_high, arr["high"][i])

                # 移動停損（獲利 ≥ trail_activate_r 後啟動；只上移不下移）
                if pos.hh_high >= pos.entry_price + pf.trail_activate_r * pos.r_value:
                    cand = pos.hh_close - pf.chandelier_mult * arr["atr"][i]
                    if arr["st_dir"][i] == 1 and not np.isnan(arr["st_line"][i]):
                        cand = max(cand, arr["st_line"][i])
                    if not np.isnan(cand):
                        pos.stop = max(pos.stop, cand)

                # 下市/資料終止：以最後可交易日收盤強制出清
                if last_valid_i.get(sym, len(cal) - 1) == i and i < len(cal) - 1:
                    price = c * (1.0 - ex.slippage)
                    cash += _record(pos, i, pos.shares, price, "DELIST")
                    del positions[sym]
                    last_exit_i[sym] = i
                    continue

                # 訊號出場（次日開盤執行）
                queued = False
                if pos.strategy in ("A", "C"):
                    sc = cfg.strat_a if pos.strategy == "A" else cfg.strat_c
                    ma20 = arr["ma20"][i]
                    pos.below_ma_count = pos.below_ma_count + 1 if (not np.isnan(ma20) and c < ma20) else 0
                    if pos.below_ma_count >= sc.ma_exit_days:
                        pending_exits.append((sym, "MA20_EXIT")); queued = True
                    elif pos.bars_held >= sc.max_hold_days:
                        pending_exits.append((sym, "TIME_MAX")); queued = True
                else:  # B
                    sb = cfg.strat_b
                    if pos.bars_held >= sb.max_hold_days:
                        pending_exits.append((sym, "TIME_MAX")); queued = True
                    elif (
                        pos.bars_held >= sb.time_stop_days
                        and pos.hh_high < pos.entry_price + sb.time_stop_r * pos.r_value
                    ):
                        pending_exits.append((sym, "TIME_NO_FOLLOW")); queued = True
                _ = queued

            # ---- 5. 新訊號 → 濾網 → 排入次日 ----
            queued_exit_syms = {s for s, _ in pending_exits}
            bull = bool(regime.iloc[i])
            for pe in candidates_by_i.get(i, []):
                if pe.symbol in positions or pe.symbol in queued_exit_syms:
                    continue
                if i + 1 - last_exit_i.get(pe.symbol, -10**9) <= pf.cooldown_days:
                    _skip("cooldown")
                    continue
                if pe.strategy in ("A", "C") and not bull:
                    _skip("regime_block")
                    continue
                scale = 1.0 if (pe.strategy != "B" or bull) else pf.bear_b_risk_scale
                pending_entries.append(
                    PendingEntry(pe.symbol, pe.strategy, pe.trigger, pe.stop, scale, pe.rank)
                )

            # ---- 6. 權益結算 ----
            invested = sum(p.shares * p.last_close for p in positions.values())
            equity_arr[i] = cash + invested
            exposure_arr[i] = invested / equity_arr[i] if equity_arr[i] > 0 else 0.0
            last_equity = equity_arr[i]

        equity = pd.Series(equity_arr, index=cal, name="equity")
        exposure = pd.Series(exposure_arr, index=cal, name="exposure")
        trades_df = pd.DataFrame(trades)
        bench_close = benchmark["close"].reindex(cal).ffill()
        final_pos = pd.DataFrame([
            {
                "symbol": p.symbol, "strategy": p.strategy, "trigger": p.trigger,
                "entry_date": p.entry_date, "entry_price": p.entry_price,
                "shares": p.shares, "stop": p.stop, "last_close": p.last_close,
                "entry_fee_ps": p.entry_fee_ps,
                "unrealized": p.shares * (p.last_close - p.entry_price) - p.shares * p.entry_fee_ps,
            }
            for p in positions.values()
        ])
        return BacktestResult(
            equity=equity, trades=trades_df, benchmark=bench_close,
            regime=regime, skip_counts=skip_counts, exposure=exposure, cfg=cfg,
            final_positions=final_pos,
        )
