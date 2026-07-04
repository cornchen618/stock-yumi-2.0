"""橫斷面動能組合（月調倉）— 預先登記規格：

- 訊號：每月最後交易日收盤，動能 = close[t-skip] / close[t-lookback] − 1（12-1：lookback=252, skip=21）
- 資格：近 20 日中位數成交量 ≥ 50 萬股、金額 ≥ 3,000 萬、收盤 ≥ 10 元、動能可計算
- 選股：動能 > 0（絕對動能閘門）中排名前 top_n（20）檔，等權
- Banding：既有持股若動能 > 0 且排名 ≤ band_rank（40）則續抱，降低換手
- 執行：次一交易日開盤（±滑價）；開盤漲幅 ≥ 9.5% 放棄買進；停牌順延
- 成本：手續費 0.1425%×0.6、賣出證交稅 0.3%、單邊滑價 0.1%
- 交易單位：1 股（盤中零股交易，2020/10 後制度存在；此為簡化假設，已於報告標註）

已知限制：資料無下市股票（存活者偏差，結果偏樂觀）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MomentumConfig:
    lookback: int = 252          # 動能遠端（交易日）
    skip: int = 21               # 跳過最近 N 日（短期反轉）
    top_n: int = 20
    band_rank: int = 40          # 持股排名低於此才賣出
    overlay_pool: int = 0        # >0：雙重排序——動能前 N 名中依 overlay 分數取 top_n（0=關閉）
    exposure_ma: int = 0         # >0：訊號日大盤 < MA(N) 時，新買部位目標金額乘以 exposure_bear_mult
    exposure_bear_mult: float = 0.5
    liq_vol_min: float = 500_000
    liq_amt_min: float = 30_000_000
    liq_window: int = 20
    min_price: float = 10.0
    initial_equity: float = 1_000_000.0
    commission_rate: float = 0.001425
    commission_discount: float = 0.6
    commission_min: float = 20.0
    tax_sell: float = 0.003
    slippage: float = 0.001
    limit_up_gap: float = 0.095


@dataclass
class MomoPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_i: int
    entry_date: pd.Timestamp
    entry_fee: float
    last_close: float


@dataclass
class MomentumResult:
    equity: pd.Series
    trades: pd.DataFrame
    benchmark: pd.Series
    exposure: pd.Series
    annual_turnover: float
    cfg: MomentumConfig
    final_positions: pd.DataFrame = field(default_factory=pd.DataFrame)


class MomentumBacktester:
    def __init__(self, cfg: MomentumConfig):
        self.cfg = cfg

    def _buy_fee(self, notional: float) -> float:
        c = self.cfg
        return max(c.commission_min, notional * c.commission_rate * c.commission_discount)

    def _sell_fee(self, notional: float) -> float:
        c = self.cfg
        return max(c.commission_min, notional * c.commission_rate * c.commission_discount) + notional * c.tax_sell

    def run(
        self,
        data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        overlay: pd.DataFrame | None = None,
    ) -> MomentumResult:
        """overlay：date × symbol 的次要分數面板（僅在 cfg.overlay_pool > 0 時使用）。"""
        cfg = self.cfg
        cal = benchmark.index
        if overlay is not None:
            overlay = overlay.reindex(cal)
        bench_bull = None
        if cfg.exposure_ma > 0:
            ma = benchmark["close"].rolling(cfg.exposure_ma, min_periods=cfg.exposure_ma).mean()
            bench_bull = (benchmark["close"] > ma).fillna(True)  # 暖機期不減碼

        # ---- 面板（全歷史，供指標暖機；主迴圈僅走 [start, end]）----
        close = pd.DataFrame({s: d["close"] for s, d in data.items()}).reindex(cal)
        open_ = pd.DataFrame({s: d["open"] for s, d in data.items()}).reindex(cal)
        volume = pd.DataFrame({s: d["volume"] for s, d in data.items()}).reindex(cal)
        amount = pd.DataFrame({s: d["amount"] for s, d in data.items()}).reindex(cal)

        mom = close.shift(cfg.skip) / close.shift(cfg.lookback) - 1.0
        med_vol = volume.rolling(cfg.liq_window, min_periods=cfg.liq_window).median()
        med_amt = amount.rolling(cfg.liq_window, min_periods=cfg.liq_window).median()
        eligible = (
            (med_vol >= cfg.liq_vol_min)
            & (med_amt >= cfg.liq_amt_min)
            & (close >= cfg.min_price)
            & mom.notna()
        )
        close_ffill = close.ffill()
        prev_close_traded = close_ffill.shift(1)
        last_valid_i = {s: int(np.nonzero(close[s].notna().to_numpy())[0][-1])
                        for s in close.columns if close[s].notna().any()}

        # 主迴圈範圍與月底訊號日
        pos_idx = np.arange(len(cal))
        in_win = np.ones(len(cal), dtype=bool)
        if start:
            in_win &= cal >= pd.Timestamp(start)
        if end:
            in_win &= cal <= pd.Timestamp(end)
        win_idx = pos_idx[in_win]
        if len(win_idx) < 40:
            raise ValueError("回測區間過短")
        month_key = cal.to_period("M")
        sig_is = [
            int(g.iloc[-1])
            for _, g in pd.Series(win_idx).groupby(month_key[win_idx])
            if int(g.iloc[-1]) < len(cal) - 1
        ]
        exec_of = {i_sig + 1: i_sig for i_sig in sig_is}

        cash = cfg.initial_equity
        last_equity = cfg.initial_equity
        positions: dict[str, MomoPosition] = {}
        pending_sells: list[tuple[str, str]] = []
        trades: list[dict] = []
        buy_notional_total = 0.0
        eq_arr = np.zeros(len(win_idx))
        expo_arr = np.zeros(len(win_idx))

        def _sell(pos: MomoPosition, i: int, price: float, reason: str) -> None:
            nonlocal cash
            notional = pos.shares * price
            fee = self._sell_fee(notional)
            pnl = pos.shares * (price - pos.entry_price) - pos.entry_fee - fee
            trades.append({
                "symbol": pos.symbol,
                "entry_date": pos.entry_date, "entry_price": round(pos.entry_price, 4),
                "exit_date": cal[i], "exit_price": round(price, 4),
                "shares": pos.shares, "hold_days": i - pos.entry_i,
                "pnl": round(pnl, 2), "pnl_pct": round(price / pos.entry_price - 1.0, 5),
                "r_multiple": np.nan, "exit_reason": reason,
            })
            cash += notional - fee
            del positions[pos.symbol]

        for k, i in enumerate(win_idx):
            # ---- 停牌順延的賣單 ----
            still: list[tuple[str, str]] = []
            for sym, reason in pending_sells:
                pos = positions.get(sym)
                if pos is None:
                    continue
                o = open_.iat[i, open_.columns.get_loc(sym)]
                if np.isnan(o):
                    still.append((sym, reason))
                else:
                    _sell(pos, i, o * (1.0 - cfg.slippage), reason)
            pending_sells = still

            # ---- 月調倉執行日 ----
            if i in exec_of:
                i_sig = exec_of[i]
                elig_row = eligible.iloc[i_sig]
                scores = mom.iloc[i_sig].where(elig_row)
                ranked = scores.dropna().sort_values(ascending=False)
                rank_map = {s: r + 1 for r, s in enumerate(ranked.index)}

                keep = {
                    s for s in positions
                    if rank_map.get(s, 10**9) <= cfg.band_rank and scores.get(s, -1.0) > 0.0
                }
                # 賣出：跌出 band 或動能轉負或失去資格
                for sym in [s for s in list(positions) if s not in keep]:
                    pos = positions[sym]
                    o = open_.iat[i, open_.columns.get_loc(sym)]
                    reason = "MOM_NEG" if scores.get(sym, -1.0) <= 0.0 else "DROP_RANK"
                    if np.isnan(o):
                        pending_sells.append((sym, reason))
                    else:
                        _sell(pos, i, o * (1.0 - cfg.slippage), reason)

                # 買進：依排名補滿 top_n（雙重排序時：動能池內依 overlay 分數排序）
                buy_order = list(ranked.index)
                if cfg.overlay_pool > 0 and overlay is not None:
                    pool = [s for s in buy_order if scores[s] > 0.0][:cfg.overlay_pool]
                    ov = overlay.iloc[i_sig].reindex(pool)
                    buy_order = sorted(pool, key=lambda s: -(ov[s] if pd.notna(ov[s]) else -1e18))
                slots = cfg.top_n - len(positions) - len(pending_sells)
                expo_mult = 1.0
                if bench_bull is not None and not bool(bench_bull.iloc[i_sig]):
                    expo_mult = cfg.exposure_bear_mult
                target_value = last_equity * expo_mult / cfg.top_n
                for sym in buy_order:
                    if slots <= 0:
                        break
                    if sym in positions or scores[sym] <= 0.0:
                        continue
                    col = open_.columns.get_loc(sym)
                    o = open_.iat[i, col]
                    if np.isnan(o):
                        continue
                    pc = prev_close_traded.iat[i, col]
                    if not np.isnan(pc) and (o / pc - 1.0) >= cfg.limit_up_gap:
                        continue  # 漲停開盤不追
                    fill = o * (1.0 + cfg.slippage)
                    shares = int(target_value // fill)
                    if shares < 1:
                        continue
                    notional = shares * fill
                    fee = self._buy_fee(notional)
                    if notional + fee > cash:
                        shares = int((cash - cfg.commission_min) / (fill * (1.0 + cfg.commission_rate)))
                        if shares < 1:
                            continue
                        notional = shares * fill
                        fee = self._buy_fee(notional)
                    cash -= notional + fee
                    buy_notional_total += notional
                    positions[sym] = MomoPosition(
                        symbol=sym, shares=shares, entry_price=fill, entry_i=i,
                        entry_date=cal[i], entry_fee=fee, last_close=fill,
                    )
                    slots -= 1

            # ---- 收盤更新／下市強制出清 ----
            for sym in list(positions):
                pos = positions[sym]
                c = close.iat[i, close.columns.get_loc(sym)]
                if not np.isnan(c):
                    pos.last_close = c
                    if last_valid_i.get(sym, len(cal) - 1) == i and i < win_idx[-1]:
                        _sell(pos, i, c * (1.0 - cfg.slippage), "DELIST")

            invested = sum(p.shares * p.last_close for p in positions.values())
            eq_arr[k] = cash + invested
            expo_arr[k] = invested / eq_arr[k] if eq_arr[k] > 0 else 0.0
            last_equity = eq_arr[k]

        # 期末以收盤清算（讓交易統計完整；標記 FINAL）
        i_last = int(win_idx[-1])
        for sym in list(positions):
            pos = positions[sym]
            _sell(pos, i_last, pos.last_close * (1.0 - cfg.slippage), "FINAL")
        invested = 0.0
        eq_arr[-1] = cash + invested

        idx = cal[win_idx]
        equity = pd.Series(eq_arr, index=idx, name="equity")
        years = len(win_idx) / 252.0
        turnover = buy_notional_total / equity.mean() / years if years > 0 else np.nan
        return MomentumResult(
            equity=equity,
            trades=pd.DataFrame(trades),
            benchmark=benchmark["close"].reindex(idx).ffill(),
            exposure=pd.Series(expo_arr, index=idx, name="exposure"),
            annual_turnover=turnover,
            cfg=cfg,
        )
