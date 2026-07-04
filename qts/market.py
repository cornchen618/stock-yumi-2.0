"""市場狀態評估：紅黃綠燈、市場廣度、曝險建議、類股強弱。

燈號規則（預先定義，門檻不因當日數據調整）：
  綠燈：TAIEX > MA200 且 > MA60 且 廣度(站上60日線比例) ≥ 50%
  紅燈：TAIEX 同時跌破 MA200 與 MA60，或 廣度 < 30%
  黃燈：其餘情況
曝險建議（治理規則，非最佳化參數）：綠 100%｜黃 60%｜紅 20%（且波段停開新倉）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketState:
    asof: pd.Timestamp
    light: str                 # "綠" / "黃" / "紅"
    exposure_pct: int          # 建議曝險（%）
    taiex_close: float
    ma60: float
    ma200: float
    breadth_ma60: float        # 站上自身60日線的個股比例
    breadth_ma20: float
    new_high_20: int           # 創20日新高家數
    new_low_20: int
    detail: list[str] = field(default_factory=list)


def assess_market(benchmark: pd.DataFrame, close_panel: pd.DataFrame) -> MarketState:
    """benchmark: 指數日線；close_panel: date × symbol 收盤價（還原）。"""
    c = benchmark["close"]
    ma60 = c.rolling(60).mean()
    ma200 = c.rolling(200).mean()
    asof = c.index[-1]

    above60 = (close_panel.iloc[-1] > close_panel.rolling(60).mean().iloc[-1])
    above20 = (close_panel.iloc[-1] > close_panel.rolling(20).mean().iloc[-1])
    valid = close_panel.iloc[-1].notna()
    breadth60 = float(above60[valid].mean())
    breadth20 = float(above20[valid].mean())

    high20 = close_panel.rolling(20).max()
    nh = int((close_panel.iloc[-1] >= high20.iloc[-1] * 0.999)[valid].sum())
    low20 = close_panel.rolling(20).min()
    nl = int((close_panel.iloc[-1] <= low20.iloc[-1] * 1.001)[valid].sum())

    above_ma60 = c.iloc[-1] > ma60.iloc[-1]
    above_ma200 = c.iloc[-1] > ma200.iloc[-1]

    if above_ma200 and above_ma60 and breadth60 >= 0.50:
        light, exposure = "綠", 100
    elif (not above_ma200 and not above_ma60) or breadth60 < 0.30:
        light, exposure = "紅", 20
    else:
        light, exposure = "黃", 60

    detail = [
        f"大盤 {c.iloc[-1]:,.0f}｜MA60 {ma60.iloc[-1]:,.0f}（{'上' if above_ma60 else '下'}）｜MA200 {ma200.iloc[-1]:,.0f}（{'上' if above_ma200 else '下'}）",
        f"廣度：{breadth60 * 100:.0f}% 個股站上 60 日線、{breadth20 * 100:.0f}% 站上 20 日線",
        f"20 日新高 {nh} 家 vs 新低 {nl} 家",
    ]
    return MarketState(asof=asof, light=light, exposure_pct=exposure,
                       taiex_close=float(c.iloc[-1]), ma60=float(ma60.iloc[-1]),
                       ma200=float(ma200.iloc[-1]), breadth_ma60=breadth60,
                       breadth_ma20=breadth20, new_high_20=nh, new_low_20=nl, detail=detail)


def sector_strength(close_panel: pd.DataFrame, industry_map: dict[str, str],
                    top_k: int = 5, min_members: int = 8) -> pd.DataFrame:
    """類股強弱：等權 20/60 日報酬、站上 60 日線比例。回傳全表（按 20 日報酬排序）。"""
    ret20 = close_panel.iloc[-1] / close_panel.iloc[-21] - 1.0 if len(close_panel) > 21 else None
    ret60 = close_panel.iloc[-1] / close_panel.iloc[-61] - 1.0 if len(close_panel) > 61 else None
    above60 = close_panel.iloc[-1] > close_panel.rolling(60).mean().iloc[-1]

    rows = []
    ind = pd.Series({s: industry_map.get(s, "未分類") for s in close_panel.columns})
    for sector, syms in ind.groupby(ind).groups.items():
        syms = [s for s in syms if pd.notna(close_panel.iloc[-1].get(s))]
        if len(syms) < min_members or sector == "未分類":
            continue
        r20 = float(ret20[syms].mean()) if ret20 is not None else float("nan")
        r60 = float(ret60[syms].mean()) if ret60 is not None else float("nan")
        strongest = ret20[syms].nlargest(3).index.tolist() if ret20 is not None else []
        rows.append({"sector": sector, "n": len(syms),
                     "ret20": r20, "ret60": r60,
                     "breadth60": float(above60[syms].mean()),
                     "leaders": strongest})
    df = pd.DataFrame(rows).sort_values("ret20", ascending=False).reset_index(drop=True)
    _ = top_k
    return df
