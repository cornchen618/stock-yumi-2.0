"""每日決策簡報：直接回答八個關鍵問題（合格量化系統的定義）。

  1. 現在市場適不適合做多？        → 紅黃綠燈（大盤位置＋市場廣度）
  2. 目前應該投入幾成資金？        → 燈號曝險建議＋動能自然曝險
  3. 哪些類股正在轉強？            → 類股 20/60 日等權報酬排行
  4. 哪些股票符合條件？            → 動能前 20＋波段最新候選
  5. 這筆交易最多可以虧多少？      → 波段 1% 風險制／動能單檔上限與歷史尾部
  6. 連續虧損要不要降低部位？      → STRATEGY.md §7.4 回撤治理表＋目前狀態
  7. 大盤轉弱要不要停止進場？      → 濾網規則＋目前狀態（自動執行中）
  8. 策略最近是否失效？            → 滾動績效 vs 基準＋失效判定框架

用法：python scripts/brief.py [--discord]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_benchmark, load_names, load_ohlcv
from qts.market import assess_market, sector_strength
from qts.scanner import STRAT_ZH


def generate_sections() -> tuple[pd.Timestamp, str, list[tuple[str, str]]]:
    """回傳 (基準日, 燈號, [(問題標題, 答案), ...×8])。"""
    equity = 1_000_000.0
    sp = ROOT / "settings.json"
    if sp.exists():
        equity = float(json.loads(sp.read_text(encoding="utf-8")).get("equity", equity))

    data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")
    names = load_names(ROOT / "data" / "universe.csv")
    uni = pd.read_csv(ROOT / "data" / "universe.csv", dtype=str)
    industry_map = dict(zip(uni["code"], uni.get("industry", pd.Series(dtype=str)).fillna("未分類")))

    close = pd.DataFrame({s: d["close"] for s, d in data.items()})
    ms = assess_market(benchmark, close)
    sectors = sector_strength(close, industry_map)

    # 動能自然曝險：合格且動能>0 的檔數是否足以填滿 20 檔
    vol = pd.DataFrame({s: d["volume"] for s, d in data.items()}).reindex(close.index)
    amt = pd.DataFrame({s: d["amount"] for s, d in data.items()}).reindex(close.index)
    mom = (close.shift(21) / close.shift(252) - 1.0).iloc[-1]
    elig = ((vol.rolling(20).median().iloc[-1] >= 5e5) & (amt.rolling(20).median().iloc[-1] >= 3e7)
            & (close.iloc[-1] >= 10.0) & mom.notna())
    mom_pos = mom.where(elig).dropna()
    n_pos = int((mom_pos > 0).sum())
    top20 = mom_pos.sort_values(ascending=False).head(20)

    # 波段最新候選
    scans = sorted((ROOT / "output").glob("scan_*.csv"))
    scan_line = "尚無掃描結果"
    if scans:
        sc = pd.read_csv(scans[-1], dtype={"symbol": str})
        cnt = sc.groupby("strategy").size()
        day = scans[-1].stem.split("_")[-1]
        scan_line = (f"{day[:4]}-{day[4:6]}-{day[6:]} 掃出 "
                     + "、".join(f"{STRAT_ZH.get(k, k)} {v} 檔" for k, v in cnt.items())
                     + "（紙上觀察）")

    # 動能回測滾動 12 個月（策略健康度參考）
    health_line = "回測輸出不存在"
    bt_eq_f = ROOT / "output" / "MOM_primary_252_top20" / "equity.csv"
    if bt_eq_f.exists():
        bt_eq = pd.read_csv(bt_eq_f, parse_dates=[0], index_col=0)["equity"]
        if len(bt_eq) > 252:
            r12 = bt_eq.iloc[-1] / bt_eq.iloc[-252] - 1.0
            b = benchmark["close"].reindex(bt_eq.index).ffill()
            b12 = b.iloc[-1] / b.iloc[-252] - 1.0
            health_line = f"策略模擬近 12 個月 {r12 * 100:+.1f}% vs 大盤 {b12 * 100:+.1f}%（差 {(r12 - b12) * 100:+.1f}pp）"

    light_icon = {"綠": "🟢", "黃": "🟡", "紅": "🔴"}[ms.light]
    sec_top = sectors.head(5)
    sec_lines = [
        f"{r.sector}（{r.n}檔）：20日 {r.ret20 * 100:+.1f}%｜{r.breadth60 * 100:.0f}%站上60日線｜"
        f"領頭：{'、'.join(f'{s} {names.get(s, '')}' for s in r.leaders)}"
        for r in sec_top.itertuples()
    ]
    top5_line = "、".join(f"{s} {names.get(s, '')}" for s in top20.head(5).index)

    sections: list[tuple[str, str]] = [
        (f"1︱市場適合做多嗎　{light_icon} {ms.light}燈", "\n".join(ms.detail)),
        (f"2︱應投入幾成資金　→ 上限 {ms.exposure_pct}%",
         f"燈號治理規則：綠100／黃60／紅20\n動能組合：{n_pos} 檔動能>0"
         f"（{'足額可滿倉' if n_pos >= 20 else f'自然縮至 {min(n_pos, 20)} 檔'}）\n"
         "註：MA200 自動減碼經檢定不採用（傷 Sharpe），總量由你依燈號控制"),
        ("3︱哪些類股轉強（20日等權前五）", "\n".join(sec_lines)),
        ("4︱哪些股票符合條件",
         f"動能前 20：{top5_line}…\n波段：{scan_line}\n完整名單與價位見掃描表格圖與儀表板"),
        ("5︱這筆最多虧多少",
         f"波段：單筆風險 = 權益×1% = {equity * 0.01:,.0f} 元（停損價逐檔列出）\n"
         f"動能：單檔投入 {equity / 20:,.0f} 元、無個股停損（回測最大單筆約 −24%，靠 20 檔分散）\n"
         f"組合層：單日新增風險上限 4% = {equity * 0.04:,.0f} 元"),
        ("6︱連續虧損降部位嗎　→ 會（自動分級）",
         "回撤>8% 波段風險減半 → >12% 波段停新倉 → >20% 動能曝險減半 → >30% 全停人工檢討\n"
         "目前狀態：尚未實盤，回撤 0%，正常層級"),
        ("7︱大盤轉弱停止進場嗎　→ 會（自動執行）",
         f"大盤<60日線 → 波段 A/C 停開新倉、B 風險減半\n"
         f"目前：大盤{'高於' if ms.taiex_close > ms.ma60 else '低於'} 60 日線 → "
         f"波段新倉{'開放' if ms.taiex_close > ms.ma60 else '停止'}中"),
        ("8︱策略失效了嗎",
         f"{health_line}\n判定框架：實盤後每季重跑比對；勝率/滑價偏離 2σ 或 PF 連兩季<1 → 停機\n"
         "目前：實績累積中（需 ≥100 筆），所有訊號已自動存檔"),
    ]
    return ms.asof, ms.light, sections


def generate_brief() -> str:
    asof, light, sections = generate_sections()
    L = [f"══ 每日決策簡報 {asof:%Y-%m-%d} ══"]
    for title, body in sections:
        L.append("")
        L.append(f"【{title}】")
        L += [f"  {ln}" for ln in body.splitlines()]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--discord", action="store_true", help="同時發送到 Discord（embed 卡片）")
    args = p.parse_args()
    asof, light, sections = generate_sections()
    text = generate_brief()
    print(text)
    out = ROOT / "output" / "brief.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")   # 供儀表板嵌入
    if args.discord:
        from qts.notify import C_GREEN, C_RED, C_YELLOW, send_embed
        color = {"綠": C_GREEN, "黃": C_YELLOW, "紅": C_RED}[light]
        send_embed(
            title=f"📋 每日決策簡報 {asof:%Y-%m-%d}",
            fields=[(t, b, False) for t, b in sections],
            color=color,
            footer="卡片顏色 = 市場燈號｜規則出處：STRATEGY.md / MOMENTUM.md",
        )


if __name__ == "__main__":
    main()
