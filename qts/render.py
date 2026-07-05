"""Discord 可視化渲染：深色主題表格 PNG（與儀表板同風格）。

matplotlib Agg 後端；中文用微軟正黑體。所有函式回傳 PNG bytes。
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#14181d"
CARD = "#161a20"
HEAD = "#1f242c"
GROUP = "#232a33"
BORDER = "#2a2f36"
FG = "#e6e6e6"
MUT = "#8a94a3"
RED = "#e05555"    # 台股慣例：紅=漲/獲利
GREEN = "#3fa66a"  # 綠=跌/虧損
GOLD = "#ffd27f"


def table_png(
    title: str,
    headers: list[str],
    rows: list[list],
    col_colors: dict[int, str] | None = None,   # 欄索引 → 文字色（資料列）
    group_rows: set[int] | None = None,          # 資料列索引（0起算）為組標題列
    footer: str = "",
) -> bytes:
    col_colors = col_colors or {}
    group_rows = group_rows or set()
    n_cols = len(headers)

    fig_w = max(7.0, min(14.0, 1.35 * n_cols + 2.5))
    fig_h = 0.85 + 0.34 * (len(rows) + 1) + (0.35 if footer else 0.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    cell_text = [[str(c) for c in headers]] + [[str(c) for c in r] for r in rows]
    tbl = ax.table(cellText=cell_text, loc="upper center", cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(list(range(n_cols)))

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.6)
        txt = cell.get_text()
        if r == 0:  # 表頭
            cell.set_facecolor(HEAD)
            txt.set_color(MUT)
            txt.set_fontweight("bold")
        elif (r - 1) in group_rows:  # 組標題列
            cell.set_facecolor(GROUP)
            if c == 0:
                txt.set_color(GOLD)
                txt.set_fontweight("bold")
                txt.set_ha("left")
                cell.set_text_props(x=0.02)
            else:
                txt.set_color(GROUP)  # 隱藏
        else:
            cell.set_facecolor(CARD)
            txt.set_color(col_colors.get(c, FG))

    ax.set_title(title, color=FG, fontsize=13, fontweight="bold", pad=14, loc="left")
    if footer:
        fig.text(0.01, 0.005, footer, color=MUT, fontsize=8.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return buf.getvalue()


def scan_table_png(candidates: pd.DataFrame, asof_label: str, strat_zh: dict, trigger_zh: dict) -> bytes:
    """波段候選（按策略分組）表格圖。"""
    headers = ["標的", "觸發", "收盤", "停損", "停利+2R", "建議", "最大虧損"]
    rows: list[list] = []
    group_idx: set[int] = set()
    d = candidates.copy()
    if "rank_score" in d.columns:
        d = d.sort_values(["strategy", "rank_score"], ascending=[True, False])
    for strat in ("A", "B", "C"):
        g = d[d["strategy"] == strat]
        if not len(g):
            continue
        group_idx.add(len(rows))
        rows.append([f"{strat_zh.get(strat, strat)}（{len(g)} 檔）", "", "", "", "", "", ""])
        for r in g.itertuples():
            shares = int(r.suggest_shares)
            lots = shares // 1000
            tgt = getattr(r, "target_partial", round(r.close + 2 * (r.close - r.init_stop), 2))
            loss = getattr(r, "max_loss", round((r.close - r.init_stop) * shares))
            rows.append([
                f"{r.symbol} {getattr(r, 'name', '')}",
                trigger_zh.get(str(r.trigger), str(r.trigger)),
                f"{r.close:g}", f"{r.init_stop:g}", f"{tgt:g}",
                f"{lots}張" if lots else "資金不足",
                f"−{loss:,.0f}" if shares else "—",
            ])
    return table_png(
        f"波段掃描候選　{asof_label}　※紙上觀察，未過上線門檻",
        headers, rows,
        col_colors={3: GREEN, 4: RED, 6: GREEN},
        group_rows=group_idx,
        footer="停損=跌破次日開盤出場｜停利=+2R先出一半、停損上移成本後吊燈追蹤｜最大虧損=(收盤−停損)×建議股數，即停損打到時的實際金額（≈權益1%）",
    )


def rebalance_table_png(reb: pd.DataFrame, asof_label: str, equity: float, top_n: int = 20) -> bytes:
    """動能換股清單表格圖。"""
    headers = ["動作", "標的", "排名", "動能", "參考價", "股數"]
    rows: list[list] = []
    group_idx: set[int] = set()
    close_col = [c for c in reb.columns if str(c).startswith("close_")]
    zh = {"KEEP": "續抱", "SELL": "賣出", "BUY": "買進"}
    for action in ("SELL", "BUY", "KEEP"):
        g = reb[reb["action"] == action]
        if not len(g):
            continue
        group_idx.add(len(rows))
        rows.append([f"{zh[action]}（{len(g)} 檔）", "", "", "", "", ""])
        for _, r in g.iterrows():
            is_buy = action == "BUY"
            mom = r.get("momentum%")
            rows.append([
                zh[action],
                f"{r['symbol']} {r.get('name', '') or ''}",
                f"#{int(r['rank'])}" if is_buy and pd.notna(r.get("rank")) else "",
                f"{float(mom):+.0f}%" if is_buy and pd.notna(mom) else "",
                f"{float(r[close_col[0]]):g}" if (is_buy and close_col and pd.notna(r.get(close_col[0]))) else "",
                f"{int(r['suggest_shares']):,}" if is_buy and pd.notna(r.get("suggest_shares")) else "全部",
            ])
    return table_png(
        f"動能組合月度換股清單　訊號日 {asof_label}　→ 次一交易日開盤執行",
        headers, rows,
        group_rows=group_idx,
        footer=f"每檔目標 {equity / top_n:,.0f} 元｜開盤漲幅≥9.5% 的買單放棄｜成交後更新 holdings.csv",
    )
