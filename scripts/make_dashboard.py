"""交易金流可視化儀表板 — 產生單檔 HTML（output/dashboard.html）。

資料來源：
  1. 動能回測（--source，預設 output/MOM_primary_252_top20）：equity.csv + trades.csv
  2. 實盤交易帳（transactions.csv，選填）：date,symbol,action,shares,price,fee
     - action = BUY / SELL；fee 留空則按預設費率估算
     - 持股市值以 data/ohlcv.parquet 最新收盤估值（平均成本法算損益）
圖表用 ECharts（CDN，開啟時需網路）。每日 17:40 盤後任務會自動重生。

用法：python scripts/make_dashboard.py [--source output/MOM_primary_252_top20]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.data import load_names

COMM = 0.001425 * 0.6
TAX = 0.003


def backtest_payload(src: Path) -> dict | None:
    eq_f, tr_f = src / "equity.csv", src / "trades.csv"
    if not eq_f.exists() or not tr_f.exists():
        return None
    eq = pd.read_csv(eq_f, parse_dates=[0], index_col=0)["equity"]
    tr = pd.read_csv(tr_f, parse_dates=["entry_date", "exit_date"], dtype={"symbol": str})

    dd = (eq / eq.cummax() - 1.0) * 100

    tr["buy_amt"] = tr["shares"] * tr["entry_price"]
    tr["sell_amt"] = tr["shares"] * tr["exit_price"]
    tr["fees"] = tr["buy_amt"] * COMM + tr["sell_amt"] * (COMM + TAX)
    m_buy = tr.groupby(tr["entry_date"].dt.to_period("M"))["buy_amt"].sum()
    m_sell = tr.groupby(tr["exit_date"].dt.to_period("M"))["sell_amt"].sum()
    months = sorted(set(m_buy.index) | set(m_sell.index))
    fees_cum = tr.sort_values("exit_date").set_index("exit_date")["fees"].cumsum()

    wins = tr[tr["pnl"] > 0]
    top = tr.nlargest(10, "pnl")[["symbol", "entry_date", "exit_date", "pnl", "pnl_pct"]]
    bot = tr.nsmallest(10, "pnl")[["symbol", "entry_date", "exit_date", "pnl", "pnl_pct"]]

    def _rows(d: pd.DataFrame, names: dict) -> list:
        return [[f"{r.symbol} {names.get(r.symbol, '')}", f"{r.entry_date:%Y-%m-%d}",
                 f"{r.exit_date:%Y-%m-%d}", round(r.pnl), f"{r.pnl_pct * 100:+.1f}%"]
                for r in d.itertuples()]

    names = load_names(ROOT / "data" / "universe.csv")
    return {
        "dates": [f"{d:%Y-%m-%d}" for d in eq.index],
        "equity": [round(v) for v in eq.values],
        "drawdown": [round(v, 2) for v in dd.values],
        "months": [str(m) for m in months],
        "m_buy": [round(float(m_buy.get(m, 0)) / 1e4) for m in months],
        "m_sell": [round(float(m_sell.get(m, 0)) / 1e4) for m in months],
        "fee_dates": [f"{d:%Y-%m-%d}" for d in fees_cum.index],
        "fee_cum": [round(v) for v in fees_cum.values],
        "pnl_list": [round(v) for v in tr["pnl"].values],
        "top": _rows(top, names), "bottom": _rows(bot, names),
        "stats": {
            "period": f"{eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}",
            "final": round(eq.iloc[-1]), "ret": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
            "mdd": round(dd.min(), 1), "n": len(tr),
            "win": round(len(wins) / len(tr) * 100, 1) if len(tr) else 0,
            "fees": round(tr["fees"].sum()),
        },
    }


def live_payload() -> dict | None:
    tx_f = ROOT / "transactions.csv"
    if not tx_f.exists():
        return None
    tx = pd.read_csv(tx_f, dtype={"symbol": str}, parse_dates=["date"]).sort_values("date")
    if not len(tx):
        return None
    names = load_names(ROOT / "data" / "universe.csv")
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date", "symbol", "raw_close"])
    last_px = px.sort_values("date").groupby("symbol")["raw_close"].last()

    pos: dict[str, dict] = {}
    realized = fees_total = invested = recovered = 0.0
    flows = []
    for r in tx.itertuples():
        amt = r.shares * r.price
        fee = float(r.fee) if "fee" in tx.columns and pd.notna(r.fee) else (
            amt * COMM if r.action == "BUY" else amt * (COMM + TAX))
        fees_total += fee
        if r.action == "BUY":
            p = pos.setdefault(r.symbol, {"sh": 0, "cost": 0.0})
            p["sh"] += r.shares
            p["cost"] += amt + fee
            invested += amt + fee
            flows.append([f"{r.date:%Y-%m-%d}", -(amt + fee)])
        else:
            p = pos.get(r.symbol)
            if not p or p["sh"] < r.shares:
                continue  # 帳目不符：略過並由表格呈現
            avg = p["cost"] / p["sh"]
            realized += (r.price * r.shares - fee) - avg * r.shares
            p["cost"] -= avg * r.shares
            p["sh"] -= r.shares
            recovered += amt - fee
            flows.append([f"{r.date:%Y-%m-%d}", amt - fee])
            if p["sh"] == 0:
                del pos[r.symbol]

    holding_rows, mkt_val, unreal = [], 0.0, 0.0
    for s, p in pos.items():
        cur = float(last_px.get(s, float("nan")))
        val = p["sh"] * cur if pd.notna(cur) else 0.0
        u = val - p["cost"]
        mkt_val += val
        unreal += u
        holding_rows.append([f"{s} {names.get(s, '')}", p["sh"], round(p["cost"] / p["sh"], 2),
                             round(cur, 2) if pd.notna(cur) else "-", round(val), round(u)])
    return {
        "flows": flows, "holdings": holding_rows,
        "stats": {"invested": round(invested), "recovered": round(recovered),
                  "mkt_val": round(mkt_val), "realized": round(realized),
                  "unreal": round(unreal), "fees": round(fees_total)},
    }


HTML = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>TWQuant 交易金流儀表板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
 body{font-family:"Microsoft JhengHei",sans-serif;margin:0;background:#111418;color:#e6e6e6}
 h1{font-size:20px;padding:16px 24px;margin:0;border-bottom:1px solid #2a2f36}
 h2{font-size:15px;margin:18px 0 8px}
 .wrap{padding:12px 24px;max-width:1200px;margin:auto}
 .cards{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
 .card{background:#1b2027;border:1px solid #2a2f36;border-radius:8px;padding:10px 16px;min-width:130px}
 .card .k{font-size:12px;color:#8a94a3}.card .v{font-size:18px;font-weight:600;margin-top:2px}
 .pos{color:#e05555}.neg{color:#3fa66a}  /* 台股慣例：紅漲綠跌 */
 .chart{height:320px;background:#161a20;border:1px solid #2a2f36;border-radius:8px;margin-bottom:14px}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:14px}
 th,td{border-bottom:1px solid #2a2f36;padding:6px 10px;text-align:right}
 th:first-child,td:first-child{text-align:left}
 th{color:#8a94a3;font-weight:500}
 .note{color:#8a94a3;font-size:12px}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 @media(max-width:900px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<h1>TWQuant 交易金流儀表板 <span class="note">產生時間 __GEN_TIME__</span></h1>
<div class="wrap">
__LIVE_SECTION__
<h2>動能組合回測（__BT_PERIOD__，含全部成本）</h2>
<div class="cards">
 <div class="card"><div class="k">期末權益</div><div class="v">__BT_FINAL__</div></div>
 <div class="card"><div class="k">總報酬</div><div class="v __BT_RET_CLS__">__BT_RET__%</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v neg">__BT_MDD__%</div></div>
 <div class="card"><div class="k">交易筆數</div><div class="v">__BT_N__</div></div>
 <div class="card"><div class="k">勝率</div><div class="v">__BT_WIN__%</div></div>
 <div class="card"><div class="k">累計費稅</div><div class="v">__BT_FEES__</div></div>
</div>
<div id="c_equity" class="chart"></div>
<div class="grid2">
 <div id="c_flow" class="chart"></div>
 <div id="c_fees" class="chart"></div>
</div>
<div id="c_hist" class="chart"></div>
<div class="grid2">
 <div><h2>獲利前十筆</h2><table id="t_top"></table></div>
 <div><h2>虧損前十筆</h2><table id="t_bot"></table></div>
</div>
<p class="note">回測資料含存活者偏差（無下市股票），數字偏樂觀；詳見 MOMENTUM.md。紅=獲利、綠=虧損（台股慣例）。</p>
</div>
<script>
const BT = __BT_JSON__;
const dark = {backgroundColor:'transparent', textStyle:{color:'#c9d1d9'}};
function mk(id, opt){ echarts.init(document.getElementById(id), null, {renderer:'canvas'}).setOption(Object.assign({}, dark, opt)); }
mk('c_equity', {
 title:{text:'權益曲線與回撤', textStyle:{fontSize:13,color:'#c9d1d9'}},
 tooltip:{trigger:'axis'}, grid:[{top:40,height:'52%'},{top:'72%',height:'20%'}],
 xAxis:[{type:'category',data:BT.dates,gridIndex:0,show:false},{type:'category',data:BT.dates,gridIndex:1}],
 yAxis:[{gridIndex:0,scale:true,name:'權益(元)'},{gridIndex:1,name:'回撤%',max:0}],
 series:[{name:'權益',type:'line',data:BT.equity,showSymbol:false,lineStyle:{width:1.5,color:'#e0a03f'}},
         {name:'回撤%',type:'line',data:BT.drawdown,xAxisIndex:1,yAxisIndex:1,showSymbol:false,
          areaStyle:{color:'rgba(63,166,106,.35)'},lineStyle:{color:'#3fa66a',width:1}}]});
mk('c_flow', {
 title:{text:'每月買賣金額（萬元）— 交易金流', textStyle:{fontSize:13,color:'#c9d1d9'}},
 tooltip:{trigger:'axis'}, legend:{top:24,textStyle:{color:'#c9d1d9'}},
 grid:{top:60}, xAxis:{type:'category',data:BT.months}, yAxis:{name:'萬元'},
 series:[{name:'買進',type:'bar',stack:'f',data:BT.m_buy.map(v=>-v),itemStyle:{color:'#e05555'}},
         {name:'賣出(回收)',type:'bar',stack:'f',data:BT.m_sell,itemStyle:{color:'#3fa66a'}}]});
mk('c_fees', {
 title:{text:'累計費稅（手續費+證交稅+滑價外的顯性成本）', textStyle:{fontSize:13,color:'#c9d1d9'}},
 tooltip:{trigger:'axis'}, grid:{top:50},
 xAxis:{type:'category',data:BT.fee_dates,axisLabel:{interval:Math.floor(BT.fee_dates.length/6)}},
 yAxis:{name:'元'},
 series:[{type:'line',data:BT.fee_cum,showSymbol:false,areaStyle:{color:'rgba(224,85,85,.25)'},lineStyle:{color:'#e05555'}}]});
(function(){
 const bins=[-1e9,-50000,-20000,-10000,-5000,0,5000,10000,20000,50000,1e9];
 const labels=['<-5萬','-5~-2萬','-2~-1萬','-1~-0.5萬','-0.5~0','0~0.5萬','0.5~1萬','1~2萬','2~5萬','>5萬'];
 const cnt=new Array(labels.length).fill(0);
 BT.pnl_list.forEach(v=>{for(let i=0;i<labels.length;i++){if(v>bins[i]&&v<=bins[i+1]){cnt[i]++;break;}}});
 mk('c_hist',{title:{text:'單筆損益分布',textStyle:{fontSize:13,color:'#c9d1d9'}},
  tooltip:{}, grid:{top:50}, xAxis:{type:'category',data:labels}, yAxis:{name:'筆數'},
  series:[{type:'bar',data:cnt.map((c,i)=>({value:c,itemStyle:{color:i<5?'#3fa66a':'#e05555'}}))}]});
})();
function fill(id, rows){
 const t=document.getElementById(id);
 t.innerHTML='<tr><th>標的</th><th>進場</th><th>出場</th><th>損益(元)</th><th>報酬</th></tr>'+
  rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="${r[3]>=0?'pos':'neg'}">${r[3].toLocaleString()}</td><td>${r[4]}</td></tr>`).join('');
}
fill('t_top', BT.top); fill('t_bot', BT.bottom);
__LIVE_SCRIPT__
</script></body></html>"""

LIVE_SECTION = """<h2>實盤帳戶（transactions.csv）</h2>
<div class="cards">
 <div class="card"><div class="k">累計投入</div><div class="v">__L_INV__</div></div>
 <div class="card"><div class="k">累計回收</div><div class="v">__L_REC__</div></div>
 <div class="card"><div class="k">持股市值</div><div class="v">__L_MKT__</div></div>
 <div class="card"><div class="k">已實現損益</div><div class="v __L_R_CLS__">__L_REAL__</div></div>
 <div class="card"><div class="k">未實現損益</div><div class="v __L_U_CLS__">__L_UNREAL__</div></div>
 <div class="card"><div class="k">累計費稅</div><div class="v">__L_FEES__</div></div>
</div>
<div id="c_live_flow" class="chart"></div>
<h2>目前持股</h2><table id="t_hold"></table>
<hr style="border-color:#2a2f36">"""

LIVE_SCRIPT = """
const LV = __LV_JSON__;
mk('c_live_flow', {
 title:{text:'實盤現金流（負=買進投入、正=賣出回收）', textStyle:{fontSize:13,color:'#c9d1d9'}},
 tooltip:{trigger:'axis'}, grid:{top:50},
 xAxis:{type:'category',data:LV.flows.map(f=>f[0])}, yAxis:{name:'元'},
 series:[{type:'bar',data:LV.flows.map(f=>({value:Math.round(f[1]),itemStyle:{color:f[1]<0?'#e05555':'#3fa66a'}}))}]});
(function(){
 const t=document.getElementById('t_hold');
 t.innerHTML='<tr><th>標的</th><th>股數</th><th>均價(含費)</th><th>現價</th><th>市值</th><th>未實現</th></tr>'+
  LV.holdings.map(r=>`<tr><td>${r[0]}</td><td>${r[1].toLocaleString()}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4].toLocaleString()}</td><td class="${r[5]>=0?'pos':'neg'}">${r[5].toLocaleString()}</td></tr>`).join('');
})();"""


def _money(v: float) -> str:
    return f"{v:,.0f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="output/MOM_primary_252_top20", help="回測輸出目錄")
    p.add_argument("--out", default="output/dashboard.html")
    args = p.parse_args()

    bt = backtest_payload(ROOT / args.source)
    if bt is None:
        raise SystemExit(f"找不到回測輸出（{args.source}/equity.csv、trades.csv）；請先跑 scripts/run_momentum.py")
    lv = live_payload()

    html = HTML
    html = html.replace("__GEN_TIME__", f"{datetime.now():%Y-%m-%d %H:%M}")
    html = html.replace("__BT_PERIOD__", bt["stats"]["period"])
    html = html.replace("__BT_FINAL__", _money(bt["stats"]["final"]))
    html = html.replace("__BT_RET_CLS__", "pos" if bt["stats"]["ret"] >= 0 else "neg")
    html = html.replace("__BT_RET__", f"{bt['stats']['ret']:+.1f}")
    html = html.replace("__BT_MDD__", str(bt["stats"]["mdd"]))
    html = html.replace("__BT_N__", str(bt["stats"]["n"]))
    html = html.replace("__BT_WIN__", str(bt["stats"]["win"]))
    html = html.replace("__BT_FEES__", _money(bt["stats"]["fees"]))
    html = html.replace("__BT_JSON__", json.dumps(bt, ensure_ascii=False))

    if lv:
        sec = LIVE_SECTION
        s = lv["stats"]
        sec = sec.replace("__L_INV__", _money(s["invested"]))
        sec = sec.replace("__L_REC__", _money(s["recovered"]))
        sec = sec.replace("__L_MKT__", _money(s["mkt_val"]))
        sec = sec.replace("__L_R_CLS__", "pos" if s["realized"] >= 0 else "neg")
        sec = sec.replace("__L_REAL__", _money(s["realized"]))
        sec = sec.replace("__L_U_CLS__", "pos" if s["unreal"] >= 0 else "neg")
        sec = sec.replace("__L_UNREAL__", _money(s["unreal"]))
        sec = sec.replace("__L_FEES__", _money(s["fees"]))
        html = html.replace("__LIVE_SECTION__", sec)
        html = html.replace("__LIVE_SCRIPT__", LIVE_SCRIPT.replace("__LV_JSON__", json.dumps(lv, ensure_ascii=False)))
    else:
        html = html.replace(
            "__LIVE_SECTION__",
            '<p class="note">實盤帳戶：尚無 transactions.csv（格式：date,symbol,action,shares,price,fee；'
            "action=BUY/SELL）。開始交易後記錄成交，本區會顯示現金流、持股市值與已實現/未實現損益。</p><hr style=\"border-color:#2a2f36\">",
        )
        html = html.replace("__LIVE_SCRIPT__", "")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"儀表板已產生：{out}")


if __name__ == "__main__":
    main()
