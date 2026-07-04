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
from qts.scanner import STRAT_ZH, TRIGGER_ZH

COMM = 0.001425 * 0.6
TAX = 0.003


def _cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def strat1_rules() -> str:
    return """
<div class="rulebox">
 <ul>
  <li><b>買進</b>：每月最後交易日收盤，12-1 動能（近 12 個月報酬、跳過最近 1 個月）&gt; 0 且排名前 20，次日開盤等權買進（每檔＝權益÷20）</li>
  <li><b>續抱</b>：動能 &gt; 0 且排名 ≤ 40</li>
  <li><b>賣出</b>：跌出前 40 名 或 動能轉負 → 月底次日開盤市價賣出</li>
  <li><b>資格</b>：20日中位數量 ≥ 50萬股、金額 ≥ 3000萬、價 ≥ 10元</li>
  <li><b>風控</b>：無個股停損（換血靠月調倉紀律）；開盤漲幅 ≥ 9.5% 放棄買單；一個月只動作一次</li>
 </ul>
</div>"""


def strat2_rules() -> str:
    return """
<div class="rulebox">
 <ul>
  <li><b>A 順勢突破</b>：多頭排列＋突破20/10日高或站回月線＋1.5倍量</li>
  <li><b>B 破底翻</b>：60日低檔跌破支撐後收復＋過昨高＋量能確認</li>
  <li><b>C 蓄勢突破</b>：壓縮整理（帶寬百分位≤30%）＋量縮 → 帶量突破</li>
  <li><b>停損</b>：訊號K低點與收盤−1.5ATR取低者（B=spring低點×0.99），跌破次日出場</li>
  <li><b>停利</b>：+2R 先出一半、停損上移至成本；獲利&gt;1R 後吊燈式移動停損（最高收盤−2.5ATR）；A/C 連2日收破月線出場</li>
  <li><b>部位</b>：單筆風險=權益1%、上限15%、大盤&lt;60日線停開新倉</li>
 </ul>
</div>"""


def momentum_snapshot(equity: float, names: dict) -> str:
    """目前動能排名前 20（即時快照，非月底正式訊號）。"""
    px = pd.read_parquet(ROOT / "data" / "ohlcv.parquet",
                         columns=["date", "symbol", "close", "raw_close", "volume", "amount"])
    px["date"] = pd.to_datetime(px["date"])
    cp = px.pivot_table(index="date", columns="symbol", values="close")
    vp = px.pivot_table(index="date", columns="symbol", values="volume")
    ap = px.pivot_table(index="date", columns="symbol", values="amount")
    rp = px.pivot_table(index="date", columns="symbol", values="raw_close")
    asof = cp.index.max()
    mom = (cp.shift(21) / cp.shift(252) - 1.0).iloc[-1]
    elig = ((vp.rolling(20).median().iloc[-1] >= 5e5)
            & (ap.rolling(20).median().iloc[-1] >= 3e7)
            & (cp.iloc[-1] >= 10.0) & mom.notna())
    top = mom.where(elig).dropna().sort_values(ascending=False)
    top = top[top > 0].head(20)

    held = set()
    hp = ROOT / "holdings.csv"
    if hp.exists():
        held = set(pd.read_csv(hp, dtype={"symbol": str})["symbol"])

    rows = []
    for i, (s, m) in enumerate(top.items(), 1):
        price = float(rp.iloc[-1][s])
        shares = int(equity / 20 // price)
        status = "持有中" if s in held else "候選"
        rows.append(f"<tr><td>{i}</td><td>{s} {names.get(s, '')}</td>"
                    f"<td class='pos'>{m * 100:+.0f}%</td><td>{price:g}</td>"
                    f"<td>{shares:,}</td><td>{status}</td></tr>")
    return f"""
<h3>目前排名前 20──下月調倉候選預覽 <span class="note">（資料日 {asof:%Y-%m-%d}；每天變動，實際買賣以月底收盤排名為準）</span></h3>
<table><tr><th>#</th><th>標的</th><th>12-1動能</th><th>收盤</th><th>建議股數</th><th>狀態</th></tr>
{''.join(rows)}</table>
<p class="note">每檔目標金額＝權益÷20＝{equity / 20:,.0f} 元｜「持有中」= holdings.csv 內的實際持股</p>"""


def scan_snapshot() -> str:
    """最新波段掃描候選（output/scan_*.csv）。"""
    files = sorted((ROOT / "output").glob("scan_*.csv"))
    if not files:
        return "<h3>最新候選</h3><p class='note'>尚無掃描結果（17:40 盤後任務會自動產生）。</p>"
    f = files[-1]
    day = f.stem.split("_")[-1]
    d = pd.read_csv(f, dtype={"symbol": str})
    if "target_partial" not in d.columns:  # 舊版 CSV 相容
        d["target_partial"] = (d["close"] + 2 * (d["close"] - d["init_stop"])).round(2)
    if "rank_score" in d.columns:
        d = d.sort_values(["strategy", "rank_score"], ascending=[True, False])
    else:
        d = d.sort_values("strategy")

    rows = []
    for strat in ("A", "B", "C"):
        g = d[d["strategy"] == strat]
        if not len(g):
            continue
        rows.append(f"<tr><td colspan='6' class='grp'>{STRAT_ZH.get(strat, strat)}（{len(g)} 檔，組內按量能強度排序）</td></tr>")
        for r in g.itertuples():
            trig = TRIGGER_ZH.get(str(r.trigger), str(r.trigger))
            lots = int(r.suggest_shares) // 1000
            size = f"{lots}張" if lots else "資金不足"
            rows.append(f"<tr><td>{r.symbol} {getattr(r, 'name', '')}</td>"
                        f"<td>{trig}</td><td>{r.close:g}</td><td class='neg'>{r.init_stop:g}</td>"
                        f"<td class='pos'>{r.target_partial:g}</td><td>{size}</td></tr>")
    return f"""
<h3>最新候選 <span class="note">（資料日 {day[:4]}-{day[4:6]}-{day[6:]}）</span></h3>
<table><tr><th>標的</th><th>觸發</th><th>收盤</th><th>停損</th><th>停利(+2R)</th><th>建議</th></tr>
{''.join(rows)}</table>
<p class="note">停損＝跌破次日開盤出場｜停利＝到價先出一半、剩餘停損上移至成本後吊燈追蹤｜建議張數以 1% 風險計算</p>"""


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
            "top10_share": round(top["pnl"].sum() / wins["pnl"].sum() * 100) if len(wins) else 0,
            "worst_pct": round(tr["pnl_pct"].min() * 100, 1) if len(tr) else 0,
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
 .rulebox{background:#1b2027;border:1px solid #2a2f36;border-radius:8px;padding:4px 16px 10px}
 .rulebox h3{font-size:14px;margin:10px 0 6px}
 .rulebox ul{margin:0;padding-left:18px;font-size:13px;line-height:1.7}
 .grp{background:#232a33;color:#ffd27f;font-weight:600;text-align:left !important;padding:8px 10px}
 .badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;margin-left:8px;vertical-align:middle}
 .b-live{background:#7a5d1e;color:#ffd27f}.b-paper{background:#37404d;color:#aab6c5}.b-sim{background:#1e4a6d;color:#9fd0f5}
 details summary{cursor:pointer;list-style:none}
 details summary h2{display:inline}
</style></head><body>
<h1>TWQuant 交易金流儀表板 <span class="note">產生時間 __GEN_TIME__</span></h1>
<div class="wrap">
__BRIEF_SECTION__
__LIVE_SECTION__
<h2>策略① 動能組合 <span class="badge b-live">主力策略・每月只動作一次</span></h2>
__S1_RULES__
__MOM_SECTION__
<hr style="border-color:#2a2f36">
<h2>策略② 波段 A/B/C <span class="badge b-paper">研究中・僅紙上追蹤，勿實際下單</span></h2>
__S2_RULES__
__SCAN_SECTION__
<hr style="border-color:#2a2f36">
<details open>
<summary><h2>策略① 歷史回測檢驗 <span class="badge b-sim">2019–2026 模擬交易・不是你的帳戶</span></h2>
<span class="note">（點標題可收合）</span></summary>
<p class="note">以下全部是「假設 2019 年起就照策略①執行 100 萬」的模擬結果。它的用途：在投入真錢之前檢驗策略是否值得執行，
並預告實際操作時會經歷什麼——勝率不到四成、最深回撤約一半、獲利靠少數大波段。看懂這些，實盤遇到時才不會慌。
（回測期間 __BT_PERIOD__，含全部成本；無下市股票資料，數字偏樂觀。紅=賺、綠=賠。）</p>
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
<h3>模擬期間最賺與最賠的十筆</h3>
<p class="note">重點不在個股，在兩個結構性事實：
① 模擬總獲利的 <b>__TOP10_SHARE__%</b> 來自最賺的十筆——動能策略靠少數大贏家吃飯，
實際操作時「提早獲利了結強勢股」等於自廢武功，這就是為什麼賣出只看排名不看獲利；
② 最大單筆虧損 <b>__WORST_PCT__%</b>——月調倉換血讓單一地雷的傷害有上限，但個股月中無停損，需有心理準備。</p>
<div class="grid2">
 <div><table id="t_top"></table></div>
 <div><table id="t_bot"></table></div>
</div>
</details>
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

    equity = 1_000_000.0
    sp = ROOT / "settings.json"
    if sp.exists():
        equity = float(json.loads(sp.read_text(encoding="utf-8")).get("equity", equity))
    names = load_names(ROOT / "data" / "universe.csv")

    brief_f = ROOT / "output" / "brief.txt"
    if brief_f.exists():
        brief_html = (
            "<details open><summary><h2>今日決策簡報（八問）"
            "<span class='badge b-live'>每日 17:40 更新</span></h2>"
            "<span class='note'>（點標題可收合）</span></summary>"
            f"<pre style='background:#161a20;border:1px solid #2a2f36;border-radius:8px;"
            f"padding:14px;font-size:13px;line-height:1.65;white-space:pre-wrap'>{brief_f.read_text(encoding='utf-8')}</pre>"
            "</details><hr style='border-color:#2a2f36'>"
        )
    else:
        brief_html = ""

    html = HTML
    html = html.replace("__BRIEF_SECTION__", brief_html)
    html = html.replace("__GEN_TIME__", f"{datetime.now():%Y-%m-%d %H:%M}")
    html = html.replace("__S1_RULES__", strat1_rules())
    html = html.replace("__S2_RULES__", strat2_rules())
    html = html.replace("__MOM_SECTION__", momentum_snapshot(equity, names))
    html = html.replace("__SCAN_SECTION__", scan_snapshot())
    html = html.replace("__TOP10_SHARE__", str(bt["stats"]["top10_share"]))
    html = html.replace("__WORST_PCT__", str(bt["stats"]["worst_pct"]))
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
