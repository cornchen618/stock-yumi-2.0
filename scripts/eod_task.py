"""盤後自動任務（排程於交易日 17:40）。

流程：
  1. 更新全市場價格資料（fetch_data --refresh，含最後一日完整性防呆）
  2. 波段掃描（共用 qts.scanner；紙上觀察，KPI Gate 未過 → 標註勿實單）
  3. 若今日為本月最後一個平日 → 動能組合換股清單（次日開盤執行）
  4. 全部結果以 Discord 友善格式發送（qts.scanner.format_*）

權益讀 settings.json 的 "equity"；成交後請維護 holdings.csv。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qts.config import Config
from qts.data import load_benchmark, load_names, load_ohlcv
from qts.notify import C_BLUE, C_GRAY, C_GREEN, C_RED, C_YELLOW, send_embed, send_png
from qts.render import rebalance_table_png, scan_table_png
from qts.scanner import STRAT_ZH, TRIGGER_ZH, scan

PY = sys.executable


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(args: list[str], timeout: int = 2400) -> tuple[int, str]:
    r = subprocess.run(
        [PY] + args, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def is_last_weekday_of_month(d: datetime) -> bool:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.month != d.month


def main() -> None:
    today = datetime.now()
    log("EOD task start")
    if today.weekday() >= 5:
        log("weekend, skip")
        return
    equity = 1_000_000.0
    sp = ROOT / "settings.json"
    if sp.exists():
        equity = float(json.loads(sp.read_text(encoding="utf-8")).get("equity", equity))

    send_embed(f"⚙️ 盤後更新開始 {today:%Y-%m-%d}", "資料下載約 3 分鐘", color=C_BLUE)

    # 1. 更新價格
    code, out = run(["scripts/fetch_data.py", "--refresh"])
    log(f"fetch_data exit={code}")
    if code != 0:
        send_embed("❌ 價格資料更新失敗", f"今日流程中止。\n```{out[-700:]}```", color=C_RED)
        return

    # 假日/資料未出偵測
    px_dates = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date"])
    last_day = pd.to_datetime(px_dates["date"]).max().date()
    if last_day != today.date():
        log(f"last data day {last_day} != today, stop")
        send_embed("💤 今日休市或資料未出", f"最新資料日為 {last_day}，流程結束。", color=C_GRAY)
        return

    # 2. 波段掃描（直接呼叫核心，不再截斷字串）
    log("scan start")
    cfg = Config()
    data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")
    names = load_names(ROOT / "data" / "universe.csv")
    try:
        res = scan(data, benchmark, cfg, equity, names)
        c = res.candidates
        n_a = int((c["strategy"] == "A").sum()) if len(c) else 0
        n_b = int((c["strategy"] == "B").sum()) if len(c) else 0
        n_c = int((c["strategy"] == "C").sum()) if len(c) else 0
        send_embed(
            f"📋 波段掃描 {res.asof:%m/%d}（紙上觀察，勿實單）",
            f"市場濾網：{'多頭 ✅ 三策略皆可' if res.bull else '空頭 ⛔ A/C 停用、B 風險減半'}\n"
            f"候選　A:{n_a}　B:{n_b}　C:{n_c}｜壓縮觀察 {len(res.watch)} 檔",
            color=C_GREEN if res.bull else C_RED,
        )
        if len(c):
            out_csv = ROOT / "output" / f"scan_{res.asof:%Y%m%d}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            c.to_csv(out_csv, index=False, encoding="utf-8-sig")
            send_png(scan_table_png(c, f"{res.asof:%Y-%m-%d}", STRAT_ZH, TRIGGER_ZH), filename="scan.png")
        log(f"scan done: {len(c)} candidates")
    except Exception as e:  # noqa: BLE001
        log(f"scan error: {e}")
        send_embed("❌ 波段掃描失敗", str(e)[:500], color=C_RED)

    # 3. 月底 → 動能換股清單
    if is_last_weekday_of_month(today):
        code, out = run([
            "scripts/scan_momentum.py", "--data", "data/ohlcv.parquet",
            "--equity", str(equity), "--holdings", "holdings.csv",
        ])
        log(f"scan_momentum exit={code}")
        csv_path = ROOT / "output" / f"momentum_rebalance_{today:%Y%m%d}.csv"
        if code == 0 and csv_path.exists():
            reb = pd.read_csv(csv_path, dtype={"symbol": str})
            if "name" not in reb.columns:
                reb["name"] = reb["symbol"].map(names)
            reb["name"] = reb["name"].fillna(reb["symbol"].map(names)).fillna("")
            send_png(rebalance_table_png(reb, f"{today:%m/%d}", equity), filename="rebalance.png",
                     content="💰 **動能組合月度換股清單**（明日開盤執行）")
            send_embed("👉 明日執行步驟",
                       "① 先掛賣單（開盤市價）② 再掛買單 ③ 開盤漲幅≥9.5% 的買單刪掉\n"
                       "④ 成交後更新 holdings.csv、settings.json", color=C_YELLOW)
        else:
            send_embed("❌ 動能掃描失敗", f"```{out[-500:]}```", color=C_RED)
    else:
        send_embed("📅 今日非月底", "動能組合無動作，盤後流程完成。", color=C_GRAY)

    # 3.5 持股除權息檢查（預告/入帳/監控校正檔；無持股時靜默）
    if (ROOT / "holdings.csv").exists():
        code, out = run(["scripts/check_dividends.py"], timeout=600)
        log(f"dividends exit={code}")
        if code != 0:
            log(f"dividends error: {out[-300:]}")

    # 4. 八問決策簡報（發 Discord＋存檔供儀表板嵌入）
    code, out = run(["scripts/brief.py", "--discord"])
    log(f"brief exit={code}")
    if code != 0:
        log(f"brief error: {out[-300:]}")

    # 5. 每日戰報 PNG（Discord 內直接顯示：大盤/持股/帳戶一張圖）
    code, out = run(["scripts/daily_report.py"])
    log(f"daily_report exit={code}")
    if code != 0:
        log(f"daily_report error: {out[-300:]}")

    # 6. 重生金流儀表板（本機瀏覽用；完整互動版在 output/dashboard.html）
    code, out = run(["scripts/make_dashboard.py"])
    log(f"dashboard exit={code}")
    if code != 0:
        log(f"dashboard error: {out[-300:]}")
    log("EOD task done")


if __name__ == "__main__":
    main()
