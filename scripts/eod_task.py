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
from qts.notify import send
from qts.scanner import format_rebalance_discord, format_scan_discord, scan

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

    send(f":gear: **盤後更新開始 {today:%Y-%m-%d}**（資料下載約 3 分鐘）")

    # 1. 更新價格
    code, out = run(["scripts/fetch_data.py", "--refresh"])
    log(f"fetch_data exit={code}")
    if code != 0:
        send(f":x: 價格資料更新失敗，今日流程中止。\n```{out[-800:]}```")
        return

    # 假日/資料未出偵測
    px_dates = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["date"])
    last_day = pd.to_datetime(px_dates["date"]).max().date()
    if last_day != today.date():
        log(f"last data day {last_day} != today, stop")
        send(f":zzz: 最新資料日為 {last_day}（今日休市或資料未出），流程結束。")
        return

    # 2. 波段掃描（直接呼叫核心，不再截斷字串）
    log("scan start")
    cfg = Config()
    data = load_ohlcv(ROOT / "data" / "ohlcv.parquet")
    benchmark = load_benchmark(ROOT / "data" / "benchmark" / "taiex.csv")
    names = load_names(ROOT / "data" / "universe.csv")
    try:
        res = scan(data, benchmark, cfg, equity, names)
        if len(res.candidates):
            out_csv = ROOT / "output" / f"scan_{res.asof:%Y%m%d}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            res.candidates.to_csv(out_csv, index=False, encoding="utf-8-sig")
        send(format_scan_discord(res))
        log(f"scan done: {len(res.candidates)} candidates")
    except Exception as e:  # noqa: BLE001
        log(f"scan error: {e}")
        send(f":x: 波段掃描失敗：{e}")

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
            send(format_rebalance_discord(reb, f"{today:%m/%d}", equity))
        else:
            send(f":x: 動能掃描失敗\n```{out[-500:]}```")
    else:
        send(":calendar: 今日非月底，動能組合無動作。盤後流程完成。")
    log("EOD task done")


if __name__ == "__main__":
    main()
