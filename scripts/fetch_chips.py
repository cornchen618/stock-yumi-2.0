"""FinMind 籌碼面資料抓取（三大法人買賣超；可選融資券）。

- 匿名限速：預設每請求間隔 9~11 秒（約 360/hr，低於官方限制）
- 可中斷續抓：data/chips_cache/{dataset}/{symbol}.parquet
- 觸發限流訊息時自動休息 15 分鐘重試
- 完成後合併輸出 data/chips_institutional.parquet（長格式：date, stock_id, name, buy, sell）

用法：
  python scripts/fetch_chips.py                        # 法人買賣超，604 檔股池
  python scripts/fetch_chips.py --datasets inst,margin # 加抓融資券
  python scripts/fetch_chips.py --merge-only           # 只合併既有快取
環境變數 FINMIND_TOKEN 存在時自動帶入（註冊免費帳號可提高限額）。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.finmindtrade.com/api/v4/data"
START_DATE = "2019-01-01"

DATASETS = {
    "inst": ("TaiwanStockInstitutionalInvestorsBuySell", "chips_institutional.parquet"),
    "margin": ("TaiwanStockMarginPurchaseShortSale", "chips_margin.parquet"),
}


def tradable_universe(min_amt: float = 15e6, min_vol: float = 300_000) -> list[str]:
    df = pd.read_parquet(ROOT / "data" / "ohlcv.parquet", columns=["symbol", "amount", "volume"])
    g = df.groupby("symbol").agg(med_amt=("amount", "median"), med_vol=("volume", "median"))
    return sorted(g[(g.med_amt >= min_amt) & (g.med_vol >= min_vol)].index.tolist())


def fetch_symbol(dataset: str, symbol: str, token: str | None,
                 start_date: str = START_DATE) -> pd.DataFrame | None:
    params = {"dataset": dataset, "data_id": symbol, "start_date": start_date}
    if token:
        params["token"] = token
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, timeout=60)
            j = r.json()
        except Exception as e:  # noqa: BLE001 - 網路錯誤重試
            print(f"    {symbol}: 網路錯誤 {e}，60s 後重試", flush=True)
            time.sleep(60)
            continue
        msg = str(j.get("msg", ""))
        if r.status_code == 200 and msg == "success":
            return pd.DataFrame(j.get("data", []))
        if "level" in msg or r.status_code in (402, 429):
            print(f"    {symbol}: 觸發限流（{msg}），休息 15 分鐘", flush=True)
            time.sleep(900)
            continue
        print(f"    {symbol}: 失敗 status={r.status_code} msg={msg}", flush=True)
        return None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="inst", help="逗號分隔：inst,margin")
    p.add_argument("--sleep", type=float, default=9.0)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--update", action="store_true",
                   help="增量更新：已快取者只補最近 10 天並去重（每日排程用）")
    args = p.parse_args()

    token = os.environ.get("FINMIND_TOKEN")
    uni = tradable_universe()
    print(f"股池 {len(uni)} 檔；token={'有' if token else '無（匿名限速）'}", flush=True)

    for key in args.datasets.split(","):
        dataset, out_name = DATASETS[key.strip()]
        cache = ROOT / "data" / "chips_cache" / key.strip()
        cache.mkdir(parents=True, exist_ok=True)

        if not args.merge_only:
            if args.update:
                todo = uni
            else:
                todo = [s for s in uni if not (cache / f"{s}.parquet").exists()]
            print(f"[{dataset}] 待抓 {len(todo)} / {len(uni)} 檔（update={args.update}）", flush=True)
            for k, sym in enumerate(todo):
                f = cache / f"{sym}.parquet"
                start = START_DATE
                old = None
                if args.update and f.exists():
                    old = pd.read_parquet(f)
                    if len(old):
                        start = (pd.to_datetime(old["date"]).max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                df = fetch_symbol(dataset, sym, token, start_date=start)
                if df is not None:
                    if old is not None and len(old):
                        df = pd.concat([old, df], ignore_index=True)
                        dedup_keys = [c for c in ("date", "stock_id", "name") if c in df.columns]
                        df = df.drop_duplicates(dedup_keys, keep="last")
                    df.to_parquet(f, index=False)
                if (k + 1) % 20 == 0:
                    print(f"  進度 {k + 1}/{len(todo)}", flush=True)
                time.sleep(args.sleep + random.uniform(0, 2))

        parts = [pd.read_parquet(f) for f in sorted(cache.glob("*.parquet"))]
        parts = [x for x in parts if len(x)]
        if parts:
            full = pd.concat(parts, ignore_index=True)
            full["date"] = pd.to_datetime(full["date"])
            out = ROOT / "data" / out_name
            full.to_parquet(out, index=False)
            print(f"[{dataset}] 合併完成：{full['stock_id'].nunique()} 檔、{len(full):,} 列 → {out}", flush=True)
        else:
            print(f"[{dataset}] 無資料可合併", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
