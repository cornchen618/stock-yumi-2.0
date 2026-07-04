"""資料抓取：TWSE ISIN 官方清單 + Yahoo Finance 日線。

流程：
  1. 從 isin.twse.com.tw 取得上市(strMode=2)＋上櫃(strMode=4)普通股代號
     （排除 ETF「00」開頭、TDR「91」開頭、特別股、權證）
  2. yfinance 分批下載 2017-01-01 至今（auto_adjust=False）
     - 還原價：以 AdjClose/Close 因子換算 O/H/L/C（含股息與分割）
     - raw_close：未還原收盤（漲跌停判定用）
     - amount：raw_close × volume（Yahoo 無成交金額，以此近似）
  3. 分塊快取（data/cache/chunk_*.parquet，可中斷續抓）→ 合併 data/ohlcv.parquet
  4. 大盤 ^TWII → data/benchmark/taiex.csv

已知限制（回測報告須如實標註）：
  - Yahoo 無已下市台股 → 存活者偏差無法完全消除
  - 股票股利還原精度不如付費資料源；成交金額為近似值

用法：
  python scripts/fetch_data.py                # 全市場
  python scripts/fetch_data.py --limit 30    # 測試用小樣本
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
START = "2017-01-01"
CHUNK = 80

ISIN_URLS = {
    "TW": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",   # 上市
    "TWO": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",  # 上櫃
}


def fetch_universe() -> pd.DataFrame:
    """回傳 DataFrame(code, name, suffix, industry)。僅 4 碼數字普通股，排除 00/91 開頭。

    ISIN 頁每列格式：代號名稱 | ISIN | 上市日 | 市場別 | 產業別 | CFI | 備註
    """
    rows = []
    for suffix, url in ISIN_URLS.items():
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = "big5"
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, flags=re.S):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
            if len(tds) < 5:
                continue
            m = re.match(r"(\d{4})　(.+)", tds[0].strip())
            if not m:
                continue
            code, name = m.group(1), m.group(2).strip()
            if code.startswith("00") or code.startswith("91"):
                continue  # ETF / TDR
            industry = re.sub(r"<[^>]+>", "", tds[4]).strip() or "未分類"
            rows.append({"code": code, "name": name, "suffix": suffix, "industry": industry})
        time.sleep(1.0)
    df = pd.DataFrame(rows).drop_duplicates("code", keep="first").sort_values("code")
    if len(df) < 500:
        raise RuntimeError(f"ISIN 清單解析異常：僅取得 {len(df)} 檔（頁面格式可能變更）")
    return df.reset_index(drop=True)


def _extract_one(raw: pd.DataFrame, ticker: str, code: str) -> pd.DataFrame | None:
    """從批次下載結果取出單一股票並轉為系統 schema。"""
    try:
        sub = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
    except KeyError:
        return None
    sub = sub.dropna(subset=["Close", "Adj Close"])
    sub = sub[(sub["Volume"] > 0) & (sub["Close"] > 0)]
    if len(sub) < 300:  # 指標暖機 260 根 + 緩衝
        return None
    factor = sub["Adj Close"] / sub["Close"]
    out = pd.DataFrame({
        "date": sub.index,
        "symbol": code,
        "open": (sub["Open"] * factor).round(4),
        "high": (sub["High"] * factor).round(4),
        "low": (sub["Low"] * factor).round(4),
        "close": sub["Adj Close"].round(4),
        "volume": sub["Volume"].astype(np.int64),
        "amount": (sub["Close"] * sub["Volume"]).round(0),
        "raw_close": sub["Close"].round(4),
    })
    bad = (out["high"] < out["low"]) | (out["close"] <= 0)
    return out[~bad].reset_index(drop=True)


def fetch_chunk(codes: list[tuple[str, str]]) -> tuple[list[pd.DataFrame], list[str]]:
    """下載一批股票。回傳 (frames, 失敗代號)。"""
    tickers = {f"{c}.{sfx}": c for c, sfx in codes}
    raw = yf.download(
        list(tickers.keys()), start=START, auto_adjust=False, actions=False,
        group_by="ticker", threads=True, progress=False,
    )
    frames, missing = [], []
    for tkr, code in tickers.items():
        got = _extract_one(raw, tkr, code) if raw is not None and len(raw) else None
        if got is None:
            missing.append(f"{code}.{tickers and tkr.split('.')[-1]}")
        else:
            frames.append(got)
    return frames, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="僅抓前 N 檔（測試用）")
    p.add_argument("--out", default=str(ROOT / "data"))
    p.add_argument("--refresh", action="store_true", help="清除分塊快取後全量重抓（日常更新用）")
    args = p.parse_args()

    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"
    if args.refresh and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("[refresh] 已清除價格快取，全量重抓", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark").mkdir(parents=True, exist_ok=True)

    print("[1/4] 抓取 TWSE ISIN 上市/上櫃普通股清單 ...", flush=True)
    uni = fetch_universe()
    if args.limit:
        uni = uni.head(args.limit)
    print(f"      共 {len(uni)} 檔（上市 {(uni['suffix'] == 'TW').sum()}、上櫃 {(uni['suffix'] == 'TWO').sum()}）", flush=True)
    uni.to_csv(out_dir / "universe.csv", index=False, encoding="utf-8-sig")

    print("[2/4] 下載大盤 ^TWII ...", flush=True)
    twii = yf.download("^TWII", start=START, auto_adjust=False, progress=False)
    if isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.get_level_values(0)
    twii = twii.dropna(subset=["Close"])
    bench = pd.DataFrame({
        "date": twii.index, "open": twii["Open"], "high": twii["High"],
        "low": twii["Low"], "close": twii["Close"], "volume": twii["Volume"],
    })
    bench.to_csv(out_dir / "benchmark" / "taiex.csv", index=False)
    print(f"      {len(bench)} 個交易日（{bench['date'].iloc[0].date()} ~ {bench['date'].iloc[-1].date()}）", flush=True)

    print(f"[3/4] 分批下載個股（每批 {CHUNK} 檔，可中斷續抓）...", flush=True)
    codes = list(zip(uni["code"], uni["suffix"]))
    all_missing: list[str] = []
    n_chunks = (len(codes) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        cache_f = cache_dir / f"chunk_{ci:03d}.parquet"
        batch = codes[ci * CHUNK:(ci + 1) * CHUNK]
        if cache_f.exists():
            print(f"      chunk {ci + 1}/{n_chunks} 已存在，跳過", flush=True)
            continue
        t0 = time.time()
        frames, missing = fetch_chunk(batch)
        all_missing += missing
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(cache_f, index=False)
        else:
            pd.DataFrame().to_parquet(cache_f, index=False)
        print(f"      chunk {ci + 1}/{n_chunks}：成功 {len(frames)}、失敗 {len(missing)}，{time.time() - t0:.0f}s", flush=True)
        time.sleep(1.5)

    print("[4/4] 合併輸出 ...", flush=True)
    parts = [pd.read_parquet(f) for f in sorted(cache_dir.glob("chunk_*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts:
        raise SystemExit("沒有任何資料成功下載")
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])

    # 資料品質防呆：最後一日檔數不足（Yahoo 日界線/盤中部分發布）→ 剔除該日
    counts = full.groupby("date")["symbol"].size()
    if len(counts) > 6:
        baseline = counts.iloc[-6:-1].median()
        while len(counts) > 1 and counts.iloc[-1] < baseline * 0.6:
            bad_day = counts.index[-1]
            full = full[full["date"] != bad_day]
            print(f"[guard] 最後一日 {bad_day.date()} 僅 {int(counts.iloc[-1])} 檔"
                  f"（基準 {int(baseline)}），資料不完整 → 已剔除", flush=True)
            counts = counts.iloc[:-1]

    full.to_parquet(out_dir / "ohlcv.parquet", index=False)

    n_sym = full["symbol"].nunique()
    print("=" * 60)
    print(f"完成：{n_sym} 檔、{len(full):,} 列 → {out_dir / 'ohlcv.parquet'}")
    print(f"日期範圍：{full['date'].min().date()} ~ {full['date'].max().date()}")
    if all_missing:
        print(f"失敗/資料不足 {len(all_missing)} 檔（多為新上市或低流動）：{', '.join(all_missing[:20])}{' ...' if len(all_missing) > 20 else ''}")
        (out_dir / "missing.txt").write_text("\n".join(all_missing), encoding="utf-8")


if __name__ == "__main__":
    main()
