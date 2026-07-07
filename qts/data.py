"""資料層：載入與驗證日線 OHLCV（STRATEGY.md 第 2 節）。

支援兩種來源：
1. 單一大檔（CSV/Parquet，含 symbol 欄位）
2. 目錄（每檔股票一個 CSV/Parquet，檔名即代號）

必要欄位：date, open, high, low, close, volume；amount 缺漏時以 close*volume 補。
可選欄位：raw_close（未還原收盤價，用於漲跌停判定；缺漏時以 close 代用並於載入時警告）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"資料缺少必要欄位 {missing}（symbol={symbol}），現有欄位：{list(df.columns)}")
    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError("單一大檔必須含 symbol 欄位")
        df["symbol"] = symbol
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    if "raw_close" not in df.columns:
        df["raw_close"] = df["close"]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    num_cols = ["open", "high", "low", "close", "volume", "amount", "raw_close"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return df[["date", "symbol"] + num_cols]


def _validate(df: pd.DataFrame) -> None:
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        n = int(bad_hl.sum())
        raise ValueError(f"資料異常：{n} 筆 high < low，範例：\n{df[bad_hl].head()}")
    bad_range = (df["close"] > df["high"] * 1.0001) | (df["close"] < df["low"] * 0.9999)
    if bad_range.any():
        n = int(bad_range.sum())
        print(f"[data] 警告：{n} 筆收盤價超出 high/low 範圍（可能為還原價 vs 原始高低價混用），已保留", file=sys.stderr)
    if (df["volume"] < 0).any() or (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("資料異常：出現負成交量或非正價格")


def load_ohlcv(path: str | Path) -> dict[str, pd.DataFrame]:
    """載入 OHLCV，回傳 {symbol: DataFrame(index=date)}。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料路徑：{path}")

    frames: list[pd.DataFrame] = []
    if path.is_dir():
        files = sorted([*path.glob("*.csv"), *path.glob("*.parquet")])
        if not files:
            raise FileNotFoundError(f"目錄 {path} 內沒有 CSV/Parquet 檔")
        for f in files:
            raw = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            frames.append(_normalize(raw, symbol=f.stem))
    else:
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frames.append(_normalize(raw))

    alldf = pd.concat(frames, ignore_index=True)
    _validate(alldf)

    out: dict[str, pd.DataFrame] = {}
    for sym, g in alldf.groupby("symbol", sort=True):
        g = g.set_index("date").sort_index()
        out[sym] = g.drop(columns=["symbol"])
    return out


def load_names(universe_csv: str | Path = "data/universe.csv") -> dict[str, str]:
    """代號 → 股票名稱（來源：TWSE ISIN 清單）。檔案不存在時回傳空 dict。"""
    p = Path(universe_csv)
    if not p.exists():
        return {}
    u = pd.read_csv(p, dtype=str)
    return dict(zip(u["code"], u["name"].fillna("")))


def load_industry(universe_csv: str | Path = "data/universe.csv") -> dict[str, str]:
    """代號 → 產業別（交易所分類，去掉冗長字尾以利表格顯示）。"""
    p = Path(universe_csv)
    if not p.exists():
        return {}
    u = pd.read_csv(p, dtype=str)
    if "industry" not in u.columns:
        return {}

    def short(x: str) -> str:
        x = (x or "").strip()
        for suf in ("工業", "事業"):
            if x.endswith(suf) and len(x) > len(suf) + 1:
                return x[: -len(suf)]
        if x.endswith("業") and len(x) > 3:
            return x[:-1]
        return x

    return {c: short(i) for c, i in zip(u["code"], u["industry"].fillna(""))}


def load_themes(themes_csv: str | Path = "themes.csv") -> dict[str, str]:
    """代號 → 題材標籤（人工維護的 themes.csv，專案根目錄、入版控）。

    注意：題材是市場敘事、會過時；缺漏顯示為空白而非猜測。使用者可自行增修。
    """
    p = Path(themes_csv)
    if not p.exists():
        return {}
    t = pd.read_csv(p, dtype=str)
    return dict(zip(t["symbol"], t["theme"].fillna("")))


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """載入大盤指數日線，回傳 DataFrame(index=date, columns 含 close)。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到指數資料：{path}")
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" not in raw.columns or "close" not in raw.columns:
        raise ValueError(f"指數資料至少需要 date, close 欄位，現有：{list(raw.columns)}")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return raw
