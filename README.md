# 台股波段量化交易系統 (qts)

策略規格、風控、回測方法論、上線門檻全部定義在 **[STRATEGY.md](STRATEGY.md)**（單一事實來源）。

## 安裝

```powershell
pip install -r requirements.txt
```

## 快速開始

```powershell
# 1. 系統自我檢查（指標正確性 / 無前視偏差 / 帳務恆等式）
python scripts/selftest.py

# 2. 合成資料跑通管線（僅驗證程式，績效無意義）
python scripts/make_sample_data.py
python scripts/run_backtest.py --data data/sample/ohlcv.parquet --benchmark data/sample/benchmark.csv

# 3. 真實資料回測（in-sample / out-of-sample 分段，OOS 只能看一次）
python scripts/run_backtest.py --data data/ohlcv --benchmark data/benchmark/taiex.csv --start 2019-01-01 --end 2023-12-31
python scripts/run_backtest.py --data data/ohlcv --benchmark data/benchmark/taiex.csv --start 2024-01-01

# 4. 每日盤後掃描（實盤 SOP 見 STRATEGY.md 第 10 節）
python scripts/run_scan.py --data data/ohlcv --benchmark data/benchmark/taiex.csv --equity 1000000
```

## 資料格式

放在 `data/` 下，CSV 或 Parquet：

- **個股日線**：單一大檔（含 `symbol` 欄）或每檔一個檔案（檔名 = 代號）。
  必要欄位 `date, open, high, low, close, volume`（volume 單位：股）；
  建議欄位 `amount`（成交金額，元）、`raw_close`（未還原收盤價，用於漲跌停判定）。
  價格請用**還原權息**價；至少涵蓋 2019 年至今、含已下市股票。
- **大盤指數**：`date, open, high, low, close, volume`（至少 `date, close`）。

## 參數調整

全部參數在 [qts/config.py](qts/config.py)（對照 STRATEGY.md 第 11 節），可用 JSON 覆寫：

```powershell
python scripts/run_backtest.py ... --config my_params.json
```

```json
{"strat_a": {"vol_mult_breakout": 2.0}, "portfolio": {"risk_per_trade": 0.005}}
```

## 實際使用流程（手動）

**每月一次 — 動能組合**（目前唯一建議實際操作的策略，見 MOMENTUM.md）：
```powershell
# 每月最後交易日收盤後（約 17:30 後）
python scripts/fetch_data.py --refresh                                    # 更新價格（約 3 分鐘）
python scripts/scan_momentum.py --data data/ohlcv.parquet --equity <目前權益>
# 次一交易日開盤按清單執行；成交後更新 holdings.csv（symbol,shares）
```

**每日盤後 — 波段掃描**（KPI Gate 未過，僅供觀察／紙上交易，勿實際下單）：
```powershell
python scripts/fetch_data.py --refresh
python scripts/run_scan.py --data data/ohlcv.parquet --benchmark data/benchmark/taiex.csv --equity <權益>
```

**籌碼資料更新**（需 FINMIND_TOKEN 環境變數）：
```powershell
python scripts/fetch_chips.py --datasets inst,margin
```

## 紀律

- `selftest.py` 沒過 → 系統不可信，禁止使用。
- 回測 KPI Gate（STRATEGY.md 第 9 節）沒全過 → 禁止實盤。
- Out-of-sample 只能看一次；看完再改參數就等於把 OOS 變成 IS。
