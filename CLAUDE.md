# CLAUDE.md — 專案工作規範

台股波段量化交易系統（qts）。以繁體中文與使用者溝通。所有工作視為 production-grade。

## 專案結構

- `qts/` 核心套件：config（全參數）→ data → indicators → signals（A突破/B破底翻/C蓄勢）→ regime → backtest / momentum（動能組合）→ metrics → scanner → notify（Discord）
- `scripts/` 執行入口：fetch_data / fetch_chips / run_backtest / run_scan / scan_momentum / preview_scan / eod_task / monitor_intraday / selftest / chip_factor_test
- 文件即規格：`STRATEGY.md`（波段策略＋KPI Gate＋驗證紀錄）、`MOMENTUM.md`（動能組合）、`OPERATIONS.md`（排程/輸出規則/下單流程）
- Windows 排程四個（TWQuant-*）：08:55 盤中監控、13:05 盤中預掃、17:40 盤後、21:30 籌碼更新

## 不可違反的領域紀律

1. **無前視**：訊號只用 T 日收盤前資料，rolling 極值一律 shift(1)；T+1 開盤成交。改訊號邏輯後必跑 `python scripts/selftest.py`（含截斷不變性檢查），沒過不得交付。
2. **預登記**：任何策略改動先寫下規則與採用標準再看數據；比較變體不得事後挑最好的報。回測含全部成本（手續費×折數、證交稅、滑價、漲停不追）。
3. **KPI Gate**（STRATEGY.md 第 9 節）沒全過的策略只能標「紙上觀察」，所有輸出須帶此標註。
4. **系統永不自動下單**；金錢相關的最後動作永遠留給使用者。
5. **祕密只放環境變數**（DISCORD_WEBHOOK_URL、FINMIND_TOKEN），嚴禁寫進程式碼或 commit；`data/`、`output/`、`logs/`、`holdings.csv`、`settings.json` 不入版控。

## 工程標準（摘自使用者的 Senior Engineer 標準，全文在其 Downloads）

- 先讀懂再改：改動前檢視相關檔案與既有慣例，遵循現有模式，不引入不必要的依賴/抽象/框架。
- 最小正確改動：不順手重構無關程式、不改無關格式；發現無關問題另行提出，不擅自修。
- 錯誤處理要刻意：不吞錯誤、失敗要浮出（參考 fetch/notify 的重試與退避模式）。
- 驗證後才宣告完成：跑 selftest / 實際執行受影響腳本；命令失敗要說明原因與影響，不得宣稱成功。
- 溝通像資深工程師：說清楚改了什麼、為什麼、風險、如何驗證；不確定就講明假設。

## 對使用者輸出的慣例

- 股票一律「代號＋名稱」；價格用原始市價並標明日期；停損/停利價直接算出來顯示。
- Discord 用等寬區塊表格、觸發條件中文化、附欄位說明；訊息不可截尾（曾因 [-1500:] 截斷把 A/B 策略砍掉）。

## 常用指令

```powershell
python scripts/selftest.py                                   # 改核心後必跑
python scripts/fetch_data.py --refresh                        # 價格更新（含最後日完整性防呆）
python scripts/run_backtest.py --data data/ohlcv.parquet --benchmark data/benchmark/taiex.csv
python scripts/run_momentum.py  --data data/ohlcv.parquet --benchmark data/benchmark/taiex.csv
python scripts/scan_momentum.py --data data/ohlcv.parquet --equity <權益>
```

## 已知資料陷阱

- Yahoo 在日界線附近會回傳不完整的最後一日（fetch_data 已有防呆，勿移除）。
- yfinance 無下市股票 → 所有回測數字偏樂觀，報告必須註明存活者偏差。
- 融資券約 21:00 才公布；法人 15:00；正式訊號鏈只依賴價格資料。
