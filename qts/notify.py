"""Discord webhook 通知。URL 讀環境變數 DISCORD_WEBHOOK_URL（勿寫進程式碼）。"""
from __future__ import annotations

import os
import time

import requests

MAX_LEN = 1900  # Discord 單則上限 2000，留餘裕


def send(text: str, retries: int = 3) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK")
    if not url:
        print(f"[notify] 未設定 DISCORD_WEBHOOK_URL，訊息僅印出：\n{text}")
        return False
    # 過長訊息切段送出
    chunks: list[str] = []
    while text:
        chunks.append(text[:MAX_LEN])
        text = text[MAX_LEN:]
    ok = True
    for chunk in chunks:
        for attempt in range(retries):
            try:
                r = requests.post(url, json={"content": chunk}, timeout=30)
                if r.status_code in (200, 204):
                    break
                if r.status_code == 429:  # rate limit
                    time.sleep(float(r.json().get("retry_after", 2)))
                    continue
                print(f"[notify] 失敗 status={r.status_code} {r.text[:200]}")
            except Exception as e:  # noqa: BLE001
                print(f"[notify] 網路錯誤 {e}，重試")
                time.sleep(5)
        else:
            ok = False
        time.sleep(0.5)
    return ok
