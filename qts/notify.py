"""Discord webhook 通知：純文字、embed 卡片、PNG 圖片上傳。

URL 讀環境變數 DISCORD_WEBHOOK_URL（勿寫進程式碼）。
"""
from __future__ import annotations

import json
import os
import time

import requests

MAX_LEN = 1900  # Discord 單則上限 2000，留餘裕

# embed 卡片色（語意色，非漲跌色）
C_GREEN = 0x3FA66A
C_YELLOW = 0xE0A03F
C_RED = 0xE05555
C_BLUE = 0x2B6CB0
C_GRAY = 0x37404D


def _url() -> str | None:
    return os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK")


def _post(payload: dict | None = None, files: dict | None = None, retries: int = 3) -> bool:
    url = _url()
    if not url:
        print(f"[notify] 未設定 DISCORD_WEBHOOK_URL，略過：{str(payload)[:120]}")
        return False
    for _ in range(retries):
        try:
            if files:
                r = requests.post(url, data={"payload_json": json.dumps(payload or {})}, files=files, timeout=60)
            else:
                r = requests.post(url, json=payload, timeout=30)
            if r.status_code in (200, 204):
                return True
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 2)))
                continue
            print(f"[notify] 失敗 status={r.status_code} {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"[notify] 網路錯誤 {e}，重試")
            time.sleep(5)
    return False


def send_embed(
    title: str,
    description: str = "",
    fields: list[tuple[str, str, bool]] | None = None,  # (name, value, inline)
    color: int = C_BLUE,
    footer: str = "",
) -> bool:
    embed: dict = {"title": title[:256], "color": color}
    if description:
        embed["description"] = description[:4000]
    if fields:
        embed["fields"] = [
            {"name": n[:256], "value": (v or "—")[:1024], "inline": inline} for n, v, inline in fields[:25]
        ]
    if footer:
        embed["footer"] = {"text": footer[:2048]}
    return _post({"embeds": [embed]})


def send_png(png: bytes, filename: str = "table.png", content: str = "") -> bool:
    payload = {"content": content[:MAX_LEN]} if content else {}
    return _post(payload, files={"file": (filename, png, "image/png")})


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
