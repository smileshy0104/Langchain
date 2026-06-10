"""验证 gpt-image-2 模型：遍历文档允许的 size / quality / stream 参数组合，统计每组的成功率与耗时。"""

import asyncio
import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

BASE_URL = os.getenv("IMAGE_BASE_URL", "https://ai-api-us.kkidc.com/v1")
API_KEY = os.getenv("IMAGE_API_KEY", "sk-c7bdHEUPJMrGOhNkZL2OgoOfkgzBMJEiKdjKr7WeCEvQT5gl-107")
OUTPUT_DIR = Path(__file__).parent / "images" / "verify"

MODEL = "gpt-image-2"
PROMPT = "生成游戏《黑道之心》登陆界面截图，5个美女各为不同的种族，名字取自任意亚洲人名，高级的心动的UI设计"
N_PER_REQUEST = 1
RESPONSE_FORMAT = "b64_json"  # 文档要求固定 b64_json

# 文档列出的合法 size
SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",   # 2K
    "2048x1152",   # 2K
    "3840x2160",   # 4K
    "2160x3840",   # 4K
]

# 文档列出的合法 quality（不同模型支持度不同，这里全跑一遍看哪些被接受）
QUALITIES = ["auto", "medium", "standard", "high", "hd"]
# 是否遍历所有 QUALITIES；False 时只用 QUALITY_SINGLE 一个值
TEST_QUALITY = False
QUALITY_SINGLE = "standard"

# 是否同时跑 stream=true（除了默认 false 之外）
TEST_STREAM = False

# 单测试的最大并发数（每个组合是一次请求）
CONCURRENCY = 40
# 单请求超时
REQUEST_TIMEOUT = 6000
# 是否保存返回的图片（True：每个成功用例落盘一份；False：只验证不保存）
SAVE_IMAGES = True
# 每次运行时，把图片放到一个带时间戳的子目录里，便于多次运行结果分开归档
RUN_DIR = OUTPUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")


@dataclass
class Case:
    size: str
    quality: str
    stream: bool

    @property
    def label(self) -> str:
        return f"{self.size}|{self.quality}|stream={'T' if self.stream else 'F'}"


@dataclass
class Result:
    case: Case
    ok: bool
    status: int
    elapsed: float
    detail: str
    saved: Optional[str] = None


def detect_ext(data: bytes) -> str:
    """根据魔术字节判断图片真实格式，避免把 jpeg/webp 错误存成 .png。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    return ".bin"


async def save_image(item: dict, base_path: Path, session: aiohttp.ClientSession) -> str:
    """把 b64_json / url 的图片解码落盘，扩展名按真实魔术字节决定。返回文件名。"""
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        async with session.get(item["url"], timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            raw = await resp.read()
    else:
        raise RuntimeError("返回数据既无 b64_json 也无 url")

    path = base_path.with_suffix(detect_ext(raw))
    path.write_bytes(raw)
    return path.name


async def parse_stream(resp: aiohttp.ClientResponse) -> dict:
    """收集 SSE 流，返回最后一条带 data 字段的 JSON。"""
    last_payload: dict = {}
    async for raw in resp.content:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("data"):
            last_payload = obj
    return last_payload


async def run_case(session: aiohttp.ClientSession, case: Case,
                   sem: asyncio.Semaphore) -> Result:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "n": N_PER_REQUEST,
        "size": case.size,
        "quality": case.quality,
        "response_format": RESPONSE_FORMAT,
        "stream": case.stream,
    }
    url = f"{BASE_URL.rstrip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if case.stream:
        headers["Accept"] = "text/event-stream"

    async with sem:
        start = time.time()
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                status = resp.status
                if status != 200:
                    text = await resp.text()
                    elapsed = time.time() - start
                    detail = text[:200].replace("\n", " ")
                    return Result(case, False, status, elapsed, detail)

                if case.stream:
                    data = await parse_stream(resp)
                else:
                    data = await resp.json(content_type=None)

                elapsed = time.time() - start
                items = (data or {}).get("data") or []
                if not items:
                    return Result(case, False, status, elapsed,
                                  f"无 data 字段 / 空: {str(data)[:160]}")

                saved_name = None
                if SAVE_IMAGES:
                    safe = case.label.replace("|", "_").replace("=", "")
                    ts = datetime.now().strftime("%H%M%S")
                    rand = hashlib.md5(os.urandom(8)).hexdigest()[:6]
                    names: list[str] = []
                    for j, item in enumerate(items):
                        suffix = f"_{j}" if len(items) > 1 else ""
                        # base_path 不带扩展名，save_image 会按真实格式补齐
                        base_path = RUN_DIR / f"{safe}_{ts}_{rand}{suffix}"
                        names.append(await save_image(item, base_path, session))
                    saved_name = ", ".join(names)
                else:
                    saved_name = "b64" if items[0].get("b64_json") else "url"

                return Result(case, True, status, elapsed,
                              f"返回 {len(items)} 张", saved_name)
        except Exception as e:
            elapsed = time.time() - start
            return Result(case, False, 0, elapsed,
                          f"{type(e).__name__}: {str(e)[:160]}")


def build_cases() -> list[Case]:
    cases: list[Case] = []
    qualities = QUALITIES if TEST_QUALITY else [QUALITY_SINGLE]
    streams = [False, True] if TEST_STREAM else [False]
    for size in SIZES:
        for quality in qualities:
            for stream in streams:
                cases.append(Case(size=size, quality=quality, stream=stream))
    return cases


async def main() -> None:
    if SAVE_IMAGES:
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    print(f"目标: {BASE_URL.rstrip('/')}/images/generations")
    print(f"模型: {MODEL}  Prompt: {PROMPT}")
    print(f"组合数: {len(cases)}  并发: {CONCURRENCY}  保存图片: {SAVE_IMAGES}")
    if SAVE_IMAGES:
        print(f"图片目录: {RUN_DIR}")
    print("=" * 96)

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start_all = time.time()
        results = await asyncio.gather(*[run_case(session, c, sem) for c in cases])
        total_time = time.time() - start_all

    print(f"\n{'#':>3}  {'状态':>4}  {'耗时':>8}  size              quality   stream  详情")
    print("-" * 96)
    ok_times: list[float] = []
    fail: list[Result] = []
    for i, r in enumerate(results, 1):
        mark = "✓" if r.ok else "✗"
        suffix = f"  -> {r.saved}" if r.saved else ""
        print(f"{i:>3}  {r.status:>4}  {r.elapsed:>7.2f}s  "
              f"{r.case.size:<17} {r.case.quality:<9} {str(r.case.stream):<6}  "
              f"{mark} {r.detail}{suffix}")
        if r.ok:
            ok_times.append(r.elapsed)
        else:
            fail.append(r)

    print("=" * 96)
    print(f"总耗时: {total_time:.2f}s")
    print(f"成功/总数: {len(ok_times)}/{len(results)}  "
          f"成功率: {len(ok_times) / len(results) * 100:.1f}%")
    if ok_times:
        print(f"最快: {min(ok_times):.2f}s  最慢: {max(ok_times):.2f}s  "
              f"平均: {sum(ok_times) / len(ok_times):.2f}s")

    if fail:
        print("\n失败明细:")
        for r in fail:
            print(f"  [{r.case.label}] status={r.status} {r.detail}")


if __name__ == "__main__":
    asyncio.run(main())
