"""并发测试 /v1/images/generations：随机提示词，统计每张图片耗时与成功率。"""

import asyncio
import base64
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import aiohttp

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://ai-api-us.kkidc.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "sk--107")
OUTPUT_DIR = Path(__file__).parent / "images"

TOTAL = 1            # 总请求数
CONCURRENCY = 1      # 同时并发数
N_PER_REQUEST = 1     # 单次请求生成几张图（>1 测试上游/proxy 是否支持多张）
MODEL = "gpt-image-2"
SIZE = "3840x2160"
QUALITY = "medium"              # auto | low | medium | high
OUTPUT_FORMAT = "png"         # png | jpeg | webp

PROMPTS = [
    "A breathtaking panoramic view of the Swiss Alps at golden hour, snow-capped peaks reflecting warm sunlight, 8K wallpaper quality",
    "日出时分的张家界石柱群，云海翻涌，金色光线穿透薄雾，超高清壁纸",
    "A stunning aerial view of turquoise glacial lakes in Banff National Park, surrounded by pine forests and rocky mountains",
    "冰岛黑沙滩上的钻石冰块，日落余晖将天空染成紫橙色，极致风光摄影",
    "A dramatic Patagonian landscape with jagged mountain peaks reflected in a perfectly still lake at dawn",
    "新西兰米尔福德峡湾的瀑布群，雨后彩虹横跨峡谷，壮丽自然风光",
    "A serene Norwegian fjord at twilight, deep blue water between towering cliffs, northern lights beginning to appear",
    "九寨沟五彩池的绚烂色彩，翠绿碧蓝的湖水倒映秋天红叶，4K壁纸",
    "A majestic waterfall cascading into a misty tropical canyon, lush green vegetation, cinematic landscape photography",
    "西藏纳木错湖边的星空银河，湖面如镜倒映漫天繁星，震撼夜景壁纸",
    "A golden autumn forest path covered in maple leaves, sunbeams filtering through the canopy, warm tones wallpaper",
    "北海道富良野的薰衣草花田延伸到天际，远处是连绵雪山，夏日壁纸",
    "A winter wonderland scene of a frozen lake surrounded by snow-covered evergreen trees under a pastel pink sunrise",
    "春天的吉野山万株樱花盛放，粉色花海覆盖整座山丘，航拍壁纸",
    "A vast lavender field in Provence at sunset, purple rows stretching to the horizon under a dramatic orange sky",
    "额济纳旗胡杨林的金秋，万亩胡杨金叶灿烂，蓝天下的绝美秋色壁纸",
    "A moody misty morning in a Pacific Northwest old-growth forest, massive moss-covered trees, ethereal atmosphere",
    "雨后的元阳梯田层层叠叠倒映天光云影，壮美的大地艺术壁纸",
    "A dramatic thunderstorm rolling over endless golden wheat fields, dark clouds with lightning, epic landscape",
    "塔克拉玛干沙漠的日落，绵延沙丘形成完美的光影曲线，极简风光壁纸",
    "A crystal clear tropical beach with overwater bungalows in the Maldives, turquoise lagoon, paradise wallpaper",
    "大堡礁的航拍心形珊瑚礁，碧蓝海水中的天然奇观，高清壁纸",
    "A breathtaking sunset over Santorini's white and blue architecture overlooking the Aegean Sea, travel wallpaper",
    "极光在冰岛教堂山上空舞动，绿紫色光带倒映在前景水面，梦幻壁纸",
    "A massive ocean wave curling perfectly at sunset, golden light shining through the water, surf photography wallpaper",
    "马尔代夫荧光海滩的夜景，蓝色荧光浮游生物点亮海岸线，奇幻壁纸",
    "A panoramic view of a double rainbow over the Grand Canyon after a storm, dramatic lighting, nature wallpaper",
    "青海茶卡盐湖的天空之镜，人影与云朵完美倒映，超现实风光壁纸",
    "A vivid Milky Way galaxy arching over Joshua Tree National Park, long exposure astrophotography wallpaper",
    "火烧云下的洱海渔船剪影，大理苍山洱海的黄昏极致美景壁纸",
    "A stunning Manhattan skyline at blue hour, city lights reflecting on the Hudson River, urban wallpaper",
    "上海陆家嘴三件套夜景，黄浦江两岸璀璨灯火，都市天际线壁纸",
    "A dramatic Hong Kong cityscape from Victoria Peak at night, neon lights and skyscrapers, cyberpunk city wallpaper",
    "东京塔在樱花季的夜景，粉色花枝前景框住璀璨铁塔，城市壁纸",
    "A breathtaking Dubai skyline with Burj Khalifa piercing through morning fog, golden sunrise, futuristic city wallpaper",
    "重庆洪崖洞的魔幻夜景，层叠的吊脚楼灯火辉煌倒映嘉陵江，赛博都市壁纸",
    "A rainy night in Tokyo's Shibuya crossing, neon reflections on wet streets, cinematic urban photography wallpaper",
    "雨后的巴黎香榭丽舍大街，埃菲尔铁塔在远处闪烁，浪漫城市壁纸",
    "A charming narrow street in Santorini with bougainvillea cascading over white walls, Mediterranean wallpaper",
    "苏州平江路的水乡古巷，粉墙黛瓦小桥流水，江南水墨城市壁纸",
    "桂林山水甲天下，漓江两岸喀斯特山峰与渔船，中国山水壁纸",
    "黄山云海日出，奇松怪石在金色晨光中若隐若现，国画风光壁纸",
    "长城在秋天红叶中蜿蜒穿越群山，航拍视角的壮美中国壁纸",
]


async def fetch_url(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        resp.raise_for_status()
        return await resp.read()


async def save_image(session: aiohttp.ClientSession, item: dict, path: Path) -> None:
    if item.get("b64_json"):
        path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        path.write_bytes(await fetch_url(session, item["url"]))
    else:
        raise RuntimeError(f"返回数据既无 b64_json 也无 url: {str(item)[:120]}")


async def send_request(session: aiohttp.ClientSession, idx: int,
                       sem: asyncio.Semaphore,
                       start_event: asyncio.Event) -> dict:
    await start_event.wait()
    async with sem:
        return await _do_request(session, idx)


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(len(ordered) * percent / 100)
    return ordered[max(0, min(rank - 1, len(ordered) - 1))]


def print_time_distribution(title: str, values: list[float]) -> None:
    if not values:
        print(f"{title}: 无数据")
        return
    print(
        f"{title}: "
        f"count={len(values)} "
        f"min={min(values):.2f}s "
        f"p50={percentile(values, 50):.2f}s "
        f"p90={percentile(values, 90):.2f}s "
        f"p95={percentile(values, 95):.2f}s "
        f"p99={percentile(values, 99):.2f}s "
        f"max={max(values):.2f}s "
        f"avg={sum(values) / len(values):.2f}s"
    )


def merge_usage(total: dict | None, usage: dict | None) -> dict | None:
    if not usage:
        return total
    if total is None:
        total = {}
    for key, value in usage.items():
        if isinstance(value, dict):
            total[key] = merge_usage(total.get(key), value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
    return total


def usage_text(usage: dict | None) -> str:
    if not usage:
        return "-"
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    )
    parts = [f"{key}={usage[key]}" for key in keys if key in usage]
    return ", ".join(parts) if parts else json.dumps(usage, ensure_ascii=False)


async def _do_request(session: aiohttp.ClientSession, idx: int) -> dict:
    prompt = random.choice(PROMPTS)
    payload = {
        "model": MODEL,
        "prompt": "生成游戏《黑道之心》登陆界面截图，5个帅哥各为不同的种族，名字取自任意亚洲人名，高级的心动的UI设计",
        "n": N_PER_REQUEST,
        "size": SIZE,
        "quality": QUALITY,
        "output_format": OUTPUT_FORMAT,
    }
    url = f"{BASE_URL.rstrip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            status = resp.status
            data = await resp.json(content_type=None)
            elapsed = time.perf_counter() - start
            usage = data.get("usage") if isinstance(data, dict) else None
            if status != 200:
                err = (data or {}).get("error", {}).get("message") if isinstance(data, dict) else None
                return {"idx": idx, "ok": False, "status": status, "elapsed": elapsed,
                        "prompt": prompt, "usage": usage, "detail": (err or str(data))[:120]}

            saved = []
            for item in data.get("data", []):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                rand_md5 = hashlib.md5(os.urandom(16)).hexdigest()
                path = OUTPUT_DIR / f"{ts}_{rand_md5}.{OUTPUT_FORMAT}"
                await save_image(session, item, path)
                saved.append(path.name)
            return {"idx": idx, "ok": True, "status": status, "elapsed": elapsed,
                    "prompt": prompt, "usage": usage, "detail": ", ".join(saved) or "(无图)"}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"idx": idx, "ok": False, "status": 0, "elapsed": elapsed,
                "prompt": prompt, "usage": None, "detail": f"{type(e).__name__}: {str(e)[:100]}"}


async def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"目标: {BASE_URL.rstrip('/')}/images/generations")
    print(f"模型: {MODEL}  尺寸: {SIZE}  质量: {QUALITY}  格式: {OUTPUT_FORMAT}")
    print(f"请求数: {TOTAL}  并发数: {CONCURRENCY}  每次 n={N_PER_REQUEST}  "
          f"预计图片数: {TOTAL * N_PER_REQUEST}")
    print(f"图片保存: {OUTPUT_DIR}")
    print("=" * 80)

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start_event = asyncio.Event()
        tasks = [
            asyncio.create_task(send_request(session, i + 1, sem, start_event))
            for i in range(TOTAL)
        ]
        await asyncio.sleep(0.1)
        start_all = time.perf_counter()
        start_event.set()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_all

    results.sort(key=lambda r: r["idx"])

    print(f"\n{'#':>3}  {'状态':>4}  {'耗时':>8}  {'Tokens':>35}  Prompt / 详情")
    print("-" * 80)
    times_ok = []
    all_times = []
    status_counter = Counter()
    usage_total = None
    for r in results:
        mark = "OK" if r["ok"] else "ERR"
        prompt_short = r["prompt"][:20].replace("\n", " ")
        token_text = usage_text(r.get("usage"))
        print(f"{r['idx']:>3}  {r['status']:>4}  {r['elapsed']:>7.2f}s  "
              f"{token_text:>35}  {mark} [{prompt_short}] {r['detail']}")
        status_counter[r["status"]] += 1
        all_times.append(r["elapsed"])
        usage_total = merge_usage(usage_total, r.get("usage"))
        if r["ok"]:
            times_ok.append(r["elapsed"])

    success = len(times_ok)
    total = len(results)
    rate_429 = status_counter[429] / total if total else 0
    print("=" * 80)
    print(f"总耗时: {total_time:.2f}s")
    print(f"成功/总数: {success}/{total}  成功率: {success / total * 100:.1f}%")
    print(f"429/总数: {status_counter[429]}/{total} ({rate_429:.2%})")
    print(f"状态码分布: {dict(sorted(status_counter.items()))}")
    print_time_distribution("全部请求耗时分布", all_times)
    print_time_distribution("成功生图耗时分布", times_ok)
    if usage_total:
        print(f"Tokens 汇总: {usage_text(usage_total)}")
        print(f"usage 汇总: {json.dumps(usage_total, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(run())
