"""并发测试 /v1/images/generations：随机提示词，统计每张图片耗时与成功率。"""

import asyncio
import base64
import hashlib
import os
import random
import time
from datetime import datetime
from pathlib import Path

import aiohttp

BASE_URL = "https://api.ai.kkidc.com/v1"
API_KEY = "sk-c7bdHEUPJMrGOhNkZL2OgoOfkgzBMJEiKdjKr7WeCEvQT5gl-184"
OUTPUT_DIR = Path(__file__).parent / "images"

TOTAL = 1             # 总请求数
CONCURRENCY = 1      # 同时并发数
N_PER_REQUEST = 1     # 单次请求生成几张图（>1 测试上游/proxy 是否支持多张）
MODEL = "gpt-image-2"
SIZE = "2048x2048"
QUALITY = "standard"          # standard | hd
RESPONSE_FORMAT = "b64_json"  # b64_json | url

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
                       sem: asyncio.Semaphore) -> dict:
    async with sem:
        return await _do_request(session, idx)


async def _do_request(session: aiohttp.ClientSession, idx: int) -> dict:
    prompt = random.choice(PROMPTS)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "n": N_PER_REQUEST,
        "size": SIZE,
        "response_format": RESPONSE_FORMAT,
    }
    url = f"{BASE_URL.rstrip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    start = time.time()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            status = resp.status
            data = await resp.json(content_type=None)
            if status != 200:
                elapsed = time.time() - start
                err = (data or {}).get("error", {}).get("message") if isinstance(data, dict) else None
                return {"idx": idx, "ok": False, "status": status, "elapsed": elapsed,
                        "prompt": prompt, "detail": (err or str(data))[:120]}

            saved = []
            for item in data.get("data", []):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                rand_md5 = hashlib.md5(os.urandom(16)).hexdigest()
                path = OUTPUT_DIR / f"{ts}_{rand_md5}.jpeg"
                await save_image(session, item, path)
                saved.append(path.name)
            elapsed = time.time() - start
            return {"idx": idx, "ok": True, "status": status, "elapsed": elapsed,
                    "prompt": prompt, "detail": ", ".join(saved) or "(无图)"}
    except Exception as e:
        elapsed = time.time() - start
        return {"idx": idx, "ok": False, "status": 0, "elapsed": elapsed,
                "prompt": prompt, "detail": f"{type(e).__name__}: {str(e)[:100]}"}


async def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"目标: {BASE_URL.rstrip('/')}/images/generations")
    print(f"模型: {MODEL}  尺寸: {SIZE}  质量: {QUALITY}  格式: {RESPONSE_FORMAT}")
    print(f"请求数: {TOTAL}  并发数: {CONCURRENCY}  每次 n={N_PER_REQUEST}  "
          f"预计图片数: {TOTAL * N_PER_REQUEST}")
    print(f"图片保存: {OUTPUT_DIR}")
    print("=" * 80)

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_request(session, i + 1, sem) for i in range(TOTAL)]
        start_all = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_all

    results.sort(key=lambda r: r["idx"])

    print(f"\n{'#':>3}  {'状态':>4}  {'耗时':>8}  Prompt / 详情")
    print("-" * 80)
    times_ok = []
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        prompt_short = r["prompt"][:20].replace("\n", " ")
        print(f"{r['idx']:>3}  {r['status']:>4}  {r['elapsed']:>7.2f}s  "
              f"{mark} [{prompt_short}] {r['detail']}")
        if r["ok"]:
            times_ok.append(r["elapsed"])

    success = len(times_ok)
    total = len(results)
    print("=" * 80)
    print(f"总耗时: {total_time:.2f}s")
    print(f"成功/总数: {success}/{total}  成功率: {success / total * 100:.1f}%")
    if times_ok:
        print(f"最快: {min(times_ok):.2f}s  最慢: {max(times_ok):.2f}s  "
              f"平均: {sum(times_ok) / len(times_ok):.2f}s")


if __name__ == "__main__":
    asyncio.run(run())
