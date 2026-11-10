"""Load and quality harness for an OpenAI-compatible completions server (vLLM).

Load: requests arrive at random (Poisson) at a fixed rate, each streamed, so time to first token
and time per output token are measured per request. Quality: the same prompts at temperature 0,
graded against the dataset's intent labels.
"""

import asyncio
import json
import random
import re
import statistics
import time

import aiohttp

# Two service levels. Strict: about 13 tokens a second per user. Readable: about 7 tokens a
# second, still faster than people read a streamed reply (roughly 4 to 6 tokens a second).
SLO_TTFT_P95 = 1.5  # seconds
SLO_TPOT_P95 = 0.075  # seconds per output token after the first
RELAXED_TTFT_P95 = 2.0
RELAXED_TPOT_P95 = 0.150


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def stream_one(session, url, model, prompt, max_tokens):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    try:
        async with session.post(url, json=body) as r:
            if r.status != 200:
                return {"error": f"http {r.status}"}
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    out_tokens = chunk["usage"].get("completion_tokens", out_tokens)
                ch = chunk.get("choices") or []
                if ch and ch[0].get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
        return {"error": type(e).__name__}
    e2e = time.perf_counter() - t0
    if ttft is None:
        return {"error": "no tokens"}
    tpot = (e2e - ttft) / (out_tokens - 1) if out_tokens > 1 else None
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "out_tokens": out_tokens}


async def run_level(url, model, prompts, rate, n, seed, max_tokens=300):
    """Send n requests with Poisson arrivals at `rate` per second; return per-request records."""
    rng = random.Random(seed)
    order = [prompts[(seed + i) % len(prompts)] for i in range(n)]
    timeout = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        tasks = []
        t_start = time.perf_counter()
        for p in order:
            tasks.append(asyncio.create_task(stream_one(s, url, model, p, max_tokens)))
            await asyncio.sleep(rng.expovariate(rate))
        res = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_start
    return res, wall


def summarise_level(rate, res, wall):
    ok = [r for r in res if "error" not in r]
    ttft = [r["ttft"] for r in ok]
    tpot = [r["tpot"] for r in ok if r["tpot"] is not None]
    toks = sum(r["out_tokens"] for r in ok)
    s = {
        "rate": rate,
        "requests": len(res),
        "errors": len(res) - len(ok),
        "wall_s": wall,
        "ttft_p50": pct(ttft, 50),
        "ttft_p95": pct(ttft, 95),
        "tpot_p50": pct(tpot, 50),
        "tpot_p95": pct(tpot, 95),
        "e2e_p95": pct([r["e2e"] for r in ok], 95),
        "out_tok_per_s": toks / wall if wall else None,
        "mean_out_tokens": statistics.mean([r["out_tokens"] for r in ok]) if ok else None,
    }

    def meets(ttft, tpot):
        return bool(
            ok
            and s["errors"] <= 0.01 * len(res)
            and s["ttft_p95"] <= ttft
            and s["tpot_p95"] is not None
            and s["tpot_p95"] <= tpot
        )

    s["passes"] = meets(SLO_TTFT_P95, SLO_TPOT_P95)
    s["passes_relaxed"] = meets(RELAXED_TTFT_P95, RELAXED_TPOT_P95)
    return s


async def sweep(url, model, prompts, rates, seed=42, log=print):
    """Raise the request rate until the service levels fail; capacity = last passing rate."""
    await run_level(url, model, prompts, 2.0, 20, seed + 999)  # warm-up, not recorded
    levels = []
    for rate in rates:
        n = int(min(1200, max(80, rate * 90)))
        res, wall = await run_level(url, model, prompts, rate, n, seed)
        s = summarise_level(rate, res, wall)
        levels.append(s)
        log(s)
        if not s["passes_relaxed"]:
            break
    strict = [lv["rate"] for lv in levels if lv["passes"]]
    relaxed = [lv["rate"] for lv in levels if lv["passes_relaxed"]]
    return {
        "levels": levels,
        "capacity_rps": max(strict) if strict else 0.0,
        "capacity_relaxed_rps": max(relaxed) if relaxed else 0.0,
    }


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(text):
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "intent" not in obj or "reply" not in obj:
        return None
    return obj


async def complete_all(url, model, prompts, max_tokens, concurrency=32):
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=900)

    async def one(s, p):
        async with sem:
            body = {"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0}
            for _ in range(3):
                try:
                    async with s.post(url, json=body) as r:
                        j = await r.json()
                        return j["choices"][0]["text"]
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2)
            return ""

    async with aiohttp.ClientSession(timeout=timeout) as s:
        return await asyncio.gather(*[one(s, p) for p in prompts])


async def quality(url, model, items):
    """items: dicts with prompt, intent, category. Returns per-item outputs and scores."""
    texts = await complete_all(url, model, [it["prompt"] for it in items], 300)
    rows = []
    for it, t in zip(items, texts):
        obj = parse_reply(t)
        pred = (obj or {}).get("intent", "")
        rows.append(
            {
                "id": it["id"],
                "intent": it["intent"],
                "category": it["category"],
                "valid": obj is not None,
                "pred": pred,
                "correct": isinstance(pred, str) and pred.strip().lower() == it["intent"],
                "reply": (obj or {}).get("reply", t),
            }
        )
    n = len(rows)
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["correct"]
        c[1] += 1
    return {
        "n": n,
        "intent_accuracy": sum(r["correct"] for r in rows) / n,
        "valid_rate": sum(r["valid"] for r in rows) / n,
        "by_category": {k: v[0] / v[1] for k, v in sorted(by_cat.items())},
        "rows": rows,
    }


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    nums = NUM_RE.findall(text or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


async def gsm8k(url, model, items):
    texts = await complete_all(url, model, [it["prompt"] for it in items], 512)
    correct = 0
    for it, t in zip(items, texts):
        got = last_number(t)
        correct += got is not None and abs(got - it["answer"]) < 1e-6
    return {"n": len(items), "accuracy": correct / len(items)}
