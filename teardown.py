"""ILLATE teardown #1: a support bot on one GPU, measured on Modal.

Run everything (all configurations in parallel):
    modal run --detach teardown.py
Run only some configurations:
    modal run --detach teardown.py --only A,B
Fetch results to ./results:
    modal volume get teardown-results / results

Each configuration starts its own vLLM server on its own GPU, runs the quality test (500 support
questions + 250 GSM8K problems at temperature 0), then raises the request rate until the service
levels fail. Results land in the `teardown-results` volume under <run>/.
"""

import json
import os
import subprocess
import sys
import time

import modal

VLLM_VERSION = "0.32.0"
BASE = "Qwen/Qwen3-8B"

# CUDA 13 devel image: vLLM 0.32 ships torch built for CUDA 13, and FlashInfer compiles some
# kernels (FP8 KV cache attention among them) at startup, which needs nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        f"vllm=={VLLM_VERSION}", "aiohttp", "datasets", "rank_bm25", "huggingface_hub", "ninja"
    )
    .env({"VLLM_LOGGING_LEVEL": "WARNING", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_file(os.path.join(os.path.dirname(__file__), "bench.py"), "/root/bench.py")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "prompt_text.py"), "/root/prompt_text.py"
    )
)

hf_cache = modal.Volume.from_name("teardown-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("teardown-results", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/results": results}

app = modal.App("illate-teardown-01", image=image)

COMMON = ["--max-model-len", "8192", "--gpu-memory-utilization", "0.92", "--seed", "0"]
TUNED = ["--kv-cache-dtype", "fp8", "--max-num-seqs", "256", "--max-num-batched-tokens", "8192"]
NGRAM = [
    "--speculative-config",
    json.dumps({"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}),
]

CONFIGS = {
    "A": {
        "gpu": "L4",
        "model": BASE,
        "prompts": "dynamic_first",
        "args": [],
        "change": "BF16, vLLM defaults, dynamic line at the top of the prompt",
    },
    "B": {
        "gpu": "L4",
        "model": BASE,
        "prompts": "static_first",
        "args": [],
        "change": "Fixed prompt first, dynamic details after it",
    },
    "C": {
        "gpu": "L4",
        "model": "Qwen/Qwen3-8B-FP8",
        "prompts": "static_first",
        "args": [],
        "change": "B + FP8 weights",
    },
    "D": {
        "gpu": "L4",
        "model": "Qwen/Qwen3-8B-AWQ",
        "prompts": "static_first",
        "args": [],
        "change": "B + 4-bit (AWQ) weights",
    },
    "E1": {
        "gpu": "L4",
        "model": "Qwen/Qwen3-8B-FP8",
        "prompts": "static_first",
        "args": TUNED,
        "change": "C + FP8 KV cache + larger batch settings",
    },
    "E2": {
        "gpu": "L4",
        "model": "Qwen/Qwen3-8B-AWQ",
        "prompts": "static_first",
        "args": TUNED,
        "change": "D + FP8 KV cache + larger batch settings",
    },
    "F": {
        "gpu": "L4",
        "model": "Qwen/Qwen3-8B-FP8",
        "prompts": "static_first",
        "args": TUNED + NGRAM,
        "change": "E1 + n-gram speculative decoding",
    },
    "G": {
        "gpu": "L40S",
        "model": BASE,
        "prompts": "static_first",
        "args": [],
        "change": "B on an L40S: twice the memory, higher hourly price",
    },
}

RATES = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]


@app.function(volumes=VOLS, cpu=4, memory=16384, timeout=3600)
def build_prompts():
    """Render every prompt once with Qwen3's chat template, so all runs send identical text."""
    sys.path.insert(0, "/root")
    import random

    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from transformers import AutoTokenizer

    import prompt_text as pt

    out = "/results/prompts"
    if os.path.exists(f"{out}/done.json"):
        print("prompts already built")
        return json.load(open(f"{out}/done.json"))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE)

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    ds = ds.shuffle(seed=42)
    intents = sorted(set(ds["intent"]))
    system = pt.system_prompt(intents)

    kb = ds.select(range(0, 20000))  # past agent answers = the help centre to retrieve from
    held = ds.select(range(20000, len(ds)))
    quality_rows = held.select(range(0, 500))
    load_rows = held.select(range(500, 2500))

    docs = kb["response"]
    bm25 = BM25Okapi([d.lower().split() for d in docs])

    def retrieve(q, budget=1500):
        scores = bm25.get_scores(q.lower().split())
        top = sorted(range(len(docs)), key=lambda i: -scores[i])[:12]
        picked, used = [], 0
        for i in top:
            n = len(tok(docs[i])["input_ids"])
            if used + n > budget and picked:
                break
            picked.append(docs[i])
            used += n
        return "\n\n".join(f"[Article {k + 1}]\n{d}" for k, d in enumerate(picked))

    rng = random.Random(7)

    def render(q, variant):
        cust = pt.customer_line(rng, rng.randint(1, 28))
        ctx = f"Help centre results:\n{retrieve(q)}\n\nCustomer message: {q}"
        if variant == "dynamic_first":
            msgs = [
                {"role": "system", "content": cust + "\n\n" + system},
                {"role": "user", "content": ctx},
            ]
        else:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": cust + "\n\n" + ctx},
            ]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    stats = {}
    for variant in ("dynamic_first", "static_first"):
        rng.seed(7)
        load = [render(r["instruction"], variant) for r in load_rows]
        with open(f"{out}/load_{variant}.jsonl", "w") as f:
            for p in load:
                f.write(json.dumps({"prompt": p}) + "\n")
        rng.seed(8)
        with open(f"{out}/quality_{variant}.jsonl", "w") as f:
            for i, r in enumerate(quality_rows):
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "prompt": render(r["instruction"], variant),
                            "intent": r["intent"],
                            "category": r["category"],
                        }
                    )
                    + "\n"
                )
        lens = [len(tok(p)["input_ids"]) for p in load[:200]]
        stats[variant] = {
            "prompt_tokens_mean": sum(lens) / len(lens),
            "min": min(lens),
            "max": max(lens),
        }

    gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(250))
    with open(f"{out}/gsm8k.jsonl", "w") as f:
        for i, r in enumerate(gsm):
            msg = [
                {
                    "role": "user",
                    "content": "Solve this problem. Keep the working short, then end "
                    "with 'The answer is <number>'.\n\n" + r["question"],
                }
            ]
            p = tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ans = float(r["answer"].split("####")[-1].strip().replace(",", ""))
            f.write(json.dumps({"id": i, "prompt": p, "answer": ans}) + "\n")

    with open(f"{out}/system_prompt.txt", "w") as f:
        f.write(system)
    stats["system_prompt_tokens"] = len(tok(system)["input_ids"])
    stats["intents"] = intents
    json.dump(stats, open(f"{out}/done.json", "w"), indent=1)
    results.commit()
    hf_cache.commit()
    print(json.dumps(stats, indent=1))
    return stats


@app.function(volumes=VOLS, cpu=4, memory=8192, timeout=3600)
def download_models():
    from huggingface_hub import snapshot_download

    for m in sorted({c["model"] for c in CONFIGS.values()}):
        snapshot_download(m)
        print("downloaded", m)
    hf_cache.commit()


def _run(name):
    sys.path.insert(0, "/root")
    import asyncio

    import bench

    cfg = CONFIGS[name]
    hf_cache.reload()
    results.reload()
    out = f"/results/{name}"
    os.makedirs(out, exist_ok=True)

    def save(fname, obj):
        json.dump(obj, open(f"{out}/{fname}", "w"), indent=1)
        results.commit()

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    cmd = (
        ["vllm", "serve", cfg["model"], "--port", "8000", "--served-model-name", "m"]
        + COMMON
        + cfg["args"]
    )
    env = {
        "name": name,
        "gpu": gpu,
        "vllm": VLLM_VERSION,
        "cmd": " ".join(cmd),
        "config": cfg,
        "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    save("env.json", env)

    log = open(f"{out}/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    import urllib.request

    t0 = time.time()
    while True:
        if proc.poll() is not None:
            log.flush()
            tail = open(f"{out}/server.log").read().strip().splitlines()[-25:]
            print(name, "server log tail:\n" + "\n".join(tail))
            env["failed"] = f"server exited with code {proc.returncode}: {tail[-1] if tail else ''}"
            env["log_tail"] = tail
            save("env.json", env)
            return env
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            break
        except Exception:  # noqa: BLE001
            if time.time() - t0 > 1800:
                env["failed"] = "server did not start in 30 minutes"
                save("env.json", env)
                proc.kill()
                return env
            time.sleep(5)
    env["startup_s"] = round(time.time() - t0)
    save("env.json", env)

    url = "http://localhost:8000/v1/completions"
    pdir = "/results/prompts"
    q_items = [json.loads(line) for line in open(f"{pdir}/quality_{cfg['prompts']}.jsonl")]
    g_items = [json.loads(line) for line in open(f"{pdir}/gsm8k.jsonl")]
    loads = [json.loads(line)["prompt"] for line in open(f"{pdir}/load_{cfg['prompts']}.jsonl")]

    q = asyncio.run(bench.quality(url, "m", q_items))
    save("quality.json", q)
    print(name, "intent accuracy", q["intent_accuracy"], "valid", q["valid_rate"])
    g = asyncio.run(bench.gsm8k(url, "m", g_items))
    save("gsm8k.json", g)
    print(name, "gsm8k", g["accuracy"])

    lines = []

    def log_level(s):
        lines.append(s)
        print(name, json.dumps(s))

    sw = asyncio.run(bench.sweep(url, "m", loads, RATES, log=log_level))
    save("load.json", sw)
    proc.terminate()
    env["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    env["capacity_rps"] = sw["capacity_rps"]
    env["capacity_relaxed_rps"] = sw["capacity_relaxed_rps"]
    save("env.json", env)
    return env


@app.function(gpu="L4", volumes=VOLS, cpu=8, memory=32768, timeout=3 * 3600)
def run_l4(name):
    return _run(name)


@app.function(gpu="L40S", volumes=VOLS, cpu=8, memory=32768, timeout=3 * 3600)
def run_l40s(name):
    return _run(name)


@app.function(volumes=VOLS, cpu=1, memory=2048, timeout=6 * 3600)
def orchestrate(names):
    """Runs on Modal, so the whole job survives the laptop disconnecting (modal run --detach)."""
    download_models.remote()
    print(json.dumps(build_prompts.remote(), indent=1))
    calls = {n: (run_l40s if CONFIGS[n]["gpu"] == "L40S" else run_l4).spawn(n) for n in names}
    print("started:", ", ".join(names), "(each on its own GPU)")
    out = {}
    for n, c in calls.items():
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001 - one failed run must not stop the others
            r = {"failed": f"{type(e).__name__}: {e}"}
        out[n] = r.get("failed") or r.get("capacity_rps")
        print(n, "failed:" if r.get("failed") else "capacity", out[n])
    print("done. fetch results with: modal volume get teardown-results / results")
    return out


@app.local_entrypoint()
def main(only: str = ""):
    names = [n.strip() for n in only.split(",") if n.strip()] or list(CONFIGS)
    unknown = set(names) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown runs: {sorted(unknown)}; choose from {list(CONFIGS)}")
    print("all work runs on Modal; safe to close this terminal once the runs have started")
    print(orchestrate.remote(names))
