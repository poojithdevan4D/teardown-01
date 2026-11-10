"""Turn results/<run>/*.json into results/summary.md, results/summary.json and two charts.

python summarise.py results
"""

import json
import math
import os
import sys

PRICE = {  # USD per GPU hour; Modal from modal.com/pricing, RunPod from runpod.io/gpu-cloud/pricing (Nov 2026)
    "L4": {"modal": 0.000222 * 3600, "runpod": 0.59},
    "L40S": {"modal": 0.000542 * 3600, "runpod": 1.09},
}
ORDER = ["A", "B", "C", "D", "E1", "E2", "F", "G"]
STRICT = (1.5, 0.075)  # p95 seconds to first token, p95 seconds per output token
READABLE = (2.0, 0.150)


def load(d, run, f):
    p = os.path.join(d, run, f)
    return json.load(open(p)) if os.path.exists(p) else None


def capacity(levels, slo):
    ok = [
        lv["rate"]
        for lv in levels
        if lv["errors"] <= 0.01 * lv["requests"]
        and lv["ttft_p95"] is not None
        and lv["tpot_p95"] is not None
        and lv["ttft_p95"] <= slo[0]
        and lv["tpot_p95"] <= slo[1]
    ]
    return max(ok) if ok else 0.0


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def per_1000(price, cap):
    return price / (cap * 3600) * 1000 if cap else None


def collect(d):
    rows = []
    for run in ORDER:
        env = load(d, run, "env.json")
        if not env:
            continue
        cfg = env["config"]
        q, g, lo = (
            load(d, run, "quality.json"),
            load(d, run, "gsm8k.json"),
            load(d, run, "load.json"),
        )
        levels = (lo or {}).get("levels", [])
        cap, cap_strict = capacity(levels, READABLE), capacity(levels, STRICT)
        at_cap = next((lv for lv in levels if lv["rate"] == cap), None)
        price = PRICE[cfg["gpu"]]
        rows.append(
            {
                "run": run,
                "gpu": cfg["gpu"],
                "change": cfg["change"],
                "failed": env.get("failed"),
                "capacity_rps": cap,
                "capacity_strict_rps": cap_strict,
                "ttft_p95_s": at_cap and at_cap["ttft_p95"],
                "tpot_p95_ms": at_cap and at_cap["tpot_p95"] * 1000,
                "out_tok_per_s": at_cap and at_cap["out_tok_per_s"],
                "cost_per_1000_runpod": per_1000(price["runpod"], cap),
                "cost_per_1000_modal": per_1000(price["modal"], cap),
                "cost_per_1000_runpod_strict": per_1000(price["runpod"], cap_strict),
                "intent_acc": q and q["intent_accuracy"],
                "valid": q and q["valid_rate"],
                "by_category": q and q["by_category"],
                "gsm8k": g and g["accuracy"],
                "levels": levels,
                "q_rows": q and q["rows"],
            }
        )
    if not rows:
        raise SystemExit(f"no runs found in {d}")

    ref = next((r for r in rows if r["run"] == "A" and r["q_rows"]), None)
    ref_ok = {x["id"]: x["correct"] for x in ref["q_rows"]} if ref else {}
    for r in rows:
        r["quality_pass"], r["mcnemar_p"] = None, None
        if r is ref:
            r["quality_pass"], r["mcnemar_p"] = True, 1.0
        elif ref and r["q_rows"]:
            mine = {x["id"]: x["correct"] for x in r["q_rows"]}
            b = sum(ref_ok[i] and not mine[i] for i in ref_ok)
            c = sum(mine[i] and not ref_ok[i] for i in ref_ok)
            r["mcnemar_p"] = mcnemar_p(b, c)
            r["quality_pass"] = bool(
                r["intent_acc"] >= ref["intent_acc"] - 0.02
                and r["valid"] >= 0.99
                and r["mcnemar_p"] >= 0.05
            )
    for r in rows:
        r.pop("q_rows", None)
    return rows


def f(x, fmt):
    return "n/a" if x is None else format(x, fmt)


def table(rows):
    lines = [
        "| Run | GPU | Change | Capacity, readable (req/s) | Capacity, strict (req/s) "
        "| p95 first token at capacity (s) | p95 per token at capacity (ms) "
        "| Cost per 1,000 conversations, RunPod / Modal ($) | Intent accuracy | Valid JSON "
        "| GSM8K | Quality |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["failed"]:
            lines.append(
                f"| {r['run']} | {r['gpu']} | {r['change']} | failed: {r['failed']} "
                "| | | | | | | | |"
            )
            continue
        qp = {True: "pass", False: "FAIL", None: "n/a"}[r["quality_pass"]]
        lines.append(
            f"| {r['run']} | {r['gpu']} | {r['change']} | {f(r['capacity_rps'], 'g')} "
            f"| {f(r['capacity_strict_rps'], 'g')} "
            f"| {f(r['ttft_p95_s'], '.2f')} | {f(r['tpot_p95_ms'], '.0f')} "
            f"| {f(r['cost_per_1000_runpod'], '.3f')} / {f(r['cost_per_1000_modal'], '.3f')} "
            f"| {f(r['intent_acc'], '.1%')} | {f(r['valid'], '.1%')} | {f(r['gsm8k'], '.1%')} "
            f"| {qp} (p={f(r['mcnemar_p'], '.2f')}) |"
        )
    note = (
        "\nReadable service level: p95 time to first token <= 2 s and p95 time per output token "
        "<= 150 ms (about 7 tokens a second, faster than people read). Strict: <= 1.5 s and "
        "<= 75 ms. Capacity = highest Poisson request rate meeting the level. Cost uses the "
        "readable capacity and assumes the GPU runs at that rate all hour. Quality pass = intent "
        "accuracy within 2 points of run A, at least 99% valid JSON, and no significant "
        "difference from A in which questions it gets right (exact McNemar test, p >= 0.05).\n"
    )
    return "\n".join(lines) + "\n" + note


def charts(d, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped charts")
        return

    ok = [r for r in rows if not r["failed"] and r["cost_per_1000_runpod"]]
    if ok:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2b6cb0" if r["quality_pass"] is not False else "#b0b0b0" for r in ok]
        ax.barh(
            [f"{r['run']} ({r['gpu']})" for r in ok],
            [r["cost_per_1000_runpod"] for r in ok],
            color=colors,
        )
        ax.invert_yaxis()
        ax.set_xlabel("USD per 1,000 conversations at RunPod prices (grey = failed quality)")
        ax.set_title("Cost per 1,000 support conversations, Qwen3-8B")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "cost.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(8, 4))
    for r in rows:
        if r["levels"]:
            ax.plot(
                [lv["rate"] for lv in r["levels"]],
                [lv["tpot_p95"] * 1000 for lv in r["levels"]],
                marker="o",
                label=f"{r['run']} ({r['gpu']})",
            )
    ax.axhline(150, color="#888", linestyle="--", linewidth=1)
    ax.text(0.01, 155, "readable: 150 ms", transform=ax.get_yaxis_transform(), color="#555")
    ax.set_xlabel("Requests per second")
    ax.set_ylabel("p95 time per output token (ms)")
    ax.set_title("Speed per user as traffic rises")
    ax.legend(frameon=False, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "latency.png"), dpi=160)
    print("charts: cost.png, latency.png")


def main(d):
    rows = collect(d)
    md = table(rows)
    open(os.path.join(d, "summary.md"), "w").write(md)
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(md)
    charts(d, rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
