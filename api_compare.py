"""Price the same support conversations on hosted APIs, using the token counts measured in the run.

    python api_compare.py results

Writes results/api_costs.md. Prices are list prices per 1M tokens, standard tier, read from the
providers' pricing pages on 10 November 2026:
    OpenAI     https://developers.openai.com/api/docs/pricing
    Anthropic  https://claude.com/pricing
    Google     https://ai.google.dev/gemini-api/docs/pricing
Quality was NOT measured for these models; this is a cost comparison only.
"""

import json
import os
import statistics
import sys

# (provider, model, input, cached input, output) in USD per 1M tokens
APIS = [
    ("OpenAI", "gpt-6-luna", 0.10, 0.01, 0.50),
    ("Anthropic", "Claude Haiku 5.5", 0.10, 0.01, 0.50),
    ("Google", "Gemini 3.1 Flash-Lite", 0.25, 0.025, 1.50),
    ("OpenAI", "gpt-6.1-terra", 1.00, 0.05, 5.00),
    ("Google", "Gemini 3.5 Flash", 1.50, 0.15, 9.00),
    ("Anthropic", "Claude Sonnet 5.5", 2.00, 0.10, 10.00),
]
GPU_HOURLY = {"L4": 0.59, "L40S": 1.09}  # RunPod, as in summarise.py


def main(d):
    stats = json.load(open(os.path.join(d, "done.json")))
    prompt = stats["static_first"]["prompt_tokens_mean"]
    cached = stats["system_prompt_tokens"]  # the fixed prefix a provider can cache
    outs = []
    for run in os.listdir(d):
        p = os.path.join(d, run, "load.json")
        if os.path.exists(p):
            outs += [
                lv["mean_out_tokens"]
                for lv in json.load(open(p))["levels"]
                if lv["mean_out_tokens"]
            ]
    out = statistics.mean(outs)

    def per_1000(inp, cin, outp, use_cache):
        c = cached if use_cache else 0
        return ((prompt - c) * inp + c * cin + out * outp) / 1e6 * 1000

    rows = [(p, m, per_1000(i, c, o, False), per_1000(i, c, o, True)) for p, m, i, c, o in APIS]
    summary = json.load(open(os.path.join(d, "summary.json")))
    best = {r["run"]: r for r in summary}
    cheapest_api = min(r[3] for r in rows)

    lines = [
        f"Per conversation: {prompt:,.0f} input tokens on average, of which {cached:,} are the fixed "
        f"system prompt a provider can cache, and {out:.0f} output tokens (measured in this run).",
        "",
        "| Provider | Model | Cost per 1,000 conversations, no caching ($) | With prompt caching ($) |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {p} | {m} | {a:.2f} | {b:.2f} |" for p, m, a, b in rows]
    lines += ["", "**Self-hosted Qwen3-8B, from this run (RunPod prices, GPU fully used):**", ""]
    for run, label in (("E1", "Tuned L4 (E1)"), ("G", "L40S, untuned (G)")):
        r = best.get(run)
        if r and r["cost_per_1000_runpod"]:
            hourly = GPU_HOURLY[r["gpu"]]
            breakeven = hourly / (cheapest_api / 1000) / 3600
            lines.append(
                f"- {label}: ${r['cost_per_1000_runpod']:.3f} per 1,000 at {r['capacity_rps']:g} req/s. "
                f"Beats the cheapest API with caching (${cheapest_api:.2f}) only when traffic averages "
                f"above {breakeven:.2f} req/s ({breakeven * 3600:,.0f} conversations an hour), "
                "before counting engineering time."
            )
    lines += [
        "",
        "Not compared: answer quality (these models were not run), response time, and data-residency "
        "needs. Cached-input prices ignore one-off cache writes and Gemini's small storage fee.",
    ]
    md = "\n".join(lines) + "\n"
    open(os.path.join(d, "api_costs.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
