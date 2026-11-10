# ILLATE teardown #1: a support bot on one GPU

Independent engineering demonstration. Not client work.

A customer support assistant on Qwen3-8B, served with vLLM on one GPU, measured eight ways. Each run
changes one thing. Every run gets the same load test and the same quality test, and counts only if
it passes both.

## Results (run 2, 10 November 2026)

- Full table: [results/summary.md](results/summary.md). Charts: [cost](results/cost.png), [speed under load](results/latency.png).
- Same conversations priced on hosted APIs: [results/api_costs.md](results/api_costs.md).
- Run 1 and what it taught us: [results-run1/README.md](results-run1/README.md).

What it shows:

1. Putting the fixed system prompt first, and per-customer details after it, made the L4 usable at all (A could not keep a readable speed even at 1 request every 4 seconds; B held 0.5 req/s).
2. FP8 weights + FP8 KV cache + larger batch settings (E1) doubled L4 capacity to 1 req/s and halved cost per conversation, with no measurable quality change. These three changes were tested together.
3. 4-bit weights alone were fine (D), but 4-bit + FP8 KV cache (E2) cut intent accuracy from 83.6% to 79.6% (exact McNemar p = 0.01). Fastest L4 run, failed quality.
4. N-gram speculative decoding (F) added nothing: support replies rephrase articles rather than copy them.
5. The untuned L40S (G) served 5 req/s, 2.7x cheaper per conversation than the best L4, when traffic is high enough to keep it busy.
6. The cheapest hosted API models cost about $0.18 per 1,000 of these conversations with prompt caching. Self-hosting this 8B model only beats that at sustained traffic above roughly 0.9 (L4) to 1.7 (L40S) requests a second, before engineering time. Quality of the API models was not measured.

## Run it

```bash
pip install modal            # once
modal setup                  # once, if this machine isn't logged in to Modal
cd teardown-01
modal run --detach teardown.py              # all runs in parallel, one GPU each
modal run --detach teardown.py --only A,B   # or just some
modal volume get teardown-results / results # fetch results when done
python summarise.py results                 # table + charts
```

## What each run changes

| Run | GPU | Change |
| --- | --- | --- |
| A | L4 | BF16, vLLM defaults, a line that changes per request at the top of the prompt |
| B | L4 | Fixed prompt first, per-request details after it (prefix caching can reuse it) |
| C | L4 | B + FP8 weights |
| D | L4 | B + 4-bit (AWQ) weights |
| E1 | L4 | C + FP8 KV cache + larger batch settings |
| E2 | L4 | D + FP8 KV cache + larger batch settings |
| F | L4 | E1 + n-gram speculative decoding |
| G | L40S | B on a bigger, more expensive GPU |

## Workload

- Questions: 2,000 (load) and 500 (quality) from the Bitext customer support dataset, never seen in retrieval.
- Prompt: about 2,000 tokens of fixed policy and intent labels, plus help-centre articles retrieved with BM25
  from 20,000 past agent answers, plus the customer's message. Output: JSON with `intent` and `reply`.
- Rendered once with Qwen3's chat template, thinking off, so every run sends identical text.

## Measurements

- Load: Poisson arrivals at rising rates. Capacity = highest rate where p95 time to first token is at most
  1.5 s and p95 time per output token is at most 75 ms.
- Cost per 1,000 conversations = GPU price per hour / (capacity x 3600) x 1000, at Modal and RunPod prices.
- Quality: intent accuracy against the dataset labels, valid JSON rate, and 250 GSM8K problems.
  Pass = within 1 point of run A, no category more than 3 points worse, at least 99% valid JSON.

## Limits

- One model, one workload, synthetic traffic. Your traffic, prompts and quality bar will differ.
- Prices change. Results record the GPU, driver and vLLM version with dates.
