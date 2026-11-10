| Run | GPU | Change | Capacity, readable (req/s) | Capacity, strict (req/s) | p95 first token at capacity (s) | p95 per token at capacity (ms) | Cost per 1,000 conversations, RunPod / Modal ($) | Intent accuracy | Valid JSON | GSM8K | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | L4 | BF16, vLLM defaults, dynamic line at the top of the prompt | 0 | 0 | n/a | n/a | n/a / n/a | 83.2% | 100.0% | 94.0% | pass (p=1.00) |
| B | L4 | Fixed prompt first, dynamic details after it | 0.5 | 0 | 1.11 | 132 | 0.328 / 0.444 | 83.4% | 100.0% | 92.8% | pass (p=1.00) |
| C | L4 | B + FP8 weights | 0.5 | 0 | 1.49 | 109 | 0.328 / 0.444 | 82.4% | 100.0% | 94.4% | pass (p=0.45) |
| D | L4 | B + 4-bit (AWQ) weights | 0.5 | 0.5 | 1.04 | 65 | 0.328 / 0.444 | 82.6% | 100.0% | 92.4% | pass (p=0.78) |
| E1 | L4 | C + FP8 KV cache + larger batch settings | failed: server exited with code 1: RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} | | | | | | | | |
| E2 | L4 | D + FP8 KV cache + larger batch settings | failed: server exited with code 1: RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} | | | | | | | | |
| F | L4 | E1 + n-gram speculative decoding | failed: server exited with code 1: RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} | | | | | | | | |
| G | L40S | B on an L40S: twice the memory, higher hourly price | 5 | 4 | 1.59 | 137 | 0.061 / 0.108 | 83.4% | 100.0% | 94.0% | pass (p=1.00) |

Readable service level: p95 time to first token <= 2 s and p95 time per output token <= 150 ms (about 7 tokens a second, faster than people read). Strict: <= 1.5 s and <= 75 ms. Capacity = highest Poisson request rate meeting the level. Cost uses the readable capacity and assumes the GPU runs at that rate all hour. Quality pass = intent accuracy within 2 points of run A, at least 99% valid JSON, and no significant difference from A in which questions it gets right (exact McNemar test, p >= 0.05).
