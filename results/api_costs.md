Per conversation: 2,757 input tokens on average, of which 1,385 are the fixed system prompt a provider can cache, and 66 output tokens (measured in this run).

| Provider | Model | Cost per 1,000 conversations, no caching ($) | With prompt caching ($) |
| --- | --- | --- | --- |
| OpenAI | gpt-6-luna | 0.31 | 0.18 |
| Anthropic | Claude Haiku 5.5 | 0.31 | 0.18 |
| Google | Gemini 3.1 Flash-Lite | 0.79 | 0.48 |
| OpenAI | gpt-6.1-terra | 3.09 | 1.77 |
| Google | Gemini 3.5 Flash | 4.73 | 2.86 |
| Anthropic | Claude Sonnet 5.5 | 6.17 | 3.54 |

**Self-hosted Qwen3-8B, from this run (RunPod prices, GPU fully used):**

- Tuned L4 (E1): $0.164 per 1,000 at 1 req/s. Beats the cheapest API with caching ($0.18) only when traffic averages above 0.89 req/s (3,206 conversations an hour), before counting engineering time.
- L40S, untuned (G): $0.061 per 1,000 at 5 req/s. Beats the cheapest API with caching ($0.18) only when traffic averages above 1.65 req/s (5,923 conversations an hour), before counting engineering time.

Not compared: answer quality (these models were not run), response time, and data-residency needs. Cached-input prices ignore one-off cache writes and Gemini's small storage fee.
