# Run 1 (9 Nov 2026)

Kept as a record. What it showed, and what changed for run 2:

- E1, E2, F crashed: FlashInfer compiles FP8 KV-cache kernels at startup and the image had no CUDA compiler. Run 2 uses the CUDA 13 devel image.
- On the L4, BF16 could not meet 75 ms per token even at 0.5 requests a second (one user at a time). The sweep started too high and stopped at the first strict failure. Run 2 starts at 0.25 req/s and also reports a readable service level (150 ms per token).
- The first quality rule (no category more than 3 points worse) failed runs that were statistically identical to A: categories hold 20 to 112 questions, so one or two answers move a category by 3 to 10 points. Run 2 uses an exact McNemar test on paired answers instead. summary.md here is recomputed with the run 2 rules.
