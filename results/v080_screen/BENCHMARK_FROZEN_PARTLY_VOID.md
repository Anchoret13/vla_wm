# `BENCHMARK_FROZEN.json` — partly void as of 2026-08-19

The file is **not edited**: an immutable artifact is superseded by a successor
that cites it (framework §14.2). This note records what no longer holds.

| field | status |
|---|---|
| `verdict: PASS` | **void for `chain1b_lr2`**, questionable for `chain2b_lr2` |
| `tasks: [chain1b_lr2, chain2b_lr2]` | `chain1b_lr2` is withdrawn as a benchmark task |
| `derived_budgets` (`W = H_max = 345`, `H`, per-class bounds, `value_estimator: MC`) | **reopened** by the 2026-08-19 audit — complete-case and context-aliased (framework §5.2.1) |
| `task_detail.*.rate` | `chain1b` 0.400 is a deadline measurement, not a capability measurement |

Stage 1R.1 (`results/v080r_1r1/POOLED_VERDICT_1R1.json`) showed `chain1b_lr2`
succeeds on **20/20** episodes by 500 steps: `@250 = 0.40`, `@350 = 0.95`,
`@500 = 1.00`. The 250-step horizon cut the completion-time distribution at its
median, so the headroom this benchmark was selected for did not exist.
`chain2b_lr2`'s official rate moved to 31/40 = 0.775 (51/70 = 0.729 pooled with
the V8.0 confirmation), above the 0.70 ceiling.

What still stands: the V8.0 *episode records* and their hashes, the harness, and
the seed-family partition. Reserved seeds `3200-3399` remain unspent.

Superseding artifact: `results/v080r_1r1/POOLED_VERDICT_1R1.json`.
