# CLAUDE.md

## Token discipline

Measured from this project's own transcripts on 2026-08-09. Two sessions, two
different failure modes — diagnose which one applies before reaching for a fix.

| Session | Shape | Dominant cost | Lever |
|---|---|---|---|
| 46 days, 5,118 turns | avg **519K** context | 91% resending context | clear the session |
| 1 day, 217 turns | avg 205K context | **51% extended thinking** | lower reasoning effort |

Output (thinking included) bills at 5x input. Cache reads bill at 0.1x. So a
long session bleeds on context resend; a short session bleeds on thinking.

### Rules

1. **Clear between milestones.** Do not carry one session across a whole
   experiment arc. Hand off through `plan_and_progress/<date>.md` — that file is
   the continuity mechanism, not the context window. The 46-day session cost 59x
   what a same-work fresh session cost.

2. **Match reasoning effort to the task.** High effort for debugging, experiment
   design, and statistical-claim review. Low for renames, print statements,
   running scripts, mechanical edits. 104 of 217 turns in one session were
   thinking-only — 3.5 reasoning turns per Edit is too many for routine work.

3. **Read narrowly.** `plan_and_progress/*.md` dailies reach 114KB. Grep to
   locate, then read the span with `offset`/`limit`. Never read a whole daily.

4. **Never Read a PNG in a long session.** Images stay in context permanently and
   are re-sent every turn afterward. View them in a throwaway session.

5. **Do not re-read after Edit.** Edit errors if it fails, so the verifying read
   is pure cost — and it invalidates the cache tail (cache-write is 32% of spend
   in the long session).

6. **Background long jobs; wait for the completion notification.** A running
   background task costs nothing. Each status check costs one full turn. Polling
   an 8-hour run every 5 min ≈ 96 turns ≈ 50M tokens at 519K context.

### Not a token lever

Disk cleanup. `results/` held 194GB and none of it ever entered context — only
what gets Read or shelled into does. Clean `results/` for disk, clear the session
for tokens. Do not confuse the two.

## Repo facts

- `.gitignore` blocks new large artifacts but not ones already committed. History
  carried 117.6GB of `.pt` blobs under paths the ignore rules now cover
  (e.g. `results/**/shards/`), because the rules were added after those commits.
- `results/` keeps compact JSON/JSONL metrics, manifests, and audit reports under
  version control. Tensors, checkpoints, caches, and rollout videos are
  reproducible from the locked contracts + hashes in the committed audit JSONs.
