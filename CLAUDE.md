# CLAUDE.md

## Goal anchor — read this before every experiment and every report

**The project goal.** Deploy a frozen VLA on tasks it has a **low success rate**
on, collect **real-world trajectories during deployment**, and use them to train a
**latent-space world model that predicts future state** — action-conditioned,
**no reconstruction** — in order to make the VLA better at deployment time.

The predictive object is framework §5.2:

```
z̃_{t+c} = T_θ(z_t, E_a(u^i))          roll the LATENT forward under a candidate action
D_θ(z̃_{t+c}) = (Δw, Δy, r, V_k, p_succ)   read heads off the PREDICTED latent
```

**Before starting any new attempt, check it against all four axes.** If an axis
fails, the work is off-mission — say so and change it, do not proceed and
rationalise afterwards.

| axis | requirement | the failure mode to catch |
|---|---|---|
| task | low baseline success rate | drifting to an easier task because it is more tractable |
| object | latent `T_θ` rolled forward under a candidate action | building a discriminative head on the *current* observation and calling it a world model |
| target | future **latent**, no pixel/obs reconstruction | adding a reconstruction loss because it is easier to fit |
| data | trajectories collected **during deployment** | pre-training offline on acquisition seeds and freezing at deploy |

**Search the literature before choosing an architecture, not just before choosing a
task.** On 2026-08-28 a latent world model was written from first principles — a
frozen alignment representation as the latent, MSE against the exact next latent,
three separate training stages — and every failure traced to that. The standard
construction (TD-MPC / Dreamer: jointly trained encoder, a transition
*distribution*, a latent required only to be sufficient for value, stop-grad
targets in place of a decoder) was never consulted. Web search is available; use
it on the method, not only on the application.

**Before reporting any result, restate the goal and say which axis the result
speaks to.** A number that improves task success but touches none of the four
axes is not progress toward this goal, however large it is.

**Recorded failure, 2026-08-27** — the cost of skipping this. A full session
produced a real, replicated deployment-time gain (0.719 → 0.938, p = 0.0013) that
was **off-mission on three of four axes**: it ran on `chain2b_lr2` at 0.797 (the
*highest*-success candidate, chosen after closing `chain3` at 0.031 and `chain1b`
at 0.438), used a discriminative classifier on the current observation with no
`T_θ` anywhere, and pre-trained everything offline. The conclusion drawn from it
— "the world model does not help" — was unsupported, because no world model was
ever built. A closure decision compounded it: two low-success testbeds were
retired on the grounds that *selection* could not help them, which says nothing
about latent state prediction. **Do not retire a testbed because one method
failed on it.**

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
