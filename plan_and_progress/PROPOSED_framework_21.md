# PROPOSED — not yet merged into framework_design.md

Two amendments to §5. Each changes what we build; everything else measured this
session is implementation detail or belongs in `CLAUDE.md`.

## §21.1 — The latent must contain what actions control (amends §5.1)

§5.1 builds LCState from `PrefixVLM(o_t, ℓ)`. That representation is trained for
language-vision alignment and is **nearly action-invariant over c = 10 steps**:

| component of the state | action gain over an action-free model | shuffled action |
|---|---:|---:|
| pooled prefix hidden (2048) | +0.81% | 0.597× identity |
| **proprioception (25)** | **+11.78%** | **1.217× identity** |

A shuffled action makes the proprioception prediction *worse than not predicting at
all* — the signature of a dynamics model, which a smoother cannot produce. The
pooled hidden shows nothing of the kind.

**Amendment:** LCState includes proprioception, and the loss is weighted per
component. Under a mean-over-dims loss the 25 informative dimensions carry ~1.2% of
the gradient and the signal vanishes — measured, +11.78% alone versus +0.84% pooled.

Without this there is no action-conditioned latent to model, and §5.2 has no object.

## §21.2 — One-step scoring cannot demonstrate the transition (amends §5.2)

§5.2 predicts `z̃_{t+c} = T_θ(z_t, E_a(u^i))` and reads `D_θ` off it to rank
candidates. **At a branch point every candidate shares `z`, so `T_θ(z, u)` is a
deterministic feature map of `(z, u)` and lies inside the function class of a critic
fed `(z, E_a(u))`.** It cannot beat one at any data scale, and more data helps the
critic at least as much.

This is not a hypothesis fitted after the fact — it explains five independent nulls
that were measured before the argument was found: deployment ablation p = 0.728;
offline ranking 0.637 vs 0.636; two joint-training variants below baseline; TD
bootstrapping −0.0102 and −0.0055 with intervals spanning zero.

**Amendment:** `D_θ` on a one-step predicted latent is a critic and must not be
described as evidence about the world model. A claim that the transition contributes
requires a use where it is not replaceable — a critic cannot generate a rollout, so
**policy improvement in imagination** or **optimisation over action sequences**.

**Corollary — model selection.** Because ranking does not need dynamics, ranking
quality cannot select for it, and here the two are anti-correlated: the model with
the best ranking AUC (0.626) had essentially no dynamics (a shuffled action cost it
0.2% at h = 1), while the model that ranked worse (0.584) predicted at 0.530×
identity with value correlation 0.88–0.95. World-model quality is measured directly:
one-step error against the identity floor, a shuffled-action control, compounding
over horizon, and the value gap.

---

**Caveat to record somewhere, not an amendment:** trajectories are
deployment-collected but **labels are not** — `Δw` comes from BDDL predicates,
success from `env.is_success`, and candidate-level labels additionally required
deterministic replay, which a real robot cannot do.
