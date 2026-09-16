#!/usr/bin/env bash
# Cross-initialisation replication of the 2026-09-13 chain3 round.
#
#   bash scripts/run_v253_replication.sh
#
# WHY THIS SCRIPT EXISTS.  The 2026-09-12 audit found that no runner existed in the
# repository, so an operator typed the commands from a docstring - and the docstring
# examples disagreed with the pre-registration on four flags, every one of which
# yields a number rather than an error.  The round is driven from here instead.
#
# WHAT IT RUNS.  Seven arms at actor restart 1 on the panel restart 0 already used
# (9100-9387), so the environment is held fixed and the initialisation is the only
# thing that moves, plus one additional untrained arm that repairs the magnitude
# confound the 09-13 run-level check fired on.
#
# PRE-REGISTRATION, sealed before any of this ran:
#   results/v251_round/chain3_lr2_2026-09-16T210743Z_dryrun/preregistration.json
# PREPARATION, with the magnitude predictor calibrated against restart 0:
#   results/v253_replication/chain3_lr2_restart1_*/prepare.json
#
# COST: 8 arms x 288 episodes x 750 steps = 1,728,000 environment steps, ~12 h.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/home/stargazer/miniconda3/envs/vf0s/bin/python

# ---- the gate that costs nothing: refuse to start on a broken driver ----------
if ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "ABORT: torch.cuda.is_available() is False."
  echo "  2026-09-16: kernel module 580.173.02 vs userspace 580.178.04 (error 804)."
  echo "  A driver upgrade landed without a reboot. Reboot, then rerun this script."
  exit 1
fi

M1=results/v251_interface/chain3_lr2_m1_2026-09-12T235743Z/actor_1.pt
M0=results/v251_interface/chain3_lr2_m0_2026-09-12T235751Z/actor_1.pt
MC=results/v251_interface/chain3_lr2_m0cont_2026-09-12T235800Z/actor_1.pt
QQ=results/v251_interface/chain3_lr2_q_2026-09-12T235808Z/actor_1.pt
MS=results/v251_interface/chain3_lr2_m1shuf_2026-09-12T235817Z/actor_1.pt
RB=$(ls -d results/v252_rand_actor/chain3_lr2_randbound_r1_*)/actor_0.pt
RM=$(ls -d results/v252_rand_actor/chain3_lr2_randmag_r1_*)/actor_0.pt

for f in "$M1" "$M0" "$MC" "$QQ" "$MS" "$RB" "$RM"; do
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done

# ---- 1. the registered seven-arm round at restart 1 ---------------------------
# rand here is the BOUND-matched control, identical in construction to restart 0's,
# so the five primary contrasts replicate the 09-13 design exactly.
"$PY" scripts/eval_v251_round.py \
  --task chain3_lr2 --panel 288 --panel-start 9100 --residual-start-chunk 26 \
  --m1 "$M1" --m0 "$M0" --m0-cont "$MC" --q "$QQ" --m1-shuf "$MS" \
  --rand "$RB" --base-actor "$M1" \
  --acknowledge-panel-reuse --execute

# ---- 2. the declared additional arm: the magnitude-matched untrained control ---
# Its contrast (M1 vs rand_mag) is the repair of the flag the 09-13 check raised on
# rand (+38.3% realised magnitude, outside the registered 25% band). It is NOT a
# member of the five-contrast primary family; it is reported on its own.
"$PY" scripts/run_v206_belief_residual_deploy.py \
  --actor "$RM" --task chain3_lr2 --panel 288 --panel-start 9100 \
  --eval-labels --fixed-horizon --residual-start-chunk 26 \
  --tag v253_r1_rand_mag

# ---- 3. the analysis the 09-13 run never reached (it died in EGL teardown) -----
"$PY" scripts/report_v251_secondary.py || \
  echo "NOTE: secondary report failed; the per-arm outcome.json are still on disk"
"$PY" scripts/report_v251_p_trajectory.py || \
  echo "NOTE: p-trajectory failed; the per-arm outcome.json are still on disk"

echo
echo "restart 1 complete. Before reading anything, restate which axis each number"
echo "speaks to, and report the restart-0 and restart-1 tables SEPARATELY as well"
echo "as pooled - cross-seed replication is not cross-init replication (rule 9)."
