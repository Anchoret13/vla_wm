#!/usr/bin/env python
"""Prove the forkable Phi is the SAME function as the episode-level original.

Action 2M.4 changes where the phase-potential memos live, not what they compute.
This asserts identity on random and structured episodes; any divergence means
the readout changed silently, which would invalidate the comparison to every
earlier Phi-based artifact.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from lcwm.phase_potential import episode_phase_potentials  # noqa: E402
from lcwm.v086_phase import PhasePotential, q_phi, discounted_delta  # noqa: E402

NAMES = ["alphabet_soup_1", "tomato_sauce_1", "basket_1"]
ATOMS = [["In", "tomato_sauce_1", "basket_1_contain_region"]]


def episode(seed: int, T: int = 60):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, 10, generator=g) * 0.2
    obj = torch.randn(T, len(NAMES), 3, generator=g) * 0.1
    bits = torch.zeros(T, len(ATOMS), dtype=torch.bool)
    if seed % 3 == 0:                     # some episodes complete partway
        bits[T // 2:] = True
    if seed % 5 == 0:                     # some regress (violation path)
        bits[T // 2: T // 2 + 3] = True
        bits[T // 2 + 3:] = False
    return q, obj, bits


def main() -> int:
    fails = []
    for seed in range(24):
        q, obj, bits = episode(seed)
        ref = episode_phase_potentials(q, obj, bits, NAMES, ATOMS)
        st = PhasePotential.at_start(q[0], obj[0], NAMES, ATOMS)
        got = [st.step(q[t], obj[t], bits[t]) for t in range(q.shape[0])]
        for t, (a, b) in enumerate(zip(ref, got)):
            if (a.phi, a.phase, a.active_atom, a.violation) != \
               (b.phi, b.phase, b.active_atom, b.violation):
                fails.append(f"seed {seed} t {t}: {a} != {b}")
                break
    print(f"1. forkable Phi == episode_phase_potentials on 24 episodes: "
          f"{'OK' if not fails else 'FAIL'}")

    # fork independence: a fork must not mutate its parent's memos
    q, obj, bits = episode(3)
    st, _ = PhasePotential.from_source_history(q[:30], obj[:30], bits[:30], NAMES, ATOMS)
    before = (list(st.ever_true), dict(st.was_lifted), dict(st.dxy_at_lift))
    child = st.fork()
    for t in range(30, 60):
        child.step(q[t], obj[t], bits[t])
    after = (list(st.ever_true), dict(st.was_lifted), dict(st.dxy_at_lift))
    if before != after:
        fails.append("fork mutated the parent's memos")
    print(f"2. fork leaves the parent's memos untouched: "
          f"{'OK' if before == after else 'FAIL'}")

    # two forks from the same parent are independent of each other
    a, b = st.fork(), st.fork()
    ra = [a.step(q[t], obj[t], bits[t]).phi for t in range(30, 60)]
    rb = [b.step(q[t], obj[t], bits[t]).phi for t in range(30, 60)]
    if ra != rb:
        fails.append("sibling forks diverged on identical input")
    print(f"3. sibling forks on identical input agree: {'OK' if ra == rb else 'FAIL'}")

    # continuing from a fork == continuing the unforked episode
    full = episode_phase_potentials(q, obj, bits, NAMES, ATOMS)
    cont = st.fork()
    tail = [cont.step(q[t], obj[t], bits[t]).phi for t in range(30, 60)]
    same = all(abs(x - full[t].phi) < 1e-12 for x, t in zip(tail, range(30, 60)))
    if not same:
        fails.append("forked continuation != unforked episode tail")
    print(f"4. forked continuation reproduces the episode tail: "
          f"{'OK' if same else 'FAIL'}  <- this is what 'not rebased at tau' means")

    # QPhi arithmetic
    phis = [0.0] + [0.1 * i for i in range(1, 91)]
    r = q_phi(phis, c=10, h=80)
    manual = sum(0.99 ** (j - 1) * (phis[j] - phis[j - 1]) for j in range(1, 11))
    manual_c = sum(0.99 ** (j - 1) * (phis[10 + j] - phis[10 + j - 1]) for j in range(1, 81))
    ok = (abs(r["GPhi_exec"] - manual) < 1e-12
          and abs(r["GPhi_cont_80"] - manual_c) < 1e-12
          and abs(r["QPhi"] - (manual + 0.99 ** 10 * manual_c)) < 1e-12)
    if not ok:
        fails.append("QPhi arithmetic")
    print(f"5. QPhi = GPhi_exec + gamma^c * GPhi_cont(80): {'OK' if ok else 'FAIL'}")

    # Phi held constant after an early terminal contributes zero
    short = [0.0, 0.5] + [0.5] * 95
    ok2 = abs(discounted_delta(short, 10, 80)) < 1e-12
    print(f"6. constant Phi after early terminal contributes 0: {'OK' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append("early-terminal handling")

    print("\nRESULT:", "PASS" if not fails else f"FAIL {fails[:3]}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
