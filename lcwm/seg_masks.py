"""Patch-grid object labels via camera projection (framework §3 selectivity).

robosuite's native camera_segmentations observable fails LIBERO's construction-time
sensor validation, so instead we project object body positions through the camera
matrix onto the SigLIP 16x16 patch grid. Coarse (center + radius disks) but
sufficient at patch granularity (1 patch = 22.5 px of a 360 px frame).

Accounts for the LiberoEnv image flip ([::-1, ::-1]) so patch coordinates match
the frames the policy (and our taps) actually see.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix

from lcwm.snapshot import get_sim

GRID = 16


def project_points(env, points_w: np.ndarray, camera: str = "agentview",
                   h: int = 360, w: int = 360) -> np.ndarray:
    """World points (n,3) -> (row, col) pixel coords in RAW render orientation.

    Empirically verified (scripts/orientation_check.py, 2026-07-20): the token
    grid lives in RAW orientation — LiberoEnv flips the obs 180° "for
    visualization" and LiberoProcessorStep flips it BACK before SigLIP. So
    projections must NOT be flip-compensated; obs frames must be rotated 180°
    when displayed under token-space maps.
    """
    sim = get_sim(env)
    mat = get_camera_transform_matrix(sim, camera, h, w)  # (4,4) world->pixel
    pts = np.concatenate([points_w, np.ones((len(points_w), 1))], axis=1)
    pix = (mat @ pts.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    col, row = pix[:, 0], pix[:, 1]
    return np.stack([row, col], axis=1)  # raw (row, col) — token space


def patch_disk_labels(
    env,
    object_positions: dict[str, np.ndarray],
    radii_px: float | dict[str, float] = 40.0,
    camera: str = "agentview",
    h: int = 360,
    w: int = 360,
) -> dict[str, np.ndarray]:
    """Per-object boolean (16,16) patch masks from projected center + pixel radius."""
    names = list(object_positions)
    centers = project_points(
        env, np.stack([object_positions[n] for n in names]), camera, h, w)
    patch = h / GRID
    rr, cc = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    patch_centers = np.stack([(rr + 0.5) * patch, (cc + 0.5) * patch], axis=-1)
    masks = {}
    for i, n in enumerate(names):
        r = radii_px[n] if isinstance(radii_px, dict) else radii_px
        d = np.linalg.norm(patch_centers - centers[i], axis=-1)
        masks[n] = d <= (r + patch / 2)
    return masks


def group_means(shift_map: np.ndarray, masks: dict[str, np.ndarray],
                groups: dict[str, list[str]]) -> dict[str, float]:
    """Mean of a (16,16) shift map over named groups of object masks + 'rest'."""
    out, used = {}, np.zeros_like(shift_map, dtype=bool)
    for gname, members in groups.items():
        m = np.zeros_like(used)
        for obj in members:
            if obj in masks:
                m |= masks[obj]
        used |= m
        out[gname] = float(shift_map[m].mean()) if m.any() else float("nan")
    out["rest"] = float(shift_map[~used].mean()) if (~used).any() else float("nan")
    return out
