"""V7.1 run/video contract (2026-08-02.md).

Every environment rollout invoked by a scheduled training-time
evaluation saves one indexed video regardless of outcome or arm.
Recording is OBSERVATIONAL: it consumes the authoritative rollout's own
observations and cannot touch seeds, noise, CRNs, timing, or metrics.

Layout frozen here: deterministic side-by-side external|wrist,
FPS 20, mp4 (imageio/ffmpeg, libx264 default). Writes go to
`<name>.partial.mp4` and are atomically renamed on close; a crash
leaves the labeled partial file as retained evidence, never a silent
deletion. `write_index_row` appends the registered video_index.jsonl
row with video/checkpoint/manifest SHAs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FPS = 20
LAYOUT = "side_by_side_external_wrist"


def _frame_of(obs: dict) -> np.ndarray:
    ext = np.asarray(obs["pixels"]["image"], dtype=np.uint8)
    wrist = obs["pixels"].get("image2")
    if wrist is None:
        return ext
    return np.concatenate(
        [ext, np.asarray(wrist, dtype=np.uint8)], axis=1)


class VideoRecorder:
    def __init__(self, path: str | Path):
        import imageio
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_name(
            self.path.stem + ".partial.mp4")
        self.writer = imageio.get_writer(
            str(self.partial), fps=FPS, codec="libx264",
            quality=7, macro_block_size=1)
        self.n_frames = 0
        self.closed = False
        self.error = None

    def add(self, obs: dict) -> None:
        try:
            self.writer.append_data(_frame_of(obs))
            self.n_frames += 1
        except Exception as e:      # recording must never kill a run
            self.error = f"{type(e).__name__}: {e}"

    def close(self, completed: bool = True) -> dict:
        if self.closed:
            return self.meta
        self.writer.close()
        self.closed = True
        if completed and self.error is None:
            self.partial.replace(self.path)
            final = self.path
        else:
            final = self.partial     # labeled partial evidence retained
        sha = hashlib.sha256(final.read_bytes()).hexdigest() \
            if final.exists() else None
        self.meta = {"video_path": str(final),
                     "video_sha256": sha,
                     "n_frames": self.n_frames,
                     "fps": FPS, "layout": LAYOUT,
                     "partial": final == self.partial,
                     "record_error": self.error}
        return self.meta

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(completed=exc_type is None)
        return False


def write_index_row(index_path: str | Path, video_meta: dict, *,
                    run_id: str, checkpoint_tag: str,
                    checkpoint_path: str | None,
                    checkpoint_sha256: str | None,
                    manifest_sha256: str, task: str, seed,
                    arm: str, split: str, steps: int, success: bool,
                    ordered_progress, damage, termination: str,
                    root: str | Path) -> None:
    row = {
        "video_path": str(Path(video_meta["video_path"]).relative_to(
            Path(root))),
        "video_sha256": video_meta["video_sha256"],
        "n_frames": video_meta["n_frames"],
        "fps": video_meta["fps"], "layout": video_meta["layout"],
        "partial": video_meta["partial"],
        "record_error": video_meta["record_error"],
        "run_id": run_id, "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "task": task, "seed": seed, "arm": arm, "split": split,
        "steps": steps, "success": success,
        "ordered_progress": ordered_progress, "damage": damage,
        "termination": termination,
    }
    with Path(index_path).open("a") as f:
        f.write(json.dumps(row) + "\n")


def video_name(run_id: str, checkpoint_tag: str, task: str, seed,
               arm: str) -> str:
    return f"{run_id}_{checkpoint_tag}_{task}_s{seed}_{arm}.mp4"
