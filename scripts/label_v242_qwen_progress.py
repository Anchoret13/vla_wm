#!/usr/bin/env python
"""Label replayed chain3 RGB frames with a frozen, local Qwen2.5-VL.

The input contract is deliberately narrow: this program reads only ``frame_id``,
``task`` and the two JPEG entries from a ``v242_rgb_replay_v1`` JSONL manifest.
It never opens a tape, a rollout summary, BDDL, simulator state, events, or success
labels.  Output is append-only JSONL so interrupted jobs can be resumed safely.

Example (one-frame smoke test)::

    python scripts/label_v242_qwen_progress.py \
      --manifest results/v242_rgb_replay/chain3_lr2_.../manifest.jsonl \
      --output results/v242_qwen_progress/smoke/labels.jsonl \
      --device cuda --dtype bfloat16 --limit 1
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
INPUT_SCHEMA = "v242_rgb_replay_v1"
OUTPUT_SCHEMA = "v242_qwen_label_v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CAMERAS = ("agentview", "eye_in_hand")
OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")
ALLOWED_LABELS = frozenset(("inside", "outside", "uncertain"))

# This is intentionally a constant rather than a CLI option.  Changing the
# question after looking at outputs would make the pseudo-label source movable.
PROMPT = """You inspect two synchronized images from one robot timestep.
Image 1 is the external agentview camera. Image 2 is the wrist-mounted eye-in-hand camera.

For each named grocery object, decide whether it is inside the basket at this timestep. Use both camera views and only visible image evidence. The fixed appearance cues are:
- alphabet_soup_1: a cylindrical metal can with a colorful label, including yellow/orange label regions.
- tomato_sauce_1: a cylindrical metal can with a red-and-green tomato-themed label.
- cream_cheese_1: a small blue-and-white rectangular box.
The object may be rotated, partially occluded, or show only its metal can top; do not require the full label to be visible.

In Image 1, the basket is the white woven container on the left side. Use occlusion geometry carefully: when a target's visible top or upper body rises from the basket opening and its lower body is hidden behind the basket's front woven wall, that is direct evidence that the target is inside. A target outside on the tabletop normally has its body and bottom visible without the basket wall crossing in front of it. Do not call a target outside merely because its top remains visible above the basket rim. If Image 1 provides this clear basket-wall occlusion, use it even when Image 2 is cropped or ambiguous. Inspect these cues silently before returning the JSON.

Allowed labels:
- \"inside\": the object is supported by the basket bottom or lies within the basket interior. It still counts as inside when the woven basket wall hides its lower part and only its top or upper part is visible.
- \"outside\": the object is on the table or held by the robot gripper, rather than supported within the basket.
- \"uncertain\": the two views do not provide enough visual evidence to decide between inside and outside.

Return exactly one JSON object with exactly these three keys and string values:
{"alphabet_soup_1":"inside|outside|uncertain","tomato_sauce_1":"inside|outside|uncertain","cream_cheese_1":"inside|outside|uncertain"}
Do not return Markdown, code fences, commentary, probabilities, or any additional keys."""


@dataclass(frozen=True)
class Frame:
    frame_id: str
    task: str
    image_paths: Mapping[str, Path]
    expected_hashes: Mapping[str, str]
    expected_sizes: Mapping[str, tuple[int, int] | None]


class InputIntegrityError(ValueError):
    """A replay JPEG does not match the immutable manifest contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(raw)


def _git(args: Sequence[str]) -> str | None:
    p = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    )
    out = p.stdout.strip()
    return out if p.returncode == 0 and out else None


def git_provenance() -> dict[str, Any]:
    status = _git(["status", "--porcelain", "--", str(Path(__file__).resolve())])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "script_dirty": bool(status),
    }


def resolve_checkpoint(spec: str, revision: str | None) -> Path:
    candidate = Path(spec).expanduser()
    if candidate.exists():
        return candidate.resolve()
    # snapshot_download performs no network request when local_files_only=True.
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec, revision=revision, local_files_only=True
        )
    ).resolve()


def _existing_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted(set(names)):
        p = root / name
        if p.is_file():
            out[name] = sha256_file(p.resolve())
    return out


def checkpoint_provenance(
    root: Path,
    checkpoint_spec: str,
    revision: str | None,
    *,
    device: str,
    dtype: str,
    attention: str,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    """Validate a complete local checkpoint and content-hash its artifacts."""
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint lacks config.json: {root}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError(
            f"expected model_type=qwen2_5_vl, got {config.get('model_type')!r}"
        )

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_names = sorted(set(index.get("weight_map", {}).values()))
        if not weight_names:
            raise ValueError(f"empty weight_map in {index_path}")
    else:
        weight_names = sorted(p.name for p in root.glob("*.safetensors"))
    if not weight_names:
        raise FileNotFoundError(f"checkpoint has no safetensors weights: {root}")
    missing = [name for name in weight_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete checkpoint; missing shards: {missing}")
    empty = [name for name in weight_names if (root / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"incomplete checkpoint; empty shards: {empty}")

    config_names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors.index.json",
    )
    tokenizer_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "chat_template.jinja",
    )
    config_hashes = _existing_hashes(root, config_names)
    tokenizer_hashes = _existing_hashes(root, tokenizer_names)
    weight_hashes = _existing_hashes(root, weight_names)
    if len(weight_hashes) != len(weight_names):
        raise RuntimeError("failed to hash every checkpoint shard")

    commit = None
    parts = root.parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            commit = parts[i + 1]

    try:
        import huggingface_hub
        import torch
        import transformers

        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        }
    except Exception:
        versions = {}

    script_hash = sha256_file(Path(__file__).resolve())
    invariant = {
        "checkpoint_commit": commit,
        "config_files_sha256": config_hashes,
        "tokenizer_files_sha256": tokenizer_hashes,
        "weight_files_sha256": weight_hashes,
        "architecture": config.get("architectures"),
        "prompt_sha256": sha256_bytes(PROMPT.encode("utf-8")),
        "script_sha256": script_hash,
        "device": device,
        "dtype": dtype,
        "attention_implementation": attention,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
        },
        "processor": {"tokenizer_padding_side": "left"},
        "versions": versions,
    }
    aggregate = canonical_sha256(invariant)
    return {
        "checkpoint_spec": checkpoint_spec,
        "resolved_checkpoint": str(root),
        "revision_requested": revision,
        "checkpoint_commit": commit,
        "local_files_only": True,
        "architecture": config.get("architectures"),
        "config_files_sha256": config_hashes,
        "tokenizer_files_sha256": tokenizer_hashes,
        "weight_files_sha256": weight_hashes,
        "checkpoint_and_runtime_sha256": aggregate,
        "prompt_sha256": invariant["prompt_sha256"],
        "script_sha256": script_hash,
        "git": git_provenance(),
        "device": device,
        "dtype": dtype,
        "attention_implementation": attention,
        "generation": invariant["generation"],
        "processor": invariant["processor"],
        "batch_size": batch_size,
        "versions": versions,
        "argv": list(sys.argv),
    }


def _resolve_manifest_path(manifest: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = [(manifest.parent / p).resolve(), (REPO / p).resolve()]
    existing = [q for q in candidates if q.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"image path {raw!r} exists neither relative to the manifest nor repo"
        )
    # If both spellings resolve to different files, accepting one silently would
    # make image provenance depend on the current directory layout.
    unique = {str(q) for q in existing}
    if len(unique) != 1:
        raise ValueError(f"ambiguous relative image path {raw!r}: {existing}")
    return existing[0]


def _forbidden_semantic_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in {
        "success",
        "successes",
        "is_success",
        "event",
        "events",
        "bddl",
        "stage",
        "stage_reached",
        "milestone",
        "milestones",
    }


def _assert_no_semantic_labels(value: Any, location: str = "row") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _forbidden_semantic_key(str(key)):
                raise ValueError(
                    f"forbidden semantic field {location}.{key}; this labeler "
                    "accepts image-only replay manifests"
                )
            _assert_no_semantic_labels(child, f"{location}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _assert_no_semantic_labels(child, f"{location}[{i}]")


def iter_frames(manifest: Path, expected_task: str | None) -> Iterator[Frame]:
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {manifest}:{lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {lineno} is not an object")
            if row.get("schema") != INPUT_SCHEMA:
                raise ValueError(
                    f"row {lineno}: expected schema={INPUT_SCHEMA!r}, "
                    f"got {row.get('schema')!r}"
                )
            _assert_no_semantic_labels(row, f"row[{lineno}]")

            frame_id = row.get("frame_id")
            task = row.get("task")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError(f"row {lineno}: invalid frame_id")
            if frame_id in seen:
                raise ValueError(f"row {lineno}: duplicate frame_id {frame_id!r}")
            seen.add(frame_id)
            if not isinstance(task, str) or not task:
                raise ValueError(f"row {lineno}: invalid task")
            if expected_task is not None and task != expected_task:
                raise ValueError(
                    f"row {lineno}: task {task!r} != expected {expected_task!r}"
                )

            images = row.get("images")
            if not isinstance(images, dict) or set(images) != set(CAMERAS):
                raise ValueError(
                    f"row {lineno}: images must have exactly {list(CAMERAS)}"
                )
            paths: dict[str, Path] = {}
            hashes: dict[str, str] = {}
            sizes: dict[str, tuple[int, int] | None] = {}
            for camera in CAMERAS:
                entry = images[camera]
                if not isinstance(entry, dict):
                    raise ValueError(f"row {lineno}: images.{camera} is not an object")
                raw_path = entry.get("path")
                expected_hash = entry.get("sha256")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError(f"row {lineno}: missing images.{camera}.path")
                if (
                    not isinstance(expected_hash, str)
                    or len(expected_hash) != 64
                    or any(c not in "0123456789abcdef" for c in expected_hash.lower())
                ):
                    raise ValueError(f"row {lineno}: invalid images.{camera}.sha256")
                path = _resolve_manifest_path(manifest, raw_path)
                if path.suffix.lower() not in (".jpg", ".jpeg"):
                    raise ValueError(f"row {lineno}: expected JPEG, got {path}")
                paths[camera] = path
                hashes[camera] = expected_hash.lower()
                width, height = entry.get("width"), entry.get("height")
                if width is None and height is None:
                    sizes[camera] = None
                elif isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                    sizes[camera] = (width, height)
                else:
                    raise ValueError(f"row {lineno}: invalid size for {camera}")
            yield Frame(frame_id, task, paths, hashes, sizes)


def load_images(frame: Frame):
    from PIL import Image

    images = []
    actual_hashes: dict[str, str] = {}
    for camera in CAMERAS:
        path = frame.image_paths[camera]
        actual = sha256_file(path)
        expected = frame.expected_hashes[camera]
        if actual != expected:
            raise InputIntegrityError(
                f"{frame.frame_id} {camera} hash mismatch: {actual} != {expected}"
            )
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise InputIntegrityError(
                        f"{frame.frame_id} {camera} is not a JPEG: {path}"
                    )
                expected_size = frame.expected_sizes[camera]
                if expected_size is not None and image.size != expected_size:
                    raise InputIntegrityError(
                        f"{frame.frame_id} {camera} size {image.size} != {expected_size}"
                    )
                images.append(image.convert("RGB").copy())
        except InputIntegrityError:
            raise
        except Exception as exc:
            raise InputIntegrityError(
                f"{frame.frame_id} {camera} cannot be decoded as JPEG: {exc}"
            ) from exc
        actual_hashes[camera] = actual
    return images, actual_hashes


def parse_response(raw: str) -> tuple[dict[str, str] | None, str | None]:
    """Strict JSON parse: no fence stripping, substring extraction, or coercion."""
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    if not isinstance(value, dict):
        return None, f"expected JSON object, got {type(value).__name__}"
    if set(value) != set(OBJECTS):
        return None, f"keys must be exactly {list(OBJECTS)}, got {sorted(value)}"
    for obj in OBJECTS:
        if not isinstance(value[obj], str) or value[obj] not in ALLOWED_LABELS:
            return None, f"{obj} must be one of {sorted(ALLOWED_LABELS)}, got {value[obj]!r}"
    return {obj: value[obj] for obj in OBJECTS}, None


def _fallback_objects() -> dict[str, dict[str, str]]:
    # ``parse_ok=false`` makes clear that these are not semantic uncertain
    # judgments; the allowed-value fallback merely keeps the row schema total.
    return {obj: {"label": "uncertain"} for obj in OBJECTS}


def make_record(
    *,
    frame: Frame,
    manifest_hash: str,
    image_hashes: Mapping[str, str],
    raw: str | None,
    labels: Mapping[str, str] | None,
    error: str | None,
    provenance: Mapping[str, Any],
    token_scores: Mapping[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    objects = (
        {obj: {"label": labels[obj]} for obj in OBJECTS}
        if labels is not None
        else _fallback_objects()
    )
    record: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "frame_id": frame.frame_id,
        "replay_manifest_sha256": manifest_hash,
        "task": frame.task,
        "image_sha256s": dict(image_hashes),
        "objects": objects,
        "raw_response": raw,
        "parse_ok": labels is not None and error is None,
        "parse_error": error,
        "prompt_sha256": sha256_bytes(PROMPT.encode("utf-8")),
        "model_provenance": dict(provenance),
        "inference": {
            "elapsed_seconds": elapsed_seconds,
            "generation_token_scores": token_scores,
        },
    }
    return record


def load_existing(
    file_obj,
    *,
    manifest_hash: str,
    prompt_hash: str,
    model_fingerprint: str,
) -> tuple[set[str], int]:
    completed: set[str] = set()
    failures = 0
    file_obj.seek(0)
    for lineno, line in enumerate(file_obj, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt existing output at line {lineno}; refusing unsafe resume: {exc}"
            ) from exc
        if row.get("schema") != OUTPUT_SCHEMA:
            raise ValueError(f"existing output line {lineno} has wrong schema")
        if row.get("replay_manifest_sha256") != manifest_hash:
            raise ValueError("existing output was produced from a different replay manifest")
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError("existing output used a different fixed prompt")
        prior_fp = row.get("model_provenance", {}).get(
            "checkpoint_and_runtime_sha256"
        )
        if prior_fp != model_fingerprint:
            raise ValueError(
                "existing output used a different model/runtime/code fingerprint; "
                "resume into a new output file"
            )
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or frame_id in completed:
            raise ValueError(f"invalid or duplicate frame_id at output line {lineno}")
        completed.add(frame_id)
        failures += int(not row.get("parse_ok", False))
    file_obj.seek(0, os.SEEK_END)
    return completed, failures


def choose_runtime(device_arg: str, dtype_arg: str) -> tuple[str, str]:
    import torch

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    if dtype_arg == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    else:
        dtype = dtype_arg
    if device == "cpu" and dtype == "float16":
        raise ValueError("float16 CPU inference is unsupported; use float32 or bfloat16")
    return device, dtype


def torch_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def prepare_batch(processor, frames: Sequence[Frame]):
    texts: list[str] = []
    flat_images = []
    hashes: list[Mapping[str, str]] = []
    for frame in frames:
        images, image_hashes = load_images(frame)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        )
        flat_images.extend(images)
        hashes.append(image_hashes)
    inputs = processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )
    return inputs, hashes


def move_inputs(inputs, device: str, dtype_name: str):
    dtype = torch_dtype(dtype_name)
    for key, value in list(inputs.items()):
        if value.is_floating_point():
            inputs[key] = value.to(device=device, dtype=dtype)
        else:
            inputs[key] = value.to(device=device)
    return inputs


def token_score_records(model, processor, generated, input_length: int):
    import torch

    if not generated.scores:
        return [None] * generated.sequences.shape[0]
    scores = model.compute_transition_scores(
        generated.sequences, generated.scores, normalize_logits=True
    )
    new_ids = generated.sequences[:, input_length:]
    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    records = []
    for ids, logps in zip(new_ids, scores):
        items = []
        for token_id, logp in zip(ids.tolist(), logps.detach().float().cpu().tolist()):
            if token_id == pad:
                break
            items.append(
                {
                    "token_id": int(token_id),
                    "token": processor.tokenizer.convert_ids_to_tokens(int(token_id)),
                    "log_probability": float(logp),
                    "probability": float(torch.exp(torch.tensor(logp)).item()),
                }
            )
            if token_id == eos:
                break
        records.append(
            {
                "kind": "greedy_generated_token_probability_not_calibrated_label_confidence",
                "tokens": items,
            }
        )
    return records


def infer_batch(
    model,
    processor,
    inputs,
    *,
    max_new_tokens: int,
    include_token_logprobs: bool,
):
    import torch

    input_length = inputs["input_ids"].shape[1]
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": include_token_logprobs,
        "output_scores": include_token_logprobs,
    }
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    sequences = generated.sequences if include_token_logprobs else generated
    new_ids = sequences[:, input_length:]
    texts = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    scores = (
        token_score_records(model, processor, generated, input_length)
        if include_token_logprobs
        else [None] * len(texts)
    )
    return texts, scores


def chunks(values: Sequence[Frame], size: int) -> Iterator[Sequence[Frame]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--expected-task", default="chain3_lr2")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    ap.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="sdpa")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of not-yet-labeled frames (use 1 for smoke)",
    )
    ap.add_argument(
        "--include-token-logprobs",
        action="store_true",
        help="store exact greedy-token probabilities; these are not calibrated label confidence",
    )
    ap.add_argument(
        "--processor-dry-run",
        action="store_true",
        help="validate/load one batch and print tensor shapes without loading the model or writing labels",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_hash = sha256_file(manifest)
    prompt_hash = sha256_bytes(PROMPT.encode("utf-8"))
    device, dtype_name = choose_runtime(args.device, args.dtype)
    checkpoint = resolve_checkpoint(args.model, args.revision)
    print(f"hashing and validating local checkpoint {checkpoint}", file=sys.stderr)
    provenance = checkpoint_provenance(
        checkpoint,
        args.model,
        args.revision,
        device=device,
        dtype=dtype_name,
        attention=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # Parse and validate the whole manifest before loading 7B parameters.  This
    # also rejects any semantic-label fields rather than accidentally consuming
    # privileged rollout annotations.
    frames = list(iter_frames(manifest, args.expected_task or None))

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if args.processor_dry_run:
        selected = frames[: min(args.batch_size, args.limit or args.batch_size)]
        if not selected:
            raise ValueError("manifest has no frames")
        inputs, hashes = prepare_batch(processor, selected)
        report = {
            "schema": "v242_qwen_processor_dry_run_v1",
            "frames": [frame.frame_id for frame in selected],
            "image_sha256s": hashes,
            "tensor_shapes": {key: list(value.shape) for key, value in inputs.items()},
            "tensor_dtypes": {key: str(value.dtype) for key, value in inputs.items()},
            "manifest_sha256": manifest_hash,
            "prompt_sha256": prompt_hash,
            "model_provenance": provenance,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # The output file itself is the lock, preventing two appenders from claiming
    # the same frame during a resumed batch job.
    with output.open("a+", encoding="utf-8") as out:
        fcntl.flock(out.fileno(), fcntl.LOCK_EX)
        completed, existing_failures = load_existing(
            out,
            manifest_hash=manifest_hash,
            prompt_hash=prompt_hash,
            model_fingerprint=provenance["checkpoint_and_runtime_sha256"],
        )
        pending = [frame for frame in frames if frame.frame_id not in completed]
        if args.limit is not None:
            pending = pending[: args.limit]
        print(
            f"manifest={len(frames)} completed={len(completed)} "
            f"pending_this_run={len(pending)} existing_failures={existing_failures}",
            file=sys.stderr,
        )
        if not pending:
            return 2 if existing_failures else 0

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration

        dtype = torch_dtype(dtype_name)
        print(
            f"loading frozen model on {device} as {dtype_name} "
            f"(local_files_only=True)",
            file=sys.stderr,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=args.attention_implementation,
        )
        model.requires_grad_(False)
        model.eval()
        model.to(device)

        new_failures = 0
        done = 0
        for batch in chunks(pending, args.batch_size):
            started = time.monotonic()
            try:
                inputs, image_hashes = prepare_batch(processor, batch)
                inputs = move_inputs(inputs, device, dtype_name)
                raw_texts, token_scores = infer_batch(
                    model,
                    processor,
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    include_token_logprobs=args.include_token_logprobs,
                )
                elapsed = time.monotonic() - started
                per_frame = elapsed / len(batch)
                rows = []
                for frame, hashes, raw, scores in zip(
                    batch, image_hashes, raw_texts, token_scores
                ):
                    labels, error = parse_response(raw)
                    if error is not None:
                        new_failures += 1
                    rows.append(
                        make_record(
                            frame=frame,
                            manifest_hash=manifest_hash,
                            image_hashes=hashes,
                            raw=raw,
                            labels=labels,
                            error=error,
                            provenance=provenance,
                            token_scores=scores,
                            elapsed_seconds=per_frame,
                        )
                    )
            except Exception as exc:
                # Integrity failures (bad/mismatched JPEGs) should stop rather
                # than be converted into labels.  Runtime inference failures are
                # recorded explicitly for every affected frame before exit.
                if isinstance(exc, (FileNotFoundError, InputIntegrityError)):
                    raise
                elapsed = time.monotonic() - started
                rows = []
                for frame in batch:
                    error = f"{type(exc).__name__}: {exc}"
                    rows.append(
                        make_record(
                            frame=frame,
                            manifest_hash=manifest_hash,
                            image_hashes=frame.expected_hashes,
                            raw=None,
                            labels=None,
                            error=error,
                            provenance=provenance,
                            token_scores=None,
                            elapsed_seconds=elapsed / len(batch),
                        )
                    )
                    new_failures += 1

            for row in rows:
                out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                done += 1
            print(
                f"labeled {done}/{len(pending)}; parse_or_inference_failures={new_failures}",
                file=sys.stderr,
                flush=True,
            )

    return 2 if (existing_failures + new_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
