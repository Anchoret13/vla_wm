#!/usr/bin/env python
"""H0 — reproducible eval_adapter assembly (replaces the one-off merge).

Constructs an lc_flow_adapter_v1 directory from a hashed WM checkpoint
bundle + trained wz adapter weights, and prints the component hashes it
consumed so the run manifest can record them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    from safetensors.torch import save_file

    from lcwm.lc_flow import LCFlowConfig, LCState, load_lc_adapter

    bundle = torch.load(args.wm, weights_only=False)
    adapter = torch.load(args.adapter, weights_only=False)
    lc = LCState()
    lc.load_state_dict(bundle["lc_state"], strict=True)
    lc.wz_hidden.load_state_dict(adapter["wz_hidden"])
    lc.wz_out.load_state_dict(adapter["wz_out"])
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in lc.state_dict().items()},
        str(args.output / "lc_flow.safetensors"),
    )
    manifest = {
        "schema": "lc_flow_adapter_v1",
        "config": dataclasses.asdict(LCFlowConfig()),
        "metadata": {
            "run": args.run_name,
            "wm_checkpoint_path": str(args.wm.resolve()),
            "wm_checkpoint_sha256": sha256(args.wm),
            "adapter_path": str(args.adapter.resolve()),
            "adapter_sha256": sha256(args.adapter),
        },
    }
    (args.output / "lc_flow_config.json").write_text(
        json.dumps(manifest, indent=2)
    )
    lc2, _ = load_lc_adapter(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eval_adapter_sha256": sha256(
                    args.output / "lc_flow.safetensors"
                ),
                "wm_sha256": manifest["metadata"]["wm_checkpoint_sha256"],
                "adapter_sha256": manifest["metadata"]["adapter_sha256"],
                "wz_out_norm": float(
                    lc2.wz_out.weight.detach().norm()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
