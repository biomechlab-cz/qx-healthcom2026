"""Recover a hardware job result by job_id (when the live script crashed post-submission).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms  # noqa: E402
from utils.config import PATHS, TRAIN  # noqa: E402
from utils.dataset import BagDataset, load_fold  # noqa: E402
from utils.quantum_hardware import (  # noqa: E402
    encode_bag, expectation_z0_from_counts, softmax_aggregate,
)
from utils.seed import provenance, write_json  # noqa: E402

# Reuse load_trained_model from script 09.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_hw09", Path(__file__).resolve().parent / "09_evaluate_hardware.py")
_hw09 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_hw09)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--bag-id", required=True, help="image_id of the recovered bag, e.g. web/14")
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    job = service.job(args.job_id)
    print(f"Job {args.job_id}: status={job.status()}  backend={job.backend().name}")
    result = job.result()

    alphas = []
    raw_counts = []
    for pub_res in result:
        counts = None
        for name in dir(pub_res.data):
            obj = getattr(pub_res.data, name, None)
            if hasattr(obj, "get_counts"):
                counts = obj.get_counts()
                break
        if counts is None:
            raise RuntimeError("could not extract counts from result")
        alphas.append(expectation_z0_from_counts(counts))
        raw_counts.append(counts)

    print(f"Recovered {len(alphas)} per-instance <Z_0> values from hardware")
    print(f"  alphas (first 5): {alphas[:5]}")
    print(f"  alphas range: [{min(alphas):.3f}, {max(alphas):.3f}]")

    # Reload model + encode the bag to get z values (deterministic).
    model, qcfg = _hw09.load_trained_model(args.fold)
    model = model.cpu()
    _, val_imgs = load_fold(args.fold)
    if args.bag_id not in val_imgs:
        raise ValueError(f"{args.bag_id} not in fold {args.fold} val set")
    _, val_tf = build_transforms()
    bag_ds = BagDataset(image_ids=[args.bag_id], transform=val_tf)
    x, y, c, _ = bag_ds[0]
    z = encode_bag(model, x, torch.device("cpu"))

    if z.shape[0] != len(alphas):
        raise ValueError(f"K mismatch: encoded {z.shape[0]} but job has {len(alphas)} circuits")

    # Compare to simulator alphas (recompute with PennyLane).
    sim_alpha = model.quantum_attention(torch.tensor(z, dtype=torch.float32)).detach().numpy()
    diffs = np.abs(np.array(alphas) - sim_alpha)
    print(f"\nSim vs hardware <Z_0> per instance:")
    print(f"  max |diff|:  {diffs.max():.4f}")
    print(f"  mean |diff|: {diffs.mean():.4f}")

    v = softmax_aggregate(alphas, z)
    v_sim = softmax_aggregate(sim_alpha, z)

    with torch.no_grad():
        v_t = torch.tensor(v, dtype=torch.float32)
        bag_logits = model.bag_head(v_t)
        bag_p_hw = float(torch.softmax(bag_logits, dim=0)[1].item())
        count_pred_hw = float(model.count_head(v_t).item())

        v_t_sim = torch.tensor(v_sim, dtype=torch.float32)
        bag_logits_sim = model.bag_head(v_t_sim)
        bag_p_sim = float(torch.softmax(bag_logits_sim, dim=0)[1].item())

    print(f"\nbag prediction:")
    print(f"  true_label    = {int(y.item())} (count={int(c.item())})")
    print(f"  sim P(DC)     = {bag_p_sim:.4f}")
    print(f"  hardware P(DC)= {bag_p_hw:.4f}")
    print(f"  hardware count pred = {count_pred_hw:.2f}")

    out_dir = PATHS.hardware / args.device / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bag_id.replace('/', '_')}.json"
    write_json(out_path, {
        "provenance": provenance(seed=TRAIN.seed, cfg={"recovered_from_job": args.job_id}),
        "bag_id": args.bag_id,
        "K": int(z.shape[0]),
        "true_label": int(y.item()),
        "true_count": int(c.item()),
        "alphas_hw": alphas,
        "alphas_sim": sim_alpha.tolist(),
        "bag_prob_dc_hw": bag_p_hw,
        "bag_prob_dc_sim": bag_p_sim,
        "count_pred_hw": count_pred_hw,
        "job_id": args.job_id,
        "max_abs_diff_alpha": float(diffs.max()),
        "mean_abs_diff_alpha": float(diffs.mean()),
        "counts": raw_counts,
    })
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
