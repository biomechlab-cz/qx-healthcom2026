"""Hardware-validate QInstance: run the per-crop quantum classifier on IBM hardware.

For each fold's val-IMAGE crops (the same crops used in bag eval), build a Qiskit
circuit with the trained QInstance quantum weights baked in, bind per-crop encoding
angles, and submit all of a fold's crops as ONE batched IBM job.

Compares per-crop sigmoid(head(alpha_hw)) vs sigmoid(head(alpha_sim)).

Output:
  results/hardware/{device}/qinst_fold_{k}.json — per-crop sim and hardware probabilities
  results/qinst_hardware_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augmentation import build_transforms
from utils.config import DATA, HARDWARE, PATHS, TRAIN
from utils.dataset import BagDataset, load_fold
from utils.models import build_resnet18_feature_extractor
from utils.quantum_aggregator import QuantumConfig
from utils.quantum_hardware import (
    bind_encoding, build_circuit_template, expectation_z0_from_counts,
)
from utils.seed import provenance, write_json
from utils.training import device

# Reuse QInstance from 07c.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_07c", Path(__file__).resolve().parent / "07c_pretrain_quantum_instance.py")
_07c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_07c)


def load_qinst(fold: int):
    ckpt = PATHS.models / f"qinst_fold_{fold}.pt"
    backbone = build_resnet18_feature_extractor(str(PATHS.models / f"resnet18_fold_{fold}.pt"))
    qcfg = QuantumConfig()
    model = _07c.QInstance(backbone, qcfg)
    model.quantum = model.quantum.cpu()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, qcfg


def collect_crops(model, fold: int, dev: torch.device) -> list[dict]:
    """For fold k's val-bag crops, compute simulator alpha + projected angles."""
    _, val_imgs = load_fold(fold)
    _, val_tf = build_transforms()
    ds = BagDataset(image_ids=val_imgs, transform=val_tf)
    out = []
    backbone = model.backbone.to(dev)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, c, bag_id = ds[i]
            x = x.to(dev)
            phi = backbone(x)
            z = model.projector(phi.cpu()) * model._scale  # projector stays on CPU
            alpha_sim = model.quantum(z).numpy()
            p_sim = torch.sigmoid(model.head(model.quantum(z).unsqueeze(-1)).squeeze(-1)).numpy()
            rows = ds.by_img[bag_id]
            for j in range(z.shape[0]):
                out.append({
                    "fold": fold,
                    "bag_id": bag_id,
                    "chrom_idx": j,
                    "true_class": int(rows[j]["class"]),
                    "z": z[j].numpy(),
                    "alpha_sim": float(alpha_sim[j]),
                    "p_sim": float(p_sim[j]),
                })
    return out


def submit_fold(fold: int, device_name: str, shots: int, model, qcfg) -> dict:
    """Submit all val-bag crops for fold k as one IBM job and reconstruct per-crop hw probs."""
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    dev = device()
    crops = collect_crops(model, fold, dev)
    if not crops:
        return {"fold": fold, "skipped": "no crops"}

    weights = model.quantum.weights.detach().numpy()
    template = build_circuit_template(weights, qcfg)

    service = QiskitRuntimeService()
    backend = service.backend(device_name)
    print(f"  fold {fold}: backend={device_name} pending={backend.status().pending_jobs}  "
          f"submitting {len(crops)} crops as one job")

    isa_list = []
    for c in crops:
        qc = bind_encoding(template, c["z"])
        isa_list.append(transpile(qc, backend=backend, optimization_level=HARDWARE.optimization_level))

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    job = sampler.run([(c,) for c in isa_list], shots=shots)
    print(f"    job_id={job.job_id()}  waiting...")
    result = job.result()

    # Extract counts -> alpha_hw -> p_hw using saved model head
    head_w = float(model.head.weight.detach()[0, 0].item())
    head_b = float(model.head.bias.detach()[0].item())

    per_crop = []
    for c, pub_res in zip(crops, result):
        counts = pub_res.data.c.get_counts() if hasattr(pub_res.data, "c") else None
        if counts is None:
            for name in dir(pub_res.data):
                obj = getattr(pub_res.data, name, None)
                if hasattr(obj, "get_counts"):
                    counts = obj.get_counts(); break
        alpha_hw = expectation_z0_from_counts(counts)
        logit_hw = head_w * alpha_hw + head_b
        p_hw = float(1.0 / (1.0 + np.exp(-logit_hw)))
        per_crop.append({
            "fold": fold,
            "bag_id": c["bag_id"],
            "chrom_idx": c["chrom_idx"],
            "true_class": c["true_class"],
            "alpha_sim": c["alpha_sim"],
            "alpha_hw": float(alpha_hw),
            "p_sim": c["p_sim"],
            "p_hw": p_hw,
        })

    out_dir = PATHS.hardware / device_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance(seed=TRAIN.seed, cfg={"qinst_fold": fold, "device": device_name}),
        "fold": fold, "device": device_name, "job_id": job.job_id(),
        "shots": shots, "n_crops": len(per_crop), "per_crop": per_crop,
    }
    write_json(out_dir / f"qinst_fold_{fold}.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default="ibm_kingston")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--submit", action="store_true",
                        help="actually submit; default = dry-run estimate only")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(DATA.n_folds))

    total_crops = 0
    payloads = []
    for k in folds:
        ckpt = PATHS.models / f"qinst_fold_{k}.pt"
        if not ckpt.exists():
            print(f"  fold {k}: ckpt missing, skip"); continue
        model, qcfg = load_qinst(k)
        crops = collect_crops(model, k, device())
        n = len(crops)
        total_crops += n
        print(f"  fold {k}: {n} crops -> {n * args.shots:,} shots")

        if args.submit:
            payloads.append(submit_fold(k, args.device, args.shots, model, qcfg))

    print(f"\nTotal: {total_crops} crops x {args.shots} shots = {total_crops * args.shots:,} shots")
    print(f"  est. QPU time on Kingston (~50k shots/s): ~{total_crops * args.shots / 50000:.1f} sec")

    if args.submit:
        write_json(PATHS.results / "qinst_hardware_summary.json",
                   {"provenance": provenance(seed=TRAIN.seed,
                                              cfg={"device": args.device, "shots": args.shots}),
                    "folds": payloads})
        # Per-crop sim/hw diff summary
        all_p_sim, all_p_hw, all_y = [], [], []
        for p in payloads:
            for c in p.get("per_crop", []):
                all_p_sim.append(c["p_sim"]); all_p_hw.append(c["p_hw"]); all_y.append(c["true_class"])
        if all_p_sim:
            diffs = np.abs(np.array(all_p_sim) - np.array(all_p_hw))
            print(f"\nsim vs hw per-crop P(DC):")
            print(f"  n={len(all_p_sim)}  mean|diff|={diffs.mean():.4f}  max|diff|={diffs.max():.4f}")
            from sklearn.metrics import roc_auc_score
            print(f"  inst AUROC sim:  {roc_auc_score(all_y, all_p_sim):.3f}")
            print(f"  inst AUROC hw:   {roc_auc_score(all_y, all_p_hw):.3f}")
    else:
        print("[dry-run only — pass --submit to push jobs to IBM Quantum]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
