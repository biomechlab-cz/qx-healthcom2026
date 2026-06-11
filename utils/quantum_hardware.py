"""Bridge from the trained PennyLane HQ-MIL model to executable Qiskit circuits.

The PennyLane forward in `utils/quantum_aggregator.py`:
  - encoding:   RY(z_j) RZ(z_j) per qubit
  - layers (L):
      RY(theta_L,j,0), RZ(theta_L,j,1) per qubit
      ring CNOTs: CNOT(j, (j+1) mod n_qubits) for j in 0..n
  - measure <Z_0>

For hardware we build the same logical circuit in Qiskit with:
  - trained `weights` BAKED IN as numerical rotation angles
  - encoding angles z_k as `Parameter` objects so we can re-bind them per instance
  - a measurement on qubit 0, no observable rotation (Z is the default basis)
  - <Z_0> reconstructed from the bitstring counts: <Z_0> = (n_zero - n_one) / shots
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from utils.quantum_aggregator import QuantumConfig


@dataclass(frozen=True)
class CircuitTemplate:
    """A parametrized circuit + the Parameter handles needed to bind per-instance encoding."""
    circuit: QuantumCircuit
    encoding_params: list[Parameter]   # length n_qubits (z_0 .. z_{n-1})
    measured_qubit: int                # always 0 for the current architecture


def build_circuit_template(weights: np.ndarray, cfg: QuantumConfig) -> CircuitTemplate:
    """Construct a Qiskit circuit with `weights` baked in and encoding angles as parameters.

    weights: shape (n_layers, n_qubits, 2) — same layout as the PennyLane tensor.
    """
    if weights.shape != (cfg.n_layers, cfg.n_qubits, 2):
        raise ValueError(f"weights shape {weights.shape} != ({cfg.n_layers}, {cfg.n_qubits}, 2)")

    qc = QuantumCircuit(cfg.n_qubits, 1)
    enc = [Parameter(f"z_{j}") for j in range(cfg.n_qubits)]

    # Encoding.
    for j in range(cfg.n_qubits):
        qc.ry(enc[j], j)
        if cfg.encoding == "ry_rz":
            qc.rz(enc[j], j)

    # Trainable layers (weights baked in as floats).
    for L in range(cfg.n_layers):
        for j in range(cfg.n_qubits):
            qc.ry(float(weights[L, j, 0]), j)
            qc.rz(float(weights[L, j, 1]), j)
        if cfg.entanglement == "ring":
            for j in range(cfg.n_qubits):
                qc.cx(j, (j + 1) % cfg.n_qubits)
        elif cfg.entanglement == "full":
            for j in range(cfg.n_qubits):
                for k in range(cfg.n_qubits):
                    if j != k:
                        qc.cx(j, k)

    # Measure qubit 0 in Z basis.
    qc.measure(0, 0)

    return CircuitTemplate(circuit=qc, encoding_params=enc, measured_qubit=0)


def bind_encoding(template: CircuitTemplate, z: np.ndarray) -> QuantumCircuit:
    """Bind the encoding angles for one instance. `z` shape: (n_qubits,)."""
    if z.shape != (len(template.encoding_params),):
        raise ValueError(f"z shape {z.shape} != ({len(template.encoding_params)},)")
    binding = {p: float(z[i]) for i, p in enumerate(template.encoding_params)}
    return template.circuit.assign_parameters(binding, inplace=False)


def bind_many(template: CircuitTemplate, zs: np.ndarray) -> list[QuantumCircuit]:
    """Bind a batch of encodings. Returns one circuit per instance."""
    return [bind_encoding(template, z) for z in zs]


def expectation_z0_from_counts(counts: dict) -> float:
    """Compute <Z> on the single measured classical bit from a Qiskit counts dict.

    Counts keys are bitstrings of length = number of classical bits (we only have 1).
    """
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    # In Qiskit bitstring convention, key '0' or '1' corresponds to classical bit 0.
    n_zero = counts.get("0", 0)
    n_one = counts.get("1", 0)
    if n_zero + n_one == 0:
        # Some backends return multi-character keys with whitespace; normalize.
        for k, v in counts.items():
            kk = k.replace(" ", "").lstrip("0")
            if kk == "" or kk == "0":
                n_zero += v
            elif kk == "1":
                n_one += v
    return (n_zero - n_one) / total


def encode_bag(model, bag: torch.Tensor, device) -> np.ndarray:
    """Run the classical front-end of the HQ-MIL model to get encoding angles for one bag.

    Returns a numpy array of shape (K, n_qubits), already scaled to (-pi/2, pi/2).
    """
    model.eval()
    with torch.no_grad():
        x = bag.to(device)
        phi = model.backbone(x)  # (K, 512)
        z = model.projector(phi) * model._encoding_scale  # (K, n_qubits)
    return z.cpu().numpy()


def softmax_aggregate(alphas: Iterable[float], z: np.ndarray) -> np.ndarray:
    """Given per-instance attention logits and projected features, return the bag feature v (n_qubits,)."""
    a = np.asarray(list(alphas), dtype=float)
    a = a - a.max()  # numerical stability
    a = np.exp(a)
    a = a / a.sum()
    return (a[:, None] * z).sum(axis=0)
