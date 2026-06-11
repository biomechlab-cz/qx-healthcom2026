"""Quantum attention aggregator for HQ-MIL.

8-qubit, 3-layer variational circuit. Per-instance forward:
  - angle encoding: R_y(z_j) R_z(z_j) on each qubit j
  - L trainable layers: R_y(theta), R_z(theta) per qubit + ring-CNOT entanglement
  - measurement: <Z_0> -> attention logit alpha_k in [-1, 1]

The TorchLayer wraps the QNode so it integrates with autograd. We use
`diff_method="adjoint"` on lightning.qubit (analytic gradient, no parameter-shift cost) —
it's exact and much faster than parameter-shift for simulator training. Hardware execution
will fall back to parameter-shift.

Total trainable quantum parameters at the default config (8 qubits, 3 layers, ry+rz per
qubit per layer):
    3 layers x 8 qubits x 2 rotations = 48
Plus an optional final RY layer (8 params) for a total of 56. Variant configurations
adjust this — see `make_quantum_aggregator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

EncodingMode = Literal["ry", "ry_rz"]
EntangleMode = Literal["ring", "full"]


@dataclass(frozen=True)
class QuantumConfig:
    n_qubits: int = 8
    n_layers: int = 3
    encoding: EncodingMode = "ry_rz"
    entanglement: EntangleMode = "ring"
    diff_method: str = "adjoint"  # "adjoint" (sim) | "parameter-shift" (hardware)

    @property
    def n_trainable(self) -> int:
        return self.n_layers * 2 * self.n_qubits  # ry + rz per qubit per layer


def _make_qnode(cfg: QuantumConfig):
    dev = qml.device("lightning.qubit", wires=cfg.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=cfg.diff_method)
    def circuit(inputs, weights):
        # `inputs` shape: (n_qubits,) — projected features, scaled to (-pi/2, pi/2)
        # `weights` shape: (n_layers, n_qubits, 2) — Ry, Rz per qubit per layer
        for j in range(cfg.n_qubits):
            qml.RY(inputs[j], wires=j)
            if cfg.encoding == "ry_rz":
                qml.RZ(inputs[j], wires=j)

        for L in range(cfg.n_layers):
            for j in range(cfg.n_qubits):
                qml.RY(weights[L, j, 0], wires=j)
                qml.RZ(weights[L, j, 1], wires=j)
            if cfg.entanglement == "ring":
                for j in range(cfg.n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % cfg.n_qubits])
            elif cfg.entanglement == "full":
                for j in range(cfg.n_qubits):
                    for k in range(cfg.n_qubits):
                        if j != k:
                            qml.CNOT(wires=[j, k])

        return qml.expval(qml.PauliZ(0))

    return circuit


class QuantumAttention(nn.Module):
    """Quantum attention logits: maps a batch (K, n_qubits) of encoded features to (K,) logits.

    We loop per-instance through the QNode rather than relying on TorchLayer's batch
    broadcasting, which mis-reshapes when the weight tensor is multi-dimensional. The
    loop is O(K) circuit evaluations per forward — about 1-2ms per evaluation on
    lightning.qubit for 8 qubits, so a K=46 bag costs ~50-100ms forward + backward.
    """

    def __init__(self, cfg: QuantumConfig | None = None):
        super().__init__()
        self.cfg = cfg or QuantumConfig()
        self.qnode = _make_qnode(self.cfg)

        # Trainable weights as a plain nn.Parameter — gives us full control.
        init = 0.1 * (torch.rand(self.cfg.n_layers, self.cfg.n_qubits, 2) * 2 - 1)
        self.weights = nn.Parameter(init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (K, n_qubits). QNode runs on CPU.
        was_cuda = z.is_cuda
        z_cpu = z.detach().cpu().float() if z.is_cuda else z.float()
        # We need gradients to flow into z (the projector output) AND into self.weights.
        # `inputs` must be a torch tensor on CPU; QNode returns a python scalar wrapped as tensor.
        # Re-attach to CPU side of the graph if it was CUDA.
        if was_cuda:
            # Keep gradients flowing: route through cpu, then move output back to cuda.
            z_in = z.cpu()
        else:
            z_in = z

        alphas = []
        for k in range(z_in.shape[0]):
            a = self.qnode(z_in[k], self.weights)
            # a is a 0-dim torch tensor on CPU.
            alphas.append(a)
        alpha = torch.stack(alphas)
        if was_cuda:
            alpha = alpha.cuda()
        return alpha


def make_quantum_aggregator(cfg: QuantumConfig | None = None) -> QuantumAttention:
    return QuantumAttention(cfg)
