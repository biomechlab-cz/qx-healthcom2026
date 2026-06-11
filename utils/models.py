"""Model factories.

build_resnet18_instance — ImageNet-pretrained ResNet-18 with conv1..layer3 frozen, fc -> 2.
build_classical_mil    — frozen ResNet-18 features + Ilse et al. gated attention + bag/count heads.
build_vgg19_modified   — Shin et al. 2024: VGG-19 with last 3 FCs replaced by GAP + FC(256) + FC(2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from utils.config import MODEL


def _freeze_until(model: nn.Module, last_frozen_block: str) -> None:
    """Freeze all params up to and including `last_frozen_block`. Train everything after."""
    train = False if last_frozen_block else True
    # Order of ResNet blocks (torchvision): conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    blocks_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    target_idx = blocks_order.index(last_frozen_block) if last_frozen_block in blocks_order else -1
    for name, child in model.named_children():
        if name in blocks_order:
            idx = blocks_order.index(name)
            requires = idx > target_idx  # train layers strictly after target
        elif name in ("fc",):
            requires = True
        else:
            requires = False
        for p in child.parameters():
            p.requires_grad = requires


def build_resnet18_instance(
    pretrained: bool = True,
    freeze_until: str = "layer3",
    num_classes: int = 2,
) -> nn.Module:
    """ResNet-18 head replaced with 2-class fc. By default freezes conv1..layer3, trains layer4 + fc."""
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    _freeze_until(model, freeze_until)
    return model


def build_resnet18_feature_extractor(state_dict_path: str | None = None) -> nn.Module:
    """ResNet-18 with fc removed, outputting 512-d feature vectors. Loads from a saved instance-classifier ckpt if given."""
    base = tvm.resnet18(weights=None)
    if state_dict_path is not None:
        # Construct a temporary model with the 2-class fc to load weights, then strip fc.
        full = build_resnet18_instance(pretrained=False, freeze_until="", num_classes=2)
        ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        full.load_state_dict(state, strict=False)
        base = full
    base.fc = nn.Identity()  # type: ignore[assignment]
    for p in base.parameters():
        p.requires_grad = False
    base.eval()
    return base


class GatedAttention(nn.Module):
    """Ilse et al. 2018 gated attention. inputs (K, D) -> attention weights (K,)."""

    def __init__(self, in_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (K, D) -> (K,)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        a = self.w(a).squeeze(-1)
        return a  # raw logits; softmax outside


class ClassicalMIL(nn.Module):
    """Frozen ResNet-18 features -> gated attention -> bag classifier + count head."""

    def __init__(self, backbone: nn.Module, feature_dim: int = 512, attn_hidden: int = 128):
        super().__init__()
        self.backbone = backbone
        self.attention = GatedAttention(feature_dim, attn_hidden)
        self.bag_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.count_head = nn.Linear(feature_dim, 1)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # bag: (K, C, H, W)
        with torch.no_grad():
            phi = self.backbone(bag)  # (K, 512)
        alpha = self.attention(phi)  # (K,)
        a = torch.softmax(alpha, dim=0)
        v = (a.unsqueeze(-1) * phi).sum(dim=0)  # (512,)
        bag_logits = self.bag_head(v)
        count = self.count_head(v).squeeze(-1)
        return bag_logits, count, a


def build_classical_mil(state_dict_path: str | None = None) -> ClassicalMIL:
    backbone = build_resnet18_feature_extractor(state_dict_path)
    return ClassicalMIL(backbone)


class HQMIL(nn.Module):
    """Hybrid quantum-classical MIL.

    bag -> frozen ResNet-18 -> [richer projector] -> n_qubits
        -> per-instance quantum attention logits alpha_k
        -> softmax(alpha / T) where T is learnable (initialised small so attention is sharp)
        -> bag feature v = sum_k a_k * z_k
        -> bag head + count head

    The temperature T prevents the K=40+ softmax from defaulting to ~uniform, which would
    average all instances and produce near-constant bag predictions across bags.
    """

    def __init__(self, backbone: nn.Module, quantum_attention: nn.Module,
                 feature_dim: int = 512, n_qubits: int = 8,
                 projector_hidden: int = 64, init_temperature: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projector_hidden),
            nn.ReLU(),
            nn.Linear(projector_hidden, n_qubits),
            nn.Tanh(),
        )
        self.quantum_attention = quantum_attention
        # Parameterise log T so T stays positive under optimisation.
        import math as _math
        self.log_attention_temp = nn.Parameter(torch.tensor(_math.log(init_temperature)))
        self.bag_head = nn.Sequential(
            nn.Linear(n_qubits, 16), nn.ReLU(), nn.Linear(16, 2)
        )
        self.count_head = nn.Linear(n_qubits, 1)
        self.n_qubits = n_qubits
        self._encoding_scale = _math.pi / 2.0

    @property
    def attention_temperature(self) -> torch.Tensor:
        return torch.exp(self.log_attention_temp)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # bag: (K, C, H, W)
        with torch.no_grad():
            phi = self.backbone(bag)              # (K, 512)
        z = self.projector(phi) * self._encoding_scale  # (K, n_qubits) in (-pi/2, pi/2)
        alpha = self.quantum_attention(z)         # (K,)
        T = self.attention_temperature
        a = torch.softmax(alpha / T, dim=0)
        v = (a.unsqueeze(-1) * z).sum(dim=0)      # (n_qubits,)
        bag_logits = self.bag_head(v)
        count = self.count_head(v).squeeze(-1)
        return bag_logits, count, a


def build_hq_mil(
    state_dict_path: str | None = None,
    quantum_cfg=None,  # utils.quantum_aggregator.QuantumConfig | None
) -> HQMIL:
    from utils.quantum_aggregator import QuantumAttention, QuantumConfig
    qcfg = quantum_cfg or QuantumConfig()
    backbone = build_resnet18_feature_extractor(state_dict_path)
    qa = QuantumAttention(qcfg)
    return HQMIL(backbone, qa, feature_dim=512, n_qubits=qcfg.n_qubits)


def build_vgg19_modified(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """VGG-19 features + GAP + FC(256) + FC(num_classes). Per Shin et al. 2024."""
    weights = tvm.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    base = tvm.vgg19(weights=weights)
    features = base.features

    class VGG19Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.pool(self.features(x)))

    return VGG19Mod()


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
