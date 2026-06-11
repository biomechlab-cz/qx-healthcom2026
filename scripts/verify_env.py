"""Environment sanity checks.

Verifies:
  1. Python version + venv interpreter
  2. PyTorch + CUDA + RTX 5090
  3. PennyLane lightning.qubit circuit
  4. Qiskit + IBM runtime service
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/00_verify_env.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import PATHS  # noqa: E402

CHECK = "[ok]"
FAIL = "[FAIL]"
SKIP = "[skip]"


def check_python() -> bool:
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    inside = Path(sys.executable).is_relative_to(PATHS.root / ".venv")
    if not inside:
        print(f"{FAIL} interpreter is not inside the repo .venv")
        return False
    print(f"{CHECK} interpreter is inside .venv")
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError as e:
        print(f"{FAIL} torch not installed: {e}")
        return False

    print(f"torch: {torch.__version__}")
    if not torch.cuda.is_available():
        print(f"{FAIL} torch.cuda.is_available() = False")
        return False

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  compute capability: {cap}")
    if cap != (12, 0):
        print(f"{FAIL} expected (12, 0) for RTX 5090 / Blackwell, got {cap}")
        return False

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    if c.shape != (1024, 1024):
        print(f"{FAIL} GPU matmul produced wrong shape: {c.shape}")
        return False
    print(f"{CHECK} GPU matmul OK")
    return True


def check_pennylane() -> bool:
    try:
        import pennylane as qml
    except ImportError as e:
        print(f"{FAIL} pennylane not installed: {e}")
        return False

    print(f"pennylane: {qml.version()}")
    try:
        dev = qml.device("lightning.qubit", wires=8)
    except Exception as e:
        print(f"{FAIL} cannot construct lightning.qubit device: {e}")
        return False

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    val = float(circuit())
    if abs(val) > 1e-9:
        print(f"{FAIL} expected <Z> = 0 after Hadamard, got {val}")
        return False
    print(f"{CHECK} lightning.qubit 8-wire OK  (<Z_0> = {val:.3g})")
    return True


def check_qiskit() -> bool:
    try:
        import qiskit
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as e:
        print(f"{FAIL} qiskit / qiskit-ibm-runtime not installed: {e}")
        return False

    print(f"qiskit: {qiskit.__version__}")

    try:
        service = QiskitRuntimeService()
    except Exception as e:
        print(f"{SKIP} IBM credentials not saved ({type(e).__name__}: {e})")
        print("       Save with: QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=..., set_as_default=True)")
        return True  # not a hard failure — hardware phase will need this later

    try:
        backends = service.backends(min_num_qubits=100, operational=True)
    except Exception as e:
        print(f"{SKIP} could not query backends: {e}")
        return True

    print(f"{CHECK} IBM runtime reachable; {len(backends)} operational >=100-qubit backends:")
    for b in backends:
        try:
            status = b.status().status_msg
        except Exception:
            status = "?"
        print(f"       - {b.name}: {b.num_qubits} qubits  status={status}")
    return True


def main() -> int:
    print("=" * 60)
    print("Phase 0 environment verification")
    print("=" * 60)
    print(f"repo root: {PATHS.root}\n")

    results = {
        "python": check_python(),
        "torch+gpu": check_torch(),
        "pennylane": check_pennylane(),
        "qiskit": check_qiskit(),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")
    print("=" * 60)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
