"""Backend qubit-layout selection.

For each target backend (ibm_boston, ibm_miami), pick the 8-qubit ring with the
lowest sum of 2-qubit gate errors. Writes the chosen layout to disk so the actual
hardware-execution script (09_*) can transpile against it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import HARDWARE, PATHS, TRAIN  # noqa: E402
from utils.seed import provenance, write_json  # noqa: E402


def edge_error(props, q1: int, q2: int) -> float | None:
    """Return the calibrated 2-qubit gate error for an undirected edge, or None."""
    for q_a, q_b in ((q1, q2), (q2, q1)):
        try:
            err = props.gate_error("ecr", [q_a, q_b])  # Heron uses ECR; falls through if unavailable
            return float(err)
        except Exception:
            pass
        for gate in ("cz", "cx"):
            try:
                err = props.gate_error(gate, [q_a, q_b])
                return float(err)
            except Exception:
                pass
    return None


def find_best_8q_ring(backend) -> dict:
    """Greedy: enumerate 8-cycles in the coupling graph, score by summed 2q error."""
    props = backend.properties()
    cmap = backend.coupling_map
    edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()})

    # Build an adjacency list once.
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Cache edge errors.
    edge_err: dict[tuple[int, int], float] = {}
    for a, b in edges:
        e = edge_error(props, a, b)
        if e is not None:
            edge_err[(a, b)] = e

    # Search for 8-cycles. We bound the search by fixing the smallest node and using DFS.
    n_target = HARDWARE_QUBITS = 8
    best = {"score": float("inf"), "qubits": None, "edges": None}

    def dfs(start: int, path: list[int], path_set: set[int]):
        if len(path) == n_target:
            # Close the cycle: need edge from path[-1] to path[0]
            last, first = path[-1], path[0]
            if first in adj[last]:
                cyc_edges = [tuple(sorted((path[i], path[i + 1])))
                             for i in range(n_target - 1)] + [tuple(sorted((last, first)))]
                try:
                    score = sum(edge_err[e] for e in cyc_edges)
                except KeyError:
                    return  # missing calibration for an edge
                if score < best["score"]:
                    best.update({"score": score, "qubits": list(path), "edges": cyc_edges})
            return
        for nxt in sorted(adj[path[-1]]):
            if nxt in path_set:
                continue
            if nxt < start:
                continue
            path.append(nxt); path_set.add(nxt)
            dfs(start, path, path_set)
            path.pop(); path_set.remove(nxt)

    nodes = sorted(adj.keys())
    for s in nodes:
        dfs(s, [s], {s})
        # Heuristic early termination if we already have a really good ring.
        if best["qubits"] is not None and best["score"] < 0.05:
            break

    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="*",
                        default=[HARDWARE.primary_backend, HARDWARE.secondary_backend])
    args = parser.parse_args()

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    except Exception as e:
        print(f"IBM credentials not available ({e}). Save with QiskitRuntimeService.save_account(...).")
        return 1

    out = {"provenance": provenance(seed=TRAIN.seed, cfg={"backends": list(args.backends)}),
           "layouts": {}}

    for name in args.backends:
        print(f"\nQuerying backend: {name}")
        try:
            backend = service.backend(name)
        except Exception as e:
            print(f"  could not load backend: {e}")
            continue

        status = backend.status()
        print(f"  status: operational={status.operational}  msg={status.status_msg}  "
              f"pending_jobs={status.pending_jobs}")

        if not status.operational:
            print(f"  not operational — skipping qubit selection")
            out["layouts"][name] = {"operational": False, "selected_qubits": None}
            continue

        print(f"  searching for best 8-qubit ring (this may take a few seconds)...")
        best = find_best_8q_ring(backend)
        if best["qubits"] is None:
            print(f"  no 8-cycle found")
            out["layouts"][name] = {"operational": True, "selected_qubits": None}
            continue

        print(f"  best 8-ring: {best['qubits']}  sum-2q-error={best['score']:.5f}")
        out["layouts"][name] = {
            "operational": True,
            "n_qubits": backend.num_qubits,
            "pending_jobs": status.pending_jobs,
            "selected_qubits": best["qubits"],
            "ring_edges": [list(e) for e in best["edges"]],
            "summed_2q_error": best["score"],
        }

    payload_path = PATHS.processed / "hardware_qubit_layouts.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload_path, out)
    print(f"\nlayouts: {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
