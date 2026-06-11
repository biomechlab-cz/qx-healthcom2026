"""Ingest the new web3 source and dedupe across web1/web2/web3.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import PATHS
from utils.data import IMG_EXTS

SOURCES = ("")
HASH_SIZE = 16  # 256-bit hash for tighter comparisons; default 8 (64-bit) is too coarse


def copy_web3_originals_to_images() -> dict:
    src = PATHS.raw / "web3" / "original"
    dst = PATHS.raw / "web3" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    stats = {"copied": 0, "skipped": 0, "failed": 0}
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        out = dst / p.name
        if out.exists():
            stats["skipped"] += 1
            continue
        try:
            shutil.copy2(p, out)
            stats["copied"] += 1
        except Exception as e:
            print(f"  failed copy {p.name}: {e}")
            stats["failed"] += 1
    return stats


def list_image_label_pairs(src: str) -> tuple[list[Path], set[str], list[Path], set[str]]:
    img_dir = PATHS.raw / src / "images"
    lbl_dir = PATHS.raw / src / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS) if img_dir.exists() else []
    lbls = sorted(p for p in lbl_dir.iterdir() if p.suffix == ".txt") if lbl_dir.exists() else []
    return imgs, {p.stem for p in imgs}, lbls, {p.stem for p in lbls}


def compute_hashes(paths: list[Path]) -> dict[str, tuple[imagehash.ImageHash, imagehash.ImageHash]]:
    out = {}
    for p in paths:
        try:
            im = Image.open(p).convert("L")  # grayscale for stable chromosome hashes
        except Exception as e:
            print(f"  hash failed {p}: {e}")
            continue
        out[str(p)] = (imagehash.phash(im, hash_size=HASH_SIZE),
                       imagehash.dhash(im, hash_size=HASH_SIZE))
    return out


def find_duplicates(
    hashes_a: dict, hashes_b: dict | None = None,
    threshold_or: int = 6, threshold_and: int = 4,
) -> list[tuple[str, str, int, int]]:
    """Return list of (path_a, path_b, phash_hamming, dhash_hamming) for near-duplicate pairs.

    If hashes_b is None, look for duplicates within hashes_a (skipping i==j and i>j).
    """
    pairs = []
    a_items = list(hashes_a.items())
    if hashes_b is None:
        for i in range(len(a_items)):
            for j in range(i + 1, len(a_items)):
                pi, (phi, dhi) = a_items[i]
                pj, (phj, dhj) = a_items[j]
                dp = phi - phj
                dd = dhi - dhj
                if dp <= threshold_or or dd <= threshold_or or (dp <= threshold_and and dd <= threshold_and):
                    pairs.append((pi, pj, dp, dd))
    else:
        b_items = list(hashes_b.items())
        for pi, (phi, dhi) in a_items:
            for pj, (phj, dhj) in b_items:
                dp = phi - phj
                dd = dhi - dhj
                if dp <= threshold_or or dd <= threshold_or or (dp <= threshold_and and dd <= threshold_and):
                    pairs.append((pi, pj, dp, dd))
    return sorted(pairs, key=lambda x: x[2] + x[3])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="actually delete duplicates (default: dry-run / report only)")
    parser.add_argument("--threshold-or", type=int, default=6,
                        help="if EITHER pHash or dHash Hamming <= this, flag as dup")
    parser.add_argument("--threshold-and", type=int, default=4,
                        help="if BOTH pHash and dHash Hamming <= this, flag as dup")
    args = parser.parse_args()

    print("=" * 70)
    print("Step 1/4: copying web3/original -> web3/images")
    stats = copy_web3_originals_to_images()
    print(f"  copied={stats['copied']}  already-present={stats['skipped']}  failed={stats['failed']}")

    print("\n" + "=" * 70)
    print("Step 2/4: orphan check (per source)")
    src_paths: dict[str, list[Path]] = {}
    for src in SOURCES:
        imgs, img_stems, lbls, lbl_stems = list_image_label_pairs(src)
        orphan_img = sorted(img_stems - lbl_stems)
        orphan_lbl = sorted(lbl_stems - img_stems)
        print(f"  {src}: {len(imgs)} images, {len(lbls)} labels; "
              f"orphan-img(no-label)={len(orphan_img)} orphan-lbl(no-image)={len(orphan_lbl)}")
        if orphan_img:
            print(f"    images without labels: {orphan_img[:10]}{' …' if len(orphan_img)>10 else ''}")
        if orphan_lbl:
            print(f"    labels without images: {orphan_lbl[:10]}{' …' if len(orphan_lbl)>10 else ''}")
        # Only keep images that HAVE a label, since we'll feed downstream pipeline
        src_paths[src] = [p for p in imgs if p.stem in lbl_stems]

    print("\n" + "=" * 70)
    print("Step 3/4: computing perceptual hashes")
    hashes = {}
    for src in SOURCES:
        print(f"  hashing {src} ({len(src_paths[src])} images)...")
        hashes[src] = compute_hashes(src_paths[src])
    print(f"  done")

    print("\n" + "=" * 70)
    print(f"Step 4/4: duplicate detection (or<= {args.threshold_or} OR and<= {args.threshold_and})")

    all_dup_pairs: list[tuple[str, str, str, str, int, int]] = []
    # within web3
    print("\n  within web3:")
    within_w3 = find_duplicates(hashes["web3"], threshold_or=args.threshold_or, threshold_and=args.threshold_and)
    if not within_w3:
        print("    (no within-web3 duplicates)")
    for a, b, dp, dd in within_w3:
        an, bn = Path(a).name, Path(b).name
        print(f"    {an} <-> {bn}  phash={dp}  dhash={dd}")
        all_dup_pairs.append(("web3", a, "web3", b, dp, dd))

    # web1 <-> web3
    print("\n  web1 (web) <-> web3:")
    cross_w1 = find_duplicates(hashes["web"], hashes["web3"],
                                threshold_or=args.threshold_or, threshold_and=args.threshold_and)
    if not cross_w1:
        print("    (no web1<->web3 duplicates)")
    for a, b, dp, dd in cross_w1:
        print(f"    web/{Path(a).name} <-> web3/{Path(b).name}  phash={dp}  dhash={dd}")
        all_dup_pairs.append(("web", a, "web3", b, dp, dd))

    # web2 <-> web3
    print("\n  web2 <-> web3:")
    cross_w2 = find_duplicates(hashes["web2"], hashes["web3"],
                                threshold_or=args.threshold_or, threshold_and=args.threshold_and)
    if not cross_w2:
        print("    (no web2<->web3 duplicates)")
    for a, b, dp, dd in cross_w2:
        print(f"    web2/{Path(a).name} <-> web3/{Path(b).name}  phash={dp}  dhash={dd}")
        all_dup_pairs.append(("web2", a, "web3", b, dp, dd))

    print("\n" + "=" * 70)
    print(f"Summary: {len(all_dup_pairs)} duplicate pair(s) flagged")

    # Deletion plan: for each pair, drop the web3 copy (keep web/web2 originals).
    # For within-web3 pairs, keep the lower-numbered stem.
    to_delete: set[Path] = set()
    for src_a, path_a, src_b, path_b, _, _ in all_dup_pairs:
        if src_a == src_b == "web3":
            try:
                if int(Path(path_a).stem) <= int(Path(path_b).stem):
                    to_delete.add(Path(path_b))
                else:
                    to_delete.add(Path(path_a))
            except ValueError:
                to_delete.add(Path(path_b))
        elif src_a in ("web", "web2") and src_b == "web3":
            to_delete.add(Path(path_b))
        elif src_b in ("web", "web2") and src_a == "web3":
            to_delete.add(Path(path_a))

    if to_delete:
        print(f"\nWould delete {len(to_delete)} image file(s) and their .txt labels:")
        for p in sorted(to_delete):
            print(f"  - {p.relative_to(PATHS.root)}")
    else:
        print("\n(nothing to delete)")

    if args.apply and to_delete:
        print("\nDeleting...")
        for p in sorted(to_delete):
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            for f in (p, lbl):
                if f.exists():
                    f.unlink()
                    print(f"  removed {f.relative_to(PATHS.root)}")
    elif not args.apply and to_delete:
        print("\n[dry-run only — pass --apply to actually delete]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
