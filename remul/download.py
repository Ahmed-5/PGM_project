"""One-shot downloader for all REMUL datasets.

Usage:
    python -m remul.download --datasets md17 motion_capture --root data/remul
    python -m remul.download --md17-molecules aspirin ethanol
    python -m remul.download            # everything (all 8 MD17 + MoCap 35 & 9)

N-body is fully synthetic (generated on the fly), so nothing is downloaded for it.
MD17 downloads via torch_geometric (quantum-machine.org). Motion Capture downloads
ASF/AMC files from the CMU database (mocap.cs.cmu.edu).
"""
from __future__ import annotations

import argparse

_MD17_ALL = ["aspirin", "benzene", "ethanol", "malonaldehyde",
             "naphthalene", "salicylic", "toluene", "uracil"]
_MOCAP_ALL = [35, 9]

_MD17_PYG = {
    "aspirin": "aspirin", "benzene": "benzene", "ethanol": "ethanol",
    "malonaldehyde": "malonaldehyde", "naphthalene": "naphthalene",
    "salicylic": "salicylic acid", "toluene": "toluene", "uracil": "uracil",
}


def download_md17(molecules, root):
    from torch_geometric.datasets import MD17
    for mol in molecules:
        name = _MD17_PYG[mol]
        print(f"[md17] {mol} ...", flush=True)
        MD17(root=f"{root}/md17/{name.replace(' ', '_')}", name=name)
        print(f"[md17] {mol} done", flush=True)


def download_mocap(subjects, root):
    from .datasets.motion_capture import _load_subject_frames
    for s in subjects:
        print(f"[mocap] subject {s} ...", flush=True)
        pos = _load_subject_frames(s, root)
        print(f"[mocap] subject {s} done: {pos.shape[0]} frames, {pos.shape[1]} joints", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["md17", "motion_capture"],
                   choices=["md17", "motion_capture", "nbody"])
    p.add_argument("--root", default="data/remul")
    p.add_argument("--md17-molecules", nargs="+", default=["all"])
    p.add_argument("--mocap-subjects", nargs="+", type=int, default=_MOCAP_ALL)
    args = p.parse_args()

    if "md17" in args.datasets:
        mols = _MD17_ALL if args.md17_molecules == ["all"] else args.md17_molecules
        download_md17(mols, args.root)
    if "motion_capture" in args.datasets:
        download_mocap(args.mocap_subjects, args.root)
    if "nbody" in args.datasets:
        print("[nbody] synthetic — nothing to download (generated at runtime).")
    print("All requested downloads complete.")


if __name__ == "__main__":
    main()
