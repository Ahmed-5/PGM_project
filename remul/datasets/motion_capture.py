"""CMU Motion Capture dataset (Section 6.2 / Appendix C.3).

Downloads ASF (skeleton) + AMC (motion) files from the CMU Motion Capture
Database (http://mocap.cs.cmu.edu) and converts joint angles to global 3D joint
positions via forward kinematics. We then form (frame t -> frame t+ΔT) position
prediction pairs (ΔT=30 on the down-sampled stream), with velocity estimated
from the previous frame.

* Subject 35 = Walking, split 200 / 600 / 600.
* Subject 9  = Running, split 200 / 240 / 240.

The ASF/AMC forward-kinematics implementation follows the standard convention
used for the CMU database (per-bone axis frame C, local Euler rotation from the
DOF channels, global rotation = parent · C · R · C⁻¹).
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch

from .common import DynamicsDataset

_BASE_URL = "http://mocap.cs.cmu.edu/subjects"

# Trials to try downloading per subject (stop once enough frames are collected).
_TRIALS = {
    35: list(range(1, 35)),   # walking
    9: list(range(1, 12)),    # running
}
_SPLITS = {
    35: (200, 600, 600),
    9: (200, 240, 240),
}


# ---------------------------------------------------------------------------
# small rotation helpers
# ---------------------------------------------------------------------------
def _deg2rad(x):
    return np.asarray(x, dtype=np.float64) * np.pi / 180.0


def _euler2mat(rad):
    """Rotation matrix from XYZ Euler angles (radians), composed Rz·Ry·Rx."""
    x, y, z = rad
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


# ---------------------------------------------------------------------------
# ASF / AMC parsing
# ---------------------------------------------------------------------------
class _Joint:
    def __init__(self, name, direction, length, axis, dof):
        self.name = name
        self.direction = np.asarray(direction, dtype=np.float64).reshape(3, 1)
        self.length = float(length)
        self.C = _euler2mat(_deg2rad(axis))
        self.Cinv = np.linalg.inv(self.C)
        self.dof = dof  # list like ['rx','ry','rz']
        self.parent = None
        self.children = []
        self.coordinate = None
        self.matrix = None

    def set_motion(self, motion):
        if self.name == "root":
            self.coordinate = np.asarray(motion["root"][:3], dtype=np.float64).reshape(3, 1)
            rot = _deg2rad(motion["root"][3:6])
            self.matrix = self.C @ _euler2mat(rot) @ self.Cinv
        else:
            rotation = np.zeros(3)
            if self.name in motion and self.dof:
                values = motion[self.name]
                for i, d in enumerate(self.dof):
                    axis = {"rx": 0, "ry": 1, "rz": 2}.get(d)
                    if axis is not None and i < len(values):
                        rotation[axis] = values[i]
            r = _euler2mat(_deg2rad(rotation))
            self.matrix = self.parent.matrix @ self.C @ r @ self.Cinv
            self.coordinate = self.parent.coordinate + self.length * (self.matrix @ self.direction)
        for c in self.children:
            c.set_motion(motion)


def _parse_asf(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    joints = {"root": _Joint("root", [0, 0, 0], 0.0, [0, 0, 0], [])}
    i = 0
    # jump to :bonedata
    while i < len(lines) and not lines[i].startswith(":bonedata"):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith(":hierarchy"):
        if lines[i] == "begin":
            i += 1
            name, direction, length, axis, dof = None, [0, 0, 0], 0.0, [0, 0, 0], []
            while lines[i] != "end":
                tok = lines[i].split()
                key = tok[0]
                if key == "name":
                    name = tok[1]
                elif key == "direction":
                    direction = [float(x) for x in tok[1:4]]
                elif key == "length":
                    length = float(tok[1])
                elif key == "axis":
                    axis = [float(x) for x in tok[1:4]]
                elif key == "dof":
                    dof = tok[1:]
                i += 1
            joints[name] = _Joint(name, direction, length, axis, dof)
        i += 1
    # hierarchy
    i += 1  # skip ':hierarchy'
    while i < len(lines) and lines[i] != "begin":
        i += 1
    i += 1
    while i < len(lines) and lines[i] != "end":
        tok = lines[i].split()
        if tok:
            parent = tok[0]
            for child in tok[1:]:
                if child in joints and parent in joints:
                    joints[child].parent = joints[parent]
                    joints[parent].children.append(joints[child])
        i += 1
    return joints


def _parse_amc(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    frames = []
    i = 0
    # skip header until first integer-only line
    while i < len(lines):
        if lines[i].isdigit():
            break
        i += 1
    while i < len(lines):
        if lines[i].isdigit():
            frame = {}
            i += 1
            while i < len(lines) and not lines[i].isdigit():
                tok = lines[i].split()
                if tok:
                    frame[tok[0]] = [float(x) for x in tok[1:]]
                i += 1
            frames.append(frame)
        else:
            i += 1
    return frames


def _joint_positions(joints, frame):
    joints["root"].set_motion(frame)
    order = sorted(joints.keys())
    return np.stack([joints[n].coordinate.reshape(3) for n in order], axis=0)  # (J, 3)


def _download(url, dest):
    if os.path.exists(dest):
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception:
        return False


def _load_subject_frames(subject, root, downsample=1, max_trials=None, needed=None):
    # downsample=1 => delta_t=30 is the classic 30-raw-frame horizon (EGNN/REMUL).
    # (Was 4 => 120-frame horizon, over which periodic walking motion returns near
    # its start, making the persistence baseline unbeatable and MSE non-discriminative.)
    subj_dir = os.path.join(root, "motion_capture", f"{subject:02d}")
    asf_path = os.path.join(subj_dir, f"{subject:02d}.asf")
    if not _download(f"{_BASE_URL}/{subject:02d}/{subject:02d}.asf", asf_path):
        raise RuntimeError(f"Could not download ASF skeleton for subject {subject}.")
    joints = _parse_asf(asf_path)

    all_pos = []
    trials = _TRIALS.get(subject, list(range(1, 20)))
    if max_trials is not None:
        trials = trials[:max_trials]
    for t in trials:
        amc_path = os.path.join(subj_dir, f"{subject:02d}_{t:02d}.amc")
        url = f"{_BASE_URL}/{subject:02d}/{subject:02d}_{t:02d}.amc"
        if not _download(url, amc_path):
            continue
        frames = _parse_amc(amc_path)
        frames = frames[::downsample]
        for fr in frames:
            all_pos.append(_joint_positions(joints, fr))
        if needed is not None and len(all_pos) > needed:
            break
    if not all_pos:
        raise RuntimeError(f"No AMC motion files could be downloaded for subject {subject}.")
    return np.stack(all_pos, axis=0)  # (T, J, 3)


def build_motion_capture_datasets(cfg):
    subject = cfg.mocap_subject
    n_train, n_val, n_test = _SPLITS.get(subject, (200, 600, 600))
    n_total = n_train + n_val + n_test
    dt = cfg.delta_t if cfg.delta_t and cfg.delta_t < 1000 else 30

    pos_seq = _load_subject_frames(
        subject, cfg.root, needed=(n_total + dt) * 2
    )  # (T, J, 3)
    pos_seq = torch.from_numpy(pos_seq).float()
    # scale down (CMU units are large); consistent linear scaling keeps equivariance
    pos_seq = pos_seq / 100.0

    T, J, _ = pos_seq.shape
    max_start = T - dt - 1
    # Disjoint window starts only (no reuse). Previously a with-replacement
    # branch was taken for short subjects (e.g. subject 9), which put the same
    # window into train AND test -> ~30% leakage and optimistically biased test
    # MSE. When frames are scarce, shrink the splits proportionally instead of
    # reusing windows.
    g = torch.Generator().manual_seed(cfg.seed)
    avail = torch.randperm(max(max_start, 1), generator=g) + 1
    if avail.numel() < n_total:
        scale = avail.numel() / n_total
        n_train = round(n_train * scale)
        n_val = round(n_val * scale)
        starts = avail
        n_total = avail.numel()
    else:
        starts = avail[:n_total]

    def gather(idx):
        pos0 = pos_seq[idx]
        posm1 = pos_seq[idx - 1]
        target = pos_seq[idx + dt]
        vel = pos0 - posm1
        com = pos0.mean(dim=1, keepdim=True)
        pos0 = pos0 - com
        target = target - com
        h = torch.ones(len(idx), J, 1)  # joints share a trivial invariant feature
        return DynamicsDataset(pos0, vel, h, target)

    tr = starts[:n_train]
    va = starts[n_train:n_train + n_val]
    te = starts[n_train + n_val:]
    datasets = {"train": gather(tr), "val": gather(va), "test": gather(te)}
    datasets["meta"] = {"num_node_features": 1, "num_nodes": J}
    return datasets
