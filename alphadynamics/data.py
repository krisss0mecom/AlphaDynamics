from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def wrap_angles(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def _scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _read_sequence(npz) -> str | None:
    for key in ("sequence", "seq", "aa_sequence", "residue_sequence"):
        if key in npz.files:
            return _scalar_str(npz[key])
    return None


def align_sequence_to_residues(
    full_sequence: str,
    n_residues: int,
    residue_indices: Iterable[int] | None = None,
) -> str:
    """Return the sequence aligned to the stored torsion residues.

    Some sequence manifests store full protein/domain sequences, while others
    already store the exact torsion-residue subsequence. The latter must not
    be sliced a second time by ``residue_indices``; doing so shifts N=98
    mdCATH sequences by one residue and appends ``X`` at the end.
    """
    seq = str(full_sequence)
    if len(seq) == n_residues:
        return seq

    if residue_indices is not None:
        indices = [int(i) for i in residue_indices]
        return "".join(seq[i] if 0 <= i < len(seq) else "X" for i in indices)

    if len(seq) > n_residues:
        raise ValueError(
            "cannot align a longer sequence without residue_indices; "
            f"sequence length={len(seq)} torsion residues={n_residues}"
        )

    return seq[:n_residues]


@dataclass
class ProteinTrajectory:
    domain_id: str
    train: np.ndarray
    val: np.ndarray
    source_path: str
    sequence: str | None = None
    temperature_K: float | None = None

    @property
    def n_residues(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.val.shape[0])

    @classmethod
    def load(
        cls,
        path: str | Path,
        max_train_frames: int = 0,
        sequences_lookup: dict | None = None,
        temperature_K: float | None = None,
    ) -> "ProteinTrajectory":
        p = Path(path)
        d = np.load(p, allow_pickle=True)
        if "train" not in d.files or "val" not in d.files:
            raise ValueError(f"{p} must contain train and val arrays")
        train = d["train"].astype(np.float32)
        val = d["val"].astype(np.float32)
        if train.ndim == 2:
            if train.shape[1] % 2 != 0:
                raise ValueError(f"{p}: flat train has odd feature count")
            train = train.reshape(train.shape[0], train.shape[1] // 2, 2)
        if val.ndim == 2:
            if val.shape[1] % 2 != 0:
                raise ValueError(f"{p}: flat val has odd feature count")
            val = val.reshape(val.shape[0], val.shape[1] // 2, 2)
        if train.ndim != 3 or val.ndim != 3 or train.shape[-1] != 2 or val.shape[-1] != 2:
            raise ValueError(f"{p}: expected train/val shapes (T, N, 2)")
        if max_train_frames > 0:
            train = train[:max_train_frames]
        if train.shape[1] != val.shape[1]:
            raise ValueError(f"{p}: train/val residue count mismatch")
        domain_id = _scalar_str(d["domain_id"]) if "domain_id" in d.files else p.stem
        sequence = _read_sequence(d)
        if sequence is None and sequences_lookup and domain_id in sequences_lookup:
            entry = sequences_lookup[domain_id]
            full_sequence = entry.get("sequence", "") if isinstance(entry, dict) else str(entry)
            if "residue_indices" in d.files:
                sequence = align_sequence_to_residues(
                    full_sequence,
                    train.shape[1],
                    d["residue_indices"],
                )
            else:
                sequence = align_sequence_to_residues(full_sequence, train.shape[1])
        if sequence is not None:
            if len(sequence) < train.shape[1]:
                sequence = sequence + "X" * (train.shape[1] - len(sequence))
            sequence = sequence[: train.shape[1]]
        temp = temperature_K if temperature_K is not None else _temperature_from_npz(d, p)
        return cls(
            domain_id=domain_id,
            train=wrap_angles(train),
            val=wrap_angles(val),
            source_path=str(p),
            sequence=sequence,
            temperature_K=temp,
        )

    def seed_train(self, n_frames: int) -> "ProteinTrajectory":
        n = min(max(int(n_frames), 0), self.n_train)
        return ProteinTrajectory(
            domain_id=self.domain_id,
            train=self.train[:n].copy(),
            val=self.val,
            source_path=self.source_path,
            sequence=self.sequence,
            temperature_K=self.temperature_K,
        )


def _temperature_from_npz(npz, path: Path) -> float | None:
    for key in ("temperature_K", "temperature", "temp_K", "T"):
        if key in npz.files:
            try:
                return float(np.asarray(npz[key]).item())
            except Exception:
                pass
    match = re.search(r"_T(\d+)", path.name)
    if match:
        return float(match.group(1))
    return None


def load_sequences_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    import json

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def discover_npz(data_dirs: Sequence[str | Path], pattern: str = "*_dihedrals.npz") -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for data_dir in data_dirs:
        for path in sorted(Path(data_dir).glob(pattern)):
            try:
                prot = ProteinTrajectory.load(path, max_train_frames=2)
            except Exception:
                continue
            key = prot.domain_id
            if key not in seen:
                paths.append(path)
                seen.add(key)
    return sorted(paths, key=lambda p: p.name)


def split_paths(paths: Sequence[Path], n_holdout: int, seed: int = 42) -> tuple[list[Path], list[Path]]:
    if n_holdout <= 0:
        return list(paths), []
    if n_holdout >= len(paths):
        raise ValueError("n_holdout must be smaller than number of discovered proteins")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    hold_idx = set(idx[:n_holdout].tolist())
    train, holdout = [], []
    for i, path in enumerate(paths):
        (holdout if i in hold_idx else train).append(path)
    return train, holdout


class TransitionBatcher:
    """Uniformly samples proteins, then uniformly samples transitions inside one protein."""

    def __init__(self, proteins: Sequence[ProteinTrajectory], batch_size: int, seed: int = 42):
        if not proteins:
            raise ValueError("empty protein list")
        self.proteins = list(proteins)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[ProteinTrajectory, np.ndarray, np.ndarray]:
        protein = self.proteins[int(self.rng.integers(0, len(self.proteins)))]
        n_pairs = protein.n_train - 1
        if n_pairs <= 0:
            raise ValueError(f"{protein.domain_id} has fewer than 2 train frames")
        bsz = min(self.batch_size, n_pairs)
        idx = self.rng.integers(0, n_pairs, size=bsz)
        return protein, protein.train[idx], protein.train[idx + 1]


def as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)


def residue_ids(sequence: str | None, n_residues: int, device: torch.device) -> torch.Tensor | None:
    if sequence is None or len(sequence) < n_residues:
        return None
    ids = [AA_TO_ID.get(aa.upper(), len(AA_ALPHABET)) for aa in sequence[:n_residues]]
    return torch.tensor(ids, dtype=torch.long, device=device)


def temperature_tensor(
    temperature_K: float | None,
    n_residues: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if temperature_K is None:
        return None
    import math

    value = math.log(max(float(temperature_K), 1.0) / 300.0)
    return torch.full((batch_size, n_residues, 1), value, dtype=torch.float32, device=device)
