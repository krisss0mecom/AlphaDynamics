"""Lazy weight downloader for AlphaDynamics pretrained checkpoints.

Heavy artifacts (model checkpoints) are NOT shipped inside the pip package.
They are downloaded on first use from the GitHub Releases of this project
and cached under ``~/.cache/alphadynamics/weights/``.

Override the cache directory with ``ALPHADYNAMICS_CACHE_DIR``.
Override the release URL base with ``ALPHADYNAMICS_RELEASE_URL``
(useful when mirroring weights to a private CDN).
"""
from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .ad_init import ADInit
from .models import AlphaDynamicsModel


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_RELEASE_TAG = "v0.3.0"
_DEFAULT_RELEASE_URL = (
    f"https://github.com/krisss0mecom/AlphaDynamics/releases/download/{_RELEASE_TAG}"
)


@dataclass(frozen=True)
class WeightSpec:
    name: str
    filename: str
    sha256: Optional[str]   # set after first release; None = skip hash check
    description: str
    kind: str               # "ad_transfer" or "ad_init"
    config: dict


# v2_clean is the headline checkpoint:
#   mean Ramachandran JSD = 0.196 vs Microsoft Timewarp 0.468 (2.39x lower)
#   3/3 wins on AAAY/AACE/AAEW canonical 4AA test set
#   ~78K params for the propagator + ~45K for AD-Init prior
_REGISTRY: dict[str, WeightSpec] = {
    "ad_transfer_v2_clean": WeightSpec(
        name="ad_transfer_v2_clean",
        filename="ad_transfer_v2_clean_best.pt",
        sha256=None,
        description=(
            "Headline AD-Transfer (v2 clean) — phase-flow ODE propagator trained on "
            "mixed-length corpus (4AA + N=48 + N=98) with explicit per-source holdout."
        ),
        kind="ad_transfer",
        config=dict(
            n_osc=64,
            n_components=8,
            hidden=128,
            rk_steps=8,
            t_max=4.0,
            use_sequence=True,
            use_temperature=False,
        ),
    ),
    "ad_init_full_1477": WeightSpec(
        name="ad_init_full_1477",
        filename="ad_init_full_1477_best.pt",
        sha256=None,
        description=(
            "AD-Init prior (mixture-of-von-Mises per residue) over 1477 mdCATH peptides. "
            "Used as a starting prior for short-context evaluation."
        ),
        kind="ad_init",
        config=dict(
            n_components=8,
            hidden=128,
            use_sequence=True,
        ),
    ),
}


def available_models() -> list[str]:
    """Return list of registered pretrained model names."""
    return sorted(_REGISTRY.keys())


def list_available_weights() -> list[dict]:
    """Return verbose info about every pretrained checkpoint."""
    return [
        dict(
            name=spec.name,
            kind=spec.kind,
            description=spec.description,
            config=dict(spec.config),
            url=_release_url(spec.filename),
        )
        for spec in _REGISTRY.values()
    ]


# --------------------------------------------------------------------------- #
# Cache + download
# --------------------------------------------------------------------------- #


def cache_dir() -> Path:
    override = os.environ.get("ALPHADYNAMICS_CACHE_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        path = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        path = path / "alphadynamics" / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _release_url(filename: str) -> str:
    base = os.environ.get("ALPHADYNAMICS_RELEASE_URL", _DEFAULT_RELEASE_URL).rstrip("/")
    return f"{base}/{filename}"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, show_progress: bool = True) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def _hook(block_num: int, block_size: int, total_size: int):
        if not show_progress or total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, 100.0 * downloaded / total_size)
        bar_w = 30
        filled = int(bar_w * pct / 100)
        bar = "#" * filled + "-" * (bar_w - filled)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stderr.write(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:6.1f} / {mb_total:6.1f} MB"
        )
        sys.stderr.flush()

    try:
        urllib.request.urlretrieve(url, tmp, _hook)
        if show_progress:
            sys.stderr.write("\n")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    shutil.move(str(tmp), str(dest))


def _ensure_cached(spec: WeightSpec) -> Path:
    target = cache_dir() / spec.filename
    if target.exists():
        if spec.sha256 is None or _hash_file(target) == spec.sha256:
            return target
        sys.stderr.write(
            f"[alphadynamics] cached {spec.filename} failed hash check, redownloading\n"
        )
        target.unlink()

    url = _release_url(spec.filename)
    sys.stderr.write(f"[alphadynamics] downloading {spec.filename} from {url}\n")
    _download(url, target)

    if spec.sha256 is not None:
        got = _hash_file(target)
        if got != spec.sha256:
            target.unlink()
            raise RuntimeError(
                f"hash mismatch for {spec.filename}: expected {spec.sha256}, got {got}"
            )
    return target


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #


def _constructor_config(cls: type[torch.nn.Module], spec_config: dict, *configs: dict) -> dict:
    """Merge checkpoint configs onto the registry default, keeping only kwargs
    the model class actually accepts. Registry config wins as the baseline,
    checkpoint configs may override individual entries when they exist."""
    cfg = dict(spec_config)
    for config in configs:
        if config:
            cfg.update({k: v for k, v in config.items() if k in cfg})

    allowed = set(inspect.signature(cls).parameters)
    return {k: v for k, v in cfg.items() if k in allowed}


def _state_dict_from_checkpoint(state, spec: WeightSpec) -> tuple[dict, dict, dict]:
    """Return ``(state_dict, checkpoint_config, args)`` for supported checkpoint layouts.

    Trainer checkpoints can wrap the state dict under ``model_state``,
    ``state_dict`` or ``model``, with config under ``model_config`` or
    ``config``. Bare state dicts are also accepted.
    """
    if not isinstance(state, dict):
        return state, {}, {}

    args = state.get("args", {}) if isinstance(state.get("args"), dict) else {}

    if "model_state" in state:
        return state["model_state"], state.get("model_config", state.get("config", {})), args
    if "state_dict" in state:
        return state["state_dict"], state.get("config", state.get("model_config", {})), args
    if "model" in state:
        return state["model"], state.get("config", state.get("model_config", {})), args

    if state and all(torch.is_tensor(v) for v in state.values()):
        return state, {}, args

    raise RuntimeError(
        f"unsupported checkpoint format for {spec.name!r}: keys={list(state.keys())[:8]}"
    )


def load_pretrained(
    name: str = "ad_transfer_v2_clean",
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Load a pretrained AlphaDynamics checkpoint, downloading on first use.

    Parameters
    ----------
    name : str
        One of :func:`available_models`. Default: ``"ad_transfer_v2_clean"``.
    device : str or torch.device
        Where to place the model.
    eval_mode : bool
        Call ``.eval()`` on the loaded model. Default: ``True``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. available: {available_models()}"
        )
    spec = _REGISTRY[name]
    path = _ensure_cached(spec)

    state = torch.load(path, map_location="cpu", weights_only=False)
    state_dict, ckpt_config, args = _state_dict_from_checkpoint(state, spec)

    if spec.kind == "ad_transfer":
        cfg = _constructor_config(AlphaDynamicsModel, spec.config, args, ckpt_config)
        model = AlphaDynamicsModel(**cfg)
    elif spec.kind == "ad_init":
        cfg = _constructor_config(ADInit, spec.config, args, ckpt_config)
        model = ADInit(**cfg)
    else:
        raise RuntimeError(f"unknown model kind {spec.kind!r}")

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model
