"""AlphaDynamics — compact sequence-only neural propagator for protein torsion dynamics.

Created by Krzysztof Gwozdz <krisss0gwo@gmail.com>.
Licensed under Apache License 2.0. See LICENSE and NOTICE.

Quick start
-----------
    >>> from alphadynamics import predict_torsion_ensemble
    >>> traj = predict_torsion_ensemble("AAAY", n_ensemble=16, rollout_steps=2500)
    >>> traj.shape
    (16, 2500, 4, 2)        # (ensemble, time, residues, [phi, psi])

Or from the command line::

    alphadynamics predict --sequence AAAY --output traj.npz
"""
from __future__ import annotations

import os as _os

__version__ = "0.4.0"
__author__ = "Krzysztof Gwozdz"
__email__ = "krisss0gwo@gmail.com"
__license__ = "Apache-2.0"
__url__ = "https://github.com/krisss0mecom/AlphaDynamics"

from .data import ProteinTrajectory, discover_npz, split_paths
from .metrics import canonical_jsd, visual_jsd
from .models import AlphaDynamicsModel
from .ad_init import ADInit
from .rollout import rollout
from .weights import load_pretrained, list_available_weights, available_models
from .api import predict_torsion_ensemble
from .geometry import (
    torsions_to_backbone,
    trajectory_to_pdb,
    radius_of_gyration,
    end_to_end_distance,
    trajectory_diagnostics,
)

__all__ = [
    "ADInit",
    "AlphaDynamicsModel",
    "ProteinTrajectory",
    "available_models",
    "canonical_jsd",
    "discover_npz",
    "end_to_end_distance",
    "list_available_weights",
    "load_pretrained",
    "predict_torsion_ensemble",
    "radius_of_gyration",
    "rollout",
    "split_paths",
    "torsions_to_backbone",
    "trajectory_diagnostics",
    "trajectory_to_pdb",
    "visual_jsd",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__version__",
]


# Show banner once per process unless ALPHADYNAMICS_NO_BANNER=1
_BANNER_SHOWN_FLAG = "_ALPHADYNAMICS_BANNER_SHOWN"
if (
    _os.environ.get("ALPHADYNAMICS_NO_BANNER", "").strip() not in {"1", "true", "True", "yes"}
    and _os.environ.get(_BANNER_SHOWN_FLAG) != "1"
):
    try:
        from .banner import print_banner as _print_banner
        _print_banner()
        _os.environ[_BANNER_SHOWN_FLAG] = "1"
        try:
            import sys as _sys
            _sys.stderr.write(
                "  NOTE: AlphaDynamics is best validated for short peptides (4-15 aa).\n"
                "        Sequences longer than 20 aa are outside the calibrated scope.\n\n"
            )
        except Exception:
            pass
    except Exception:
        # Never let banner crash the import
        pass

    # Background-style update check (cached 24h, 2s timeout, never blocks)
    try:
        from .version_check import check_for_update as _check_for_update
        _check_for_update()
    except Exception:
        # Never let version check crash the import
        pass
