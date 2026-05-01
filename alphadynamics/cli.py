"""Command-line interface for AlphaDynamics.

Usage::

    alphadynamics --help
    alphadynamics version
    alphadynamics info
    alphadynamics models
    alphadynamics predict --sequence AAAY --output traj.npz
    alphadynamics predict --sequence AAAY --n-ensemble 16 --rollout-steps 2500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __email__, __license__, __url__, __version__, __author__
from .banner import banner_text


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"alphadynamics {__version__}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    # The banner has already been printed by __init__.py at import time.
    print()
    print("Package information")
    print("-------------------")
    print(f"  version : {__version__}")
    print(f"  author  : {__author__}")
    print(f"  email   : {__email__}")
    print(f"  license : {__license__}")
    print(f"  url     : {__url__}")
    print()
    print("Headline result (canonical Ramachandran JSD on 4AA test set)")
    print("------------------------------------------------------------")
    print("  AlphaDynamics v2_clean : 0.196   (~123K params)")
    print("  Microsoft Timewarp     : 0.468   (~396M params)")
    print("  Improvement            : 2.39x lower JSD, 3/3 peptide wins")
    print("                           3000x fewer parameters")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .weights import list_available_weights
    print("Available pretrained models:")
    for spec in list_available_weights():
        print()
        print(f"  {spec['name']}  ({spec['kind']})")
        print(f"    {spec['description']}")
        print(f"    URL: {spec['url']}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import numpy as np

    from .api import predict_torsion_ensemble

    out_path = Path(args.output) if args.output else None

    print(f"[predict] sequence={args.sequence!r}  n_ensemble={args.n_ensemble}  "
          f"rollout_steps={args.rollout_steps}  model={args.model}", file=sys.stderr)

    traj = predict_torsion_ensemble(
        args.sequence,
        model_name=args.model,
        n_ensemble=args.n_ensemble,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
        kappa_mult=args.kappa_mult,
        temperature_K=args.temperature,
        show_progress=True,
    )

    print(f"[predict] output shape={traj.shape}  (ensemble, time, residues, [phi,psi] rad)",
          file=sys.stderr)

    if out_path is None:
        # Default destination
        out_path = Path(f"alphadynamics_{args.sequence.upper()}_torsions.npz")

    np.savez_compressed(
        out_path,
        sequence=args.sequence.upper(),
        torsions=traj,
        torsion_units="radians",
        torsion_axes="(ensemble, time, residues, [phi, psi])",
        n_ensemble=args.n_ensemble,
        rollout_steps=args.rollout_steps,
        model_name=args.model,
        alphadynamics_version=__version__,
    )
    print(f"[predict] wrote {out_path}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# Argparser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphadynamics",
        description=(
            "AlphaDynamics — compact sequence-only neural propagator for protein "
            "torsion dynamics. Created by Krzysztof Gwozdz, Apache-2.0."
        ),
    )
    parser.add_argument("--version", action="version", version=f"alphadynamics {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print package version").set_defaults(func=_cmd_version)
    sub.add_parser("info", help="show banner, headline metrics and credits").set_defaults(
        func=_cmd_info
    )
    sub.add_parser("models", help="list pretrained weights").set_defaults(func=_cmd_models)

    p_predict = sub.add_parser(
        "predict",
        help="predict torsion-angle ensemble for a sequence",
    )
    p_predict.add_argument(
        "--sequence", required=True, type=str,
        help="One-letter amino-acid sequence, e.g. AAAY",
    )
    p_predict.add_argument(
        "--n-ensemble", type=int, default=16, dest="n_ensemble",
        help="Number of independent rollouts (default: 16)",
    )
    p_predict.add_argument(
        "--rollout-steps", type=int, default=2500, dest="rollout_steps",
        help="Number of trajectory frames per rollout (default: 2500)",
    )
    p_predict.add_argument(
        "--model", type=str, default="ad_transfer_v2_clean",
        help="Pretrained checkpoint name (default: ad_transfer_v2_clean)",
    )
    p_predict.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed for reproducibility (default: 42)",
    )
    p_predict.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, cpu, or auto if omitted",
    )
    p_predict.add_argument(
        "--kappa-mult", type=float, default=1.0, dest="kappa_mult",
        help="Multiplier on von Mises concentration (default: 1.0)",
    )
    p_predict.add_argument(
        "--temperature", type=float, default=None,
        help="Optional temperature conditioning in Kelvin",
    )
    p_predict.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .npz path (default: alphadynamics_<SEQ>_torsions.npz)",
    )
    p_predict.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
