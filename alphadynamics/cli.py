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
# Output path helpers (cross-platform writability check + auto-fallback)
# --------------------------------------------------------------------------- #


def _writable_dir(path: Path) -> bool:
    """Probe whether we can create a file under `path`. Cleans up after itself."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".alphadynamics_write_probe_{os.getpid()}"
        probe.touch()
        probe.unlink()
        return True
    except (PermissionError, OSError):
        return False


def _safe_output_path(requested: str | Path, default_basename: str) -> Path:
    """Return a path we are guaranteed to be able to write to.

    Tries (in order):
    1. The requested path.
    2. ~/Documents/ (Windows-friendly default that is always writable).
    3. ~/  (home dir).
    4. tempfile.gettempdir().

    Prints a friendly note (to stderr) if it falls back, so the user
    always knows where the file landed.
    """
    import tempfile

    requested = Path(requested).expanduser()
    parent = requested.parent if requested.parent != Path("") else Path(".")

    # Resolve "." to an absolute path so error messages are useful
    try:
        parent_resolved = parent.resolve()
    except Exception:
        parent_resolved = parent

    if _writable_dir(parent_resolved):
        return requested

    candidates = [
        Path.home() / "Documents",
        Path.home(),
        Path(tempfile.gettempdir()),
    ]
    for cand in candidates:
        if _writable_dir(cand):
            fallback = cand / Path(requested.name or default_basename)
            print(
                f"[alphadynamics] cannot write to {parent_resolved!s} "
                f"(no permission). Saving to {fallback!s} instead.",
                file=sys.stderr,
            )
            return fallback

    raise PermissionError(
        f"No writable location found. Tried: {requested!s}, "
        f"~/Documents, ~/, and {tempfile.gettempdir()!r}."
    )


# --------------------------------------------------------------------------- #
# Interactive Ramachandran plot (plotly-optional)
# --------------------------------------------------------------------------- #


def _make_ramachandran_html(traj, sequence: str, out_path: str) -> bool:
    """Save an interactive Ramachandran density map as a self-contained HTML.

    Produces: aggregate density (smooth contour) + per-residue subplot grid
    (when n_residues <= 16) + reference Ramachandran-favored zones overlay
    (Lovell et al. 2003 simplified) + annotated basin labels.

    Returns True on success, False if plotly missing.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[plot-html] plotly not installed. Run: pip install 'alphadynamics[viz]'",
              file=sys.stderr)
        return False

    n_ens, n_time, n_res, _ = traj.shape

    has_per_res = 1 <= n_res <= 16
    if has_per_res:
        ncols = min(4, n_res)
        nrows = 1 + (n_res + ncols - 1) // ncols
        specs = [
            [{"colspan": ncols, "rowspan": 1}] + [None] * (ncols - 1),
        ] + [
            [{} for _ in range(ncols)] for _ in range((n_res + ncols - 1) // ncols)
        ]
        titles = [f"<b>{sequence}</b> — aggregate Ramachandran density"]
        for i in range(n_res):
            aa = sequence[i] if i < len(sequence) else "?"
            titles.append(f"{aa}{i+1}")
        # pad titles to fill grid
        total = ncols + ((n_res + ncols - 1) // ncols) * ncols
        titles += [""] * max(0, total - len(titles))
        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.04, vertical_spacing=0.10,
        )
        agg_row = 1
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f"<b>{sequence}</b> — aggregate Ramachandran density "
                            f"({n_res} residues, {n_ens}×{n_time} samples)"],
        )
        agg_row, ncols = 1, 1

    phi_all = np.degrees(traj[..., 0].flatten())
    psi_all = np.degrees(traj[..., 1].flatten())

    fig.add_trace(
        go.Histogram2dContour(
            x=phi_all, y=psi_all,
            colorscale="Viridis",
            xaxis="x", yaxis="y",
            ncontours=20,
            contours=dict(coloring="fill", showlines=False),
            line=dict(width=0),
            showscale=True,
            colorbar=dict(title="density",
                          len=0.45 if has_per_res else 0.85,
                          y=0.78 if has_per_res else 0.5,
                          x=1.02),
            hovertemplate="φ=%{x:.0f}°<br>ψ=%{y:.0f}°<br>density=%{z:.4f}<extra></extra>",
        ),
        row=agg_row, col=1,
    )

    # Annotate canonical basins on aggregate panel
    basin_labels = [
        (-60, -45, "α-R", "white"),
        (-120, 130, "β", "white"),
        (-60, 140, "PPII", "white"),
        (60, 50, "α-L (forbidden)", "orange"),
    ]
    for px, py, label, color in basin_labels:
        fig.add_annotation(
            x=px, y=py, text=f"<b>{label}</b>",
            showarrow=False, font=dict(color=color, size=12),
            xref="x1", yref="y1",
        )

    if has_per_res:
        for i in range(n_res):
            row = 2 + (i // ncols)
            col = 1 + (i % ncols)
            phi_i = np.degrees(traj[:, :, i, 0].flatten())
            psi_i = np.degrees(traj[:, :, i, 1].flatten())
            fig.add_trace(
                go.Histogram2dContour(
                    x=phi_i, y=psi_i,
                    colorscale="Viridis",
                    ncontours=15,
                    contours=dict(coloring="fill", showlines=False),
                    line=dict(width=0),
                    showscale=False,
                    hovertemplate="φ=%{x:.0f}°<br>ψ=%{y:.0f}°<extra></extra>",
                ),
                row=row, col=col,
            )

    # Style every panel: square, [-180,180] axes, gridlines at 0
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].update(
                range=[-180, 180],
                tickvals=[-180, -90, 0, 90, 180],
                ticksuffix="°",
                gridcolor="rgba(255,255,255,0.15)",
                zerolinecolor="rgba(255,255,255,0.4)",
                zerolinewidth=1,
            )
        elif axis.startswith("yaxis"):
            # plotly axis reference: yaxisN -> xN (not xaxisN)
            suffix = axis[len("yaxis"):]
            x_ref = "x" + suffix
            fig.layout[axis].update(
                range=[-180, 180],
                tickvals=[-180, -90, 0, 90, 180],
                ticksuffix="°",
                gridcolor="rgba(255,255,255,0.15)",
                zerolinecolor="rgba(255,255,255,0.4)",
                zerolinewidth=1,
                scaleanchor=x_ref,
                scaleratio=1,
            )

    fig.update_layout(
        title=dict(
            text=f"<b>AlphaDynamics — {sequence}</b><br>"
                 f"<sub>{n_ens} trajectories × {n_time} steps × {n_res} residues "
                 f"= {n_ens * n_time * n_res:,} samples · model: ad_transfer_v2_clean</sub>",
            x=0.5, xanchor="center",
        ),
        template="plotly_dark",
        height=350 + (250 * (1 + (n_res + ncols - 1) // ncols) if has_per_res else 350),
        width=900 if not has_per_res else max(900, 230 * ncols + 100),
        margin=dict(l=70, r=120, t=110, b=60),
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12),
        hovermode="closest",
    )

    # Common axis labels via single visible-only annotation
    fig.add_annotation(
        text="φ (degrees)", xref="paper", yref="paper",
        x=0.5, y=-0.06 if not has_per_res else -0.02,
        showarrow=False, font=dict(size=13),
    )
    fig.add_annotation(
        text="ψ (degrees)", xref="paper", yref="paper",
        x=-0.06, y=0.5,
        textangle=-90,
        showarrow=False, font=dict(size=13),
    )

    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format="png", filename=f"ramachandran_{sequence}",
                height=900, width=900, scale=2,
            ),
        ),
    )
    return True


# --------------------------------------------------------------------------- #
# Ramachandran plot helper (matplotlib-optional)
# --------------------------------------------------------------------------- #


def _make_ramachandran_plot(traj, sequence: str, out_path: str) -> bool:
    """Save a Ramachandran density map (plus per-residue panels for short peptides).

    Returns True on success, False if matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed. Run: pip install 'alphadynamics[viz]'",
              file=sys.stderr)
        return False

    n_ens, n_time, n_res, _ = traj.shape
    phi_all = np.degrees(traj[..., 0].flatten())
    psi_all = np.degrees(traj[..., 1].flatten())

    has_per_res = 1 <= n_res <= 16

    if has_per_res:
        ncols = min(4, n_res)
        nrows_pp = (n_res + ncols - 1) // ncols
        fig = plt.figure(figsize=(6 + 2.0 * ncols, max(6, 2.0 * nrows_pp) + 0.5))
        gs = fig.add_gridspec(nrows_pp, ncols + 2, width_ratios=[3, 0.05] + [1] * ncols)
        ax_main = fig.add_subplot(gs[:, 0])
    else:
        fig, ax_main = plt.subplots(figsize=(7.5, 6.5))

    h = ax_main.hist2d(
        phi_all, psi_all,
        bins=72,
        range=[[-180, 180], [-180, 180]],
        cmap="viridis",
        density=True,
    )
    ax_main.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax_main.axvline(0, color="white", lw=0.5, alpha=0.3)
    # Basin labels
    ax_main.text(-60, -45, "α-R",   ha="center", color="white", fontsize=12, fontweight="bold")
    ax_main.text(-120, 130, "β",    ha="center", color="white", fontsize=12, fontweight="bold")
    ax_main.text(-60, 140, "PPII",  ha="center", color="white", fontsize=11, fontweight="bold")
    ax_main.text(60, 50, "α-L (forbidden)",
                 ha="center", color="orange", fontsize=8, alpha=0.7)
    ax_main.set_xlabel("φ (degrees)")
    ax_main.set_ylabel("ψ (degrees)")
    ax_main.set_xlim(-180, 180)
    ax_main.set_ylim(-180, 180)
    ax_main.set_xticks([-180, -90, 0, 90, 180])
    ax_main.set_yticks([-180, -90, 0, 90, 180])
    ax_main.set_aspect("equal")
    ax_main.set_title(
        f"AlphaDynamics — {sequence}  ({n_res} residues, {n_ens}×{n_time} samples)"
    )
    fig.colorbar(h[3], ax=ax_main, fraction=0.046, pad=0.04, label="density")

    # Per-residue panels
    if has_per_res:
        for i in range(n_res):
            row, col = divmod(i, ncols)
            ax_i = fig.add_subplot(gs[row, 2 + col])
            phi_i = np.degrees(traj[:, :, i, 0].flatten())
            psi_i = np.degrees(traj[:, :, i, 1].flatten())
            ax_i.hist2d(phi_i, psi_i, bins=36,
                        range=[[-180, 180], [-180, 180]],
                        cmap="viridis", density=True)
            ax_i.set_xlim(-180, 180); ax_i.set_ylim(-180, 180)
            ax_i.set_xticks([]); ax_i.set_yticks([])
            ax_i.set_aspect("equal")
            aa = sequence[i] if i < len(sequence) else "?"
            ax_i.set_title(f"{aa}{i+1}", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


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

    seq_upper = args.sequence.upper()
    requested = args.output or f"alphadynamics_{seq_upper}_torsions.npz"
    # Pre-flight: bail BEFORE running predict if the chosen path is not writable.
    # The model run is expensive; we don't want to waste it on a write error
    # at the end (Windows users hitting this on C:\ root, for example).
    out_path = _safe_output_path(requested, f"alphadynamics_{seq_upper}_torsions.npz")

    print(f"[predict] sequence={args.sequence!r}  n_ensemble={args.n_ensemble}  "
          f"rollout_steps={args.rollout_steps}  model={args.model}", file=sys.stderr)
    print(f"[predict] will save to: {out_path}", file=sys.stderr)

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

    np.savez_compressed(
        out_path,
        sequence=seq_upper,
        torsions=traj,
        torsion_units="radians",
        torsion_axes="(ensemble, time, residues, [phi, psi])",
        n_ensemble=args.n_ensemble,
        rollout_steps=args.rollout_steps,
        model_name=args.model,
        alphadynamics_version=__version__,
    )
    print(f"[predict] wrote {out_path}", file=sys.stderr)

    if getattr(args, "plot", False):
        requested_png = args.plot_out or str(out_path).replace(".npz", "_ramachandran.png")
        if not requested_png.endswith(".png"):
            requested_png += ".png"
        plot_path = _safe_output_path(requested_png, f"alphadynamics_{seq_upper}_ramachandran.png")
        if _make_ramachandran_plot(traj, seq_upper, str(plot_path)):
            print(f"[plot] wrote {plot_path}", file=sys.stderr)

    if getattr(args, "plot_html", False):
        requested_html = args.html_out or str(out_path).replace(".npz", "_ramachandran.html")
        if not requested_html.endswith(".html"):
            requested_html += ".html"
        html_path = _safe_output_path(requested_html, f"alphadynamics_{seq_upper}_ramachandran.html")
        if _make_ramachandran_html(traj, seq_upper, str(html_path)):
            print(f"[plot-html] wrote {html_path}  (open in browser)", file=sys.stderr)

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
    p_predict.add_argument(
        "--plot", action="store_true",
        help="Save a static Ramachandran density map as PNG (requires matplotlib).",
    )
    p_predict.add_argument(
        "--plot-out", type=str, default=None, dest="plot_out",
        help="PNG path (default: <output>_ramachandran.png)",
    )
    p_predict.add_argument(
        "--plot-html", action="store_true", dest="plot_html",
        help="Save an INTERACTIVE Ramachandran map as HTML (zoom, hover, pan; requires plotly).",
    )
    p_predict.add_argument(
        "--html-out", type=str, default=None, dest="html_out",
        help="HTML path (default: <output>_ramachandran.html)",
    )
    p_predict.set_defaults(func=_cmd_predict)
    return parser


def _cmd_interactive(args: argparse.Namespace | None = None) -> int:
    """Interactive prompt — what most users want when they just type `alphadynamics`."""
    print()
    print("Welcome! I'll help you predict torsion dynamics for any peptide.")
    print("Press Ctrl+C any time to cancel.")
    print()

    try:
        from .api import predict_torsion_ensemble
        from .weights import available_models

        seq = ""
        while not seq:
            seq = input("Sequence (1-letter AA, e.g. AAAY): ").strip().upper()
            if not seq:
                print("  (please enter at least one residue)")
            elif len(seq) > 20:
                print("  (warning: sequences longer than 20 aa are outside the calibrated scope)")

        ne_str = input("How many independent trajectories? [16]: ").strip() or "16"
        rs_str = input("How many timesteps per trajectory? [2500]: ").strip() or "2500"
        dev_in = input("Device (cuda/cpu/auto) [auto]: ").strip().lower() or "auto"
        out_default = f"alphadynamics_{seq}_torsions.npz"
        out_path = input(f"Output file [{out_default}]: ").strip() or out_default
        plot_in = input("Save Ramachandran plot? [n=no, p=PNG, h=HTML interactive, b=both] [b]: ").strip().lower()
        if plot_in in ("", "b", "both"):
            want_plot, want_html = True, True
        elif plot_in in ("p", "png"):
            want_plot, want_html = True, False
        elif plot_in in ("h", "html"):
            want_plot, want_html = False, True
        else:
            want_plot, want_html = False, False

        try:
            n_ensemble = int(ne_str)
            rollout_steps = int(rs_str)
        except ValueError:
            print("ERROR: ensemble size and timesteps must be integers.")
            return 1

        # Pre-flight: verify we can write BEFORE running predict.
        # The model run takes minutes and we don't want to lose it on
        # PermissionError at the end (Windows users running from C:\ root).
        safe_npz = _safe_output_path(out_path, f"alphadynamics_{seq}_torsions.npz")
        if str(safe_npz) != str(out_path):
            out_path = str(safe_npz)

        device = None if dev_in == "auto" else dev_in

        print()
        print(f"Predicting torsions for {seq!r}: "
              f"{n_ensemble} trajectories x {rollout_steps} steps "
              f"on {device or 'auto'}...")
        print(f"Will save to: {out_path}")
        print()

        traj = predict_torsion_ensemble(
            seq,
            n_ensemble=n_ensemble,
            rollout_steps=rollout_steps,
            seed=42,
            device=device,
            show_progress=True,
        )

        import math
        import numpy as np

        np.savez_compressed(
            out_path,
            sequence=seq,
            torsions=traj,
            torsion_units="radians",
            torsion_axes="(ensemble, time, residues, [phi, psi])",
            n_ensemble=n_ensemble,
            rollout_steps=rollout_steps,
            model_name="ad_transfer_v2_clean",
            alphadynamics_version=__version__,
        )

        print(f"\nWrote {out_path}")
        print(f"Trajectory shape: {traj.shape}  (ensemble, time, residues, [phi,psi] rad)")
        print()

        # Quick basin analysis on the spot
        phi = np.degrees(traj[..., 0].flatten())
        psi = np.degrees(traj[..., 1].flatten())

        def _b(plo, phi_, slo, shi):
            return float(((phi >= plo) & (phi <= phi_) & (psi >= slo) & (psi <= shi)).mean()) * 100.0

        print("Ramachandran basin populations:")
        print(f"  alpha-helix R     (phi ~-60, psi ~-45):  {_b(-130,-30,-90,30):.1f}%")
        print(f"  beta-sheet        (phi ~-120, psi ~120): {_b(-180,-90,70,180):.1f}%")
        print(f"  PPII extended     (phi ~-60, psi ~140):  {_b(-90,-30,100,180):.1f}%")
        print(f"  alpha-helix L     (sterically forbidden): {_b(30,100,-10,90):.1f}%")
        print()

        if want_plot:
            png_path = out_path.replace(".npz", "_ramachandran.png")
            if not png_path.endswith(".png"):
                png_path += ".png"
            if _make_ramachandran_plot(traj, seq, png_path):
                print(f"Wrote static Ramachandran (PNG): {png_path}")

        if want_html:
            html_path = out_path.replace(".npz", "_ramachandran.html")
            if not html_path.endswith(".html"):
                html_path += ".html"
            if _make_ramachandran_html(traj, seq, html_path):
                print(f"Wrote interactive Ramachandran (HTML): {html_path}")
                print("  → open in browser: zoom, pan, hover for density values")

        if want_plot or want_html:
            print()

        print("Tip: load in Python with")
        print(f"    import numpy as np")
        print(f"    d = np.load('{out_path}', allow_pickle=True)")
        print(f"    traj = d['torsions']  # shape {traj.shape}")
        return 0

    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130


def main(argv: list[str] | None = None) -> int:
    # No subcommand at all → friendly interactive mode (what most users expect
    # when they just type `alphadynamics`).
    if argv is None and len(sys.argv) == 1:
        return _cmd_interactive()
    if argv is not None and len(argv) == 0:
        return _cmd_interactive()

    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
