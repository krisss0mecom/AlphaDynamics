"""ASCII banner shown on first import / CLI entry."""
from __future__ import annotations

import os
import sys

from . import __version__


# ANSI color codes (no extra deps). Disabled if NO_COLOR is set or stderr
# is not a TTY (so piping into a file gets clean text).
def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ALPHADYNAMICS_NO_COLOR", "").strip() in {"1", "true", "True", "yes"}:
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


_C_CYAN = "\033[36m"
_C_BOLD = "\033[1m"
_C_DIM = "\033[2m"
_C_RESET = "\033[0m"


# ANSI Shadow font, "Alpha" stacked above "Dynamics" so the brand is
# clearly readable on any 80-column terminal. Same visual family as
# Claude Code's block banner. Generated with `pyfiglet -f ansi_shadow`.
_ASCII_LOGO = r"""
 █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗
██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗
███████║██║     ██████╔╝███████║███████║
██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║
██║  ██║███████╗██║     ██║  ██║██║  ██║
╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝

██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗██╗ ██████╗███████╗
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██║██╔════╝██╔════╝
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║██║     ███████╗
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║██║     ╚════██║
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║██║╚██████╗███████║
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝ ╚═════╝╚══════╝
"""

_TAGLINE = (
    "  Compact sequence-only neural propagator for protein torsion dynamics\n"
    "  2.39x lower JSD than Microsoft Timewarp · 3000x fewer parameters\n"
)

_AUTHOR_TEMPLATE = (
    "  Created by Krzysztof Gwozdz  <krisss0gwo@gmail.com>\n"
    "  https://github.com/krisss0mecom/AlphaDynamics\n"
    "  Apache License 2.0\n"
)


def banner_text(short: bool = False) -> str:
    """Return the banner as a string. short=True for one-liner."""
    if short:
        return f"AlphaDynamics v{__version__} — by Krzysztof Gwozdz (Apache-2.0)"

    if _colors_enabled():
        logo = f"{_C_BOLD}{_C_CYAN}{_ASCII_LOGO}{_C_RESET}"
        version_line = f"  {_C_DIM}Version {__version__}{_C_RESET}\n"
    else:
        logo = _ASCII_LOGO
        version_line = f"  Version {__version__}\n"

    parts = [logo, _TAGLINE, version_line, _AUTHOR_TEMPLATE]
    return "\n".join(parts)


def print_banner(short: bool = False, file=None) -> None:
    """Print banner unless ALPHADYNAMICS_NO_BANNER=1 is set."""
    if os.environ.get("ALPHADYNAMICS_NO_BANNER", "").strip() in {"1", "true", "True", "yes"}:
        return
    print(banner_text(short=short), file=file or sys.stderr, flush=True)
