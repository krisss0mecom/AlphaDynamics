"""ASCII banner shown on first import / CLI entry."""
from __future__ import annotations

import os
import sys

from . import __version__

_ASCII_LOGO = r"""
   _____  __        __         ____                                _
  /  _  \|  |  ____ |  |__  _____  ___ _   _ _ __   __ _ _ __ ___ (_) ___ ___
 /  /_\  \  | / __ \|  __ \(____ \/ _ \ | | | '_ \ / _` | '_ ` _ \| |/ __/ __|
/    |    \ |/ /_/ /|  ___/__   \\(_) ) |_| | | | | (_| | | | | | | | (__\__ \
\____|__  /__\____/ |_|     \____/\___/ \__, |_| |_|\__,_|_| |_| |_|_|\___|___/
        \/                              |___/
"""

_TAGLINE = (
    "    Compact sequence-only neural propagator for protein torsion dynamics\n"
    "    2.39x lower JSD than Microsoft Timewarp · 3000x fewer parameters\n"
)

_AUTHOR_LINE = (
    "    Created by Krzysztof Gwozdz  <krisss0gwo@gmail.com>\n"
    "    https://github.com/krisss0mecom/AlphaDynamics\n"
    "    Licensed under Apache License 2.0\n"
)


def banner_text(short: bool = False) -> str:
    """Return the banner as a string."""
    if short:
        return f"AlphaDynamics v{__version__} — by Krzysztof Gwozdz (Apache-2.0)"
    parts = [_ASCII_LOGO, _TAGLINE, f"    Version {__version__}\n", _AUTHOR_LINE]
    return "\n".join(parts)


def print_banner(short: bool = False, file=None) -> None:
    """Print banner unless ALPHADYNAMICS_NO_BANNER=1 is set."""
    if os.environ.get("ALPHADYNAMICS_NO_BANNER", "").strip() in {"1", "true", "True", "yes"}:
        return
    print(banner_text(short=short), file=file or sys.stderr, flush=True)
