"""Background check for newer version on PyPI.

Behavior:
- On import (after banner), make ONE HTTP GET to https://pypi.org/pypi/alphadynamics/json
- Compare with installed __version__
- If newer is available, print a yellow upgrade hint to stderr
- Cache the result for 24h in ~/.cache/alphadynamics/version_check.json
  (so we don't hit PyPI on every command)
- Silently disabled when:
    * ALPHADYNAMICS_NO_VERSION_CHECK=1
    * ALPHADYNAMICS_NO_BANNER=1
    * CI env var set
    * stderr is not a TTY (logs / pipes)
    * Network failure (we never block)

Network timeout: 2 seconds. If PyPI is slow / unreachable, we silently skip.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from . import __version__


PYPI_URL = "https://pypi.org/pypi/alphadynamics/json"
CACHE_TTL_SECONDS = 24 * 3600
TIMEOUT_SECONDS = 2.0


def _cache_path() -> Path:
    override = os.environ.get("ALPHADYNAMICS_CACHE_DIR")
    if override:
        cache_dir = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        cache_dir = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        cache_dir = cache_dir / "alphadynamics"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cache_dir / "version_check.json"


def _check_disabled() -> bool:
    for var in ("ALPHADYNAMICS_NO_VERSION_CHECK", "ALPHADYNAMICS_NO_BANNER"):
        if os.environ.get(var, "").strip() in {"1", "true", "True", "yes"}:
            return True
    if os.environ.get("CI"):
        return True
    try:
        if not sys.stderr.isatty():
            return True
    except Exception:
        return True
    return False


def _parse_version(v: str) -> tuple[int, ...]:
    """Simple semver tuple. Returns (0,0,0) on parse failure."""
    try:
        return tuple(int(p.split("+")[0].split("-")[0]) for p in v.split(".")[:3])
    except Exception:
        return (0, 0, 0)


def _fetch_latest_pypi_version(timeout: float = TIMEOUT_SECONDS) -> str | None:
    try:
        req = urllib.request.Request(
            PYPI_URL,
            headers={"User-Agent": f"alphadynamics/{__version__} version-check"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data.get("info", {}).get("version")
    except Exception:
        return None


def _print_upgrade_hint(latest: str) -> None:
    """Print yellow upgrade hint. Falls back to plain text if no color support."""
    try:
        from .banner import _colors_enabled
        color = _colors_enabled()
    except Exception:
        color = False

    yellow = "\033[33m" if color else ""
    bold = "\033[1m" if color else ""
    reset = "\033[0m" if color else ""

    print(
        f"\n{yellow}{bold}  A new alphadynamics version is available: "
        f"{__version__} -> {latest}{reset}\n"
        f"{yellow}  Update with: {bold}pip install --upgrade alphadynamics{reset}\n"
        f"  (set ALPHADYNAMICS_NO_VERSION_CHECK=1 to silence this)\n",
        file=sys.stderr,
        flush=True,
    )


def check_for_update() -> str | None:
    """Main entry. Returns latest version string if update available, else None."""
    if _check_disabled():
        return None

    cache_file = _cache_path()
    now = time.time()
    latest: str | None = None

    # Try cache first (avoid network on every command)
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if cache.get("checked_at", 0) > now - CACHE_TTL_SECONDS:
                latest = cache.get("latest_version")
        except Exception:
            pass

    # Cache miss / stale -> fetch
    if latest is None:
        latest = _fetch_latest_pypi_version()
        if latest:
            try:
                cache_file.write_text(json.dumps({
                    "checked_at": now,
                    "latest_version": latest,
                }))
            except Exception:
                pass

    if not latest:
        return None

    if _parse_version(latest) > _parse_version(__version__):
        _print_upgrade_hint(latest)
        return latest
    return None
