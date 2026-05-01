# Privacy Policy — AlphaDynamics

**Last updated:** 2026-05-01
**Maintainer:** Krzysztof Gwozdz &lt;krisss0gwo@gmail.com&gt;

This document describes the privacy practices of **AlphaDynamics**, both
the [`alphadynamics` Python package on PyPI](https://pypi.org/project/alphadynamics/)
and the [Claude Code plugin](https://github.com/krisss0mecom/AlphaDynamics)
distributed from this repository.

## TL;DR — no data collection

AlphaDynamics does not collect, transmit, store, or share any personal
data. It runs entirely on the user's local machine. There are no
analytics, no telemetry, no usage reporting, and no third-party
services involved beyond the one-time weight download described below.

## What information is processed locally

The user provides:

- A protein/peptide sequence string (one-letter amino acid code), and
- Configuration options (ensemble size, rollout steps, output path,
  device).

These inputs are passed to the local model and are never transmitted
off the user's machine.

The model output (a NumPy `.npz` trajectory file) is written to the
user's local filesystem at the path the user chooses (default:
the current working directory).

## Network activity

The package performs **exactly one type of network request**:

- On first use, it downloads pretrained model weights (~0.5 MB total)
  from the public
  [GitHub Releases page of this repository](https://github.com/krisss0mecom/AlphaDynamics/releases),
  via standard HTTPS, into `~/.cache/alphadynamics/weights/`.

This download happens once per machine (cached locally afterward) and
involves no authentication or user identifiers beyond the standard
HTTP request metadata that GitHub itself collects (see
[GitHub's privacy policy](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)).

The user can override the download URL with the
`ALPHADYNAMICS_RELEASE_URL` environment variable (e.g. to mirror
weights to a private CDN), and can override the cache directory with
`ALPHADYNAMICS_CACHE_DIR`.

No further network calls are made during prediction, evaluation,
or any other CLI/API operation.

## What is stored locally

- Pretrained weights cached at `~/.cache/alphadynamics/weights/`
  (override with `ALPHADYNAMICS_CACHE_DIR`).
- Output trajectory files at the path the user specifies.

Nothing else is written outside the user's chosen output paths.

## Third parties

None. The package depends on PyTorch, NumPy, and tqdm for runtime
computation — these are standard scientific Python libraries that
do not perform telemetry by default.

## Children's privacy

The package is a research tool for protein dynamics and does not
target or interact with children.

## Changes to this policy

Any future changes will be committed to this file in the repository
and announced in release notes.

## Contact

For questions about this policy:

**Krzysztof Gwozdz** &lt;krisss0gwo@gmail.com&gt;
[https://github.com/krisss0mecom/AlphaDynamics/issues](https://github.com/krisss0mecom/AlphaDynamics/issues)
