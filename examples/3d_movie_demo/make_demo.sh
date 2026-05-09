#!/bin/bash
# Reproducible AlphaDynamics 3D movie demo.
# Generates torsion .npz + backbone .pdb files for 3 sample peptides.
#
# Usage:
#   pip install -U alphadynamics  (>=0.4.0)
#   bash make_demo.sh
#   python -m http.server 8000   # serve viewer.html locally
#   # then open http://localhost:8000/viewer.html
#
set -e

cd "$(dirname "$0")"

echo "AlphaDynamics 3D movie demo — generating PDB files..."
echo

# 1. KLVFFAE — amyloid β16-22 (β-aggregating)
echo "[1/3] KLVFFAE (7 aa, amyloid β fragment)..."
alphadynamics predict --sequence KLVFFAE \
  --n-ensemble 4 --rollout-steps 200 --device cpu \
  -o /tmp/_demo_klvffae.npz
alphadynamics rebuild /tmp/_demo_klvffae.npz -s KLVFFAE \
  --frames 50 -o klvffae_amyloid.pdb --diagnostics

# 2. Trp-cage NLYIQWLKDGGPSSGRPPPS — mini-fold benchmark
echo
echo "[2/3] NLYIQWLKDGGPSSGRPPPS (Trp-cage, 20 aa)..."
alphadynamics predict --sequence NLYIQWLKDGGPSSGRPPPS \
  --n-ensemble 4 --rollout-steps 200 --device cpu \
  -o /tmp/_demo_trpcage.npz
alphadynamics rebuild /tmp/_demo_trpcage.npz -s NLYIQWLKDGGPSSGRPPPS \
  --frames 50 -o trpcage_minifold.pdb --diagnostics

# 3. AAAY — minimum 4-AA paper benchmark
echo
echo "[3/3] AAAY (4 aa, paper benchmark)..."
alphadynamics predict --sequence AAAY \
  --n-ensemble 4 --rollout-steps 200 --device cpu \
  -o /tmp/_demo_aaay.npz
alphadynamics rebuild /tmp/_demo_aaay.npz -s AAAY \
  --frames 50 -o aaay_4aa.pdb --diagnostics

echo
echo "Done. PDB files:"
ls -la *.pdb
echo
echo "View in browser:"
echo "  python -m http.server 8000"
echo "  open http://localhost:8000/viewer.html"
echo
echo "Or in PyMOL:"
echo "  pymol klvffae_amyloid.pdb"
