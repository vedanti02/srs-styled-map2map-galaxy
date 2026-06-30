"""Power spectrum P(k) of a 3D scalar density field, computed from a 6-channel
displacement+velocity cube on a Lagrangian grid.

Two density estimators are supported:
  - "div"  : delta ≈ -∇·Ψ via finite differences (cheap, linear-theory).
  - "cic"  : positions = lattice + displacement → CIC mass assignment → delta = ρ/ρ̄ - 1.
            Needs a Lbox so positions live in physical units; if your data is
            normalized, pass --denormalize first or set --lbox to match.

Output Pk format: numpy structured array with fields (k, Pk, n_modes).
"""
import argparse
import glob
import os
import re
import numpy as np


def divergence_density(disp, lbox):
    """delta ≈ -∇·Ψ via central differences. disp: (3, N, N, N) in same units as lbox/N.
    Returns delta: (N, N, N).
    """
    N = disp.shape[-1]
    cell = lbox / N
    # central difference with periodic wrap
    dpsi_dx = (np.roll(disp[0], -1, axis=0) - np.roll(disp[0], 1, axis=0)) / (2 * cell)
    dpsi_dy = (np.roll(disp[1], -1, axis=1) - np.roll(disp[1], 1, axis=1)) / (2 * cell)
    dpsi_dz = (np.roll(disp[2], -1, axis=2) - np.roll(disp[2], 1, axis=2)) / (2 * cell)
    return -(dpsi_dx + dpsi_dy + dpsi_dz)


def cic_density(disp, lbox):
    """delta from CIC assignment of displaced lattice particles. disp must be in
    the same length units as lbox.
    """
    Nc, N, _, _ = disp.shape
    cell = lbox / N
    grid_1d = np.arange(N) * cell + 0.5 * cell
    # Lagrangian lattice positions; broadcast to full (3, N, N, N).
    lx = grid_1d[:, None, None]
    ly = grid_1d[None, :, None]
    lz = grid_1d[None, None, :]
    px = (disp[0] + lx) % lbox
    py = (disp[1] + ly) % lbox
    pz = (disp[2] + lz) % lbox

    pos = np.stack([px, py, pz], axis=0).reshape(3, -1) / cell  # in cell units
    rho = np.zeros((N, N, N), dtype=np.float64)

    i = np.floor(pos).astype(np.int64) % N
    f = pos - np.floor(pos)
    ip1 = (i + 1) % N
    one = 1.0 - f

    # 8 corners
    for dx, ix, wx in [(0, i[0], one[0]), (1, ip1[0], f[0])]:
        for dy, iy, wy in [(0, i[1], one[1]), (1, ip1[1], f[1])]:
            for dz, iz, wz in [(0, i[2], one[2]), (1, ip1[2], f[2])]:
                w = wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)

    return rho / rho.mean() - 1.0


def power_spectrum(delta, lbox, n_bins=32):
    """Compute spherically averaged P(k) of `delta` on an N^3 grid.
    Returns (k_centers, Pk, n_modes).
    """
    N = delta.shape[-1]
    delta_k = np.fft.fftn(delta) / N**3                  # discrete normalization
    pk_grid = (np.abs(delta_k) ** 2) * (lbox ** 3)       # P(k) in (Mpc/h)^3

    kx = np.fft.fftfreq(N, d=lbox / N) * 2 * np.pi
    kgrid = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kx[None, None, :] ** 2)

    k_nyq = np.pi * N / lbox
    k_min = 2 * np.pi / lbox
    bins = np.logspace(np.log10(k_min * 1.01), np.log10(k_nyq), n_bins + 1)

    n_modes, _ = np.histogram(kgrid, bins=bins)
    pk_sum, _ = np.histogram(kgrid, bins=bins, weights=pk_grid)
    k_sum, _ = np.histogram(kgrid, bins=bins, weights=kgrid)

    mask = n_modes > 0
    k_centers = np.zeros(n_bins)
    pk = np.zeros(n_bins)
    k_centers[mask] = k_sum[mask] / n_modes[mask]
    pk[mask] = pk_sum[mask] / n_modes[mask]
    return k_centers, pk, n_modes


def counts_to_overdensity(cube, nbar=None):
    """1-channel halo-count cube → overdensity delta = n/nbar - 1 (per-box mean)."""
    c = np.asarray(cube, dtype=np.float64)
    if c.ndim == 4:        # (1, N, N, N) -> (N, N, N)
        c = c[0]
    if nbar is None:
        nbar = max(c.mean(), 1e-6)
    return c / nbar - 1.0


def cube_pk_counts(cube, lbox, n_bins=32, nbar=None):
    """Eulerian count field → P(k) on its overdensity (no displacement transform)."""
    return power_spectrum(counts_to_overdensity(cube, nbar), lbox, n_bins=n_bins)


def cube_pk(cube, lbox, estimator="div", n_bins=32):
    """One-shot cube → P(k). estimator 'counts' = 1-ch density; 'div'/'cic' = 6-ch displacement."""
    if estimator == "counts":
        return cube_pk_counts(cube, lbox, n_bins=n_bins)
    disp = cube[:3]
    if estimator == "div":
        delta = divergence_density(disp, lbox)
    elif estimator == "cic":
        delta = cic_density(disp, lbox)
    else:
        raise ValueError(estimator)
    return power_spectrum(delta, lbox, n_bins=n_bins)


# ---------------- CLI ----------------

_SET_RE = re.compile(r"set(\d+)")


def _load_cube_from_stitched(stitched_root, sid, kind, snap):
    base = os.path.join(stitched_root, f"set{sid}_{kind}", snap)
    disp = np.load(os.path.join(base, "disp.npy"))
    vel = np.load(os.path.join(base, "vel.npy"))
    return np.concatenate([disp, vel], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stitched", "transformed"], required=True,
                   help="stitched: read pairs from stitched/ root. transformed: read .npy outputs of transform.py.")
    p.add_argument("--input", required=True,
                   help="stitched/ root, OR directory of transform.py outputs.")
    p.add_argument("--output", required=True, help="Output directory for Pk arrays.")
    p.add_argument("--lbox", type=float, default=1000.0, help="Box size in Mpc/h.")
    p.add_argument("--estimator", choices=["div", "cic", "counts"], default="div")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--snap", default="PART_009")
    p.add_argument("--kind", choices=["quijote", "quijotelike"], default="quijote",
                   help="When mode=stitched, which side of the pair to use.")
    p.add_argument("--ids", nargs="*", type=int, default=None,
                   help="Restrict to these set IDs (default: all found).")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "stitched":
        if args.ids is None:
            entries = os.listdir(args.input)
            ids = sorted({
                int(_SET_RE.match(e).group(1))
                for e in entries
                if _SET_RE.match(e) and e.endswith(f"_{args.kind}")
            })
        else:
            ids = args.ids
        for sid in ids:
            cube = _load_cube_from_stitched(args.input, sid, args.kind, args.snap)
            k, pk, nm = cube_pk(cube, args.lbox, args.estimator, args.n_bins)
            out = os.path.join(args.output, f"pk_{args.kind}_set{sid}.npz")
            np.savez(out, k=k, pk=pk, n_modes=nm)
        print(f"wrote {len(ids)} Pk files for kind={args.kind}")

    else:  # transformed
        files = sorted(glob.glob(os.path.join(args.input, "*.npy")))
        if args.ids is not None:
            files = [f for f in files if any(f"set{sid}_" in f for sid in args.ids)]
        for f in files:
            cube = np.load(f)
            k, pk, nm = cube_pk(cube, args.lbox, args.estimator, args.n_bins)
            base = os.path.splitext(os.path.basename(f))[0]
            np.savez(os.path.join(args.output, f"pk_{base}.npz"), k=k, pk=pk, n_modes=nm)
        print(f"wrote {len(files)} Pk files")


if __name__ == "__main__":
    main()
