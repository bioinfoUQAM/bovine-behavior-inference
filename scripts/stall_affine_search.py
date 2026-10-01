"""
stall_affine_search.py
======================
Thesis sections 3.6.1 and 4.3.2 (affine registration - parameter search).

Numba-accelerated exhaustive grid search over a 7-parameter affine transform
(tx, ty, rotation, sx, sy, skew_x, skew_y) that maps the normalised stall
corners of a video ("floating" stall, see 06_normalize_to_stall.py) onto the
canonical stall (x in [-0.8, 0.8], y in [-1, 1], i.e. a 4:5 width:height
ratio). The cost is the sum of squared distances between the transformed
floating corners and the canonical corners. All local minima of the grid
(7-D 3^7-1 neighbourhood) are listed and the global minimum is returned.

Importable module - used by 07_register_bboxes.py and the QC scripts.

Usage (standalone test):
    python stall_affine_search.py

CLI arguments (standalone mode):
    --floating     8 floats: x1 y1 x2 y2 x3 y3 x4 y4  (LT LB RB RT, normalised)
    --standard     8 floats: x1 y1 x2 y2 x3 y3 x4 y4  (LT LB RB RT, canonical)
    --nr           Grid resolution per dimension (default: 7)
"""

import argparse
import time
import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Numba kernels (JIT-compiled, cached)
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def _dist_for_params(points, target, tx, ty, rot_c, rot_s, sx, sy, skx, sky):
    a00 = sx * (rot_c + sky * rot_s)
    a01 = sx * (skx * rot_c + rot_s)
    a10 = sy * (-rot_s + sky * rot_c)
    a11 = sy * (-skx * rot_s + rot_c)
    a02 = tx
    a12 = ty
    s = 0.0
    for i in range(points.shape[0]):
        x = points[i, 0];  y = points[i, 1]
        xt = x * a00 + y * a01 + a02
        yt = x * a10 + y * a11 + a12
        dx = xt - target[i, 0];  dy = yt - target[i, 1]
        s += dx * dx + dy * dy
    return s


@nb.njit(cache=True)
def _unravel_index(idx, shape, out):
    for i in range(len(shape) - 1, -1, -1):
        out[i] = idx % shape[i]
        idx //= shape[i]


@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_dist_grid(dist_grid, points, target,
                       tx_vals, ty_vals, rot_cos, rot_sin,
                       sx_vals, sy_vals, skx_vals, sky_vals):
    shape = dist_grid.shape
    total = 1
    for k in range(len(shape)):
        total *= shape[k]
    for flat in nb.prange(total):
        idx_vec = np.empty(7, dtype=np.int64)
        _unravel_index(flat, shape, idx_vec)
        ix0, ix1, ix2, ix3, ix4, ix5, ix6 = (
            idx_vec[0], idx_vec[1], idx_vec[2], idx_vec[3],
            idx_vec[4], idx_vec[5], idx_vec[6])
        d = _dist_for_params(
            points, target,
            tx_vals[ix0], ty_vals[ix1],
            rot_cos[ix2], rot_sin[ix2],
            sx_vals[ix3], sy_vals[ix4],
            skx_vals[ix5], sky_vals[ix6])
        dist_grid[ix0, ix1, ix2, ix3, ix4, ix5, ix6] = d


@nb.njit(parallel=True, cache=True)
def _compute_minima_mask(dist_grid, offsets):
    shape = dist_grid.shape
    total = 1
    for k in range(len(shape)):
        total *= shape[k]
    mask = np.ones(shape, dtype=np.uint8)
    for flat in nb.prange(total):
        idx_vec = np.empty(7, dtype=np.int64)
        _unravel_index(flat, shape, idx_vec)
        center_val = dist_grid[idx_vec[0], idx_vec[1], idx_vec[2],
                               idx_vec[3], idx_vec[4], idx_vec[5], idx_vec[6]]
        for k in range(offsets.shape[0]):
            n0 = idx_vec[0] + offsets[k, 0];  n1 = idx_vec[1] + offsets[k, 1]
            n2 = idx_vec[2] + offsets[k, 2];  n3 = idx_vec[3] + offsets[k, 3]
            n4 = idx_vec[4] + offsets[k, 4];  n5 = idx_vec[5] + offsets[k, 5]
            n6 = idx_vec[6] + offsets[k, 6]
            Nr = shape[0]
            if (n0 < 0 or n0 >= Nr or n1 < 0 or n1 >= Nr or
                n2 < 0 or n2 >= Nr or n3 < 0 or n3 >= Nr or
                n4 < 0 or n4 >= Nr or n5 < 0 or n5 >= Nr or
                n6 < 0 or n6 >= Nr):
                continue
            v = dist_grid[n0, n1, n2, n3, n4, n5, n6]
            if v < center_val:
                mask[idx_vec[0], idx_vec[1], idx_vec[2],
                     idx_vec[3], idx_vec[4], idx_vec[5], idx_vec[6]] = 0
                break
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level parameter arrays (populated by find_local_minima)
tx_vals = ty_vals = rot_vals = sx_vals = sy_vals = skx_vals = sky_vals = None


def find_local_minima(floating, standard_stall, Nr=7,
                      tx_range=(-0.5, 0.5), ty_range=(-0.5, 0.5),
                      rot_range=(-0.5, 0.5),
                      sx_range=(0.5, 1.5), sy_range=(0.5, 1.5),
                      skx_range=(-0.5, 0.5), sky_range=(-0.5, 0.5)):
    """
    Search a 7-DOF affine parameter grid for the transform that minimises the
    sum of squared distances between transformed floating points and target.

    Parameters
    ----------
    floating : array-like, shape (N, 2)
        Normalised stall corners from the video (LT, LB, RB, RT).
    standard_stall : array-like, shape (N, 2)
        Canonical stall corners to align to.
    Nr : int
        Grid resolution per dimension (default: 7). Total cells = Nr^7.
    *_range : (float, float)
        Search range [min, max] for each parameter.

    Returns
    -------
    mins : list of (index_7d, value)
        All local minima found in the grid.
    (best_idx, best_val) : tuple
        Index and value of the global best (argmin over the full grid).
    """
    global tx_vals, ty_vals, rot_vals, sx_vals, sy_vals, skx_vals, sky_vals

    Nt = 7
    floating       = np.ascontiguousarray(floating,       dtype=np.float64)
    standard_stall = np.ascontiguousarray(standard_stall, dtype=np.float64)

    if floating.shape != standard_stall.shape or floating.ndim != 2 or floating.shape[1] != 2:
        raise ValueError("floating and standard_stall must be matching (N, 2) arrays.")
    if Nr < 2:
        raise ValueError("Nr must be >= 2.")

    est_cells = Nr ** Nt
    print(f"Grid: Nr={Nr}, cells={est_cells} (~{est_cells * 8 / 1e6:.1f} MB)")

    tx_vals  = np.linspace(*tx_range,  Nr)
    ty_vals  = np.linspace(*ty_range,  Nr)
    rot_vals = np.linspace(*rot_range, Nr)
    sx_vals  = np.linspace(*sx_range,  Nr)
    sy_vals  = np.linspace(*sy_range,  Nr)
    skx_vals = np.linspace(*skx_range, Nr)
    sky_vals = np.linspace(*sky_range, Nr)

    rot_cos = np.cos(rot_vals)
    rot_sin = np.sin(rot_vals)

    # Build [-1,0,1]^7 neighborhood offsets (excluding center)
    grid    = np.array([-1, 0, 1], dtype=np.int8)
    offsets = np.empty((3**Nt - 1, Nt), dtype=np.int8)
    idx = 0
    for a in grid:
        for b in grid:
            for c in grid:
                for d in grid:
                    for e in grid:
                        for f in grid:
                            for g in grid:
                                if (a | b | c | d | e | f | g) != 0:
                                    offsets[idx] = [a, b, c, d, e, f, g]
                                    idx += 1

    dist_grid = np.empty((Nr,) * Nt, dtype=np.float64)

    print("Computing distance grid (parallel)...")
    t0 = time.time()
    _compute_dist_grid(dist_grid, floating, standard_stall,
                       tx_vals, ty_vals, rot_cos, rot_sin,
                       sx_vals, sy_vals, skx_vals, sky_vals)
    print(f"  Done in {time.time() - t0:.2f}s")

    print("Searching local minima (parallel)...")
    t1 = time.time()
    mask = _compute_minima_mask(dist_grid, offsets)
    print(f"  Done in {time.time() - t1:.2f}s")

    mins = []
    it = np.nditer(mask, flags=["multi_index"])
    for m in it:
        if m.item() == 1:
            idx7 = it.multi_index
            mins.append((idx7, float(dist_grid[idx7])))

    best_flat = int(np.argmin(dist_grid))
    best_idx  = np.unravel_index(best_flat, dist_grid.shape)
    best_val  = float(dist_grid[best_idx])

    print(f"Local minima found: {len(mins)}  |  Best value: {best_val:.6f}")
    return mins, (best_idx, best_val)


# ---------------------------------------------------------------------------
# Convenience helpers (shared by 07_register_bboxes.py and the QC scripts)
# ---------------------------------------------------------------------------

STANDARD_STALL = np.array([[-0.8, -1.0], [-0.8, 1.0], [0.8, 1.0], [0.8, -1.0]],
                          dtype=np.float64)          # LT, LB, RB, RT


def affine_coeffs_from_params(tx, ty, rot_c, rot_s, sx, sy, skx, sky):
    """2x3 affine coefficients (a00, a01, a02, a10, a11, a12) of the search parametrisation."""
    a00 = sx * (rot_c + sky * rot_s)
    a01 = sx * (skx * rot_c + rot_s)
    a10 = sy * (-rot_s + sky * rot_c)
    a11 = sy * (-skx * rot_s + rot_c)
    return a00, a01, tx, a10, a11, ty


def apply_affine(points_xy, a00, a01, a02, a10, a11, a12):
    x, y = points_xy[:, 0], points_xy[:, 1]
    return np.stack([x * a00 + y * a01 + a02, x * a10 + y * a11 + a12], axis=1)


def best_affine(floating, standard=STANDARD_STALL, Nr=7):
    """
    Run the grid search and return (params, coeffs, cost) for the global minimum.
    params = dict(tx, ty, rot, sx, sy, skx, sky); coeffs = 2x3 affine coefficients.
    """
    _, (bi, cost) = find_local_minima(floating, standard, Nr=Nr)
    params = dict(tx=tx_vals[bi[0]], ty=ty_vals[bi[1]], rot=rot_vals[bi[2]],
                  sx=sx_vals[bi[3]], sy=sy_vals[bi[4]],
                  skx=skx_vals[bi[5]], sky=sky_vals[bi[6]])
    params = {k: float(v) for k, v in params.items()}
    coeffs = affine_coeffs_from_params(params["tx"], params["ty"],
                                       np.cos(params["rot"]), np.sin(params["rot"]),
                                       params["sx"], params["sy"],
                                       params["skx"], params["sky"])
    return params, coeffs, cost


# ---------------------------------------------------------------------------
# CLI entry point (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="7-DOF affine grid search for stall alignment.")
    parser.add_argument("--floating", type=float, nargs=8,
                        metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                        help="Floating stall corners (LT LB RB RT, normalised).")
    parser.add_argument("--standard", type=float, nargs=8,
                        default=[-0.8,-1.0, -0.8,1.0, 0.8,1.0, 0.8,-1.0],
                        metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                        help="Standard stall corners (default: ratio 4:5).")
    parser.add_argument("--nr", type=int, default=7,
                        help="Grid resolution per dimension (default: 7).")
    args = parser.parse_args()

    if args.floating is None:
        # built-in test case
        floating = np.array([(-0.98727,-0.979144),(-0.995843,0.979144),
                              (0.98727,0.979144),(1.027275,-1.062267)], dtype=np.float64)
        print("Using built-in test data.")
    else:
        floating = np.array(list(zip(args.floating[0::2], args.floating[1::2])), dtype=np.float64)

    standard = np.array(list(zip(args.standard[0::2], args.standard[1::2])), dtype=np.float64)

    t0 = time.time()
    mins, (best_idx, best_val) = find_local_minima(floating, standard, Nr=args.nr)
    print(f"\nBest params:")
    print(f"  value={best_val:.6f}")
    print(f"  tx={tx_vals[best_idx[0]]:.4f}  ty={ty_vals[best_idx[1]]:.4f}"
          f"  rot={rot_vals[best_idx[2]]:.4f}")
    print(f"  sx={sx_vals[best_idx[3]]:.4f}  sy={sy_vals[best_idx[4]]:.4f}"
          f"  skx={skx_vals[best_idx[5]]:.4f}  sky={sky_vals[best_idx[6]]:.4f}")
    print(f"Total time: {time.time() - t0:.3f}s")
