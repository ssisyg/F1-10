#!/usr/bin/env python3
"""Plan a racing line (x, y, speed) from a ROS occupancy map of a closed track.

Pipeline:
  1. Free space of the map -> centerline (skeleton of the corridor, smoothed)
  2. Corridor width to the left/right of the centerline
  3. Minimum-curvature line inside the corridor (bounded linear least squares,
     re-linearised a few times), keeping a safety margin to the walls
  4. Speed profile from the curvature (lateral acceleration limit) with forward/
     backward passes for the acceleration and braking limits

Output CSV columns: x, y, v   (map frame, meters and m/s). The first point is the
one closest to --start, and the direction of travel follows --start-heading.

Usage:
    python3 tools/plan_racing_line.py maps/track.yaml --out racing_line.csv --plot racing_line.png

The map must contain the empty track as a closed corridor (build it without
obstacles in the way). Requirements: numpy, scipy, scikit-image, pyyaml, pillow
(and matplotlib for --plot).
"""
import argparse
import math
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage, sparse
from scipy.interpolate import splev, splprep
from scipy.optimize import lsq_linear
from skimage.morphology import skeletonize


# --------------------------------------------------------------------------- map
class GridMap:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img = np.array(Image.open(img_path).convert('L'), dtype=float)
        img = np.flipud(img)  # row 0 = smallest y
        p = img / 255.0 if meta.get('negate', 0) else (255.0 - img) / 255.0
        self.resolution = float(meta['resolution'])
        self.origin = np.array(meta['origin'][:2], dtype=float)
        self.free = p < float(meta.get('free_thresh', 0.196))
        self.occupied = p > float(meta.get('occupied_thresh', 0.65))
        self.shape = self.free.shape

    def to_world(self, ij):
        """(col, row) pixel coordinates (float) -> world x, y."""
        ij = np.asarray(ij, dtype=float)
        return self.origin + (ij + 0.5) * self.resolution

    def to_pixel(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.floor((xy - self.origin) / self.resolution).astype(int)

    def is_free(self, xy):
        """Vectorised free-space lookup for an array of world points (N, 2)."""
        ij = self.to_pixel(xy)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[0])
        out = np.zeros(len(xy), dtype=bool)
        out[inside] = self.free[ij[inside, 1], ij[inside, 0]]
        return out


# ----------------------------------------------------------------- centerline
def largest_component(mask):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        raise ValueError('The map has no free space.')
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def prune_spurs(skel, max_iter=500):
    """Remove branches that end in an endpoint until only closed loops remain."""
    skel = skel.copy()
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    for _ in range(max_iter):
        neighbours = ndimage.convolve(skel.astype(int), kernel, mode='constant')
        endpoints = skel & (neighbours <= 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def order_loop(skel):
    """Walk the pixels of a one-pixel-wide closed loop and return them in order (col, row)."""
    pts = set(map(tuple, np.argwhere(skel)[:, ::-1]))  # (col, row)
    if not pts:
        raise ValueError('Skeleton is empty after pruning: no closed corridor found.')
    start = min(pts)
    order = [start]
    visited = {start}
    cur = start
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]  # 4-neighbours first
    while True:
        nxt = None
        for dx, dy in steps:
            cand = (cur[0] + dx, cur[1] + dy)
            if cand in pts and cand not in visited:
                nxt = cand
                break
        if nxt is None:
            break
        order.append(nxt)
        visited.add(nxt)
        cur = nxt
    coverage = len(visited) / len(pts)
    return np.array(order, dtype=float), coverage


def smooth_closed(points, ds, smoothing):
    """Periodic smoothing spline through points, resampled at uniform arc length ds."""
    pts = np.vstack([points, points[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing * len(pts), per=True)
    u = np.linspace(0, 1, 20 * len(pts))
    x, y = splev(u, tck)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    n = max(int(round(s[-1] / ds)), 8)
    target = np.linspace(0, s[-1], n, endpoint=False)
    u2 = np.interp(target, s, u)
    x, y = splev(u2, tck)
    dx, dy = splev(u2, tck, der=1)
    ddx, ddy = splev(u2, tck, der=2)
    return np.stack([x, y], 1), np.stack([dx, dy], 1), np.stack([ddx, ddy], 1)


def extract_centerline(gm, ds, smoothing=0.02):
    free = largest_component(gm.free)
    skel = prune_spurs(skeletonize(free))
    order, coverage = order_loop(skel)
    if coverage < 0.9:
        print('warning: only %.0f%% of the skeleton is one loop; check the map has a closed corridor' % (100 * coverage))
    world = gm.to_world(order)
    return smooth_closed(world, ds, smoothing)[0]


# --------------------------------------------------------------- path helpers
def tangents_normals(path):
    d = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], 1)  # left normal
    return t, n


def corridor_widths(gm, path, normals, max_w=4.0, step=0.02):
    """Distance from each path point to the nearest non-free cell, left (+n) and right (-n)."""
    widths = []
    for sign in (1.0, -1.0):
        w = np.full(len(path), max_w)
        alive = np.ones(len(path), dtype=bool)
        for k in range(1, int(max_w / step) + 1):
            d = k * step
            probe = path + sign * d * normals
            hit = alive & ~gm.is_free(probe)
            w[hit] = d - step
            alive &= ~hit
            if not alive.any():
                break
        widths.append(w)
    return widths[0], widths[1]


def second_diff_matrix(n):
    """Circular second difference operator (n x n)."""
    rows = np.arange(n)
    D = sparse.coo_matrix((np.full(n, -2.0), (rows, rows)), shape=(n, n))
    D = D + sparse.coo_matrix((np.ones(n), (rows, (rows - 1) % n)), shape=(n, n))
    D = D + sparse.coo_matrix((np.ones(n), (rows, (rows + 1) % n)), shape=(n, n))
    return D.tocsr()


def min_curvature_step(gm, base, margin, smooth_weight):
    """One linearised minimum-curvature solve around the path `base`."""
    n = len(base)
    _, normals = tangents_normals(base)
    wl, wr = corridor_widths(gm, base, normals)
    lo, hi = -(wr - margin), (wl - margin)
    if np.any(hi < lo):
        raise ValueError('Corridor is narrower than 2 * margin (%.2f m) at some points.' % (2 * margin))

    D2 = second_diff_matrix(n)
    # circular first difference of alpha (keeps the lateral offset smooth)
    rows = np.arange(n)
    D1 = sparse.coo_matrix((np.ones(n), (rows, rows)), shape=(n, n)) \
        - sparse.coo_matrix((np.ones(n), (rows, (rows + 1) % n)), shape=(n, n))
    D1 = D1.tocsr()

    Ax = D2 @ sparse.diags(normals[:, 0])
    Ay = D2 @ sparse.diags(normals[:, 1])
    A = sparse.vstack([Ax, Ay, smooth_weight * D1]).tocsr()
    b = np.concatenate([-(D2 @ base[:, 0]), -(D2 @ base[:, 1]), np.zeros(n)])
    res = lsq_linear(A, b, bounds=(lo, hi), lsmr_tol='auto', max_iter=500)
    return base + normals * res.x[:, None]


def resample(path, ds, smoothing=0.0):
    return smooth_closed(path, ds, smoothing)


# ------------------------------------------------------------- speed profile
def curvature(d1, d2):
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-9) ** 3


def speed_profile(path_len_step, kappa, v_max, a_lat, a_acc, a_dec):
    n = len(kappa)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)))
    for _ in range(2):  # two laps around the loop so the passes wrap correctly
        for i in range(2 * n):
            j, k = i % n, (i + 1) % n
            v[k] = min(v[k], math.sqrt(v[j] ** 2 + 2 * a_acc * path_len_step))
        for i in range(2 * n, 0, -1):
            j, k = i % n, (i - 1) % n
            v[k] = min(v[k], math.sqrt(v[j] ** 2 + 2 * a_dec * path_len_step))
    return v


# ------------------------------------------------------------------ main plan
def plan(gm, ds=0.1, margin=0.25, iterations=3, smooth_weight=0.3,
         v_max=1.5, a_lat=2.0, a_acc=1.5, a_dec=2.5, start=(0.0, 0.0), start_heading_deg=0.0):
    center = extract_centerline(gm, ds)
    path = center
    for _ in range(iterations):
        path = min_curvature_step(gm, path, margin, smooth_weight)
        path, _, _ = resample(path, ds, smoothing=0.0005)

    path, d1, d2 = resample(path, ds, smoothing=0.0005)
    ds_actual = float(np.mean(np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)))
    kappa = curvature(d1, d2)
    v = speed_profile(ds_actual, kappa, v_max, a_lat, a_acc, a_dec)

    # Start at the point closest to `start`, travelling along the start heading
    i0 = int(np.argmin(np.linalg.norm(path - np.asarray(start), axis=1)))
    heading = np.array([math.cos(math.radians(start_heading_deg)), math.sin(math.radians(start_heading_deg))])
    tangent = path[(i0 + 3) % len(path)] - path[i0]
    if np.dot(tangent, heading) < 0:
        path, kappa, v, center = path[::-1], kappa[::-1], v[::-1], center[::-1]
        i0 = int(np.argmin(np.linalg.norm(path - np.asarray(start), axis=1)))
    path, kappa, v = np.roll(path, -i0, axis=0), np.roll(kappa, -i0), np.roll(v, -i0)
    return {'center': center, 'path': path, 'kappa': kappa, 'v': v, 'ds': ds_actual}


def path_length(p):
    return float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))


def save_csv(filename, path, v):
    with open(filename, 'w') as f:
        f.write('x,y,v\n')
        for (x, y), s in zip(path, v):
            f.write('%.4f,%.4f,%.3f\n' % (x, y, s))


def save_plot(filename, gm, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ext = [gm.origin[0], gm.origin[0] + gm.shape[1] * gm.resolution,
           gm.origin[1], gm.origin[1] + gm.shape[0] * gm.resolution]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    img = np.where(gm.occupied, 0.0, np.where(gm.free, 1.0, 0.8))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=ext)
    c = result['center']
    ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), '--', color='tab:blue', lw=1, label='centerline')
    p, v = result['path'], result['v']
    sc = ax.scatter(p[:, 0], p[:, 1], c=v, s=6, cmap='viridis', label='racing line (color = speed)')
    ax.arrow(p[0, 0], p[0, 1], p[3, 0] - p[0, 0], p[3, 1] - p[0, 1], color='red', head_width=0.25)
    fig.colorbar(sc, ax=ax, label='speed [m/s]')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(filename, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_yaml')
    ap.add_argument('--out', default='racing_line.csv')
    ap.add_argument('--plot', default=None, help='optional PNG of the result')
    ap.add_argument('--ds', type=float, default=0.1, help='point spacing in meters')
    ap.add_argument('--margin', type=float, default=0.35, help='min distance to walls in meters (car half width + safety)')
    ap.add_argument('--v-max', type=float, default=1.5)
    ap.add_argument('--a-lat', type=float, default=2.0, help='max lateral acceleration m/s^2')
    ap.add_argument('--a-acc', type=float, default=1.5, help='max acceleration m/s^2')
    ap.add_argument('--a-dec', type=float, default=2.5, help='max braking m/s^2')
    ap.add_argument('--start', type=float, nargs=2, default=[0.0, 0.0], help='start position x y in the map frame')
    ap.add_argument('--start-heading', type=float, default=0.0, help='heading at the start in degrees')
    ap.add_argument('--centerline-only', action='store_true', help='skip the optimisation and use the centerline')
    a = ap.parse_args()

    gm = GridMap(a.map_yaml)
    kw = dict(ds=a.ds, margin=a.margin, v_max=a.v_max, a_lat=a.a_lat, a_acc=a.a_acc, a_dec=a.a_dec,
              start=tuple(a.start), start_heading_deg=a.start_heading)
    result = plan(gm, iterations=0 if a.centerline_only else 3, **kw)

    p, v = result['path'], result['v']
    edt = ndimage.distance_transform_edt(gm.free) * gm.resolution
    clearance = edt[gm.to_pixel(p)[:, 1], gm.to_pixel(p)[:, 0]]
    lap = path_length(p) / np.mean(v)
    print('points: %d   length: %.2f m (centerline %.2f m)' % (len(p), path_length(p), path_length(result['center'])))
    print('min wall clearance: %.2f m   max |curvature|: %.2f 1/m (min radius %.2f m)' %
          (clearance.min(), np.abs(result['kappa']).max(), 1 / max(np.abs(result['kappa']).max(), 1e-9)))
    print('speed: %.2f .. %.2f m/s   estimated lap time: %.1f s' % (v.min(), v.max(), lap))
    save_csv(a.out, p, v)
    print('wrote', a.out)
    if a.plot:
        save_plot(a.plot, gm, result)
        print('wrote', a.plot)


if __name__ == '__main__':
    main()
