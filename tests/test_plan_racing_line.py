"""Tests for the racing line planner. Run with:  python3 tests/test_plan_racing_line.py"""
import os
import sys
import tempfile

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import make_test_track
import plan_racing_line as pl
from PIL import Image


def build_map(width, tmpdir):
    img, (x0, y0) = make_test_track.make_map(width, 0.05, 0.15)
    base = os.path.join(tmpdir, 'track')
    Image.fromarray(img).save(base + '.pgm')
    with open(base + '.yaml', 'w') as f:
        f.write('image: track.pgm\nresolution: 0.05\norigin: [%f, %f, 0.0]\nnegate: 0\n'
                'occupied_thresh: 0.65\nfree_thresh: 0.196\n' % (x0, y0))
    return pl.GridMap(base + '.yaml')


def check(width, margin, v_max, a_lat, a_acc, a_dec):
    with tempfile.TemporaryDirectory() as d:
        gm = build_map(width, d)
        r = pl.plan(gm, margin=margin, v_max=v_max, a_lat=a_lat, a_acc=a_acc, a_dec=a_dec)
    p, v, k, ds = r['path'], r['v'], r['kappa'], r['ds']

    # 1. inside the corridor with the requested margin (small tolerance for map resolution)
    edt = ndimage.distance_transform_edt(gm.free) * gm.resolution
    ij = gm.to_pixel(p)
    clearance = edt[ij[:, 1], ij[:, 0]]
    assert clearance.min() >= margin - 0.05, 'clearance %.3f < margin %.2f' % (clearance.min(), margin)

    # 2. closed loop with even spacing
    gap = np.linalg.norm(p[0] - p[-1])
    assert gap < 2 * ds, 'loop not closed (gap %.3f)' % gap

    # 3. starts near the origin (the racing line may sit up to half a corridor away from the
    #    centerline) heading +x
    assert np.linalg.norm(p[0]) < width / 2, 'start %s is not near the origin' % p[0]
    assert (p[3] - p[0])[0] > 0, 'start heading is not +x'

    # 4. no longer than the centerline, and smoother
    assert pl.path_length(p) <= pl.path_length(r['center']) + 1e-6
    kc = pl.curvature(*pl.smooth_closed(r['center'], ds, 0.0005)[1:])
    assert np.abs(k).max() <= np.abs(kc).max() * 1.05

    # 5. speed profile respects its limits
    assert v.max() <= v_max + 1e-6 and v.min() > 0
    assert np.all(v ** 2 <= a_lat / np.maximum(np.abs(k), 1e-6) + 1e-6)
    dv2 = (np.roll(v, -1) ** 2 - v ** 2) / (2 * ds)
    assert dv2.max() <= a_acc * 1.02 and dv2.min() >= -a_dec * 1.02, (dv2.min(), dv2.max())
    return pl.path_length(p), pl.path_length(r['center']), clearance.min(), v.min(), v.max()


if __name__ == '__main__':
    for width in (1.8, 1.4):
        L, Lc, cl, vmin, vmax = check(width, 0.25, 3.0, 2.0, 1.5, 2.5)
        print('PASS corridor %.1f m: line %.2f m vs centerline %.2f m, clearance %.2f m, speed %.2f..%.2f m/s'
              % (width, L, Lc, cl, vmin, vmax))
    # corridor narrower than 2 * margin must be rejected, not silently produce a bad line
    try:
        check(0.4, 0.25, 3.0, 2.0, 1.5, 2.5)
        raise SystemExit('FAIL: narrow corridor was accepted')
    except ValueError as e:
        print('PASS narrow corridor rejected:', e)
