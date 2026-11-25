#!/usr/bin/env python3
"""Generate a closed-loop test track as a ROS occupancy map (pgm + yaml).

The map follows the map_server convention: free = 254, walls = 0, unknown = 205.
It is only used for dry runs without the real car (fake_car.py) and for testing
the racing line planner. On the real car you build the map with SLAM instead.

Usage:
    python3 tools/make_test_track.py --out maps/test_track
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def centerline(n=4000):
    """A smooth closed curve (meters), starting at the origin heading +x. Minimum radius about 1.9 m."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 8.5 * np.cos(t)
    y = 5.0 * np.sin(t) + 0.5 * np.sin(2 * t)
    c = np.stack([x, y], axis=1)

    # Put the first point at the map origin with the car heading along +x, like a real
    # mapping run where the map frame is defined by the car's starting pose.
    heading = c[1] - c[0]
    a = -np.arctan2(heading[1], heading[0])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (c - c[0]) @ rot.T


def make_map(width, resolution, wall_thickness, margin=2.0):
    c = centerline()
    x0, y0 = c.min(axis=0) - (width / 2 + margin)
    x1, y1 = c.max(axis=0) + (width / 2 + margin)
    w = int(np.ceil((x1 - x0) / resolution))
    h = int(np.ceil((y1 - y0) / resolution))

    # Rasterize the centerline, then use a distance transform to get the corridor
    center_mask = np.zeros((h, w), dtype=bool)
    ij = np.floor((c - [x0, y0]) / resolution).astype(int)
    center_mask[ij[:, 1], ij[:, 0]] = True
    dist = ndimage.distance_transform_edt(~center_mask) * resolution  # meters to centerline

    free = dist <= width / 2
    wall = (dist > width / 2) & (dist <= width / 2 + wall_thickness)

    img = np.full((h, w), 205, dtype=np.uint8)  # unknown
    img[free] = 254
    img[wall] = 0
    img = np.flipud(img)  # image row 0 is the top of the map (max y)
    return img, (x0, y0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='maps/test_track', help='output path without extension')
    p.add_argument('--width', type=float, default=1.8, help='corridor width in meters')
    p.add_argument('--resolution', type=float, default=0.05)
    p.add_argument('--wall', type=float, default=0.15, help='wall thickness in meters')
    a = p.parse_args()

    img, (x0, y0) = make_map(a.width, a.resolution, a.wall)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(img).save(a.out + '.pgm')
    with open(a.out + '.yaml', 'w') as f:
        f.write('image: %s\n' % (os.path.basename(a.out) + '.pgm'))
        f.write('resolution: %.3f\n' % a.resolution)
        f.write('origin: [%.3f, %.3f, 0.0]\n' % (x0, y0))
        f.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    print('wrote %s.pgm / .yaml  (%dx%d px, corridor %.1f m)' % (a.out, img.shape[1], img.shape[0], a.width))


if __name__ == '__main__':
    main()
