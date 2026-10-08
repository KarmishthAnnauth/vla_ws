#!/usr/bin/env python3
"""Make a cleaned copy of an occupancy map: keep only a region, straighten walls
and boxes, drop stray points.

Inside the region polygon (map metres):
  * big occupied objects (>= --min-structure cells) are replaced by straight
    lines (walls) or rectangles (boxes) fitted to their cells;
  * small objects are kept only if the recorded drives saw them every time
    (posts on the islands), so the person who walked with the car during
    mapping disappears;
  * everything that is not occupied becomes free, except the inside of boxes.
Outside the region everything becomes unknown. The image size, resolution and
origin stay the same, so the map frame and the aligned Lanelet2 map still apply.

    source /opt/ros/humble/setup.bash
    python3 scripts/clean_map.py --map src/particle_filter/maps/track_20260930.yaml \
        --region "x1,y1 x2,y2 ..." --out src/particle_filter/maps/track_20260930_clean

The "seen every time" test needs the bags (default data/bags/*, see
align_osm_to_map.py); with --no-bags every small object is kept.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_osm_to_map import GridMap, static_cells_and_paths  # noqa: E402

FREE, UNKNOWN, OCCUPIED = 254, 205, 0
LINE_TOL = 0.11          # a cell belongs to a wall line if its centre is this close [m] (walls are 2-3 cells thick)
MIN_LINE_CELLS = 12
MAX_LINE_GAP = 0.5       # a longer gap splits a wall line into two segments [m]
MIN_BOX_SIDE = 0.5       # shorter structures are treated as walls [m]


def cell_polygon_mask(grid, verts):
    cols = (np.asarray(verts)[:, 0] - grid.ox) / grid.res
    rows = grid.h - (np.asarray(verts)[:, 1] - grid.oy) / grid.res
    mask = np.zeros((grid.h, grid.w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.c_[cols, rows]).astype(np.int32)], 1)
    return mask.astype(bool)


def to_cells(grid, xy):
    """metres -> (col, row) pixel coordinates for cv2 drawing (cell centres at .0)."""
    return np.c_[(xy[:, 0] - grid.ox) / grid.res - 0.5, grid.h - 0.5 - (xy[:, 1] - grid.oy) / grid.res]


def ransac_lines(points, rng):
    """Greedy RANSAC: returns (segments, leftover points); a segment is (p0, p1) in metres."""
    segments = []
    pts = points.copy()
    while len(pts) >= MIN_LINE_CELLS:
        best = None
        for _ in range(400):
            a, b = pts[rng.choice(len(pts), 2, replace=False)]
            d = b - a
            if np.linalg.norm(d) < 0.15:
                continue
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            inl = np.abs((pts - a) @ n) < LINE_TOL
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < MIN_LINE_CELLS:
            break
        # refine with PCA on the inliers, then split along the line at gaps
        c = pts[best].mean(0)
        u = np.linalg.svd(pts[best] - c)[2][0]
        n = np.array([-u[1], u[0]])
        inl = np.abs((pts - c) @ n) < LINE_TOL
        s = np.sort((pts[inl] - c) @ u)
        breaks = np.nonzero(np.diff(s) > MAX_LINE_GAP)[0]
        used = False
        for lo, hi in zip(np.r_[0, breaks + 1], np.r_[breaks, len(s) - 1]):
            if hi - lo + 1 >= MIN_LINE_CELLS:
                segments.append((c + u * s[lo], c + u * s[hi]))
                used = True
        if not used:
            break
        pts = pts[~inl]
    return segments, pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', required=True, help='map yaml')
    ap.add_argument('--region', required=True, help='polygon "x,y x,y ..." in map metres')
    ap.add_argument('--out', required=True, help='output prefix (writes .pgm, .yaml, .png)')
    ap.add_argument('--bags', nargs='+', default=None)
    ap.add_argument('--no-bags', action='store_true', help='keep every small object')
    ap.add_argument('--min-structure', type=int, default=40, help='cells; bigger objects get straightened')
    args = ap.parse_args()

    grid = GridMap(args.map)
    verts = np.array([[float(v) for v in p.split(',')] for p in args.region.split()])
    region = cell_polygon_mask(grid, verts)
    if args.no_bags:
        static = np.ones_like(region)
    else:
        bags = args.bags or [b for b in sorted(glob.glob('data/bags/*')) if 'sl_lap_145448' not in b]
        static, _ = static_cells_and_paths(bags, grid)

    out = np.where(region, FREE, UNKNOWN).astype(np.uint8)
    occ = grid.occupied & region
    labels, n = label(occ, structure=np.ones((3, 3)))
    rng = np.random.default_rng(0)
    n_boxes = n_kept = n_dropped = 0
    wall_pts = []
    for k in range(1, n + 1):
        cells = labels == k
        if cells.sum() < args.min_structure:
            if (cells & static).any():
                out[cells] = OCCUPIED
                n_kept += 1
            else:
                n_dropped += 1
            continue
        pts = grid.centres(cells & static) if (cells & static).sum() >= MIN_LINE_CELLS else grid.centres(cells)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32))
        if min(w, h) >= MIN_BOX_SIDE:
            box = cv2.boxPoints(((cx, cy), (w + grid.res, h + grid.res), ang))
            poly = np.round(to_cells(grid, box)).astype(np.int32)
            cv2.fillPoly(out, [poly], UNKNOWN)
            cv2.polylines(out, [poly], True, OCCUPIED, 1)
            n_boxes += 1
        else:
            wall_pts.append(pts)
    # walls: lines through all elongated structures together, so a wall that the map
    # shows in pieces becomes one straight line
    segments, rest = ransac_lines(np.vstack(wall_pts), rng) if wall_pts else ([], np.zeros((0, 2)))
    for p0, p1 in segments:
        a, b = np.round(to_cells(grid, np.array([p0, p1]))).astype(int)
        cv2.line(out, tuple(a), tuple(b), OCCUPIED, 1)
    if len(rest):                                   # static cells that fit no line stay as they are
        r, c = grid.cells(rest[:, 0], rest[:, 1])
        keep = static[r, c]
        out[r[keep], c[keep]] = OCCUPIED
    n_walls = len(segments)
    print(f'{n} objects in the region: {n_boxes} boxes, {n_walls} wall segments, '
          f'{n_kept} small objects kept, {n_dropped} dropped')

    Image.fromarray(out).save(args.out + '.pgm')
    Image.fromarray(out).save(args.out + '.png')
    with open(args.map) as f:
        meta = yaml.safe_load(f)
    meta['image'] = os.path.basename(args.out) + '.pgm'
    with open(args.out + '.yaml', 'w') as f:
        f.write(f'# cleaned copy of {os.path.basename(args.map)} by scripts/clean_map.py, region:\n'
                f'# {args.region}\n')
        yaml.safe_dump(meta, f, sort_keys=False)
    print('wrote', args.out + '.pgm/.yaml/.png')


if __name__ == '__main__':
    main()
