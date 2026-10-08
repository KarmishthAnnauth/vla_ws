#!/usr/bin/env python3
"""Put a Lanelet2 OSM map into the frame of a SLAM occupancy map.

The lane markings are flat, so the LiDAR map does not contain them and the two
maps share no features. The alignment (rotation + translation, scale 1) is
fitted from what the recorded drives do show:

  * objects that every drive sees at the same place (walls, boxes, the posts on
    the islands) must not stand on a lanelet;
  * the car was driven on the road, so the driven paths must lie on lanelets.

Steps: re-localise every scan of every bag against the occupancy map, mark the
cells hit in (nearly) all bags as static, fit the pose of the OSM frame in the
map frame, then write

  <out>.osm              the same lanelets, nodes moved into the map frame
                         (tags local_x / local_y = map metres; lat / lon are the
                         same numbers as degrees on a sphere, as in the input)
  <out>.png              overlay to check the result by eye
  <out>_alignment.yaml   the fitted pose and the fit statistics

    source /opt/ros/humble/setup.bash        # only the fit needs ROS (rclpy)
    python3 scripts/align_osm_to_map.py

Pass --alignment <file>_alignment.yaml to skip the fit and reuse a stored pose
(e.g. after editing the OSM in JOSM). Run from the workspace root.
"""
import argparse
import glob
import math
import os
import sqlite3
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares

EARTH_R = 6378137.0       # lat/lon <-> metres on a sphere (what JOSM shows near 0, 0)
LASER_X = 0.27            # base_link -> laser (bringup_launch.py)
CAR_CENTRE_X = 0.16       # base_link -> middle of the wheelbase
ODOM_GAIN = 1.13          # wheel odometry reads ~10 % short (data/README.md)
STATIC_HIT_FRAC = 0.02    # a cell is "seen" by a bag if this share of its scans hits it
STATIC_BAG_FRAC = 0.8     # ... and static if this share of the bags sees it
MIN_SCAN_FIT = 0.85       # keep a pose if this share of its beams ends within 0.1 m of a wall
ROAD_MARGIN = 0.10        # the car centre should stay this far inside the road edge
LOSS_SCALE = 0.05         # soft-L1 scale of the fit [m]


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# --------------------------------------------------------------------------- maps

class GridMap:
    """nav2 map_server occupancy image; pixel (col, row) centre is at
    x = ox + (col + 0.5) res, y = oy + (H - row - 0.5) res."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        self.image = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])))
        self.res = float(meta['resolution'])
        self.ox, self.oy = meta['origin'][:2]
        self.h, self.w = self.image.shape
        self.occupied = self.image == 0
        self.wall_dist = distance_transform_edt(~self.occupied) * self.res

    def cells(self, x, y):
        return (self.h - 1 - np.floor((y - self.oy) / self.res)).astype(int), \
            np.floor((x - self.ox) / self.res).astype(int)

    def centres(self, mask):
        rows, cols = np.nonzero(mask)
        return np.c_[self.ox + (cols + 0.5) * self.res, self.oy + (self.h - rows - 0.5) * self.res]

    def dist_to_wall(self, x, y):
        return map_coordinates(self.wall_dist, [self.h - 0.5 - (y - self.oy) / self.res,
                                                (x - self.ox) / self.res - 0.5], order=1, mode='nearest')

    @property
    def extent(self):
        return [self.ox, self.ox + self.w * self.res, self.oy, self.oy + self.h * self.res]


class LaneletMap:
    def __init__(self, path):
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        self.nodes = {n.get('id'): np.array([math.radians(float(n.get('lon'))) * EARTH_R,
                                             math.radians(float(n.get('lat'))) * EARTH_R])
                      for n in root.findall('node')}
        self.ways = {w.get('id'): ([nd.get('ref') for nd in w.findall('nd')],
                                   {t.get('k'): t.get('v') for t in w.findall('tag')})
                     for w in root.findall('way')}
        self.lanelets = {}
        for rel in root.findall('relation'):
            if any(t.get('k') == 'type' and t.get('v') == 'lanelet' for t in rel.findall('tag')):
                self.lanelets[rel.get('id')] = {m.get('role'): m.get('ref') for m in rel.findall('member')}
        pts = np.array(list(self.nodes.values()))
        self.lo, self.hi = pts.min(0), pts.max(0)
        self.centre = (self.lo + self.hi) / 2

    def way_xy(self, way_id):
        return np.array([self.nodes[i] for i in self.ways[way_id][0]])

    def polygon(self, lanelet_id):
        """Outline of a lanelet; the OSM ways may point either way along the lane."""
        left, right = (self.way_xy(self.lanelets[lanelet_id][k]) for k in ('left', 'right'))
        d = np.linalg.norm
        if d(left[0] - right[0]) + d(left[-1] - right[-1]) > d(left[0] - right[-1]) + d(left[-1] - right[0]):
            right = right[::-1]
        return np.vstack([left, right[::-1]])

    def road_sdf(self, cell=0.01, pad=2.0):
        """Signed distance to the road edge, > 0 on the lanelets."""
        lo = self.lo - pad
        nx, ny = np.ceil((self.hi + pad - lo) / cell).astype(int)
        img = Image.new('1', (int(nx), int(ny)), 0)
        draw = ImageDraw.Draw(img)
        for lid in self.lanelets:
            draw.polygon([tuple(p) for p in (self.polygon(lid) - lo) / cell], fill=1)
        road = np.array(img, dtype=bool)
        sdf = (distance_transform_edt(road) - distance_transform_edt(~road)) * cell
        return lambda p: map_coordinates(sdf, [(p[:, 1] - lo[1]) / cell - 0.5, (p[:, 0] - lo[0]) / cell - 0.5],
                                         order=1, mode='nearest')


# --------------------------------------------------------------------------- bags

def read_bag(bag_dir):
    """/scan, /pf/pose/odom and /odom from a rosbag2 sqlite3 file (read directly:
    the bags were recorded on Foxy, the reader here is Humble)."""
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan

    db = sqlite3.connect(glob.glob(os.path.join(bag_dir, '*.db3'))[0])
    topic_id = {name: i for i, name in db.execute('select id, name from topics')}

    def messages(topic, msg_type):
        if topic not in topic_id:
            return []
        rows = db.execute('select data from messages where topic_id = ? order by timestamp', (topic_id[topic],))
        return [deserialize_message(data, msg_type) for (data,) in rows]

    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def poses(topic):
        out = []
        for m in messages(topic, Odometry):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            if abs(p.x) < 1e3 and abs(p.y) < 1e3:      # a lost particle filter publishes garbage
                out.append((stamp(m), p.x, p.y, yaw))
        out = np.array(out).reshape(-1, 4)
        out[:, 3] = np.unwrap(out[:, 3])
        return out

    scans = messages('/scan', LaserScan)
    angles = scans[0].angle_min + scans[0].angle_increment * np.arange(len(scans[0].ranges))
    scans = [(stamp(m), np.array(m.ranges, dtype=np.float32)) for m in scans]
    odom = poses('/odom')                             # base_link in odom -> laser in odom
    odom[:, 1] += LASER_X * np.cos(odom[:, 3])
    odom[:, 2] += LASER_X * np.sin(odom[:, 3])
    return scans, angles, poses('/pf/pose/odom'), odom


def pose_at(track, t):
    return np.array([np.interp(t, track[:, 0], track[:, k]) for k in (1, 2, 3)])


def track_bag(bag_dir, grid, every=2, max_range=12.0):
    """Laser pose in the map for every `every`-th scan, by matching the scan to the
    occupancy map. Start values: the recorded localiser pose and the previous result
    advanced by odometry, so drives where the recorded localiser got lost still work."""
    scans, angles, recorded, odom = read_bag(bag_dir)
    sources = [(s, g) for s, gains in ((odom, (1.0, ODOM_GAIN)), (recorded, (1.0,))) if len(s) for g in gains]
    poses, prev = [], None
    for t, ranges in scans[::every]:
        if not all(s[0, 0] <= t <= s[-1, 0] for s, _ in sources):
            continue
        beams = np.nonzero(np.isfinite(ranges) & (ranges > 0.15) & (ranges < max_range))[0]
        lx, ly = ranges[beams] * np.cos(angles[beams]), ranges[beams] * np.sin(angles[beams])

        def residual(q, lx=lx[::3], ly=ly[::3]):
            c, s = math.cos(q[2]), math.sin(q[2])
            return grid.dist_to_wall(q[0] + c * lx - s * ly, q[1] + s * lx + c * ly)

        now = [pose_at(s, t) for s, _ in sources]
        starts = [pose_at(recorded, t)]
        if prev is not None:
            for (_, gain), a, b in zip(sources, prev[1], now):
                step = rot(prev[0][2] - a[2]) @ (b[:2] - a[:2]) * gain
                starts.append(np.r_[prev[0][:2] + step, prev[0][2] + b[2] - a[2]])
        best = None
        for start in starts:
            sol = least_squares(residual, start, loss='soft_l1', f_scale=0.05, x_scale=[0.05, 0.05, 0.01],
                                diff_step=[1e-3, 1e-3, 1e-3])
            r = residual(sol.x)
            score = np.mean(np.minimum(r, 0.2))
            if best is None or score < best[0]:
                best = (score, sol.x, np.mean(r < 0.1))
        prev = (best[1], now)
        if best[2] >= MIN_SCAN_FIT:
            c, s = math.cos(best[1][2]), math.sin(best[1][2])
            poses.append((best[1], np.c_[best[1][0] + c * lx - s * ly, best[1][1] + s * lx + c * ly]))
    return poses


def static_cells_and_paths(bag_dirs, grid):
    seen = np.zeros((grid.h, grid.w))
    paths = []
    for bag in bag_dirs:
        poses = track_bag(bag, grid)
        hits = np.zeros((grid.h, grid.w))
        for _, points in poses:
            rows, cols = grid.cells(points[:, 0], points[:, 1])
            ok = (rows >= 0) & (rows < grid.h) & (cols >= 0) & (cols < grid.w)
            once = np.zeros((grid.h, grid.w), dtype=bool)
            once[rows[ok], cols[ok]] = True
            hits += once
        seen += hits / len(poses) > STATIC_HIT_FRAC
        laser = np.array([p for p, _ in poses])
        centre = laser[:, :2] - (LASER_X - CAR_CENTRE_X) * np.c_[np.cos(laser[:, 2]), np.sin(laser[:, 2])]
        keep = [0]
        for i in range(1, len(centre)):               # one point every 5 cm, so parking does not add weight
            if np.linalg.norm(centre[i] - centre[keep[-1]]) > 0.05:
                keep.append(i)
        paths.append(centre[keep])
        print(f'  {os.path.basename(bag)}: {len(poses)} scans matched, {len(keep) * 0.05:.0f} m driven')
    return seen >= math.ceil(STATIC_BAG_FRAC * len(bag_dirs)), paths


# ---------------------------------------------------------------------------- fit

def fit_alignment(lanelets, static_xy, path_xy):
    """Pose (yaw, cx, cy) of the OSM frame in the map: map = R(yaw) (osm - osm_centre) + (cx, cy)."""
    sdf = lanelets.road_sdf()
    # the OSM may cover only part of the floor: ignore paths beyond its outline (checked per pose)
    span = (lanelets.hi - lanelets.lo) / 2

    def to_osm(q, p):
        return (p - q[1:3]) @ rot(q[0]) + lanelets.centre          # R(-yaw) (p - c) + centre

    def covered(path):
        return np.all(np.abs(path - lanelets.centre) < span - 0.2, axis=1)

    def residual(q, stride=1):
        path = to_osm(q, path_xy[::stride])
        return np.r_[np.maximum(0, sdf(to_osm(q, static_xy[::stride]))),
                     np.where(covered(path), np.maximum(0, ROAD_MARGIN - sdf(path)), ROAD_MARGIN)]

    def cost(q, stride=1):
        return np.mean(np.minimum(residual(q, stride), 0.3))

    # coarse search over all headings, then refine the best candidates
    mid = (path_xy.min(0) + path_xy.max(0)) / 2
    shifts = np.arange(-3.0, 3.01, 0.15)
    coarse = sorted((cost((yaw, mid[0] + dx, mid[1] + dy), 5), yaw, mid[0] + dx, mid[1] + dy)
                    for yaw in np.radians(np.arange(0, 360, 5)) for dx in shifts for dy in shifts)
    best = None
    for _, *start in coarse[:30]:
        sol = least_squares(residual, start, loss='soft_l1', f_scale=LOSS_SCALE, x_scale=[0.01, 0.05, 0.05],
                            diff_step=[2e-4, 2e-3, 2e-3])
        if best is None or sol.cost < best.cost:
            best = sol
    path = to_osm(best.x, path_xy)
    depth = sdf(path[covered(path)])                  # path points inside the OSM outline
    stats = {
        'static_cells': len(static_xy),
        'static_cells_on_road': int(np.sum(sdf(to_osm(best.x, static_xy)) > 0.03)),
        'path_points': len(depth),
        'path_points_off_road': int(np.sum(depth < 0)),
        'path_points_near_edge': int(np.sum(depth < ROAD_MARGIN)),
    }
    return best.x, stats


# ------------------------------------------------------------------------- output

def write_osm(lanelets, to_map, path):
    for node in lanelets.tree.getroot().findall('node'):
        x, y = to_map(lanelets.nodes[node.get('id')][None])[0]
        node.set('lat', f'{math.degrees(y / EARTH_R):.11f}')
        node.set('lon', f'{math.degrees(x / EARTH_R):.11f}')
        for tag in node.findall('tag'):
            if tag.get('k') in ('local_x', 'local_y'):
                node.remove(tag)
        ET.SubElement(node, 'tag', k='local_x', v=f'{x:.4f}')
        ET.SubElement(node, 'tag', k='local_y', v=f'{y:.4f}')
    ET.indent(lanelets.tree, space='  ')
    lanelets.tree.write(path, encoding='UTF-8', xml_declaration=True)


def write_overlay(lanelets, to_map, grid, static, paths, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shade = np.where(grid.occupied, 150, np.where(grid.image > 250, 255, 225)).astype(np.uint8)
    if static is not None:
        shade[static] = 0
    fig, ax = plt.subplots(figsize=(grid.w * grid.res * 1.6, grid.h * grid.res * 1.6))
    ax.imshow(shade, cmap='gray', vmin=0, vmax=255, extent=grid.extent, interpolation='nearest')
    for ids, tags in lanelets.ways.values():
        p = to_map(np.array([lanelets.nodes[i] for i in ids]))
        ax.plot(p[:, 0], p[:, 1], color='tab:blue', ls='-' if tags.get('subtype') == 'solid' else '--',
                lw=2.2 if tags.get('type') == 'line_thick' else 1.0)
    for name, p in paths or []:
        ax.plot(p[:, 0], p[:, 1], lw=0.8, label=name)
    ax.plot(0, 0, 'r+', ms=18, mew=2, label='map origin')
    ax.set_xlabel('map x [m]')
    ax.set_ylabel('map y [m]')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--osm', default='testtrack_base.osm')
    ap.add_argument('--map', default='src/particle_filter/maps/track_20260930.yaml')
    ap.add_argument('--bags', nargs='+', default=None, help='rosbag2 directories (default: data/bags/*, '
                    'without sl_lap_145448, where the car stands still)')
    ap.add_argument('--alignment', help='reuse a stored *_alignment.yaml instead of fitting')
    ap.add_argument('--out', default=None, help='output prefix (default: <map>_lanelet2)')
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.map)[0] + '_lanelet2'
    grid = GridMap(args.map)
    lanelets = LaneletMap(args.osm)
    static, paths = None, None
    if args.alignment:
        with open(args.alignment) as f:
            stored = yaml.safe_load(f)
        assert np.allclose(stored['osm_centre'], lanelets.centre, atol=1e-3), 'alignment is for another OSM'
        pose = np.r_[math.radians(stored['yaw_deg']), stored['centre_in_map']]
        stats = stored.get('fit', {})
    else:
        bags = args.bags or [b for b in sorted(glob.glob('data/bags/*')) if 'sl_lap_145448' not in b]
        print(f'matching scans of {len(bags)} bags to {args.map}')
        static, path_list = static_cells_and_paths(bags, grid)
        paths = [(os.path.basename(b), p) for b, p in zip(bags, path_list)]
        print('fitting')
        pose, stats = fit_alignment(lanelets, grid.centres(static), np.vstack(path_list))

    def to_map(p):
        return (p - lanelets.centre) @ rot(pose[0]).T + pose[1:3]

    yaw_deg = math.degrees(pose[0]) % 360
    print(f'OSM frame in map: yaw {yaw_deg:.2f} deg, OSM centre at ({pose[1]:.3f}, {pose[2]:.3f}) m')
    print('fit:', stats)
    write_osm(lanelets, to_map, out + '.osm')
    write_overlay(lanelets, to_map, grid, static, paths, out + '.png',
                  f'{os.path.basename(args.osm)} on {os.path.basename(args.map)}: yaw {yaw_deg:.2f} deg'
                  + (' (black: seen in all drives, grey: mapping run only)' if static is not None else ''))
    if not args.alignment:
        with open(out + '_alignment.yaml', 'w') as f:
            f.write('# map = R(yaw) * (osm - osm_centre) + centre_in_map, all in metres;\n'
                    '# osm = (lon, lat) in radians * 6378137. Written by scripts/align_osm_to_map.py.\n')
            yaml.safe_dump({'osm': os.path.basename(args.osm), 'map': os.path.basename(args.map),
                            'yaw_deg': round(yaw_deg, 6),
                            'osm_centre': [round(float(v), 6) for v in lanelets.centre],
                            'centre_in_map': [round(float(v), 6) for v in pose[1:3]],
                            'bags': [os.path.basename(b) for b in bags], 'fit': stats}, f, sort_keys=False)
    print('wrote', out + '.osm,', out + '.png' + ('' if args.alignment else ', ' + out + '_alignment.yaml'))


if __name__ == '__main__':
    main()
