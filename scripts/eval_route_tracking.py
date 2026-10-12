#!/usr/bin/env python3
"""How far did the car stray from the global route? Metrics for a recorded run.

Reads the trajectory files written by record_run_trajectory.py during a run
(log/<model>/route_*_traj.csv; started by the run_*_route.sh scripts) and compares the
driven path, base_link in ``map``, with the route it was given. Only poses while the car
was released count (e-stop off); every route of a run file is scored on its own.
No ROS needed (numpy; matplotlib + Pillow for --plot).

The route CSVs have no timing (the velocity column is informational), so the
time-indexed ADE/FDE of trajectory prediction do not apply as such. Their spatial
counterparts:

  CTE mean / RMS / p95 / max   cross-track error: distance of the driven path to the route
                               polyline, sampled every 5 cm of driven path (a car standing
                               still does not weigh in), up to where the car passes the
                               route's last waypoint. Matched against the route section
                               the car has reached, so a route passing close to itself
                               (route 5) does not hide an error. "CTE mean" is the
                               path-following ADE.
  signed                       mean signed offset, + = left of the route (a constant bias,
                               e.g. a camera/localisation offset, shows here)
  FDE                          distance from the car's last released pose to the route's
                               last waypoint
  WP mean / max                for every route waypoint the closest the driven path came to
                               it, averaged / worst: the minADE-style view (each ground-truth
                               point against its best-matching driven point). Waypoints the
                               car never got to count with their full distance.
  completion                   furthest progress along the route / route length; progress
                               only counts while the car is within --on-route (0.5 m) of it

    scripts/eval_route_tracking.py log/orion/route_1_fast_*_traj.csv
    scripts/eval_route_tracking.py log/*/route_*_traj.csv --csv results.csv --plot
"""

import argparse
import csv
import os
import sys

import numpy as np

STEP = 0.05          # m, resampling of the driven path
STILL = 0.01         # m, pose noise below this is not driven distance
WINDOW_BACK = 0.5    # m of route behind / ahead of the progress so far that a pose may match
WINDOW_AHEAD = 3.0
ON_ROUTE = 0.5       # m, progress only advances this close to the route (--on-route)


# ── reading ──────────────────────────────────────────────────────────────────

def read_run(path):
    """Segments [{'route': name, 'points': [N,2], 't','xy','estop': arrays}] of one run file."""
    segs = []
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("# route "):
                _, _, name, pts = line.split(" ", 3)
                points = np.array([[float(v) for v in p.split()] for p in pts.split(";")])
                cur = {"route": name, "points": points, "rows": []}
                segs.append(cur)
            elif not line or line.startswith("#") or line.startswith("t,"):
                continue
            elif cur is not None:
                f = line.split(",")
                try:
                    cur["rows"].append((float(f[0]), float(f[1]), float(f[2]), int(f[5])))
                except (ValueError, IndexError):
                    continue                      # a line cut by a freeze
    for s in segs:
        a = np.array(s.pop("rows"), dtype=np.float64).reshape(-1, 4)
        s["t"], s["xy"], s["estop"] = a[:, 0], a[:, 1:3], a[:, 3].astype(bool)
    return segs


# ── geometry ─────────────────────────────────────────────────────────────────

def resample(xy, step):
    """Points every ``step`` metres along a path; pose jitter below STILL is dropped first."""
    keep = [xy[0]]
    for p in xy[1:]:
        if np.hypot(*(p - keep[-1])) >= STILL:
            keep.append(p)
    keep = np.array(keep)
    if len(keep) < 2:
        return keep, 0.0
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(keep, axis=0).T))])
    q = np.arange(0.0, s[-1], step)
    out = np.stack([np.interp(q, s, keep[:, 0]), np.interp(q, s, keep[:, 1])], axis=1)
    return np.vstack([out, keep[-1]]), float(s[-1])


def project(p, a, b, first, last):
    """Closest point of p on segment a-b (the first/last segment extended as a ray).
    Returns (distance signed + left, fraction along a-b, unclamped fraction)."""
    d = b - a
    L2 = float(d @ d)
    u = float((p - a) @ d) / L2 if L2 > 0 else 0.0
    u_c = u
    if not (first and u < 0):
        u_c = max(u_c, 0.0)
    if not (last and u > 1):
        u_c = min(u_c, 1.0)
    c = a + u_c * d
    dist = float(np.hypot(*(p - c)))
    side = np.sign(d[0] * (p - a)[1] - d[1] * (p - a)[0]) or 1.0
    return side * dist, u_c, u


def cross_track(path, route, on_route=ON_ROUTE):
    """Signed cross-track error and route progress (m) for the points of ``path`` up to
    where the car crosses the line through the route's end (perpendicular to its last
    segment), matched against the route window around the progress so far. Progress
    only advances while the car is within ``on_route`` of the route and drives along
    it (not against it): a car that has left the route does not drag the progress
    along the nearest route section."""
    seg_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    n_seg = len(route) - 1
    s_cur = 0.0
    errs, prog = [], []
    for k, p in enumerate(path):
        heading = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
        best = None
        for i in range(n_seg):
            if seg_s[i + 1] < s_cur - WINDOW_BACK or seg_s[i] > s_cur + WINDOW_AHEAD:
                continue
            e, u, u_raw = project(p, route[i], route[i + 1], i == 0, i == n_seg - 1)
            if best is None or abs(e) < abs(best[0]):
                best = (e, seg_s[i] + u * (seg_s[i + 1] - seg_s[i]), i == n_seg - 1 and u_raw > 1.0,
                        float(heading @ (route[i + 1] - route[i])) > 0.0)
        e, s, past_end, along = best
        if past_end and errs:
            break                              # beyond the route: nothing left to follow
        if abs(e) <= on_route and along:
            s_cur = max(s_cur, min(max(s, 0.0), seg_s[-1]))
        errs.append(e)
        prog.append(s_cur)
    return np.array(errs), np.array(prog), float(seg_s[-1])


def point_to_path(p, path):
    """Distance of p to the polyline ``path``."""
    if len(path) == 1:
        return float(np.hypot(*(p - path[0])))
    a, b = path[:-1], path[1:]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-12)
    u = np.clip(((p - a) * d).sum(1) / L2, 0.0, 1.0)
    c = a + u[:, None] * d
    return float(np.hypot(*(p - c).T).min())


# ── metrics ──────────────────────────────────────────────────────────────────

def evaluate(seg, on_route=ON_ROUTE):
    route = seg["points"]
    rel = ~seg["estop"]
    m = {"route": seg["route"], "route_m": float(np.hypot(*np.diff(route, axis=0).T).sum()),
         "poses": int(rel.sum())}
    if rel.sum() < 2:
        return m
    xy = seg["xy"][rel]
    t = seg["t"][rel]
    path, driven = resample(xy, STEP)
    if len(path) < 2:
        return m
    e, prog, L = cross_track(path, route, on_route)
    path = path[:len(e)]
    a = np.abs(e)
    wp = np.array([point_to_path(w, path) for w in route])
    released = float(np.sum(np.minimum(np.diff(t), 0.5)))     # a gap = e-stopped in between
    m.update({
        "driven_m": driven, "released_s": released,
        "cte_mean": float(a.mean()), "cte_rms": float(np.sqrt((e ** 2).mean())),
        "cte_p95": float(np.percentile(a, 95)), "cte_max": float(a.max()),
        "cte_max_at": float(prog[int(a.argmax())]), "signed": float(e.mean()),
        "fde": float(np.hypot(*(xy[-1] - route[-1]))),
        "wp_mean": float(wp.mean()), "wp_max": float(wp.max()), "wp_worst": int(wp.argmax()),
        "completion": float(prog.max() / L) if L > 0 else 0.0,
        "_path": path, "_err": e, "_end": xy[-1],
    })
    return m


def report(path, metrics):
    print(f"{path}")
    for i, m in enumerate(metrics):
        head = f"  [{i}] {m['route']} ({m['route_m']:.1f} m)"
        if "cte_mean" not in m:
            print(f"{head}: never released ({m['poses']} poses while released)")
            continue
        print(f"{head}: drove {m['driven_m']:.1f} m in {m['released_s']:.0f} s, "
              f"completion {100 * m['completion']:.0f} %")
        print(f"      CTE mean {m['cte_mean']:.3f} m  RMS {m['cte_rms']:.3f}  p95 {m['cte_p95']:.3f}  "
              f"max {m['cte_max']:.3f} (at {m['cte_max_at']:.1f} m)  signed {m['signed']:+.3f} (+ = left)")
        print(f"      FDE {m['fde']:.3f} m   WP mean {m['wp_mean']:.3f} m  max {m['wp_max']:.3f} "
              f"(waypoint {m['wp_worst']})")


COLUMNS = ["file", "segment", "route", "route_m", "driven_m", "released_s", "completion",
           "cte_mean", "cte_rms", "cte_p95", "cte_max", "signed", "fde", "wp_mean", "wp_max"]


def write_csv(out, rows):
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "") for c in COLUMNS])


# ── plot ─────────────────────────────────────────────────────────────────────

def load_map(yaml_path):
    """(image, extent) of a map_server yaml, or None."""
    try:
        from PIL import Image
        meta = {}
        for line in open(yaml_path):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        img = np.asarray(Image.open(os.path.join(os.path.dirname(yaml_path), meta["image"])).convert("L"))
        res = float(meta["resolution"])
        ox, oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        h, w = img.shape
        return img, (ox, ox + w * res, oy, oy + h * res)
    except (OSError, KeyError, ValueError, ImportError):
        return None


def plot(out, title, metrics, map_yaml):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    segs = [m for m in metrics if "cte_mean" in m]
    if not segs:
        return False
    fig, axes = plt.subplots(1, len(segs), figsize=(7 * len(segs), 6), squeeze=False)
    bg = load_map(map_yaml) if map_yaml else None
    vmax = max(0.3, max(m["cte_max"] for m in segs))
    for ax, m in zip(axes[0], segs):
        if bg is not None:
            ax.imshow(bg[0], extent=bg[1], cmap="gray", vmin=0, vmax=255, origin="upper", alpha=0.6)
        r = m.get("_route")
        ax.plot(r[:, 0], r[:, 1], "o--", color="0.2", ms=4, lw=1, label="route")
        p = m["_path"]
        lc = LineCollection(np.stack([p[:-1], p[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(0, vmax), linewidths=2.5)
        lc.set_array(np.abs(m["_err"][:-1]))
        ax.add_collection(lc)
        ax.plot(*r[-1], "s", color="tab:red", ms=8, label="route end")
        ax.plot(*m["_end"], "x", color="tab:red", ms=9, mew=2, label=f"car end (FDE {m['fde']:.2f} m)")
        pad = 1.0
        lo = np.minimum(r.min(0), p.min(0)) - pad
        hi = np.maximum(r.max(0), p.max(0)) + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(f"{m['route']}: CTE mean {m['cte_mean']:.2f} m, max {m['cte_max']:.2f} m, "
                     f"{100 * m['completion']:.0f} % done", fontsize=10)
        ax.set_xlabel("map x [m]"); ax.set_ylabel("map y [m]")
        ax.legend(loc="best", fontsize=8)
        fig.colorbar(lc, ax=ax, label="cross-track error [m]", shrink=0.8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="trajectory files from record_run_trajectory.py")
    ap.add_argument("--csv", help="append one row per route attempt to this CSV")
    ap.add_argument("--on-route", type=float, default=ON_ROUTE,
                    help="m from the route within which driving counts as progress (default %(default)s)")
    ap.add_argument("--plot", action="store_true", help="save <run>.png next to each run file")
    ap.add_argument("--map", default=os.path.join(here, "..", "src", "particle_filter", "maps", "track_20260930.yaml"),
                    help="map yaml drawn under the plot ('' = none)")
    a = ap.parse_args()

    rows = []
    for path in a.runs:
        segs = read_run(path)
        if not segs:
            print(f"{path}: no route in the file", file=sys.stderr)
            continue
        metrics = [evaluate(s, a.on_route) for s in segs]
        for m, s in zip(metrics, segs):
            m["_route"] = s["points"]
        report(path, metrics)
        if a.plot:
            png = os.path.splitext(path)[0] + ".png"
            if plot(png, os.path.basename(path), metrics, a.map):
                print(f"  plot: {png}")
        for i, m in enumerate(metrics):
            rows.append(dict(m, file=os.path.basename(path), segment=i))
    if a.csv and rows:
        write_csv(a.csv, rows)
        print(f"{len(rows)} row(s) appended to {a.csv}")


if __name__ == "__main__":
    main()
