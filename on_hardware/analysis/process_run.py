"""
Post-process a OneMap hardware run (.rrd written by on_hardware/server_side/main.py).

    python3 on_hardware/analysis/process_run.py hardware_test/<run>.rrd [--cell 0.2]

Writes hardware_test/<run>_report/:
    summary.json        duration, steps, distance, detections, stuck time, phases
    events.csv          planner log (path_updates) with run-relative time
    trajectory.csv      robot position per step (map px and metres from start)
    trajectory.png      path over the final similarity + explored maps
    timeline.png        planner state, speed and detections over time
    detections.png      camera frames where the target was detected (with boxes)
    final_maps.png      final similarity / explored / confidence maps
"""
import argparse, io, json, os, re, shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rerun as rr


def column(rec, entity, component):
    """All rows of one component of one entity, as (times, values)."""
    df = rec.view(index="log_time", contents={entity: [component]}).select().read_all().to_pandas()
    cols = [c for c in df.columns if c.endswith(component) or component in c]
    if not cols:
        return pd.Series(dtype="datetime64[ns]"), []
    df = df.dropna(subset=[cols[0]])
    return df["log_time"].reset_index(drop=True), list(df[cols[0]])


def first(v):
    """Rerun cells are lists of instances; return the first instance."""
    return v[0] if len(v) else None


def decode_blob(cell):
    data = np.asarray(first(cell), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def target_reached(rrd, n_steps, max_steps=2000):
    final = os.path.join(os.path.dirname(os.path.abspath(rrd)), "similarities", "final_sim_hardware_0.png")
    if not os.path.exists(final) or n_steps >= max_steps:
        return False
    return abs(os.path.getmtime(final) - os.path.getmtime(os.path.realpath(rrd))) < 10


def classify(msg):
    if "initial scan" in msg:
        return "scan_done"
    if "stuck" in msg:
        return "stuck"
    if "chair" in msg or "detected" in msg or "Path to object" in msg:
        return "target"
    if "lost similarity" in msg:
        return "target_lost"
    if "explorer" in msg:
        return "explore"
    if "exploiter" in msg:
        return "exploit"
    if "Resetting checked map" in msg:
        return "no_path"
    if "Frontier" in msg:
        return "frontier_invalid"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rrd")
    ap.add_argument("--cell", type=float, default=0.2, help="metres per map cell (size / n_points)")
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()

    run = os.path.splitext(os.path.basename(args.rrd))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(args.rrd)), f"{run}_report")
    os.makedirs(out, exist_ok=True)
    rec = rr.dataframe.load_recording(args.rrd)

    # --- trajectory -------------------------------------------------------
    tp, pos = column(rec, "/map/agent_0/position", "Position2D")
    P = np.array([np.asarray(first(p), float).reshape(2) for p in pos])
    t0 = tp.iloc[0]
    rel = lambda ts: (ts - t0).dt.total_seconds().to_numpy()
    tpos = rel(tp)
    xy_m = (P - P[0]) * args.cell
    step_d = np.r_[0, np.linalg.norm(np.diff(xy_m, axis=0), axis=1)]
    speed = np.r_[0, step_d[1:] / np.maximum(np.diff(tpos), 1e-3)]
    pd.DataFrame({"t_s": tpos, "px_x": P[:, 0], "px_y": P[:, 1],
                  "x_m": xy_m[:, 0], "y_m": xy_m[:, 1], "speed_mps": speed}).to_csv(
        f"{out}/trajectory.csv", index=False)

    # --- planner events ---------------------------------------------------
    te, ev = column(rec, "/path_updates", "Text")
    ev = [str(first(e)) for e in ev]
    events = pd.DataFrame({"t_s": rel(te), "message": ev})
    events = events[~events.message.str.fullmatch(r"\d+")]       # drop bare step counters
    events["kind"] = events.message.map(classify)
    events.to_csv(f"{out}/events.csv", index=False)

    # stuck intervals: consecutive "stuck" events less than 1 s apart
    stuck_t = events.t_s[events.kind == "stuck"].to_numpy()
    stuck = []
    if len(stuck_t):
        s = e = stuck_t[0]
        for t in stuck_t[1:]:
            if t - e > 1.0:
                stuck.append((s, e)); s = t
            e = t
        stuck.append((s, e))

    # --- detections -------------------------------------------------------
    td, _ = column(rec, "/object_detections", "Text")
    det_t = rel(td) if len(td) else np.array([])
    target_ev = events[events.kind.isin(["target", "target_lost"])]

    # --- summary ----------------------------------------------------------
    ts, steps = column(rec, "/step", "Text")
    duration = float(tpos[-1])
    summary = {
        "run": run,
        "duration_s": round(duration, 1),
        "steps": len(steps),
        "steps_per_s": round(len(steps) / max(duration, 1e-6), 2),
        "distance_travelled_m": round(float(step_d.sum()), 2),
        "start_to_end_m": round(float(np.linalg.norm(xy_m[-1])), 2),
        "max_dist_from_start_m": round(float(np.linalg.norm(xy_m, axis=1).max()), 2),
        "target_detections": int(len(det_t)),
        "first_detection_s": round(float(det_t[0]), 1) if len(det_t) else None,
        # main.py only leaves the search loop when the object is found or max_steps is hit, and then
        # writes similarities/final_sim_*.png; Ctrl+C skips that. So: final images written within a few
        # seconds of the end of this recording and fewer than max_steps (2000) steps => found.
        # The image is overwritten by every run, so also accept the logged "< 3 cells to the object"
        # condition (navigator.check_object_reached) for older runs.
        "target_reached": target_reached(args.rrd, len(steps)) or bool(
            len(target_ev) and re.search(r"Path to object .* of length [0-2] computed", target_ev.message.iloc[-1])),
        "last_target_event_s": round(float(target_ev.t_s.iloc[-1]), 1) if len(target_ev) else None,
        "stuck_intervals_s": [[round(a, 1), round(b, 1)] for a, b in stuck],
        "stuck_total_s": round(sum(b - a for a, b in stuck), 1),
        "initial_scan_done_s": round(float(events.t_s[events.kind == "scan_done"].iloc[0]), 1) if (events.kind == "scan_done").any() else None,
        "events_by_kind": events.kind.value_counts().to_dict(),
    }
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # keep this run's end-of-search images; main.py overwrites them on the next run
    if target_reached(args.rrd, len(steps)):
        sim_dir = os.path.join(os.path.dirname(os.path.abspath(args.rrd)), "similarities")
        for name in ("final_sim_hardware_0.png", "path_hardware_0_0.svg"):
            if os.path.exists(f"{sim_dir}/{name}"):
                shutil.copy2(f"{sim_dir}/{name}", f"{out}/{name}")

    # --- final maps -------------------------------------------------------
    maps = {}
    for name in ["similarity", "explored", "confidence"]:
        _, blobs = column(rec, f"/map/{name}", "Blob")
        if blobs:
            maps[name] = decode_blob(blobs[-1])
    if maps:
        fig, axs = plt.subplots(1, len(maps), figsize=(5 * len(maps), 5.6))
        for ax, (k, im) in zip(np.atleast_1d(axs), maps.items()):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(f"final {k}"); ax.axis("off")
        fig.tight_layout(); fig.savefig(f"{out}/final_maps.png", dpi=120); plt.close(fig)

    # --- trajectory over map (zoomed to visited area) --------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    if "similarity" in maps:
        ax.imshow(maps["similarity"])
    if "explored" in maps:
        ex = maps["explored"]
        ex = ex[..., :3].mean(-1) if ex.ndim == 3 else ex
        ax.imshow(np.ma.masked_where(ex < 1, ex), cmap="Greys", alpha=0.25)
    sc = ax.scatter(P[:, 0], P[:, 1], c=tpos, cmap="viridis", s=12, zorder=3)
    ax.plot(P[:, 0], P[:, 1], "-", color="white", lw=1, alpha=0.7, zorder=2)
    ax.plot(*P[0], "o", ms=12, mfc="none", mec="lime", mew=2, label="start")
    ax.plot(*P[-1], "s", ms=11, mfc="none", mec="red", mew=2, label="end")
    for t in det_t[:1]:
        i = np.searchsorted(tpos, t)
        ax.plot(*P[min(i, len(P) - 1)], "*", ms=18, color="gold", mec="k", label="first chair detection")
    pad = 25
    ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
    ax.set_ylim(P[:, 1].max() + pad, P[:, 1].min() - pad)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="time (s)")
    ax.set_title(f"{run}\n{summary['distance_travelled_m']} m travelled in {summary['duration_s']} s "
                 f"(1 cell = {args.cell} m)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/trajectory.png", dpi=130); plt.close(fig)

    # --- timeline ---------------------------------------------------------
    lanes = ["exploit", "explore", "no_path", "stuck", "target", "target_lost", "frontier_invalid", "scan_done"]
    colors = dict(exploit="tab:blue", explore="tab:green", no_path="tab:gray", stuck="tab:red",
                  target="gold", target_lost="tab:orange", frontier_invalid="tab:purple", scan_done="tab:cyan")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.3, 1]))
    for i, k in enumerate(lanes):
        tt = events.t_s[events.kind == k]
        a1.scatter(tt, np.full(len(tt), i), s=14, color=colors[k], marker="|" if k != "target" else "*")
    for a, b in stuck:
        a1.axvspan(a, b, color="tab:red", alpha=0.08); a2.axvspan(a, b, color="tab:red", alpha=0.08)
    a1.set_yticks(range(len(lanes)), lanes); a1.set_title("planner events (red band = stuck)")
    a2.plot(tpos, pd.Series(speed).rolling(5, min_periods=1).mean(), color="k", lw=1)
    for t in det_t:
        a2.axvline(t, color="gold", lw=0.8)
    a2.set_ylabel("speed (m/s, 5-step avg)"); a2.set_xlabel("time since start (s)")
    fig.tight_layout(); fig.savefig(f"{out}/timeline.png", dpi=120); plt.close(fig)

    # --- detection frames -------------------------------------------------
    trgb, rgbs = column(rec, "/agent_0/camera/rgb", "ImageBuffer")
    tb, centers = column(rec, "/agent_0/camera/detection", "Position2D")
    _, halves = column(rec, "/agent_0/camera/detection", "HalfSize2D")
    _, labels = column(rec, "/agent_0/camera/detection", "Text")
    trgb_s, tb_s = rel(trgb), rel(tb)
    if len(det_t) and len(rgbs):
        pick = det_t[np.linspace(0, len(det_t) - 1, min(args.max_frames, len(det_t))).astype(int)]
        n = len(pick); cols = 4; rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows))
        for ax in np.atleast_1d(axs).ravel():
            ax.axis("off")
        for ax, t in zip(np.atleast_1d(axs).ravel(), pick):
            i = int(np.argmin(np.abs(trgb_s - t)))
            img = np.asarray(first(rgbs[i]), np.uint8).reshape(480, 640, 3)
            ax.imshow(img)
            j = int(np.argmin(np.abs(tb_s - t))) if len(tb_s) else None
            if j is not None and abs(tb_s[j] - t) < 1.0 and j < len(halves):
                for c, h, l in zip(centers[j], halves[j], labels[j] if j < len(labels) else []):
                    ax.add_patch(plt.Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1],
                                               fill=False, ec="lime", lw=2))
                    ax.text(c[0] - h[0], c[1] - h[1] - 4, str(l), color="lime", fontsize=8)
            ax.set_title(f"t = {trgb_s[i]:.1f} s", fontsize=9)
        fig.suptitle("frames at target detections"); fig.tight_layout()
        fig.savefig(f"{out}/detections.png", dpi=110); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
