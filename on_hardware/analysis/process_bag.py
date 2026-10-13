"""
Post-process the robot-side MCAP bag of a OneMap hardware run (record_bag.sh), aligned to
the server's .rrd so times match process_run.py (t = 0 at the first OneMap step).

    python3 on_hardware/analysis/process_bag.py hardware_test/bags/<run> hardware_test/<run>.rrd [--offset 0.036]

Needs the ROS environment (rosbag2_py). Writes hardware_test/<rrd name>_report/:
    bag_summary.json   collisions, emergency-stop activity, stalls, speeds
    bag_timeline.csv   10 Hz: commanded v/w, odom speed, pose, nearest obstacle, path size
    bag_timeline.png   speed (commanded vs actual), estop distance, IMU jolts
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rerun as rr
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

NS = "/j100_0751"
TOPICS = ["/estop_0", f"{NS}/cmd_vel", f"{NS}/platform/odom/filtered", "/path_0", f"{NS}/sensors/imu_0/data"]
JOLT = 15.0          # m/s^2 horizontal, above this counts as an impact
STOP_GAP_CAM = 0.43  # depth_guard stop distance from the camera
SLOW_CAM = 1.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("rrd")
    ap.add_argument("--offset", type=float, default=0.036, help="robot clock - laptop clock (s)")
    args = ap.parse_args()

    run = os.path.splitext(os.path.basename(args.rrd))[0]   # run_06.rrd link -> run_06_report
    out = os.path.join(os.path.dirname(os.path.abspath(args.rrd)), f"{run}_report")
    os.makedirs(out, exist_ok=True)

    t = rr.dataframe.load_recording(args.rrd).view(
        index="log_time", contents={"/map/agent_0/position": ["Position2D"]}).select().read_all().to_pandas()["log_time"].dropna()
    T0 = t.iloc[0].value / 1e9 + args.offset

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: x.type for x in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[x for x in TOPICS if x in types]))
    E, C, O, P, I = [], [], [], [], []
    while r.has_next():
        tp, d, ts = r.read_next()
        m = deserialize_message(d, get_message(types[tp])); tt = ts / 1e9 - T0
        if tp == "/estop_0": E.append((tt, m.data))
        elif tp.endswith("cmd_vel"): C.append((tt, m.linear.x, m.angular.z))
        elif tp.endswith("filtered"):
            q = m.pose.pose.orientation
            O.append((tt, m.pose.pose.position.x, m.pose.pose.position.y,
                      np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)), m.twist.twist.linear.x))
        elif tp == "/path_0": P.append((tt, len(m.data) // 2))
        else: I.append((tt, m.linear_acceleration.x, m.linear_acceleration.y))

    C = pd.DataFrame(C, columns=["t", "cmd_v", "cmd_w"])
    O = pd.DataFrame(O, columns=["t", "x", "y", "yaw", "odom_v"])
    P = pd.DataFrame(P, columns=["t", "path_n"])
    I = pd.DataFrame(I, columns=["t", "ax", "ay"])
    E = pd.DataFrame(E, columns=["t", "obstacle_m"]) if E else pd.DataFrame({"t": C.t, "obstacle_m": np.nan})
    m = pd.merge_asof(C, E, on="t")
    m = pd.merge_asof(m, O, on="t")
    m = pd.merge_asof(m, P, on="t")
    m.to_csv(f"{out}/bag_timeline.csv", index=False)

    I["jolt"] = np.hypot(I.ax - I.ax.rolling(50, min_periods=1).median(), I.ay - I.ay.rolling(50, min_periods=1).median())
    impacts = []
    for _, row in I[I.jolt > JOLT].iterrows():          # merge spikes within 1 s
        if not impacts or row.t - impacts[-1][0] > 1.0:
            impacts.append([row.t, row.jolt])
        else:
            impacts[-1][1] = max(impacts[-1][1], row.jolt)

    run_part = m[m.t >= 0]
    active = run_part[run_part.path_n > 0]
    dist = active.obstacle_m
    stall = run_part[(run_part.cmd_v > 0.05) & (run_part.odom_v.abs() < 0.03)]
    d_xy = np.hypot(np.diff(O[O.t >= 0].x), np.diff(O[O.t >= 0].y)).sum()
    summary = {
        "run": run,
        "bag_span_s": [round(m.t.min(), 1), round(m.t.max(), 1)],
        "distance_odom_m": round(float(d_xy), 2),
        "impacts_over_%.0f_mps2" % JOLT: [[round(a, 1), round(b / 9.81, 1)] for a, b in impacts],
        "impact_unit": "[t_s, peak_g]",
        "estop_share_while_navigating_pct": {
            "blind": round(100 * (dist == 0).mean(), 1),
            "stopped": round(100 * ((dist > 0) & (dist <= STOP_GAP_CAM)).mean(), 1),
            "slowed": round(100 * ((dist > STOP_GAP_CAM) & (dist < SLOW_CAM)).mean(), 1),
            "clear": round(100 * (dist >= SLOW_CAM).mean(), 1),
        },
        "closest_obstacle_cam_m": round(float(dist[dist > 0].min()), 2) if (dist > 0).any() else None,
        "stall_s (cmd>0.05, odom<0.03)": round(len(stall) * 0.1, 1),
        "max_cmd_v": round(float(run_part.cmd_v.max()), 2),
        "max_odom_v": round(float(run_part.odom_v.max()), 2),
    }
    with open(f"{out}/bag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    ax[0].plot(m.t, m.cmd_v, lw=1, label="commanded v"); ax[0].plot(m.t, m.odom_v, lw=1, label="odom v")
    ax[0].set_ylabel("m/s"); ax[0].legend(fontsize=8)
    ax[1].plot(m.t, m.obstacle_m.clip(upper=3), lw=1, color="k")
    ax[1].axhline(STOP_GAP_CAM, color="r", ls="--", lw=0.8, label="stop"); ax[1].axhline(SLOW_CAM, color="orange", ls="--", lw=0.8, label="slow")
    ax[1].set_ylabel("nearest obstacle\n(camera, m)"); ax[1].legend(fontsize=8)
    ax[2].plot(I.t, I.jolt, lw=0.6, color="tab:red"); ax[2].axhline(JOLT, color="k", ls=":", lw=0.8)
    ax[2].set_ylabel("IMU jolt (m/s²)"); ax[2].set_xlabel("time since OneMap start (s)")
    for a, _ in impacts:
        for x in ax: x.axvline(a, color="red", alpha=0.4)
    fig.suptitle(f"{run}: bag (robot side)"); fig.tight_layout(); fig.savefig(f"{out}/bag_timeline.png", dpi=110)

    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
