#!/usr/bin/env bash
# Robot side: record an MCAP bag of a OneMap hardware run, locally on the robot
# (no Wi-Fi bandwidth used). Pair it with the server's .rrd by using the same RUN name:
#
#   robot : bash ~/onemap_jackal/record_bag.sh run_03            [--lidar]
#   laptop: ONEMAP_RUN_NAME=run_03 python3 -m on_hardware.server_side.main -c config/mon/eval_multi_conf_hardware.yaml
#
# Size: ~1.5 MB/s (~5 GB/h) without lidar, ~8 MB/s (~28 GB/h) with --lidar.
#
# Stop with Ctrl+C before powering the robot off. If it was cut off anyway (no metadata.yaml),
# the data is still there: ros2 bag reindex ~/onemap_bags/<RUN> -s mcap
set -e
RUN=${1:-onemap_$(date +%Y%m%d_%H%M%S)}
OUT=~/onemap_bags/${RUN}
mkdir -p ~/onemap_bags
[ -e "$OUT" ] && { echo "$OUT already exists, pick another run name"; exit 1; }

NS=/j100_0751
TOPICS=(
    # OneMap I/O
    /rgbs_0/compressed /depths_0/compressedDepth   # camera as sent to the server
    /camera/camera/color/camera_info               # intrinsics, for offline replay
    /odometries_0 /path_0 /target_object_0 /spin_0 # robot pose -> server, waypoints/target/scan <- server
    /estop_0                                       # jackal_node emergency stop: nearest obstacle ahead (m)
    # base
    $NS/platform/odom $NS/platform/odom/filtered $NS/platform/cmd_vel_unstamped $NS/cmd_vel
    $NS/platform/joint_states $NS/sensors/imu_0/data
    $NS/tf $NS/tf_static
    $NS/platform/emergency_stop $NS/platform/bms/state $NS/diagnostics_agg
    $NS/sensors/lidar3d_0/scan                     # 2D scan is small, always on
)
[ "$2" = "--lidar" ] && TOPICS+=($NS/sensors/lidar3d_0/points)

echo "Recording ${#TOPICS[@]} topics to $OUT (Ctrl+C to stop)"
exec ros2 bag record -s mcap -o "$OUT" "${TOPICS[@]}"
