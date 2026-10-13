#!/usr/bin/env bash
# Server side: turn the robot's compressed camera streams back into the raw
# rgbs_0 / depths_0 topics that server_side/ros_topics.py subscribes to.
#
# Robot side: ros2 launch ~/onemap_minimal.launch.py   (compressed:=true is the default)
#   /rgbs_0/compressed         JPEG   -> /rgbs_0   (sensor_msgs/Image, rgb8)
#   /depths_0/compressedDepth  PNG    -> /depths_0 (sensor_msgs/Image, 16UC1, lossless)
#
# Needs: sudo apt install ros-humble-image-transport-plugins
set -e
source /opt/ros/humble/setup.bash

ros2 run image_transport republish compressed raw --ros-args \
    -r __node:=rgb_decompress_0 \
    -r in/compressed:=/rgbs_0/compressed -r out:=/rgbs_0 &
RGB=$!

ros2 run image_transport republish compressedDepth raw --ros-args \
    -r __node:=depth_decompress_0 \
    -r in/compressedDepth:=/depths_0/compressedDepth -r out:=/depths_0 &
DEPTH=$!

# `ros2 run` wrappers don't forward signals, so stop their republish children too
trap 'trap - INT TERM EXIT; pkill -P $RGB; pkill -P $DEPTH; kill $RGB $DEPTH 2>/dev/null' INT TERM EXIT
echo "Decompressing /rgbs_0 and /depths_0 (Ctrl+C to stop)"
wait
