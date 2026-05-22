#!/bin/bash

MAX_JOBS=4

PARAMS=(
# "config/mon/eval_multi_conf_1.yaml"

# "config/mon/eval_multi_conf_2_certainty_greed.yaml"
"config/mon/eval_multi_conf_2_div_greed.yaml"
# "config/mon/eval_multi_conf_2_greed.yaml"
# "config/mon/eval_multi_conf_2_size_greed.yaml"

# "config/mon/eval_multi_conf_3_certainty_greed.yaml"
"config/mon/eval_multi_conf_3_div_greed.yaml"
# "config/mon/eval_multi_conf_3_greed.yaml"
# "config/mon/eval_multi_conf_3_size_greed.yaml"

# "config/mon/eval_multi_conf_4_certainty_greed.yaml"
"config/mon/eval_multi_conf_4_div_greed.yaml"
# "config/mon/eval_multi_conf_4_greed.yaml"
# "config/mon/eval_multi_conf_4_size_greed.yaml"
)

running=0

for CONFIG in "${PARAMS[@]}"; do
    python3 read_results_multi.py --config "$CONFIG" &

    ((running++))

    if (( running >= MAX_JOBS )); then
        wait -n   # wait for one job to finish
        ((running--))
    fi
done

wait
echo "Done."