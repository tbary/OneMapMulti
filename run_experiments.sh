#!/bin/bash

# while [ ! -f "results_multi_one/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1.yaml
# done

# while [ ! -f "results_multi_two/greed/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_2_greed.yaml
# done

# while [ ! -f "results_multi_two/certainty_greed/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_2_certainty_greed.yaml
# done

# # while [ ! -f "results_multi_two/size_greed/state/state_235.txt" ]; do
# #     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_2_size_greed.yaml
# # done

while [ ! -f "results_multi_two/div_greed/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_2_div_greed.yaml
done

# while [ ! -f "results_multi_three/greed/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_3_greed.yaml
# done

# while [ ! -f "results_multi_three/certainty_greed/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_3_certainty_greed.yaml
# done

while [ ! -f "results_multi_three/div_greed/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_3_div_greed.yaml
done

# while [ ! -f "results_multi_three/size_greed/state/state_235.txt" ]; do
#     python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_3_size_greed.yaml
# done

while [ ! -f "results_multi_four/div_greed/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_4_div_greed.yaml
done