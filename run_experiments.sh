#!/bin/bash

while [ ! -f "results_multi_one/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1.yaml
done

while [ ! -f "results_multi_two/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_2.yaml
done

while [ ! -f "results_multi_three/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_3.yaml
done
