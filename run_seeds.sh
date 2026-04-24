#!/bin/bash

while [ ! -f "results_multi_one/seed_0/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1_seed_0.yaml
done

while [ ! -f "results_multi_one/seed_1/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1_seed_1.yaml
done

while [ ! -f "results_multi_one/seed_2/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1_seed_2.yaml
done

while [ ! -f "results_multi_one/seed_3/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1_seed_3.yaml
done

while [ ! -f "results_multi_one/seed_4/state/state_235.txt" ]; do
    python3 eval_habitat_multi.py --config config/mon/eval_multi_conf_1_seed_4.yaml
done