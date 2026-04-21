import os
from eval.multi_results_read import MultiResultsReader
from config import load_eval_config
import numpy as np

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = MultiResultsReader(eval_config.EvalConf)

    results_dir = eval_config.EvalConf.results_path
    
    data = os.path.join(results_dir, "data.pkl") if os.path.exists(os.path.join(results_dir, "data.pkl")) else None
    evaluator.read_results(results_dir, "Episode Success",  data)

if __name__ == "__main__":
    np.seterr(all='raise')
    main()
