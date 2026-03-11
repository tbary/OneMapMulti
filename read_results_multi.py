from eval.habitat_multi_evaluator import HabitatMultiEvaluator
from config import load_eval_config

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, None)
    evaluator.read_results('results_multi/', "s",  None)

if __name__ == "__main__":
    main()
