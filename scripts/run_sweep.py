import subprocess
import json
import wandb
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["WANDB_DIR"] = str(PROJECT_ROOT / "wandb")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a W&B sweep")
    parser.add_argument(
        "--sweep-spec",
        type=Path,
        required=True,
        help="Path to a YAML file describing your sweep (parameters & metric)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=9,
        help="Number of total trials to run",
    )
    return parser.parse_args()


def load_sweep_spec(path: Path):
    """YAML example:
    method: grid
    metric:
      name: centralized_evaluate_accuracy
      goal: maximize
    parameters:
      l2_norm_clip: {values: [1.0, 2.0, 4.0]}
      noise_multiplier: {values: [0.5, 1.0, 2.0]}
    """
    with open(path) as fp:
        return yaml.safe_load(fp)


def find_latest_run_dir():
    today = datetime.now().strftime("%Y-%m-%d")
    base = PROJECT_ROOT / "outputs" / today
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def run_trial():
    run = wandb.init(reinit=True, project="fl-iot")
    cfg = run.config

    run_cfg_str = (
        f"l2-norm-clip={cfg.l2_norm_clip} "
        f"noise-multiplier={cfg.noise_multiplier} "
    )

    subprocess.run(
        ["flwr", "run", ".", "fl-iot-local", "--run-config", run_cfg_str],
        check=True,
    )

    base_dir = find_latest_run_dir()
    evals = json.load(open(base_dir / "evaluation_results.json"))["centralized_evaluate"][-1]
    dp = json.load(open(base_dir / "dp_results.json"))["dp_metrics"][-1]

    run.log({
        "centralized_evaluate_loss": evals["centralized_evaluate_loss"],
        "centralized_evaluate_accuracy": evals["centralized_evaluate_accuracy"],
        **dp,
    })
    run.summary["centralized_evaluate_accuracy"] = evals["centralized_evaluate_accuracy"]
    if "dp_epsilon" in dp:
        run.summary["dp_epsilon"] = dp["dp_epsilon"]
    run.finish()


def main():
    args = parse_args()
    spec = load_sweep_spec(args.sweep_spec)

    sweep_id = wandb.sweep(spec, project="fl-iot")
    wandb.agent(sweep_id, function=run_trial, count=args.count)


if __name__ == "__main__":
    main()
