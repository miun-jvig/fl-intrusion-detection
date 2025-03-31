import numpy as np
import json
import os
from flwr.common.typing import UserConfig
from pathlib import Path
from datetime import datetime

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def one_hot_encode(y_test, predicted_classes):
    predicted_classes = np.argmax(predicted_classes, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    return predicted_classes, y_test_labels
