import json
from logging import INFO
import wandb
from apps.task import load_model, create_run_dir
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement, compute_dp_sgd_privacy

PROJECT_NAME = "fl-iot"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.total_rounds = run_config["num-server-rounds"]
        self.use_dp = run_config["use-dp"]
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.eval_results = {}
        self.fit_results = {}
        self.dp_results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, metric_type: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        metric_mapping = {
            'evaluation': (self.eval_results, f"{self.save_path}/evaluation_results.json"),
            'fit': (self.fit_results, f"{self.save_path}/fit_results.json"),
            'dp': (self.dp_results, f"{self.save_path}/dp_results.json")
        }
        results_dict_to_store, file_path = metric_mapping.get(metric_type, (None, None))

        if tag in results_dict_to_store:
            results_dict_to_store[tag].append(results_dict)
        else:
            results_dict_to_store[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(results_dict_to_store, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = load_model()
            model.set_weights(ndarrays)
            # Save the PyTorch model
            file_name = (
                    self.save_path
                    / f"model_state_acc_{accuracy:.3f}_round_{round}.keras"
            )
            model.save(file_name)

    def _compute_and_store_privacy(self, num_examples: int, batch_size: int, local_epochs: int, noise_multiplier: float,
                                   delta: float, round_num: int):
        """Compute (ε,δ) for on the final round and log it to W&B / store_results."""
        if round_num == self.total_rounds and self.use_dp:
            dp_report = compute_dp_sgd_privacy_statement(
                number_of_examples=num_examples,
                batch_size=batch_size,
                num_epochs=local_epochs,
                noise_multiplier=noise_multiplier,
                delta=delta,
            )

            eps, opt_order = compute_dp_sgd_privacy(
                n=num_examples,
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                epochs=local_epochs,
                delta=delta,
            )

            # tidigare {"Report": dp_report}
            self._store_results(
                tag="dp_metrics",
                metric_type="dp",
                results_dict={"dp_epsilon": eps, "dp_delta": delta},
            )

    def store_results_and_log(self, server_round: int, tag: str, metric_type: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            metric_type=metric_type,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def aggregate_fit(self, server_round, results, failures):
        loss, metrics = super().aggregate_fit(server_round, results, failures)

        self._compute_and_store_privacy(metrics["num_examples"], metrics["batch_size"], metrics["local_epochs"],
                                        metrics["noise_multiplier"], metrics["delta"], server_round)

        fit_metrics = {
            "training_loss": metrics["training_loss"],
            "training_accuracy": metrics["training_accuracy"],
            "val_loss": metrics["val_loss"],
            "val_accuracy": metrics["val_accuracy"],
        }

        # Store and log fit metrics
        self.store_results_and_log(
            server_round=server_round,
            tag="fit_metrics",
            metric_type="fit",
            results_dict=fit_metrics,
        )
        return loss, metrics

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_evaluate_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            metric_type="evaluation",
            results_dict={"centralized_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            metric_type="evaluation",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
