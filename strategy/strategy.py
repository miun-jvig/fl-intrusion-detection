import json
from logging import INFO
import wandb
from apps.task import load_model, create_run_dir
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement

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
        self.run_config = run_config
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.total_rounds = run_config["num-server-rounds"]
        self.use_dp = run_config["use-dp"]
        self.l2_norm_clip = run_config["l2-norm-clip"]
        self.noise_multiplier = run_config["noise-multiplier"]
        self.use_wandb = use_wandb
        self.dp_args = dict(use_dp=self.use_dp, l2_norm_clip=self.l2_norm_clip, noise_multiplier=self.noise_multiplier)

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
            model = load_model(**self.dp_args)
            model.set_weights(ndarrays)

            # Save the PyTorch model
            file_name = (self.save_path / f"model_state_acc_{accuracy:.3f}_round_{round}.h5")
            model.save_weights(str(file_name))

    def _compute_and_store_privacy(self, num_examples: int, batch_size: int, local_epochs: int, noise_multiplier: float,
                                   delta: float):
        """Compute (ε,δ) for on the final round and log it to W&B / store_results."""
        total_epochs = local_epochs * self.total_rounds

        report = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            num_epochs=total_epochs,
            delta=delta,
        )

        self._store_results(
            tag="dp_metrics",
            metric_type="dp",
            results_dict={"Report": report},
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
        # first call the parent so you still get the global loss & metrics
        loss, metrics = super().aggregate_fit(server_round, results, failures)

        if server_round == self.total_rounds and self.use_dp:
            self._compute_and_store_privacy(
                metrics["num_examples"],
                self.run_config["batch-size"],
                self.run_config["local-epochs"],
                self.run_config["noise-multiplier"],
                self.run_config["delta"],
            )

        client_dict = {}
        for client_proxy, fit_res in results:
            pid = fit_res.metrics.get("partition_id", client_proxy.cid)
            m = fit_res.metrics
            client_dict[f"client_{pid}/train_acc"] = m["accuracy"]
            client_dict[f"client_{pid}/val_acc"] = m["val_accuracy"]
            client_dict[f"client_{pid}/train_loss"] = m["loss"]
            client_dict[f"client_{pid}/val_loss"] = m["val_loss"]

        self.store_results_and_log(
            server_round=server_round,
            tag=f"client_fit",
            metric_type="fit",
            results_dict=client_dict
        )

        self.store_results_and_log(
            server_round=server_round,
            tag="all_clients_fit",
            metric_type="fit",
            results_dict={**metrics}
        )

        return loss, metrics

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        if server_round != 0:
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
