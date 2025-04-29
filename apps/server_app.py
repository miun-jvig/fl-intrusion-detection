import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple
from strategy.strategy import CustomFedAvg
from apps.task import load_model
from data_loading.data_loader import load_data
from pathlib import Path
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_evaluate_fn(x_test, y_test):
    # The `evaluate` function will be called by Flower after every round
    def evaluate(server_round, parameters_ndarrays, config):
        model = load_model()
        model.set_weights(parameters_ndarrays)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"centralized_evaluate_accuracy": accuracy}

    return evaluate


def fit_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Aggregate client metrics and extract accuracy and val_accuracy
    aggregated = {
        "training_accuracy": sum(m["accuracy"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "val_accuracy": sum(m["val_accuracy"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "training_loss": sum(m["loss"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "val_loss": sum(m["val_loss"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
    }
    worst_n = min(num_examples for num_examples, _ in metrics)
    aggregated["num_examples"] = worst_n
    return aggregated


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""
    # Read run_config to fetch hyperparameters relevant to this run
    run_config = context.run_config
    fraction_fit = run_config["fraction-fit"]
    use_wandb = run_config["use-wandb"]
    use_dp = run_config["use-dp"]
    fraction_evaluate = run_config["fraction-evaluate"]
    num_server_rounds = run_config["num-server-rounds"]
    l2_norm_clip = run_config["l2-norm-clip"]
    noise_multiplier = run_config["noise-multiplier"]

    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Test data_loading
    # test_path = Path.home() / 'fl-intrusion-detection' / 'datasets' / 'global_test.csv'
    test_path = PROJECT_ROOT / 'datasets' / 'global_test.csv'
    _, x_test, y_test = load_data(test_path)

    # Define the strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        fraction_fit=fraction_fit,
        use_wandb=use_wandb,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(x_test, y_test),
        fit_metrics_aggregation_fn=fit_metrics_fn,
        initial_parameters=parameters,
    )
    if use_dp:
        dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
            strategy=strategy,
            initial_clipping_norm=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_sampled_clients=5,
        )
        strategy = dp_strategy

    config = ServerConfig(num_rounds=num_server_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
