import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping
from typing import List, Tuple
from strategy.strategy import CustomFedAvg
from apps.task import load_model
from data_loading.data_loader import load_data


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
    aggregated_metrics = {
        "training_accuracy": sum(m["accuracy"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "val_accuracy": sum(m["val_accuracy"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "training_loss": sum(m["loss"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
        "val_loss": sum(m["val_loss"] * num_examples for num_examples, m in metrics) / sum(
            num_examples for num_examples, _ in metrics),
    }
    return aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""
    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Test data_loading
    # test_path = '/home/joelv/fl-intrusion-detection/datasets/global_test.csv'
    test_path = r'C:\Users\joelv\PycharmProjects\thesis-ML-FL\datasets\global_test.csv'
    _, x_test, y_test = load_data(test_path)

    # Define the strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        fraction_fit=context.run_config["fraction-fit"],
        use_wandb=context.run_config["use-wandb"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(x_test, y_test),
        fit_metrics_aggregation_fn=fit_metrics_fn,
        initial_parameters=parameters,
    )
    # Wrap the strategy with DifferentialPrivacyServerSideAdaptiveClipping
    dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
        strategy,
        noise_multiplier=0.05,
        num_sampled_clients=5,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
