import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple, Dict, Optional
from strategy.strategy import CustomFedAvg
from model.model import load_model
from data.data_loader import load_data


def get_evaluate_fn():
    test_path = '/home/joelv/fl-iot/datasets/global_test.csv'
    x_test, y_test = load_data(test_path)

    # The `evaluate` function will be called by Flower after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar], ) \
            -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = load_model()
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit results using weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""

    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        fraction_fit=context.run_config["fraction-fit"],
        use_wandb=context.run_config["use-wandb"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
