import flwr as fl
import csv
from flwr.common import Metrics, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple, Dict, Optional
from model.model import create_model
from data.data_loader import load_data
from strategy.strategy import FedAvgWithModelSaving


# data
test_path = "/home/joelv/fl-iot/datasets/global_test.csv"
x_test, y_test = load_data(test_path)
input_dim = x_test.shape[1]
num_classes = y_test.shape[1]

# model
global_model = create_model(input_dim, num_classes)
history_file = "evaluation_history.csv"
history = []


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit results using weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# The `evaluate` function will be called by Flower after every round
def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar],)\
        -> Optional[Tuple[float, Dict[str, Scalar]]]:
    global_model.set_weights(parameters)
    loss, accuracy = global_model.evaluate(x_test, y_test)

    history.append((server_round, accuracy, loss))  # Store in memory
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")

    # Save history to file every round
    with open(history_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round_number", "accuracy", "loss"])
        writer.writerows(history)

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""

    # Initialize model parameters
    ndarrays = global_model.get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvgWithModelSaving(
        save_path='checkpoints',
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
