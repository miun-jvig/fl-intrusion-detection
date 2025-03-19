import flwr as fl
import tensorflow as tf
from flwr.common import Metrics, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
from model.model import create_model
from data.data_loader import load_data

# run on GPU if possible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print('Training on GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# data
test_path = "datasets/global_test.csv"
x_test, y_test = load_data(test_path)
input_dim = x_test.shape[1]
num_classes = y_test.shape[1]

# model
global_model = create_model(input_dim, num_classes)


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
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""

    # Initialize model parameters
    ndarrays = global_model.get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
