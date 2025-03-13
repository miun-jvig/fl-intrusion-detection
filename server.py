import flwr as fl
import tensorflow as tf
from flwr.common import Metrics, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
from config.config_loader import server_cfg, client_cfg
from model.model import create_model
from data.data_loader import load_data

# run on GPU if possible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print('Training on GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# datasets
num_clients = int(client_cfg['num_clients'])
min_available_clients = int(server_cfg['min_available_clients'])
input_dim = int(server_cfg['input_dim'])
num_classes = int(server_cfg['num_classes'])
num_rounds = int(server_cfg['num_rounds'])
test_path = server_cfg['test_path']

# model
global_model = create_model(97, 15)


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
    x_test, y_test = load_data(test_path)
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
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn)
