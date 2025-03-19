import flwr as fl
import pickle
import numpy as np
from flwr.common import Metrics, NDArrays, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple, Dict, Optional
from model.model import create_model
from data.data_loader import load_data
from logging import INFO
from pathlib import Path


# data
test_path = "/datasets/global_test.csv"
x_test, y_test = load_data(test_path)
input_dim = x_test.shape[1]
num_classes = y_test.shape[1]

# model
global_model = create_model(input_dim, num_classes)


class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """
    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""

        # convert parameters to list of NumPy arrays
        # this will make things easy if you want to load them into a
        # PyTorch or TensorFlow model later
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'globa_parameters': ndarrays}
        filename = str(self.save_path/f"parameters_round_{server_round}.pkl")
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)


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
    weights = np.array(parameters, dtype=object)
    global_model.set_weights(weights)
    loss, accuracy = global_model.evaluate(x_test, y_test)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""

    # Define the strategy
    strategy = FedAvgWithModelSaving(
        save_path='checkpoints',
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
