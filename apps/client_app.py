from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from apps.task import load_model
from data_loading.data_loader import load_dataset


class FlowerClient(NumPyClient):
    def __init__(self, model, partition_id, data, local_epochs, batch_size):
        self.model = model
        self.partition_id = partition_id
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val)
        )
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {'accuracy': float(accuracy)}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    partition_id = context.node_config["partition-id"]
    # Read the node_config to know where dataset is located
    dataset_path = context.node_config["dataset-path"]
    # dataset_path = fr'C:\Users\joelv\PycharmProjects\thesis-ML-FL\datasets\preprocessed_{partition_id}.csv'
    data = load_dataset(dataset_path)
    model = load_model()

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(model, partition_id, data, local_epochs, batch_size).to_client()


app = ClientApp(client_fn)
