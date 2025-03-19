from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from model.model import create_model
from data.data_loader import load_dataset


class FlowerClient(NumPyClient):
    def __init__(self, model, data, local_epochs, batch_size):
        self.model = model
        self.data = data
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.data['x_train'],
            self.data['y_train'],
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            validation_data=(self.data['x_val'], self.data['y_val'])
        )
        evaluation_metrics = {'loss': history.history['loss'], 'accuracy': history.history['accuracy']}
        return self.model.get_weights(), len(self.data['y_val']), evaluation_metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['x_test'], self.data['y_test'])
        return loss, len(self.data['y_test']), {'accuracy': float(accuracy)}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to know where dataset is located
    dataset_path = context.node_config["dataset-path"]
    data = load_dataset(dataset_path)
    input_dim = data['x_train'].shape[1]
    num_classes = data['y_train'].shape[1]
    model = create_model(input_dim, num_classes)

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(model, data, local_epochs, batch_size).to_client()


app = ClientApp(client_fn)
