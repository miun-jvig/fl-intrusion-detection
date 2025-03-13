from flwr.client import ClientApp, Client, NumPyClient
from flwr.common import Context
from model.model import create_model
from config.config_loader import client_cfg
from data.data_loader import load_dataset

dataset_path = client_cfg['dataset_path']
client_epochs = int(client_cfg['client_epochs'])
batch_size = int(client_cfg['batch_size'])


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, model, data, local_epochs, batch_size):
        self.cid = partition_id
        self.model = model
        self.data = data
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def get_parameters(self, config):
        print('[Client {}] get_parameters'.format(self.cid))
        return self.model.get_weights()

    def fit(self, parameters, config):
        print('[Client {}] fit'.format(self.cid))
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
        print('[Client {}] evaluate, config: {}'.format(self.cid, config))
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['x_test'], self.data['y_test'])
        return loss, len(self.data['y_test']), {'accuracy': float(accuracy)}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # print('Creating [Client {}]'.format(cid))
    data = load_dataset(dataset_path)

    input_dim = data['x_train'].shape[1]
    num_classes = data['y_train'].shape[1]
    model = create_model(input_dim, num_classes)

    # Read the node_config to fetch datasets partition associated to this node
    partition_id = context.node_config["partition-id"]

    return FlowerClient(partition_id, model, data, client_epochs, batch_size).to_client()


client = ClientApp(client_fn=client_fn)
