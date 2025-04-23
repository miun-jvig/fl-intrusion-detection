from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger
from apps.task import load_model, load_dp_model
from data_loading.data_loader import load_dataset
from logging import INFO
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, model, data, local_epochs, batch_size, noise_multiplier, delta):
        self.partition_id = partition_id
        self.model = model
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.delta = delta

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
            # pass the raw DP params along to be used later for dp evaluation
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
        }
        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {'accuracy': float(accuracy)}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read run_config to fetch hyperparameters relevant to this run
    partition_id = context.node_config["partition-id"]
    run_config = context.run_config
    use_dp = run_config["use-dp"]
    l2_norm_clip = run_config["l2-norm-clip"]
    noise_multiplier = run_config["noise-multiplier"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    delta = context.run_config["delta"]

    # Read the node_config to know where dataset is located
    dataset_path = context.node_config["dataset-path"]
    data = load_dataset(dataset_path)

    # Load model
    if use_dp:
        logger.log(INFO, "⚙️ Using DP sequential model.")
        model = load_dp_model(l2_norm_clip, noise_multiplier)
    else:
        logger.log(INFO, f"⚙️ Using non-DP sequential model.")
        model = load_model()

    return FlowerClient(model, partition_id, data, local_epochs, batch_size, noise_multiplier, delta).to_client()


app = ClientApp(client_fn)
