Note: this project is actively being developed.

# Introduction
The aim of this project is to explore federated learning and differential privacy for intrusion detection in a practical implementation on IoT-devices and see how it compares to centralized learning. The implementation uses Flower, TensorFlow, tensorflow_privacy, Raspberry Pi's, and wandb for tracking and visualization. The dataset used is the Edge-IIoTset Cyber Security Dataset of IoT & IIoT (https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).

# Requisites
The code can be run using either embedded devices (I used five Raspberry Pi 3 with 16GB SD-card) or through simulation (see instructions).

* A Linux-based server is required when running on embedded devices (I used Ubuntu WSL)
* Python 3.10+
* The Edge-IIoTset dataset

# Dependencies
Install the dependencies defined in `pyproject.toml`.

```
# From a new python environment, run:
pip install -e .
```

# Instructions

Start by cloning this project, then:

## 1. Preprocessing
Download the Edge-IIoTset and create a folder called /datasets under /fl-intrusion-detection-dp/, then extract the contents of the Edge-IIoTset into that folder. Open preprocessing.py and run the code to get a stratified split of the set (change num_clients to how many clients you want). Copy the preprocessed dataset (output from preprocessing.py) to your Raspberry Pi in a location of your choice. If you just want to simulate a run, then you can keep the preprocessed dataset in /datasets.

## 2. Execution and logging
**To simulate the setup**

1. Change options.num-supernodes in pyproject.toml to how many clients you want (should be equal to the number of stratified splits you did in preprocessing)
2. Update the variable "dataset_path" to `'.../datasets/preprocessed_{partition_id}.csv'` in client_fn()
3. Open a terminal to start the simulation with the command `flwr run . fl-iot-local`

**To run on embedded devices**

To run using embedded devices such as Raspberry Pi, you first need to set it up by following https://github.com/adap/flower/blob/main/examples/embedded-devices/device_setup.md. Then install TensorFlow==2.18.0, flwr, scikit-learn and pandas on the devices.

After that:

1. Start a flower superlink on your Linux-based server with the command `flower-superlink --insecure`
3. Start your supernodes on your embedded devices with the command `flower-supernode --insecure --superlink="SERVER_IP:9092" --node-config="dataset-path='LOCAL_DEVICE_DATA_LOCATION/preprocessed_PARTITION-ID.csv', partition-id='PARTITION-ID'"`
4. Start the FL process on the server with the command `flwr run . fl-iot --stream`

Logs will be created in .../outputs/DATE/TIMESTAMP/ and will contain:

* fit_results.json: aggregated training and validation accuracy
* evaluation_results.json: aggregated federated and centralized evaluation accuracy
* dp_results.json: a privacy statement containing information about the DP guarantee
* The saved .keras models that achieved the best accuracy during your run

Additionally, wandb is also logging these results.

## 3. Visualization
Visualization can be done by using wandb or by running the script in visualization.py.

# Results
The model (non-DP) is currently achieving a 96 % federated/centralized evaluation accuracy on the global test set using 40 FL rounds, 3 local epochs, and a batch size of 64. In comparison, my centralized implementation achieves a 97 % accuracy on the same test set.
