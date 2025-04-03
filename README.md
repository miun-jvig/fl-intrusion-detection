Note: this project is actively being developed. Some features such as the visualization script are still in progress.

# Introduction
The aim of this project is to explore federated learning for intrusion detection in a practical implementation on IoT-devices and see how it compares to centralized learning. The implementation uses Flower, TensorFlow, Raspberry Pi's, differential privacy, and wandb for tracking and visualization. The dataset used is the Edge-IIoTset Cyber Security Dataset of IoT & IIoT (https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).

# Requisites
The code can be run using either embedded devices (I used five Raspberry Pi 3 with 16GB SD-card) or through simulation (see instructions).

* A Linux-based server is required when running on embedded devices (I used Ubuntu WSL)
* Python 3.10+
* Edge-IIoTset Dataset in /datasets

# Instructions

## 1. Preprocessing
Download the Edge-IIoTset and extract the contents into "/datasets". Open preprocessing.py and run the code to get a stratified split of the set (change num_clients to how many clients you want). Copy the preprocessed dataset (output from preprocessing.py) to your Raspberry Pi in a location of your choice.

## 2. Execution and logging
**To simulate the setup**

1. Change options.num-supernodes in pyproject.toml to how many clients you want 
2. Update the variable "dataset_path" to '/datasets/preprocessed_{partition_id}.csv' in client_fn()
3. Open a terminal to start the simulation with the command `flwr run . fl-iot-local`

**To run on embedded devices**

To run using embedded devices such as Raspberry Pi, you first need to set it up by following https://github.com/adap/flower/blob/main/examples/embedded-devices/device_setup.md, as well as installing pandas and scikit-learn on the devices.

After that:

1. Start a flower superlink on your server with the command `flower-superlink --insecure`
3. Start your supernodes on your embedded devices with the command `flower-supernode --insecure --superlink="SERVER_IP:9092" --node-config="dataset-path='LOCAL_DEVICE_DATA_LOCATION/preprocessed_i.csv', partition-id='i'"`
4. Start the FL process with the command `flwr run . fl-iot --stream`

Logs will be created in .../outputs/DATE/TIMESTAMP/ and will contain:

* fit_results.json: aggregated training and validation accuracy
* evaluation_results.json: aggregated federated and centralized evaluation accuracy
* The saved .keras models that achieved the best accuracy during your run

Additionally, wandb is also logging these results.

## 3. Visualization
Visualization can be done by using wandb or by running the (unfinished) script in visualization.py.

# Results
The model is currently achieving a 96 % federated evaluation accuracy on the global test set using 40 FL rounds, 3 local epochs, and a batch size of 64. In comparison, my centralized implementation achieves a 97 % accuracy on the same test set.
