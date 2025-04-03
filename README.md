Note: this project is actively being developed. Some features such as the visualization script are still in progress.

# Introduction
The aim of this project is to explore federated learning for intrusion detection in a practical implementation on IoT-devices and see how it compares to centralized learning. The implementation uses Flower, TensorFlow, Raspberry Pi's, differential privacy, and wandb for tracking and visualization. The dataset used is the Edge-IIoTset Cyber Security Dataset of IoT & IIoT (https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot), made by Mohamed Amine FERRAG et al.

# Requisites
The code can be run using either embedded devices (I used five Raspberry Pi 3 with 16GB SD-card) or through simulation (see instructions).

* A Linux-based server is required when running on embedded devices due to dependency compatibility and network constraints
* Python 3.10+
* Edge-IIoTset Dataset in /datasets

# Instructions

## 1. Preprocessing
Download the Edge-IIoTset and extract the contents into "/datasets". Open preprocessing.py and run the code to get a stratified split of the set (change num_clients to how many clients you want). Copy the preprocessed dataset (output from preprocessing.py) to your Raspberry Pi in a location of your choice.

## 2. Execution and logging
To simulate, change options.num-supernodes in pyproject.toml to num_client and open a terminal to start the simulation with the command "flwr run . fl-iot-local".

To run using the embedded devices:

1. Start a flower superlink on your server with the command _flower-superlink --insecure_
2. Start your supernodes on your embedded devices with the command _flower-supernode --insecure --superlink="YOUR_IP:9092" --node-config="dataset-path='YOUR_DATA_LOCATION/preprocessed_i.csv', partition-id='i'"_
3. Start the FL process with the command _flwr run . fl-iot --stream_

Logs will now be created and put into .../outputs/DATE/TIMESTAMP and will contain:

* fit_results.json: training and validation accuracy
* evaluation_results.json: federated and centralized evaluation accuracy
* The saved .keras models that achieved the best accuracy during your run

Additionally, wandb is also logging these results.

## 3. Visualization
Visualization can be done by using wandb or by running the script in visualization.py (which is not finished yet).

# Results
The model is currently achieving a 96 % federated evaluation accuracy on the global test set using 40 FL rounds, 3 local epochs, and a batch size of 64. In comparison, my centralized implementation achieves a 97 % accuracy on the same test set.
