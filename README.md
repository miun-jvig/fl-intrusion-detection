# Introduction
The aim of this project is to explore federated learning and differential privacy for intrusion detection in a practical implementation on IoT-devices and see how it compares to centralized learning. The implementation uses Flower, TensorFlow, tensorflow_privacy, Raspberry Pi's, and wandb for tracking and visualization. The dataset used is the Edge-IIoTset Cyber Security Dataset of IoT & IIoT (https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).

The implementation is able to perform federated learning and federated learning with differential privacy on the Edge-IIoTset. It also logs data locally on the server as well as on wandb. In addition, it also offers a way to help with hyperparameter tuning by introducing run_sweep.py, which uses wandb to display grid/random paralell coordinates plots. It also has built-in visualization.

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
Download the Edge-IIoTset and create a folder called /datasets under /fl-intrusion-detection-dp/, then extract the contents of the Edge-IIoTset into that folder. Run `python scripts/preprocessing.py --test-size x --num-clients y` to get a stratified split of the set with x test-size and y number of clients. Copy the preprocessed dataset (output from preprocessing.py) to your Raspberry Pi in a location of your choice. If you just want to simulate a run, then you can keep the preprocessed dataset in /datasets.

## 2. Execution and logging
**To simulate the setup**

1. Change options.num-supernodes in pyproject.toml to how many clients you want (should be equal to the number of stratified splits you did in preprocessing)
2. Update the variable "dataset_path" to `'.../datasets/preprocessed_{partition_id}.csv'` in client_fn() under `apps/client_app.py`
3. Open a terminal to start the simulation with the command `flwr run . fl-iot-local`

**To run on embedded devices**

To run using embedded devices such as Raspberry Pi, you first need to set it up by following https://github.com/adap/flower/blob/main/examples/embedded-devices/device_setup.md. Then install TensorFlow==2.18.0, tensorflow_privacy, flwr, scikit-learn and pandas on the devices.

After that:

1. Start a flower superlink on your Linux-based server with the command `flower-superlink --insecure`
3. Start your supernodes on your embedded devices with the command `flower-supernode --insecure --superlink="SERVER_IP:9092" --node-config="dataset-path='LOCAL_DEVICE_DATA_LOCATION/preprocessed_PARTITION-ID.csv', partition-id='PARTITION-ID'"`. Note that partition-ID should match the ID on the preprocessed file
4. Start the FL process on the server with the command `flwr run . fl-iot --stream`

Logs will be created in .../outputs/DATE/TIMESTAMP/ and will contain:

* fit_results.json: aggregated training and validation accuracy
* evaluation_results.json: aggregated federated and centralized evaluation accuracy
* dp_results.json: a privacy statement containing information about the DP guarantee
* The saved .keras models that achieved the best accuracy during your run

Additionally, wandb is also logging these results.

## 3. Hyperparameter tuning
To visualize and make hyperparameter tuning easier:

1. Open `scripts/sweep.yaml` and edit the 'parameters:' section. **⚠️** Make sure each key matches exactly what’s defined in `pyproject.toml`.
2. Run the script using:

    ```
    python scripts/run_sweep.py \
    --sweep-spec scripts/sweep.yaml \
    --count TOTAL_TRIALS
    ```

This will start a W&B sweep with your specified trials. Results and metrics will be streamed to your W&B project. **Note:** I know it's a bit unintuitive but `use-wandb` must be set to `false` in `pyproject.toml`.

## 4. Visualization
Visualization is done automatically if use-wandb is set to true in `pyproject.toml` or by running `python -m scripts.visualize`. Note that you have to change some variables in `scripts/visualize.py` to match the output folder and the model you wish to visualize.

# Results
### FL
- **Evaluation Accuracy:** 100 % on 2-class, 95 % on 6-class, and 94 % on 15-class classification.
- **Setup:** 25 rounds, 3 local epochs, batch size = 800

### FL + DP
- **Evaluation Accuracy:** 95 % on 2-class, 85 % on 6-class, and 82 % on 15-class classification.
- **Privacy Budget:** ε ≈ 10, δ = 1 × 10⁻⁷  
- **DP Parameters:** L₂-norm clip = 1.0, noise multiplier = 0.293
- **Setup:** 25 rounds, 5 local epochs, batch size = 800
