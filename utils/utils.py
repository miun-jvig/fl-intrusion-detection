import json
import numpy as np


def one_hot_encode(y_test, predicted_classes):
    predicted_classes = np.argmax(predicted_classes, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    return predicted_classes, y_test_labels


def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)


def print_data_sizes(x_train, y_train, x_val, y_val, x_test, y_test):
    print(f"Training data: {x_train.shape}, {y_train.shape}")
    print(f"Validation data: {x_val.shape}, {y_val.shape}")
    print(f"Testing data: {x_test.shape}, {y_test.shape}")
