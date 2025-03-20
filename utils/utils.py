import pandas as pd
import numpy as np


def one_hot_encode(y_test, predicted_classes):
    predicted_classes = np.argmax(predicted_classes, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    return predicted_classes, y_test_labels


def save_history(history, round_num, filepath):
    """Saves history of training round to a .csv file, later used for creating training/loss history graphs"""
    hist_df = pd.DataFrame(history.history)
    hist_df['round'] = round_num
    if round_num == 1:
        with open(filepath, mode='w') as f:
            hist_df.to_csv(f, index=False)
    else:
        with open(filepath, mode='a') as f:
            hist_df.to_csv(f, header=False, index=False)
