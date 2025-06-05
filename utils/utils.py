import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def one_hot_encode(y_test, predicted_classes):
    predicted_classes = np.argmax(predicted_classes, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    return predicted_classes, y_test_labels


def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)


def make_class_weight(y_onehot):
    y_inds = np.argmax(y_onehot, axis=1)
    classes = np.unique(y_inds)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_inds,
    )
    return dict(zip(classes.tolist(), weights.tolist()))
