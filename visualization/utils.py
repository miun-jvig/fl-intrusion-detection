import numpy as np
from sklearn.preprocessing import LabelEncoder
from apps.task import one_hot_encode, load_model
from data_loading.data_loader import load_data
from pathlib import Path
import json


def load_run_config(run_dir: Path):
    return json.load((run_dir / "run_config.json").open())


def load_predictions_and_classes(model_path, test_data_path):
    # Load data
    df_test, x_test, y_test = load_data(test_data_path)

    # Variables
    model = load_model()

    model.load_weights(model_path)
    predictions = model.predict(x_test)

    predicted_classes, true_classes = one_hot_encode(y_test, predictions)
    class_names = get_class_names(df_test['Attack_type'])

    return class_names, predicted_classes, true_classes


def get_class_names(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le.classes_


def group_classes(classes, class_map):
    return np.array([class_map[c] for c in classes])
