import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from apps.task import one_hot_encode
from data_loading.data_loader import load_data


def load_predictions_and_classes(model_path, test_data_path):
    df_test, x_test, y_test = load_data(test_data_path)

    model = tf.keras.models.load_model(model_path)
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
