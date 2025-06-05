import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils.utils import one_hot_encode
from pathlib import Path
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_plot(ax, x, y, y_label, x_label, title, labels, fontsize=12):
    """Create plot"""
    ax.plot(x, 'b', y, 'r')
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(labels, fontsize=fontsize, loc='best')
    x_ticks = range(0, len(x), 5)
    ax.set_xticks(x_ticks)


def plot_hist(history, filename):
    """
    Plots training and validation accuracy and loss histories as two separate histograms.

    Args:
        history (tf.keras.callbacks.History or dict): The training history object or dictionary containing
                                                      'accuracy', 'val_accuracy', 'loss', and 'val_loss'.
        filename (str): The path to save the resulting plot image.
    """
    if isinstance(history, tf.keras.callbacks.History):
        history = history.history

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    # first plot
    create_plot(ax1, history['accuracy'], history['val_accuracy'], 'Accuracy Rate', 'Epoch',
                'Categorical Cross Entropy', ['Training Accuracy', 'Validation Accuracy'])

    # second plot
    create_plot(ax2, history['loss'], history['val_loss'], 'Loss', 'Epoch', 'Learning Curve',
                ['Training Loss', 'Validation Loss'])

    # save figure
    fig.tight_layout()
    plt.savefig(filename)


def plot_binary_matrix(df_test, y_true, y_pred, filename):
    le = LabelEncoder()
    le.fit(df_test['Attack_type'])
    class_names = le.classes_

    binary_predicted = np.array([0 if class_names[p] == 'Normal' else 1 for p in y_pred])
    binary_true = np.array([0 if class_names[t] == 'Normal' else 1 for t in y_true])

    report = classification_report(binary_true, binary_predicted, target_names=['Normal', 'Attack'])
    with open(PROJECT_ROOT / 'classification_report_binary.txt', "w") as f:
        f.write(report)

    labels = ['Normal', 'Attack']
    cm = confusion_matrix(binary_true, binary_predicted)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            annot[i, j] = f'{count}'

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_normalized * 100, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('2-Class Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_six_class_matrix(y_true, y_pred, filename):
    # Mapping from original class indices to new group indices
    class_map = {
        1: 0, 2: 0, 3: 0, 4: 0,  # DoS/DDoS → 0
        5: 1, 9: 1, 13: 1,  # Info Gathering → 1
        6: 2,               # MITM → 2
        11: 3, 12: 3, 14: 3,  # Injection → 3
        0: 4, 8: 4, 10: 4,  # Malware → 4
        7: 5  # Normal → 5
    }

    # Apply mapping
    y_pred_grouped = np.array([class_map[i] for i in y_pred])
    y_true_grouped = np.array([class_map[i] for i in y_true])

    labels = ['DDoS', 'Scanning', 'MITM', 'Injection', 'Malware', 'Normal']
    report = classification_report(y_true_grouped, y_pred_grouped, target_names=labels)
    with open(PROJECT_ROOT / 'classification_report_six.txt', "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true_grouped, y_pred_grouped)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('6-Class Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)


def plot_conf_matrix(y_test, predicted_classes, filename):
    """
    Plots a confusion matrix comparing predicted classes with the true labels.

    Args:
        y_test (list or np.array): The true labels of the test data.
        predicted_classes (list or np.array): The predicted classes of the test data.
        filename (str): The path to save the resulting confusion matrix image.
    """
    labels_15 = ['Back', 'HTTP', 'ICMP', 'TCP', 'UDP', 'Fing', 'MITM', 'Normal', 'Pwd', 'Port', 'Rans', 'SQL',
                 'Upload', 'Scan', 'XSS']
    predicted_classes, y_test_labels = one_hot_encode(y_test, predicted_classes)
    cm = confusion_matrix(y_test_labels, predicted_classes)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_15, yticklabels=labels_15)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
