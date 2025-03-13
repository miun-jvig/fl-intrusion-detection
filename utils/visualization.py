import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.utils import one_hot_encode


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


def plot_conf_matrix(y_test, predicted_classes, filename):
    """
    Plots a confusion matrix comparing predicted classes with the true labels.

    Args:
        y_test (list or np.array): The true labels of the test datasets.
        predicted_classes (list or np.array): The predicted classes of the test datasets.
        filename (str): The path to save the resulting confusion matrix image.
    """
    labels_2 = ['Normal', 'Attack']
    labels_6 = ['DDoS', 'Injection', 'MITM', 'Malware', 'Normal', 'Scanning']
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
    plt.savefig(filename)
