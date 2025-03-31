import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import one_hot_encode
from data.data_loader import load_data
from sklearn.metrics import classification_report


test_path = "C:/Users/joelv/PycharmProjects/thesis-ML-FL/datasets/global_test.csv"
x_test, y_test = load_data(test_path)
input_dim = x_test.shape[1]
num_classes = y_test.shape[1]


def create_plot(ax, x, y, y_label, x_label, title, labels, fontsize=12):
    """Create plot"""
    ax.plot(x, 'b', y, 'r')
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(labels, fontsize=fontsize, loc='best')
    x_ticks = range(0, len(x), 5)
    ax.set_xticks(x_ticks)


def plot_fl_rounds(data_file, save_filename):
    df = pd.read_csv(data_file)

    # Extract data
    rounds = df['round_number']
    accuracy = df['accuracy']
    loss = df['loss']
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    # Accuracy plot
    ax1.plot(rounds, accuracy, 'b-', label='Accuracy')
    ax1.set_ylabel('Accuracy Rate')
    ax1.set_xlabel('Federated Round')
    ax1.set_title('Accuracy per Federated Round')
    ax1.legend(loc='best')

    # Loss plot
    ax2.plot(rounds, loss, 'r-', label='Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Federated Round')
    ax2.set_title('Learning Curve per Federated Round')
    ax2.legend(loc='best')

    # Adjust layout and save figure
    fig.tight_layout()
    plt.savefig(save_filename)
    plt.show()


def plot_hist(history, filename):
    """
    Plots training and validation accuracy and loss histories as two separate histograms.

    Args:
        history (tf.keras.callbacks.History or dict): The training history object or dictionary containing
                                                      'accuracy', 'val_accuracy', 'loss', and 'val_loss'.
        filename (str): The path to save the resulting plot image.
    """
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


def plot_conf_matrix(yt, predicted_classes, filename):
    """
    Plots a confusion matrix comparing predicted classes with the true labels.

    Args:
        yt (list or np.array): The true labels of the test datasets.
        predicted_classes (list or np.array): The predicted classes of the test datasets.
        filename (str): The path to save the resulting confusion matrix image.
    """
    labels_2 = ['Normal', 'Attack']
    labels_6 = ['DDoS', 'Injection', 'MITM', 'Malware', 'Normal', 'Scanning']
    labels_15 = ['Back', 'HTTP', 'ICMP', 'TCP', 'UDP', 'Fing', 'MITM', 'Normal', 'Pwd', 'Port', 'Rans', 'SQL',
                 'Upload', 'Scan', 'XSS']
    predicted_classes, y_test_labels = one_hot_encode(yt, predicted_classes)
    cm = confusion_matrix(y_test_labels, predicted_classes)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_15, yticklabels=labels_15)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)


model = tf.keras.models.load_model('../checkpoints/global_model_5.keras')
test_predictions = model.predict(x_test)
plot_conf_matrix(y_test, test_predictions, '../results/conf.png')
plot_fl_rounds('../logs/evaluation_history.csv', '../results/fl_rounds.png')
predicted, y_lab = one_hot_encode(y_test, test_predictions)
report = classification_report(y_lab, predicted)
print("\nClassification Report:\n", report)
