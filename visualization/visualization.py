import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix
from apps.task import one_hot_encode
from data_loading.data_loader import load_data
from sklearn.metrics import classification_report

home_path = 'C:/Users/joelv/PycharmProjects/thesis-ML-FL/'
test_path = home_path + 'datasets/global_test.csv'
current_output = 'outputs/2025-04-03/15-25-12/'
best_model = 'model_state_acc_0.963_round_18.keras'


def create_plot(ax, x, y, y_label, x_label, title, labels, fontsize=12):
    ax.plot(x, 'b', label=labels[0])
    ax.plot(y, 'r', label=labels[1])
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(labels, fontsize=fontsize, loc='best')
    x_ticks = range(0, len(x), 5)
    ax.set_xticks(x_ticks)


def plot_eval_data(data_file, save_filename):
    with open(data_file, 'r') as file:
        data = json.load(file)
    # Extract data
    centralized_accuracy = [entry["centralized_accuracy"] for entry in data["centralized_evaluate"]]
    federated_accuracy = [entry["federated_evaluate_accuracy"] for entry in data["federated_evaluate"]]
    centralized_loss = [entry["centralized_loss"] for entry in data["centralized_evaluate"]]
    federated_loss = [entry["federated_evaluate_loss"] for entry in data["federated_evaluate"]]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    # First plot: Accuracy
    create_plot(ax1, centralized_accuracy, federated_accuracy, 'Accuracy', 'Round',
                'Aggregated Accuracy', ['Centralized Accuracy', 'Federated Accuracy'])

    # Second plot: Validation accuracy
    create_plot(ax2, centralized_loss, federated_loss, 'Loss', 'Round', 'Aggregated Loss',
                ['Centralized Loss', 'Federated Loss'])

    # Save figure
    fig.tight_layout()
    plt.savefig(save_filename)


def plot_fit_results(data_file, save_filename):
    with open(data_file, 'r') as file:
        data = json.load(file)
    # Extract data
    accuracy = [entry['accuracy'] for entry in data['fit_metrics']]
    val_accuracy = [entry['val_accuracy'] for entry in data['fit_metrics']]
    loss = [entry['loss'] for entry in data['fit_metrics']]
    val_loss = [entry['val_loss'] for entry in data['fit_metrics']]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    # First plot: Accuracy
    create_plot(ax1, accuracy, val_accuracy, 'Accuracy', 'Round',
                'Aggregated Accuracy', ['Training Accuracy', 'Validation Accuracy'])

    # Second plot: Validation accuracy
    create_plot(ax2, loss, val_loss, 'Loss', 'Round', 'Aggregated Loss',
                ['Loss', 'Validation loss'])

    # Save figure
    fig.tight_layout()
    plt.savefig(save_filename)


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


x_test, y_test = load_data(test_path)
model = tf.keras.models.load_model(home_path + current_output + best_model)
test_predictions = model.predict(x_test)
plot_conf_matrix(y_test, test_predictions, 'conf.png')
plot_eval_data(home_path + current_output + 'evaluation_results.json', 'evaluation_plot.png')
plot_fit_results(home_path + current_output + 'fit_results.json', 'fit_plot.png')
predicted, y_lab = one_hot_encode(y_test, test_predictions)
report = classification_report(y_lab, predicted)
with open("classification_report.txt", "w") as f:
    f.write(report)
