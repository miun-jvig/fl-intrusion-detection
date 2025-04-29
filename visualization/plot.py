import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D


def plot_confusion_matrix(y_true, y_pred, labels, filename, title="Confusion Matrix", figsize=(9, 7)):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_binary_confusion(y_true, y_pred, filename):
    labels = ['Normal', 'Attack']
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = str(cm[i, j])

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_normalized * 100, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('2-Class Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_plot(ax, x, y, y_label, x_label, title, labels, fontsize=12):
    ax.plot(x, 'b', label=labels[0])
    ax.plot(y, 'r', label=labels[1])
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(labels, fontsize=fontsize, loc='best')
    ax.set_xticks(range(0, len(x), 5))


def plot_eval_data(data_file, save_filename):
    with open(data_file, 'r') as file:
        data = json.load(file)

    centralized_accuracy = [entry["centralized_evaluate_accuracy"] for entry in data["centralized_evaluate"]]
    federated_accuracy = [entry["federated_evaluate_accuracy"] for entry in data["federated_evaluate"]]
    centralized_loss = [entry["centralized_evaluate_loss"] for entry in data["centralized_evaluate"]]
    federated_loss = [entry["federated_evaluate_loss"] for entry in data["federated_evaluate"]]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    create_plot(ax1, centralized_accuracy, federated_accuracy, 'Accuracy', 'Round',
                'Aggregated Accuracy', ['Centralized Accuracy', 'Federated Accuracy'])
    create_plot(ax2, centralized_loss, federated_loss, 'Loss', 'Round', 'Aggregated Loss',
                ['Centralized Loss', 'Federated Loss'])

    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close()


def plot_aggregated_fit_results(data_file, save_filename):
    with open(data_file, 'r') as file:
        data = json.load(file)

    # Use only the 'all_clients_fit' entries, ignore aggregated 'client_fit'
    entries = data.get('all_clients_fit', [])
    if not entries:
        raise ValueError("No 'all_clients_fit' data found in the file.")

    accuracy = [entry['training_accuracy'] for entry in data['fit_metrics']]
    val_accuracy = [entry['val_accuracy'] for entry in data['fit_metrics']]
    loss = [entry['training_loss'] for entry in data['fit_metrics']]
    val_loss = [entry['val_loss'] for entry in data['fit_metrics']]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    create_plot(ax1, accuracy, val_accuracy, 'Accuracy', 'Round',
                'Aggregated Accuracy', ['Training Accuracy', 'Validation Accuracy'])
    create_plot(ax2, loss, val_loss, 'Loss', 'Round', 'Aggregated Loss',
                ['Loss', 'Validation Loss'])

    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close()


def plot_fit_results_clients(data_file, save_filename):
    # Load JSON data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Use only the 'client_fit' entries, ignore aggregated 'all_clients_fit'
    entries = data.get('client_fit', [])
    if not entries:
        raise ValueError("No 'client_fit' data found in the file.")

    # Identify client IDs from keys in the first entry
    sample = entries[0]
    client_ids = sorted({key.split('/')[0] for key in sample.keys() if key != 'round'})

    # Assign one distinct color per client
    cmap = plt.get_cmap('tab10')
    colors = {cid: cmap(i % cmap.N) for i, cid in enumerate(client_ids)}

    # Prepare the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Extract rounds once
    rounds = [e['round'] for e in entries]

    # Plot per-client accuracy and loss
    for cid in client_ids:
        train_acc = [e[f'{cid}/train_acc'] for e in entries]
        val_acc = [e[f'{cid}/val_acc'] for e in entries]
        train_loss = [e[f'{cid}/train_loss'] for e in entries]
        val_loss = [e[f'{cid}/val_loss'] for e in entries]

        color = colors[cid]
        ax1.plot(rounds, train_acc, marker='o', linestyle='-', color=color, alpha=0.8)
        ax1.plot(rounds, val_acc, marker='o', linestyle='--', color=color, alpha=0.8)

        ax2.plot(rounds, train_loss, marker='o', linestyle='-', color=color, alpha=0.8)
        ax2.plot(rounds, val_loss, marker='o', linestyle='--', color=color, alpha=0.8)

    # Set titles and labels
    ax1.set_title('Per-Client Accuracy by Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2.set_title('Per-Client Loss by Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')

    # Build and attach shared legend
    handles = [Line2D([0], [0], color=colors[cid], lw=2, label=cid) for cid in client_ids]
    ax1.legend(handles=handles, title='Clients', loc='best')
    ax2.legend(handles=handles, title='Clients', loc='best')

    # Adjust layout, save and close
    fig.tight_layout()
    plt.savefig(save_filename, dpi=300)
    plt.close(fig)
