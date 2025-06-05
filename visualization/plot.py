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


def plot_eval_data_client(data_file, save_filename, fontsize=12):
    # 1) Load JSON
    with open(data_file, 'r') as file:
        data = json.load(file)

    # 2) Build dicts for centralized eval by round
    central_acc = {
        e["round"]: e["centralized_evaluate_accuracy"]
        for e in data.get("centralized_evaluate", [])
    }
    central_loss = {
        e["round"]: e["centralized_evaluate_loss"]
        for e in data.get("centralized_evaluate", [])
    }

    # 3) Load per‐client entries
    entries = data.get("client_evaluate", [])
    if not entries:
        raise ValueError("No 'client_evaluate' data found")

    # 4) Discover all client IDs from the first entry
    sample = entries[0]
    client_ids = sorted({
        key.split("/")[0]
        for key in sample.keys()
        if key not in ("round",)
    })

    # 5) Prepare plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Rounds common to client data
    rounds = [e["round"] for e in entries]

    # 6) Plot centralized as solid black line
    ax1.plot(rounds, [central_acc[r] for r in rounds], color="black", linestyle="-", linewidth=2,
             label="Aggregated Eval")
    ax2.plot(rounds, [central_loss[r] for r in rounds], color="black", linestyle="-", linewidth=2,
             label="Aggregated Loss")

    # 7) Plot per‐client eval as dashed colored lines
    cmap = plt.get_cmap("tab10")
    for i, cid in enumerate(client_ids):
        # Gather only rounds where both metrics exist
        rd, accs, losses = [], [], []
        for e in entries:
            a_key, l_key = f"{cid}/eval_accuracy", f"{cid}/eval_loss"
            if a_key in e and l_key in e:
                rd.append(e["round"])
                accs.append(e[a_key])
                losses.append(e[l_key])

        color = cmap(i % cmap.N)
        ax1.plot(rd, accs, linestyle="--", color=color, alpha=0.8, label=cid)
        ax2.plot(rd, losses, linestyle="--", color=color, alpha=0.8, label=cid)

    ax1.grid(True, alpha=0.5)
    ax2.grid(True, alpha=0.5)
    # 8) Labels & legend
    ax1.set_title("Per-Client Evaluation Accuracy by Round", fontsize=fontsize)
    ax1.set_xlabel("FL Round", fontsize=fontsize)
    ax1.set_ylabel("Accuracy", fontsize=fontsize)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    ax2.set_title("Per-Client Evaluation Loss by Round", fontsize=fontsize)
    ax2.set_xlabel("Round", fontsize=fontsize)
    ax2.set_ylabel("Loss", fontsize=fontsize)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close()


def plot_fit_results_clients(data_file, save_filename, fontsize=12):
    with open(data_file) as f:
        data = json.load(f)

    entries = data.get('client_fit', [])
    if not entries:
        raise ValueError("No 'client_fit' data found")

    # Discover clients & assign colors
    client_ids = sorted({k.split('/')[0] for k in entries[0] if k != 'round'})
    cmap = plt.get_cmap('tab10')
    colors = {cid: cmap(i % cmap.N) for i, cid in enumerate(client_ids)}
    rounds = [e['round'] for e in entries]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot each client (no labels here)
    for cid in client_ids:
        c = colors[cid]
        ax1.plot(rounds, [e[f'{cid}/train_acc'] for e in entries],
                 linestyle='-', color=c, alpha=0.8)
        ax1.plot(rounds, [e[f'{cid}/val_acc'] for e in entries],
                 linestyle='--', color=c, alpha=0.8)
        ax2.plot(rounds, [e[f'{cid}/train_loss'] for e in entries],
                 linestyle='-', color=c, alpha=0.8)
        ax2.plot(rounds, [e[f'{cid}/val_loss'] for e in entries],
                 linestyle='--', color=c, alpha=0.8)

    ax1.grid(True, alpha=0.5)
    ax2.grid(True, alpha=0.5)
    # Build the style‐legend (train vs. val)
    style_handles = [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Training'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='Validation'),
    ]
    # Anchor it at the right, up near the top
    style_legend = ax1.legend(handles=style_handles,
                              title='Split',
                              loc='upper left',
                              bbox_to_anchor=(1.02, 0.85),
                              fontsize="small")
    ax1.add_artist(style_legend)

    # Build the client‐color legend
    client_handles = [
        Line2D([0], [0], color=colors[cid], lw=2, label=cid)
        for cid in client_ids
    ]
    ax1.legend(handles=client_handles,
               title='Client',
               loc='upper left',
               bbox_to_anchor=(1.02, 0.50),
               fontsize="small")

    # Repeat legends on loss subplot
    # style
    style_legend2 = ax2.legend(handles=style_handles,
                               title='Split',
                               loc='upper left',
                               bbox_to_anchor=(1.02, 0.85),
                               fontsize="small")
    ax2.add_artist(style_legend2)
    # clients
    ax2.legend(handles=client_handles,
               title='Client',
               loc='upper left',
               bbox_to_anchor=(1.02, 0.50),
               fontsize="small")

    ax1.set_title("Per-Client Training/Validation Accuracy by Round", fontsize=fontsize)
    ax1.set_xlabel("FL Round", fontsize=fontsize)
    ax1.set_ylabel("Accuracy", fontsize=fontsize)
    ax2.set_title("Per-Client Training/Validation Loss by Round", fontsize=fontsize)
    ax2.set_xlabel("FL Round", fontsize=fontsize)
    ax2.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(save_filename)
    plt.close(fig)
