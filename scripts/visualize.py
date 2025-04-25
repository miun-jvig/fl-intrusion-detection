from visualization.plot import plot_confusion_matrix, plot_binary_confusion, plot_eval_data, plot_fit_results
from visualization.utils import load_predictions_and_classes, group_classes
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path
from visualization.utils import load_run_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_OUTPUT = PROJECT_ROOT / 'outputs' / '2025-04-25' / '14-56-41'
BEST_MODEL = CURRENT_OUTPUT / 'model_state_acc_0.779_round_1.h5'
TEST_PATH = PROJECT_ROOT / 'datasets' / 'global_test.csv'
VISUALIZATION_PATH = PROJECT_ROOT / 'visualization'
cfg = load_run_config(CURRENT_OUTPUT)

class_names, predicted, true = load_predictions_and_classes(BEST_MODEL, TEST_PATH, cfg)


def binary_visualize():
    binary_pred = np.array([0 if class_names[p] == 'Normal' else 1 for p in predicted])
    binary_true = np.array([0 if class_names[t] == 'Normal' else 1 for t in true])

    plot_binary_confusion(binary_true, binary_pred, VISUALIZATION_PATH / 'conf_2.png')
    report = classification_report(binary_true, binary_pred)
    with open(VISUALIZATION_PATH / 'classification_report_binary.txt', "w") as f:
        f.write(report)


def six_class_visualize():
    six_class_map = {
        1: 0, 2: 0, 3: 0, 4: 0,        # DoS/DDoS → 0
        5: 1, 9: 1, 13: 1,             # Scanning → 1
        6: 2,                          # MITM → 2
        11: 3, 12: 3, 14: 3,           # Injection → 3
        0: 4, 8: 4, 10: 4,             # Malware → 4
        7: 5                           # Normal → 5
    }

    labels = ['DDoS', 'Scanning', 'MITM', 'Injection', 'Malware', 'Normal']
    y_pred_grouped = group_classes(predicted, six_class_map)
    y_true_grouped = group_classes(true, six_class_map)

    plot_confusion_matrix(y_true_grouped, y_pred_grouped, labels, VISUALIZATION_PATH / 'conf_6.png',
                          title='6-Class Confusion Matrix')
    report = classification_report(y_true_grouped, y_pred_grouped, target_names=labels)
    with open(VISUALIZATION_PATH / 'classification_report_six.txt', "w") as f:
        f.write(report)


def multiclass_visualize():
    labels = ['Back', 'HTTP', 'ICMP', 'TCP', 'UDP', 'Fing', 'MITM', 'Normal',
              'Pwd', 'Port', 'Rans', 'SQL', 'Upload', 'Scan', 'XSS']

    plot_confusion_matrix(true, predicted, labels, VISUALIZATION_PATH / 'conf_15.png',
                          title='15-Class Confusion Matrix')
    report = classification_report(true, predicted, target_names=labels)
    with open(VISUALIZATION_PATH / 'classification_report_multi.txt', "w") as f:
        f.write(report)


def plot_metrics():
    plot_eval_data(CURRENT_OUTPUT / 'evaluation_results.json', VISUALIZATION_PATH / 'evaluation_plot.png')
    plot_fit_results(CURRENT_OUTPUT / 'fit_results.json', VISUALIZATION_PATH / 'fit_plot.png')


# --- Choose what to run ---
if __name__ == "__main__":
    binary_visualize()
    multiclass_visualize()
    six_class_visualize()
    plot_metrics()
