from visualization.plot import plot_confusion_matrix, plot_binary_confusion, plot_eval_data, plot_fit_results
from visualization.utils import load_predictions_and_classes, group_classes
import numpy as np
from sklearn.metrics import classification_report


home_path = 'C:/Users/joelv/PycharmProjects/thesis-ML-FL/'
test_path = home_path + 'datasets/global_test.csv'
current_output = 'outputs/2025-04-03/23-55-34/'
best_model = 'model_state_acc_0.953_round_22.keras'

class_names, predicted, true = load_predictions_and_classes(home_path + current_output + best_model, test_path)


def binary_visualize():
    binary_pred = np.array([0 if class_names[p] == 'Normal' else 1 for p in predicted])
    binary_true = np.array([0 if class_names[t] == 'Normal' else 1 for t in true])

    plot_binary_confusion(binary_true, binary_pred, 'conf_2.png')
    report = classification_report(binary_true, binary_pred)
    with open("classification_report_binary.txt", "w") as f:
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

    plot_confusion_matrix(y_true_grouped, y_pred_grouped, labels, 'conf_6.png', title='6-Class Confusion Matrix')
    report = classification_report(y_true_grouped, y_pred_grouped, target_names=labels)
    with open("classification_report_six.txt", "w") as f:
        f.write(report)


def multiclass_visualize():
    labels = ['Back', 'HTTP', 'ICMP', 'TCP', 'UDP', 'Fing', 'MITM', 'Normal',
              'Pwd', 'Port', 'Rans', 'SQL', 'Upload', 'Scan', 'XSS']

    plot_confusion_matrix(true, predicted, labels, 'conf_15.png', title='15-Class Confusion Matrix')
    report = classification_report(true, predicted, target_names=labels)
    with open("classification_report_multi.txt", "w") as f:
        f.write(report)


def plot_metrics():
    plot_eval_data(home_path + current_output + 'evaluation_results.json', 'evaluation_plot.png')
    plot_fit_results(home_path + current_output + 'fit_results.json', 'fit_plot.png')


# --- Choose what to run ---
binary_visualize()
multiclass_visualize()
six_class_visualize()
plot_metrics()
