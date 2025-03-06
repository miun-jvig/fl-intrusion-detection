import tensorflow as tf
from preprocessing.preprocessing import preprocess_data, load_data
from sklearn.model_selection import train_test_split
from utils.utils import print_data_sizes, save_history, one_hot_encode
from utils.visualization import plot_hist, plot_conf_matrix
from cl.model import create_model
from sklearn.metrics import classification_report

# variables
EPOCHS = 25
BATCH_SIZE = 128


def main():
    # train on GPU if possible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print('Training on GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # preprocess_data('DNN-EdgeIIoT-dataset.csv')
    x_data, y_data = load_data('preprocessed_DNN.csv')

    # 95 % train, 5 % test, 5 % val
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.1, random_state=42,
                                                        stratify=y_data)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42,
                                                    stratify=y_temp)
    print_data_sizes(x_train, y_train, x_val, y_val, x_test, y_test)

    # model
    input_dim = x_train.shape[1]
    num_classes = y_train.shape[1]
    model = create_model(input_dim, num_classes)

    # training
    with tf.device('/gpu:0'):
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
    model.save('model.h5')
    # model.keras.saving.save.load_model('model.h5')
    save_history(history, 'training_history.json')

    # evaluation
    plot_hist(history, 'loss.png')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    test_predictions = model.predict(x_test)
    plot_conf_matrix(y_test, test_predictions, 'conf.png')
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    predicted_classes, y_test_labels = one_hot_encode(y_test, test_predictions)
    report = classification_report(y_test_labels, predicted_classes)
    print("\nClassification Report:\n", report)


if __name__ == '__main__':
    main()
