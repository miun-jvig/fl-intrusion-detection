import tensorflow as tf
from preprocessing.preprocessing import preprocess_data
from data_loading.data_loader import load_data, load_dataset
from utils.utils import save_history, one_hot_encode, make_class_weight
from utils.visualization import plot_hist, plot_conf_matrix, plot_binary_matrix, plot_six_class_matrix
from model.model import load_model

# variables
EPOCHS = 25
BATCH_SIZE = 800


def main():
    # train on GPU if possible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print('Training on GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    preprocess_data('DNN-EdgeIIoT-dataset.csv')
    df, _, _ = load_data('preprocessed_DNN.csv')
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('preprocessed_DNN.csv')

    # model
    model = load_model()
    class_weight = make_class_weight(y_train)

    # training
    with tf.device('/gpu:0'):
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
                            class_weight=class_weight)
    model.save('model.h5')
    model = tf.keras.models.load_model('model.h5')
    save_history(history, 'training_history.json')

    # evaluation
    plot_hist(history, 'loss.png')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    test_predictions = model.predict(x_test)
    predicted_classes, y_test_labels = one_hot_encode(y_test, test_predictions)
    plot_binary_matrix(df, y_test_labels, predicted_classes, 'conf_2.png')
    plot_six_class_matrix(y_test_labels, predicted_classes, 'conf_6.png')
    plot_conf_matrix(y_test, test_predictions, 'conf_15.png')
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == '__main__':
    main()
