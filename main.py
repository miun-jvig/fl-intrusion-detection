import tensorflow as tf
from preprocessing.preprocessing import preprocess_data, load_data
from sklearn.model_selection import train_test_split
from utils.utils import print_data_sizes


def main():
    # train on GPU if possible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print('Training on GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # preprocess_data('DNN-EdgeIIoT-dataset.csv')
    x_data, y_data = load_data('preprocessed_DNN.csv')

    # 90 % train, 5 % test, 5 % val
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.1, random_state=42,
                                                        stratify=y_data)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    print_data_sizes(x_train, y_train, x_val, y_val, x_test, y_test)

    # model


if __name__ == '__main__':
    main()
