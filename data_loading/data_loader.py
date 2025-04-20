import pandas as pd
from sklearn.model_selection import train_test_split
from keras import utils
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path):
    _, x_data, y_data = load_data(file_path)
    # 90 % train, 5 % test, 5 % val
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.1, random_state=42,
                                                        stratify=y_data)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42,
                                                    stratify=y_temp)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df.head(5)
    labels = 'Attack_type'

    x = df.drop(columns=[labels]).to_numpy().astype('float32')  # Features: all columns except 'Attack_type'
    y = utils.to_categorical(LabelEncoder().fit_transform(df[labels]))  # Label: 'Attack_type', one hot encoded
    return df, x, y
