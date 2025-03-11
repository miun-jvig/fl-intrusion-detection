from keras.models import Sequential
from keras.layers import (Dense, Dropout, Input, BatchNormalization)
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def create_model(input_dim, num_classes):
    lambda_val = 1e-4
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(lambda_val)),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(lambda_val)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
