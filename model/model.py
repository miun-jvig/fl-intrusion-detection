from keras import Sequential, Input
from keras import layers
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def load_model():
    lambda_val = 1e-4
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    model = Sequential([
        Input(shape=(95,)),
        layers.Dense(90, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Dense(90, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Dropout(0.3),
        layers.Dense(15, activation="softmax")
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
