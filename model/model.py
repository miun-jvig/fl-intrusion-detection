from keras import Sequential, Input
from keras import layers
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def create_model():
    lambda_val = 1e-4
    input_dim = 96
    num_classes = 15
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    model = Sequential([
        Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
