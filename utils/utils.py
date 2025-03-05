
def print_data_sizes(x_train, y_train, x_val, y_val, x_test, y_test):
    print(f"Training data: {x_train.shape}, {y_train.shape}")
    print(f"Validation data: {x_val.shape}, {y_val.shape}")
    print(f"Testing data: {x_test.shape}, {y_test.shape}")
