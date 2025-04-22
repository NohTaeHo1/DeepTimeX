from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense, LeakyReLU, MaxPooling1D, Dropout, Add
from tensorflow.keras.activations import softmax

def Multi_Layer_Perceptron(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Dense(128)(inputs)
    x = LeakyReLU(0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def Fully_Convolutional_Network(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=128, kernel_size=8, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters=256, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def One_Dimensional_CNN(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=5, padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def residual_block(x, filters, kernel_sizes):
    shortcut = x

    for kernel_size in kernel_sizes:
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Conv1D(filters=filters, kernel_size=kernel_sizes[-1], padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = LeakyReLU(0.2)(x)
    return x

def Residual_Network(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(3):
        x = residual_block(x, filters=64, kernel_sizes=[8, 5, 3])

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model



# Test code
if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    from src.data_loader import load_ucr_dataset

    X_train, y_train, X_test, y_test = load_ucr_dataset("FordA")

    X_small_flat = X_train[:10] 
    X_small_seq = X_train[:10].reshape(10, 500, 1) 
    y_small = y_train[:10]

    models = [
        ("Multi_Layer_Perceptron", Multi_Layer_Perceptron, X_small_flat, (500,)),
        ("Fully_Convolutional_Network", Fully_Convolutional_Network, X_small_seq, (500, 1)),
        ("One_Dimensional_CNN", One_Dimensional_CNN, X_small_seq, (500, 1)),
        ("Residual_Network", Residual_Network, X_small_seq, (500, 1)),
    ]

    for name, model_fn, X_input, input_shape in models:
        print(f"\n🧪 Testing: {name}")
        model = model_fn(input_shape=input_shape, num_classes=2)
        model.summary()
        preds = model.predict(X_input)
        print("Prediction shape:", preds.shape)  # 예: (10, 2)

