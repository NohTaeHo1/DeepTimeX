import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from src.data_loader import load_ucr_dataset
from src.augmentation import jitter, scaling

from src.model import (
    Multi_Layer_Perceptron,
    Fully_Convolutional_Network,
    One_Dimensional_CNN,
    Residual_Network,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from src.augmentation import jitter, scaling

def get_model(name, input_shape, num_classes):
    if name == "mlp":
        return Multi_Layer_Perceptron(input_shape, num_classes)
    elif name == "fcn":
        return Fully_Convolutional_Network(input_shape, num_classes)
    elif name == "cnn":
        return One_Dimensional_CNN(input_shape, num_classes)
    elif name == "resnet":
        return Residual_Network(input_shape, num_classes)
    else:
        raise ValueError("Unknown model name")

def main(args):
    print(f"데이터셋: {args.dataset} / 모델: {args.model}\n")

    X_train, y_train, X_test, y_test = load_ucr_dataset(args.dataset)

    X_train_jitter = jitter(X_train)
    X_train_scaling = scaling(X_train)

    y_train_jitter = y_train.copy()
    y_train_scaling = y_train.copy()

    X_train = np.concatenate([X_train, X_train_jitter, X_train_scaling], axis=0)
    y_train = np.concatenate([y_train, y_train_jitter, y_train_scaling], axis=0)

    if args.model in ["fcn", "cnn", "resnet"]:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = get_model(args.model, input_shape, num_classes)
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 모델 별 배치 사이즈
    if args.model == "mlp":
        batch_size = 64
    elif args.model == "fcn":
        batch_size = 128
    elif args.model == "cnn":
        batch_size = 32
    elif args.model == "resnet":
        batch_size = 64

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=batch_size,
        verbose=1,
    )

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    os.makedirs("results/loss_graph", exist_ok=True)
    save_path = f"results/loss_graph/{args.dataset}_{args.model}_loss_plot.png"
    plt.savefig(save_path)
    plt.close()

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n Accuracy:  {accuracy_score(y_test, y_pred_classes):.4f}")
    print(f" Precision: {precision_score(y_test, y_pred_classes):.4f}")
    print(f" Recall:    {recall_score(y_test, y_pred_classes):.4f}")
    print(f" F1 Score:  {f1_score(y_test, y_pred_classes):.4f}")

    save_path = f"results/checkpoints/{args.dataset}_{args.model}.h5"
    model.save(save_path)
    print(f"모델 저장 완료: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FordA")
    parser.add_argument("--model", type=str, choices=["mlp", "fcn", "cnn", "resnet"], default="resnet")
    parser.add_argument("--augmentation", type=str, choices=["none", "jitter", "scaling"], default="none")
    args = parser.parse_args()
    main(args)
