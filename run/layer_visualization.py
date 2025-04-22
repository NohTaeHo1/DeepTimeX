import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model, Model
from src.data_loader import load_ucr_dataset


def visualize_intermediate_layers(model, sample, layer_names, save_dir, prefix):
    # 지정된 Conv1D 레이어들의 출력 시각화
    for layer_name in layer_names:
        intermediate_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer_name).output)
        output = intermediate_model.predict(sample[np.newaxis, ...])[0]  # shape: (T, C)

        plt.figure(figsize=(12, 6))
        for i in range(min(output.shape[-1], 8)):
            plt.plot(output[:, i], label=f'Channel {i}')
        plt.title(f"Layer: {layer_name}")
        plt.xlabel("Time")
        plt.ylabel("Activation")
        plt.legend()
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"시각화 저장됨: {save_path}")


def run_visualization(args):
    model_path = f"results/checkpoints/{args.dataset}_{args.model}_{args.augmentation}.h5"
    model = load_model(model_path)

    _, _, X_test, y_test = load_ucr_dataset(args.dataset)
    if args.model in ["cnn", "fcn", "resnet"]:
        X_test = X_test[..., np.newaxis]

    sample = X_test[args.sample_id]
    label = y_test[args.sample_id]

    if args.model != "resnet":
        raise ValueError("현재 레이어 시각화는 resnet 모델에서만 지원")

    conv_layer_names = [layer.name for layer in model.layers if "conv1d" in layer.name]

    save_dir = f"results/layers"
    prefix = f"{args.dataset}_{args.model}_{args.augmentation}_sample{args.sample_id}"

    visualize_intermediate_layers(model, sample, conv_layer_names, save_dir, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FordA")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--augmentation", type=str, default="none")
    parser.add_argument("--sample_id", type=int, default=0)
    args = parser.parse_args()

    run_visualization(args)
