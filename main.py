from run.train import main as train_main
from run.evaluate import evaluate_model
from run.layer_visualization import run_visualization as layer_visualization

import argparse
import os


def run_pipeline(datasets, models):
    for dataset in datasets:
        for model in models:
            print(f"실행: dataset={dataset}, model={model}")

            try:
                train_args = argparse.Namespace(dataset=dataset, model=model)
                train_main(train_args)
            except Exception as e:
                print(f"학습 중 에러 발생: {e}")
                continue

            try:
                model_path = f"results/checkpoints/{dataset}_{model}.h5"
                evaluate_model(model_path, dataset, model)
            except Exception as e:
                print(f"평가 중 에러 발생: {e}")
                continue

            try:
                if model == "resnet":
                    cam_args = argparse.Namespace(
                        dataset=dataset,
                        model=model,
                        sample_id=0
                    )
                    layer_visualization(cam_args)

            except Exception as e:
                print(f"CAM 시각화 중 에러 발생: {e}")


if __name__ == "__main__":
    datasets = ["FordA", "FordB", "Wafer"]
    models = ["mlp", "fcn", "cnn", "resnet"]

    run_pipeline(datasets, models)