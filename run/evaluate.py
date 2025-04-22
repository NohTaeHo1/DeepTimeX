import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from tensorflow.keras.models import load_model
from src.data_loader import load_ucr_dataset

def evaluate_model(model_path, dataset, model_type):
    model = load_model(model_path)
    _, _, X_test, y_test = load_ucr_dataset(dataset)

    if model_type in ["cnn", "fcn", "resnet"]:
        X_test = X_test[..., np.newaxis]

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    prec = precision_score(y_test, y_pred_classes, zero_division=0)
    rec = recall_score(y_test, y_pred_classes, zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, zero_division=0)

    print(f"\n[Evaluation Result] {model_path}")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1 Score:  {f1:.4f}")

    os.makedirs("results/logs", exist_ok=True)
    summary_path = "results/logs/evaluation_summary.csv"
    result_row = pd.DataFrame([{
        "dataset": dataset,
        "model": model_type,
        "model_path": model_path,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }])

    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        result_row = pd.concat([prev, result_row], ignore_index=True)

    result_row.to_csv(summary_path, index=False)
    print(f"누적 결과 저장: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--augmentation", type=str, default="none")
    args = parser.parse_args()

    model_path = f"results/checkpoints/{args.dataset}_{args.model}_{args.augmentation}.h5"
    evaluate_model(model_path, args.dataset, args.model)
