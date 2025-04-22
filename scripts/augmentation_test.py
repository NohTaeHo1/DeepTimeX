import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from data_loader import load_ucr_dataset
from augmentation import jitter, scaling
import matplotlib.pyplot as plt
import numpy as np

def save_plot(fig, filename, folder='results/plots'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path)
    print(f"Saved plot: {path}")

def plot_overlayed_augmentation(X_sets, labels, dataset_name, class_label, y, filename_prefix):
    idx = np.where(y == class_label)[0][0]
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ['red', 'steelblue', 'darkorange']
    for X, label, color in zip(X_sets, labels, colors):
        ax.plot(X[idx], label=label, color=color, alpha=0.8)


    ax.set_title(f"{dataset_name} - Class {class_label} (One Sample, Overlayed)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()

    filename = f"{filename_prefix}_class{class_label}_overlayed.png"
    save_plot(fig, filename)


datasets = ['FordA', 'FordB', 'Wafer']

for name in datasets:
    print(f"\nProcessing {name}...")
    X_train, y_train, X_test, y_test = load_ucr_dataset(name)

    X_jittered = jitter(X_train)
    X_scaled = scaling(X_train)

    for c in np.unique(y_train):
        plot_overlayed_augmentation(
            X_sets=[X_train, X_jittered, X_scaled],
            labels=['Original', 'Jitter', 'Scaling'],
            dataset_name=name,
            class_label=c,
            y=y_train,
            filename_prefix=name
        )

