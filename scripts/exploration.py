# In[0]
import os
import sys

# 프로젝트 루트로 이동 (scripts → ../)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

# src 폴더를 import 경로에 추가
sys.path.append(os.path.join(project_root, 'src'))

from data_loader import load_ucr_dataset

import matplotlib.pyplot as plt
import numpy as np


# 그래프 저장 함수
def save_plot(fig, filename, folder='results/plots'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path)
    print(f"그래프 저장됨: {path}")


# 시각화 함수
def plot_samples(X, y, title, dataset_name, n_per_class=1):
    classes = np.unique(y)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']  # 최대 6 클래스까지

    fig = plt.figure()
    for ci, c in enumerate(classes):
        idxs = np.where(y == c)[0][:n_per_class]
        for i, idx in enumerate(idxs):
            plt.plot(X[idx], label=f"Class {c}" if i == 0 else "", color=colors[ci], alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filename = f"{dataset_name}_visualization.png"
    save_plot(fig, filename)

    plt.show()

# In[1]
# 확인할 데이터셋 목록
datasets = ['FordA', 'FordB', 'Wafer']

for name in datasets:
    print(f"\n Loading {name}")
    X_train, y_train, X_test, y_test = load_ucr_dataset(name)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    plot_samples(X_train, y_train, title=f"{name} - Training Samples", dataset_name=name)
