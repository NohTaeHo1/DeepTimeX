import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_ucr_dataset(dataset_name, root='data/raw'):
    train_path = os.path.join(root, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(root, dataset_name, f"{dataset_name}_TEST.tsv")

    print("불러오려는 경로:", train_path)
    print("현재 작업 디렉토리:", os.getcwd())

    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)

    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values

    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    if set(np.unique(y_train)) == {-1, 1}:
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
