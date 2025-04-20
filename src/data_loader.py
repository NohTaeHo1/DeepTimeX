import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_ucr_dataset(dataset_name, root='data/raw'):
    """  
    Args:
        dataset_name (str): 데이터셋 이름 (예: 'FordA')
        root (str): 데이터셋이 저장된 루트 경로

    Returns:
        X_train (np.ndarray): 정규화된 학습 입력 데이터
        y_train (np.ndarray): 학습 레이블
        X_test (np.ndarray): 정규화된 테스트 입력 데이터
        y_test (np.ndarray): 테스트 레이블
    """
    
    # 파일 경로 설정
    train_path = os.path.join(root, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(root, dataset_name, f"{dataset_name}_TEST.tsv")

    print("불러오려는 경로:", train_path)
    print("현재 작업 디렉토리:", os.getcwd())

    # TSV 파일 불러오기
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)

    # 레이블과 입력 분리
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values

    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    # 라벨이 -1/1이면 0/1로 바꾸기
    if set(np.unique(y_train)) == {-1, 1}:
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)

    # Z-score 정규화 (평균 0, 표준편차 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
