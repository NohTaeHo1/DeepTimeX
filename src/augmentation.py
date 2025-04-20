import numpy as np

def jitter(X, sigma=0.05):
    """
    각 시계열 데이터에 정규분포 기반 노이즈를 추가한다.

    Parameters:
        X (np.ndarray): 입력 데이터 (배열 형태: [n_samples, n_timestamps])
        sigma (float): 노이즈의 표준편차

    Returns:
        np.ndarray: 노이즈가 추가된 시계열 데이터
    """
    return X + np.random.normal(loc=0.0, scale=sigma, size=X.shape)


def scaling(X, sigma=0.1):
    """
    각 시계열 데이터에 랜덤한 스케일 값을 곱한다.

    Parameters:
        X (np.ndarray): 입력 데이터 (배열 형태: [n_samples, n_timestamps])
        sigma (float): 스케일 값의 표준편차

    Returns:
        np.ndarray: 스케일이 적용된 시계열 데이터
    """
    scaling_factors = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1))
    return X * scaling_factors
