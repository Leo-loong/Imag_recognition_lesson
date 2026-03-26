import numpy as np
from scipy.optimize import nnls
from PIL import Image
from scipy.fft import dct
import matplotlib.pyplot as plt
import cv2
import pywt
def omp(X, y, n_nonzero_coefs=None, tol=1e-3, nonneg=False):
    """
    正交匹配追踪算法实现

    参数:
    X (np.ndarray): 特征矩阵，形状为 (n_samples, n_features)
    y (np.ndarray): 目标向量，形状为 (n_samples,)
    n_nonzero_coefs (int, 可选): 非零系数的最大数量。如果未提供，则默认为特征数量的一半
    tol (float, 可选): 收敛容差，当残差的范数小于该值时停止迭代，默认为 1e-3
    nonneg (bool, 可选): 是否强制系数为非负，默认为 True

    返回:
    np.ndarray: 稀疏系数向量，形状为 (n_features,)
    """
    # 初始化
    n_samples, n_features = X.shape
    if n_nonzero_coefs is None:
        n_nonzero_coefs = n_features // 2

    # 初始化系数向量为零向量
    coef = np.zeros(n_features)
    # 初始化活动集为空列表
    active_set = []
    # 初始化残差为目标向量
    residual = y.copy()

    for _ in range(n_nonzero_coefs):
        # 计算残差与每个特征的相关性
        correlations = np.abs(np.dot(X.T, residual))

        if nonneg:
            # 选择相关性最大的特征索引
            best_index = np.argmax(correlations)
        else:
            # 也可以考虑绝对值最大的相关性
            best_index = np.argmax(np.abs(correlations))

        # 将最佳特征索引添加到活动集
        if best_index not in active_set:
            active_set.append(best_index)

        ##########################
        # 从活动集中选择对应的特征矩阵#
        X_active = X[:, active_set]
        ##########################

        if nonneg:
            # 使用非负最小二乘法求解活动集上的系数
            coef_active, _ = nnls(X_active, y)
        else:
            # 使用普通最小二乘法求解活动集上的系数
            coef_active, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)

        # 更新系数向量
        coef[active_set] = coef_active

        ##########################
        # 更新残差#
        residual = y - np.dot(X, coef)
        ##########################

        # 检查收敛条件
        if np.linalg.norm(residual) < tol:
            break

    return coef


if __name__ == "__main__":
    # 读取图像
    image_path = 'Imgs/glass_tiles_ms_22.png'  # 替换为你的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 调整图像大小
    image = cv2.resize(image, (64, 64))

    # 将图像矩阵展平为一维向量
    image_vector = image.flatten()

    # 构建小波字典
    wavelet_name = 'db4'  # 选择小波基，这里使用 Daubechies 4
    wavelet = pywt.Wavelet(wavelet_name)
    level = pywt.dwt_max_level(len(image_vector), wavelet.dec_len)

    # 进行小波分解，获取系数
    coeffs = pywt.wavedecn(image_vector, wavelet, level=level)
    # 计算所有系数的数量
    total_coeffs = int(sum([np.prod(np.array(c).shape) for c in pywt.coeffs_to_array(coeffs)[0].flatten()]))

    dictionary = np.zeros((len(image_vector), total_coeffs))



    for i in range(total_coeffs):
        basis = np.zeros(total_coeffs)
        basis[i] = 1
        # 将一维的 basis 重新转换为小波系数结构
        coeffs_basis = pywt.array_to_coeffs(basis, pywt.coeffs_to_array(coeffs)[1], output_format='wavedecn')
        # 重构小波系数
        reconstructed = pywt.waverecn(coeffs_basis, wavelet)
        dictionary[:, i] = reconstructed.flatten()



    # 调用 omp 函数进行稀疏编码
    n_nonzero_coefs = 800 # 可以根据实际情况调整
    sparse_coefficients = omp(dictionary, image_vector, n_nonzero_coefs=n_nonzero_coefs)

    # 重构图像
    reconstructed_image_vector = np.dot(dictionary, sparse_coefficients)
    reconstructed_image_array = reconstructed_image_vector.reshape(image.shape)

    # 确保重构图像的数据类型为 uint8，并且像素值在 0 - 255 范围内
    reconstructed_image_array = np.clip(reconstructed_image_array, 0, 255).astype(np.uint8)
    # 显示原始图像和重构图像
    cv2.imwrite('1.png',image)
    cv2.imwrite('2.png',reconstructed_image_array)