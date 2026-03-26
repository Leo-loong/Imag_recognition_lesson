from sklearn import linear_model
from OMP import omp
import cv2
import numpy as np
import pywt


def ksvd(Y, D, max_iter=10, sparsity_target=200):
    """
    KSVD 算法实现

    参数:
    Y (np.ndarray): 数据矩阵，形状为 (n_samples, n_features)
    D (np.ndarray): 初始字典，形状为 (n_samples, n_atoms)
    max_iter (int, 可选): 最大迭代次数，默认为 10
    sparsity_target (int, 可选): 稀疏系数的目标非零数量，默认为 200

    返回:
    np.ndarray: 学习到的字典，形状为 (n_samples, n_atoms)
    np.ndarray: 稀疏系数矩阵，形状为 (n_atoms, n_features)
    """
    for iter in range(max_iter):
        # 稀疏编码阶段：使用 OMP 算法求解稀疏系数
        X = np.zeros((D.shape[1], Y.shape[1]))

        for i in range(Y.shape[1]):
            #########################
            # 稀疏编码初始更新化阶段
            #########################
            x_i = omp(D, Y[:, i], sparsity_target)
            X[:, i] = x_i.flatten()
            pass

        # 字典更新阶段
        for j in range(D.shape[1]):
            # 找到使用第 j 个原子的样本索引
            omega = np.nonzero(X[j, :])[0]
            if len(omega) == 0:
                continue

            # 从数据矩阵和稀疏系数矩阵中提取相关部分
            Y_omega = Y[:, omega]
            X_omega = X[:, omega]
            X_omega_j = X_omega.copy()
            X_omega_j[j, :] = 0

            #########################
            # 计算误差矩阵
            #########################
            E = Y_omega - np.dot(D, X_omega_j)

            #########################
            # 对误差矩阵进行 SVD 分解 #
            #########################
            U, S, Vt = np.linalg.svd(E, full_matrices=False)

            #########################
            # 更新字典的第 j 列
            #########################
            D[:, j] = U[:, 0]
            X[j, omega] = S[0] * Vt[0, :]
            #########################
            # 更新稀疏系数矩阵的第 j 行
            #########################


    return D, X


if __name__ == "__main__":

    # 读取图像
    image_path = 'Imgs/huangmo1.png'  # 替换为你的图像路径
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

    # 添加代码：显示小波字典中的部分原子
    # 选择要显示的原子数量
    num_atoms_to_show = total_coeffs
    # 计算显示网格的大小
    grid_size = int(np.ceil(np.sqrt(num_atoms_to_show)))
    # 创建一个大图像来放置所有原子图像
    dict_display = np.zeros((grid_size * image.shape[0], grid_size * image.shape[1]), dtype=np.uint8)

    for i in range(min(num_atoms_to_show, dictionary.shape[1])):
        # 获取第i个原子
        atom = dictionary[:, i]
        # 将原子重塑为图像大小
        atom_image = atom.reshape(image.shape)
        # 归一化到0-255范围以便显示
        atom_image = (
                    (atom_image - np.min(atom_image)) / (np.max(atom_image) - np.min(atom_image) + 1e-8) * 255).astype(
            np.uint8)
        # 计算在网格中的位置
        row = i // grid_size
        col = i % grid_size
        # 放置到显示网格中
        dict_display[row * image.shape[0]: (row + 1) * image.shape[0],
        col * image.shape[1]: (col + 1) * image.shape[1]] = atom_image

    # 保存字典可视化结果
    cv2.imwrite('wavelet_dictionary.png', dict_display)
    print(
        f"小波字典可视化已保存为 'wavelet_dictionary.png'，显示了前{min(num_atoms_to_show, dictionary.shape[1])}个原子")

    # 调用 ksvd 函数进行字典学习
    learned_dictionary, sparse_coefficients = ksvd(image_vector.reshape(-1, 1), dictionary, max_iter=4,
                                                   sparsity_target=50)

    # 重构图像
    reconstructed_image_vector = np.dot(learned_dictionary, sparse_coefficients)
    reconstructed_image_array = reconstructed_image_vector.reshape(image.shape)

    # 确保重构图像的数据类型为 uint8，并且像素值在 0 - 255 范围内
    reconstructed_image_array = np.clip(reconstructed_image_array, 0, 255).astype(np.uint8)

    # 显示原始图像和重构图像
    cv2.imwrite('original.png', image)
    cv2.imwrite('reconstructed.png', reconstructed_image_array)
