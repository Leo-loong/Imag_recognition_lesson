import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_blocks(blocks, output_folder, block_size=3):
    """
    将图像块保存到指定文件夹
    :param blocks: 图像块列表
    :param output_folder: 保存图像块的文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, block in enumerate(blocks):
        block = block.reshape((block_size, block_size))
        block_path = os.path.join(output_folder, f"block_{i}.png")
        cv2.imwrite(block_path, block)
def omp(D, y, n_nonzero_coefs):
    """
    正交匹配追踪（OMP）算法实现
    :param D: 字典矩阵
    :param y: 信号向量
    :param n_nonzero_coefs: 非零系数的目标数量
    :return: 稀疏系数向量
    """
    residual = y.copy()
    indices = []
    X = np.zeros(D.shape[1])
    for _ in range(n_nonzero_coefs):
        correlations = np.abs(D.T @ residual)
        best_index = np.argmax(correlations)
        indices.append(best_index)
        selected_atoms = D[:, indices]
        coefficients = np.linalg.pinv(selected_atoms) @ y
        residual = y - selected_atoms @ coefficients
        X[indices] = coefficients
    return X
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
            X[:, i] = omp(D, Y[:, i], n_nonzero_coefs=sparsity_target)

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

            # 计算误差矩阵
            E = Y_omega - np.dot(D, X_omega_j)

            # 对误差矩阵进行 SVD 分解
            U, S, Vt = np.linalg.svd(E)

            # 更新字典的第 j 列
            D[:, j] = U[:, 0]

            # 更新稀疏系数矩阵的第 j 行
            X[j, omega] = S[0] * Vt[0, :]

    return D, X
def pad_image(image, block_size=3, overlap=2):
    """
    对图像进行填充，使得图像能够被完整地分割成图像块
    :param image: 输入的图像
    :param block_size: 图像块的大小
    :param overlap: 图像块间的重叠像素数
    :return: 填充后的图像，填充的高度和宽度
    """
    height, width = image.shape[:2]
    pad_height = (block_size - (height % (block_size - overlap))) % (block_size - overlap)
    pad_width = (block_size - (width % (block_size - overlap))) % (block_size - overlap)
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_REPLICATE)
    return padded_image, pad_height, pad_width
def unpad_image(image, pad_height, pad_width):
    """
    去除图像的填充部分
    :param image: 填充后的图像
    :param pad_height: 填充的高度
    :param pad_width: 填充的宽度
    :return: 去除填充后的图像
    """
    height, width = image.shape[:2]
    return image[:height - pad_height, :width - pad_width]


def split_image_into_blocks(image, block_size=3, overlap=2):
    """
    将输入图像拆分成 3x3 图像块，块间重叠 1 像素
    :param image: 输入的图像
    :param block_size: 图像块的大小，默认为 3
    :param overlap: 图像块间的重叠像素数，默认为 1
    :return: 图像块列表
    """
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height - block_size + 1, block_size - overlap):
        for x in range(0, width - block_size + 1, block_size - overlap):
            block = image[y:y + block_size, x:x + block_size]
            if len(block.shape) == 3:
                block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
            block = block.flatten()
            blocks.append(block)
    return np.array(blocks)
def reconstruct_image_from_blocks(blocks, original_shape, block_size=3, overlap=2):
    """
    从图像块重建图像
    :param blocks: 图像块列表
    :param original_shape: 原始图像的形状
    :param block_size: 图像块的大小
    :param overlap: 图像块间的重叠像素数
    :return: 重建后的图像
    """
    height, width = original_shape
    result = np.zeros((height, width))
    count = np.zeros((height, width))

    block_index = 0
    for y in range(0, height - block_size + 1, block_size - overlap):
        for x in range(0, width - block_size + 1, block_size - overlap):
            block = blocks[block_index].reshape((block_size, block_size))
            result[y:y + block_size, x:x + block_size] += block
            count[y:y + block_size, x:x + block_size] += 1
            block_index += 1

    # 处理重叠部分，取平均值
    result = result / count
    return result.astype(np.uint8)

def split_image_into_blocks_test(image, block_size=3, overlap=2):
    """
    将输入图像拆分成 3x3 图像块，块间重叠 1 像素
    :param image: 输入的图像
    :param block_size: 图像块的大小，默认为 3
    :param overlap: 图像块间的重叠像素数，默认为 1
    :return: 图像块列表
    """
    height, width = image.shape[:2]
    blocks = []
    positions = []
    for y in range(0, height - block_size + 1, block_size - overlap):
        for x in range(0, width - block_size + 1, block_size - overlap):
            block = image[y:y + block_size, x:x + block_size]
            if len(block.shape) == 3:
                block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
            block = block.flatten()
            blocks.append(block)
            positions.append((y, x))
    return np.array(blocks), positions
def reconstruct_image_from_blocks_test(blocks, positions, original_shape, block_size=3, overlap=2):
    """
    从图像块重建图像
    :param blocks: 图像块列表
    :param positions: 每个图像块在原始图像中的位置
    :param original_shape: 原始图像的形状
    :param block_size: 图像块的大小
    :param overlap: 图像块间的重叠像素数
    :return: 重建后的图像
    """
    height, width = original_shape
    result = np.zeros((height, width))
    count = np.zeros((height, width))

    for block, (y, x) in zip(blocks, positions):
        block = block.reshape((block_size, block_size))
        result[y:y + block_size, x:x + block_size] += block
        count[y:y + block_size, x:x + block_size] += 1

    # 处理重叠部分，取平均值
    result = np.divide(result, count, out=np.zeros_like(result), where=count != 0)
    return result.astype(np.uint8)
#######################Stage1: train############################
#
# 读取成对的低质量和高质量图像
low_quality_folder = 'lr_train/'
high_quality_folder = 'hr_train/'
# low_quality_image = cv2.imread(low_quality_image_path, cv2.IMREAD_GRAYSCALE)
# high_quality_image = cv2.imread(high_quality_image_path, cv2.IMREAD_GRAYSCALE)
# low_quality_folder = 'low_quality_images'
# high_quality_folder = 'high_quality_images'

# 初始化空列表用于存储所有低质量和高质量图像块
all_low_quality_blocks = []
all_high_quality_blocks = []

# 遍历低质量图像文件夹
for filename in os.listdir(low_quality_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 构建低质量图像文件路径
        low_quality_image_path = os.path.join(low_quality_folder, filename)
        # 构建对应的高质量图像文件路径
        high_quality_image_path = os.path.join(high_quality_folder, filename)

        # 检查对应的高质量图像文件是否存在
        if os.path.exists(high_quality_image_path):
            # 读取低质量和高质量图像
            low_quality_image = cv2.imread(low_quality_image_path, cv2.IMREAD_GRAYSCALE)
            high_quality_image = cv2.imread(high_quality_image_path, cv2.IMREAD_GRAYSCALE)

            # # 对图像进行填充
            # low_quality_image, _, _ = pad_image(low_quality_image)
            # high_quality_image, _, _ = pad_image(high_quality_image)

            # 拆分图像为图像块
            low_quality_blocks = split_image_into_blocks(low_quality_image)
            high_quality_blocks = split_image_into_blocks(high_quality_image)

            # 将当前图像的图像块添加到总列表中
            all_low_quality_blocks.extend(low_quality_blocks)
            all_high_quality_blocks.extend(high_quality_blocks)

# 将列表转换为 numpy 数组
all_low_quality_blocks = np.array(all_low_quality_blocks)
all_high_quality_blocks = np.array(all_high_quality_blocks)
print(all_high_quality_blocks.shape)
# 保存图像块到新文件夹
low_quality_blocks_folder = 'lr_blocks'
high_quality_blocks_folder = 'hr_blocks'
# save_blocks(all_low_quality_blocks, low_quality_blocks_folder)
# save_blocks(all_high_quality_blocks, high_quality_blocks_folder)
# 初始化字典
n_components = 100
n_features = all_low_quality_blocks.shape[1]
print(all_low_quality_blocks.shape,all_high_quality_blocks.shape)
# # 拼接低质量和高质量图像块
combined_blocks = np.hstack((all_low_quality_blocks, all_high_quality_blocks)).T
print('combined_blocks',combined_blocks.shape)
D_combined_initial = np.random.randn(n_features*2, n_components)
# 使用 KSVD 学习大字典
D_combined, sparse_co = ksvd(combined_blocks, D_combined_initial, max_iter=10, sparsity_target=20)
print(D_combined.shape)
half_components = n_features
Dl = D_combined[:n_features,:]
Dh = D_combined[n_features:,:]
print(Dl.shape,sparse_co.shape)

'''
Dl_initial = np.random.randn(n_features, n_components)
Dh_initial = np.random.randn(n_features, n_components)
# 使用 KSVD 学习字典
Dl, _ = ksvd(all_low_quality_blocks.T, Dl_initial, max_iter=10, sparsity_target=20)
Dh, _ = ksvd(all_high_quality_blocks.T, Dh_initial, max_iter=10, sparsity_target=20)
print("低质量图像块字典形状:", Dl.shape)
print("高质量图像块字典形状:", Dh.shape)
'''

# 拼接低质量和高质量图像块矩阵
combined_blocks = np.hstack((all_low_quality_blocks.T, all_high_quality_blocks.T))
# 初始化联合字典
n_components = 100
D_combined_initial = np.random.randn(n_features * 2, n_components)
# 用KSVD训练联合字典
D_combined, _ = ksvd(combined_blocks, D_combined_initial, max_iter=10, sparsity_target=20)
# 拆分得到低质量字典Dl和高质量字典Dh
Dl = D_combined[:n_features, :]
Dh = D_combined[n_features:, :]

# 保存 Dl 和 Dh 矩阵
np.save('实验内容/Dl_test.npy', Dl)
np.save('实验内容/Dh_test.npy', Dh)
print("Dl 和 Dh 矩阵已保存为 Dl_test.npy 和 Dh_test.npy")

#


######################Stage2: test############################
new_low_quality_image_path = 'lr_test/1a.jpg'
new_low_quality_image = cv2.imread(new_low_quality_image_path, cv2.IMREAD_GRAYSCALE)
# 读取 Dl 和 Dh 矩阵（可在需要时使用）
Dl = np.load('实验内容/Dl_test.npy')
Dh = np.load('实验内容/Dh_test.npy')
print("Dl 和 Dh 矩阵已读取")
# 对测试图像进行填充
padded_low_quality_image, pad_height, pad_width = pad_image(new_low_quality_image)
cv2.imwrite('padding.png',padded_low_quality_image)
# 拆分新的低质量图像为图像块
# new_low_quality_blocks = split_image_into_blocks(padded_low_quality_image)
new_low_quality_blocks, new_positions = split_image_into_blocks_test(padded_low_quality_image)
# 学习稀疏系数 a
sparse_coefficients = []

for i in range(new_low_quality_blocks.shape[0]):
    block = new_low_quality_blocks.T[:,i]
    a = omp(Dl, block, n_nonzero_coefs=50)
    sparse_coefficients.append(a)
sparse_coefficients = np.array(sparse_coefficients).T
print('sparse_coefficients ',sparse_coefficients.shape)
# 用稀疏系数 a 乘上 Dh 得到高质量的图像块
high_quality_blocks_predicted = np.dot(Dh, sparse_coefficients)
high_quality_blocks_predicted = np.clip(high_quality_blocks_predicted, 0, 255).astype(np.uint8)
# 从高质量图像块重建图像
# reconstructed_image = reconstruct_image_from_blocks(high_quality_blocks_predicted.T, new_low_quality_image.shape)
reconstructed_image = reconstruct_image_from_blocks_test(high_quality_blocks_predicted.T, new_positions, padded_low_quality_image.shape)
# 去除填充部分
final_image = unpad_image(reconstructed_image, pad_height, pad_width)
# 保存重建后的高质量图像
cv2.imwrite('reconstructed_high_quality_image.png', final_image)
print("重建后的高质量图像已保存为 reconstructed_high_quality_image.png")