import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_toeplitz_matrix(kernel, image_height, image_width):
    """
    根据卷积核和图像尺寸创建Toeplitz矩阵
    :param kernel: 3x3卷积核
    :param image_height: 图像的高度
    :param image_width: 图像的宽度
    :return: Toeplitz矩阵
    """
    image_height = image_height + kernel.shape[0] - 1
    image_width = image_width + kernel.shape[1] - 1
    k_height, k_width = kernel.shape
    output_height = image_height - k_height + 1
    output_width = image_width - k_width + 1
    # 总输出元素数量
    output_size = output_height * output_width
    # 总输入元素数量
    input_size = image_height * image_width

    toeplitz_matrix = np.zeros((output_size, input_size))
    for i in range(output_height):
        for j in range(output_width):
            output_index = i * output_width + j
            for ki in range(k_height):
                for kj in range(k_width):
                    input_row = i + ki
                    input_col = j + kj
                    if 0 <= input_row < image_height and 0 <= input_col < image_width:
                        input_index = input_row * image_width + input_col
                        toeplitz_matrix[output_index, input_index] = kernel[ki, kj]

    return toeplitz_matrix


def image_to_vector(image):
    """
    将图像转换为向量
    :param image: 输入图像
    :return: 图像向量
    """
    return image.flatten()


def vector_to_image(vector, output_height, output_width):
    """
    将向量转换为图像
    :param vector: 输入向量
    :param output_height: 输出图像的高度
    :param output_width: 输出图像的宽度
    :return: 输出图像
    """
    return vector.reshape((output_height, output_width))


# 为了更好地可视化，使用较小的卷积核和图像尺寸
# 定义一个5x5卷积核（更小以便于可视化）
kernel = np.random.rand(5, 5)
# 也可以使用简单的预定义卷积核
# kernel = np.array([
#     [1, 2, 1],
#     [0, 0, 0],
#     [-1, -2, -1]
# ])
flipped_kernel = np.flip(kernel)
# 定义一个较小的示例图像
image = np.random.randint(0, 256, size=(16, 16))

# 二维卷积操作
convolved_image = convolve2d(image, flipped_kernel, mode='same')

# 构建Toeplitz矩阵
image_height, image_width = image.shape
toeplitz_matrix = create_toeplitz_matrix(kernel, image_height, image_width)

# 将图像转换为向量
# image_vector = image_to_vector(image)
if kernel.shape[0] % 2 == 1:
    pad_height_top = (kernel.shape[0] - 1)//2
else:
    pad_height_top = kernel.shape[0] // 2
if kernel.shape[1] % 2 == 1:
    pad_width_top = (kernel.shape[1] - 1)//2
else:
    pad_width_top = kernel.shape[1]//2
image_vector = image_to_vector(np.pad(image, (( pad_height_top, kernel.shape[0] - 1 - pad_height_top), (pad_width_top, kernel.shape[1] - 1 - pad_width_top)), mode='constant', constant_values=0))

# 进行Toeplitz矩阵乘法
output_vector = np.dot(toeplitz_matrix, image_vector)

# 计算输出图像的正确尺寸
# output_height = image_height - kernel.shape[0] + 1
# output_width = image_width - kernel.shape[1] + 1
output_height = image_height
output_width = image_width

# 确保向量长度与输出尺寸匹配
if output_vector.size == output_height * output_width:
    # 将结果向量转换为图像
    toeplitz_result = vector_to_image(output_vector, output_height, output_width)
    # 比较两种方法的结果
    is_equal = np.allclose(convolved_image, toeplitz_result)
    print(f"二维卷积和Toeplitz矩阵乘法的结果是否相等: {is_equal}")
    print("二维卷积结果:")
    print(convolved_image)
    print("Toeplitz矩阵乘法结果:")
    print(toeplitz_result)
    
    # 打印卷积核和Toeplitz矩阵的形状对比
    print(f"\n形状对比:")
    print(f"卷积核形状: {kernel.shape}")
    print(f"输入图像形状: {image.shape}")
    print(f"Toeplitz矩阵形状: {toeplitz_matrix.shape}")
    print(f"输出图像形状: {toeplitz_result.shape}")
    
    # 创建一个自定义的颜色映射，黑色表示0，白色表示非零值
    colors = [(0, 0, 0), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list('binary_cmap', colors, N=256)
    
    # 可视化Toeplitz矩阵
    plt.figure(figsize=(15, 10))
    
    # 显示Toeplitz矩阵的稀疏表示
    plt.subplot(1, 2, 1)
    # 为了更好地可视化，我们只显示矩阵的一部分（如果太大）
    max_display_size = 100  # 最大显示尺寸
    toeplitz_display = toeplitz_matrix.copy()
    # if toeplitz_matrix.shape[0] > max_display_size or toeplitz_matrix.shape[1] > max_display_size:
    #     toeplitz_display = toeplitz_matrix[:min(max_display_size, toeplitz_matrix.shape[0]), 
    #                                      :min(max_display_size, toeplitz_matrix.shape[1])]
    #     plt.title(f'Toeplitz Matrix (showing first {max_display_size}x{max_display_size})')
    # else:
    #     plt.title('Toeplitz Matrix')
    plt.title('Toeplitz Matrix')
    plt.imshow(toeplitz_display != 0, cmap=cmap)
    plt.colorbar(label='Non-zero Elements')
    plt.xlabel('Input Element Index')
    plt.ylabel('Output Element Index')
    
    # 显示卷积核
    plt.subplot(1, 2, 2)
    plt.imshow(kernel, cmap='viridis')
    plt.title('Convolution Kernel')
    plt.colorbar()
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # # 显示原始图像
    # plt.subplot(2, 2, 3)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.colorbar()
    # plt.xlabel('Width')
    # plt.ylabel('Height')
    
    # # 显示卷积结果
    # plt.subplot(2, 2, 4)
    # plt.imshow(convolved_image, cmap='gray')
    # plt.title('Convolution Result')
    # plt.colorbar()
    # plt.xlabel('Width')
    # plt.ylabel('Height')
    
    plt.tight_layout()
    plt.show()
    
    # # 如果Toeplitz矩阵不是特别大，可以显示其值分布的热力图
    # plt.figure(figsize=(10, 8))
    # plt.title('Value Distribution of Toeplitz Matrix')
    # # Use logarithmic scale for better visualization of non-zero values
    # plt.imshow(np.log10(np.abs(toeplitz_display) + 1e-10), cmap='hot')
    # plt.colorbar(label='log10(|value|+1e-10)')
    # plt.xlabel('Input Element Index')
    # plt.ylabel('Output Element Index')
    # plt.tight_layout()
    # plt.show()
else:
    print("向量长度与输出尺寸不匹配，请检查Toeplitz矩阵的构建。")
