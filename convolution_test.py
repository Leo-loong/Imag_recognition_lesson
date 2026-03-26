import numpy as np
import cv2
import time
from scipy.signal import convolve2d

def custom_convolve2d(image, kernel):
    """
    自定义二维卷积函数，支持 mode='same'
    :param image: 输入的二维图像
    :param kernel: 卷积核
    :param mode: 卷积模式，默认为 'same'
    :return: 卷积后的图像
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算零填充的大小
    if kernel_height % 2 == 1:
        pad_height_top = (kernel_height - 1) // 2
    else:
        pad_height_top = kernel_height // 2
    pad_height_bottom = kernel_height - 1 - pad_height_top
    if kernel_width % 2 == 1:
        pad_width_left = (kernel_width - 1) // 2
    else:
        pad_width_left = kernel_width // 2
    pad_width_right = kernel_width - 1 - pad_width_left

    # 对图像进行零填充
    padded_image = np.pad(image, ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)), mode='constant', constant_values=0)

    # 初始化输出图像
    output = np.zeros((image_height, image_width))

    # 进行卷积操作
    for i in range(image_height):
        for j in range(image_width):
            window = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(window * kernel)

    return output


# 加载图像
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# 定义滤波器
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#sobel_x = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
#sobel_x = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# 实验 1: 计算效率对比
# 分块运算
block_size = 3
blocked_result = np.zeros_like(image)
start_time_block = time.time()
for i in range(0, image.shape[0], block_size):
    for j in range(0, image.shape[1], block_size):
        block = image[i:i+block_size, j:j+block_size]
        if block.size > 0:
            blocked_result[i:i+block_size, j:j+block_size] = convolve2d(block, sobel_x, mode='same')
            # blocked_result[i:i+block_size, j:j+block_size] = custom_convolve2d(block, sobel_x)
end_time_block = time.time()
time_block = end_time_block - start_time_block

# 卷积运算
start_time_conv = time.time()
conv_result = convolve2d(image, sobel_x, mode='same')
# conv_result = custom_convolve2d(image, sobel_x)
end_time_conv = time.time()
time_conv = end_time_conv - start_time_conv

print(f"分块运算时间: {time_block} 秒")
print(f"卷积运算时间: {time_conv} 秒")

# 实验 2: 特征提取能力对比
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(blocked_result, cmap='gray')
plt.title('Blocked Result')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(conv_result, cmap='gray')
plt.title('Convolution Result')
plt.axis('off')

plt.show()

# 实验 3: 计算灵活性对比
# 改变滤波器大小
new_sobel_x = np.array([[-1, 0, 1, 0], [-2, 0, 2, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])

# 分块运算
block_size = 4
blocked_result_new = np.zeros_like(image)
for i in range(0, image.shape[0], block_size):
    for j in range(0, image.shape[1], block_size):
        block = image[i:i+block_size, j:j+block_size]
        if block.size > 0:
            blocked_result_new[i:i+block_size, j:j+block_size] = convolve2d(block, new_sobel_x, mode='same')
            # blocked_result_new[i:i+block_size, j:j+block_size] = custom_convolve2d(block, new_sobel_x)

# 卷积运算
conv_result_new = convolve2d(image, new_sobel_x, mode='same')
# conv_result_new = custom_convolve2d(image, new_sobel_x)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(blocked_result_new, cmap='gray')
plt.title('Blocked Result (new filter)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(conv_result_new, cmap='gray')
plt.title('Convolution Result (new filter)')
plt.axis('off')

plt.show()

# 创建一个包含两个子图的图形，并排显示两种残差
plt.figure(figsize=(16, 7))

# 获取最小尺寸以确保残差计算的尺寸一致性
min_height = min(blocked_result.shape[0], blocked_result_new.shape[0], 
                conv_result.shape[0], conv_result_new.shape[0])
min_width = min(blocked_result.shape[1], blocked_result_new.shape[1],
                conv_result.shape[1], conv_result_new.shape[1])

# 第一子图：分块结果的残差
plt.subplot(1, 2, 1)
blocked_result_cropped = blocked_result[:min_height, :min_width]
blocked_result_new_cropped = blocked_result_new[:min_height, :min_width]
blocked_residual = blocked_result_cropped - blocked_result_new_cropped
plt.imshow(blocked_residual, cmap='gray')
plt.title('Blocked Processing Residual')
plt.colorbar(label='Residual Value')
plt.axis('off')

# 计算分块残差的统计信息
blocked_max_residual = np.max(np.abs(blocked_residual))
blocked_mean_residual = np.mean(np.abs(blocked_residual))
plt.figtext(0.25, 0.01, 
            f'Blocked Max Residual: {blocked_max_residual:.4f}\nBlocked Mean Residual: {blocked_mean_residual:.4f}', 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# 第二子图：卷积结果的残差
plt.subplot(1, 2, 2)
conv_result_cropped = conv_result[:min_height, :min_width]
conv_result_new_cropped = conv_result_new[:min_height, :min_width]
conv_residual = conv_result_cropped - conv_result_new_cropped
plt.imshow(conv_residual, cmap='gray')
plt.title('Convolution Processing Residual')
plt.colorbar(label='Residual Value')
plt.axis('off')

# 计算卷积残差的统计信息
conv_max_residual = np.max(np.abs(conv_residual))
conv_mean_residual = np.mean(np.abs(conv_residual))
plt.figtext(0.75, 0.01, 
            f'Conv Max Residual: {conv_max_residual:.4f}\nConv Mean Residual: {conv_mean_residual:.4f}', 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.suptitle('Comparison of Residuals Between Original and New Filters', fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()