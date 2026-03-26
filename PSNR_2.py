import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import log10

plt.rcParams["font.family"] = ["Arial Unicode MS"]
# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def add_gaussian_noise(image, mean=0, var=0.001):
    """给图像添加高斯噪声"""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    # 确保像素值在0-255范围内
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def psnr(original, processed):
    """计算两个图像之间的PSNR值"""
    # 将图像转换为浮点数以避免溢出
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    # 计算MSE (均方误差)
    mse = np.mean((original - processed) ** 2)

    # 如果MSE为0，说明两个图像完全相同
    if mse == 0:
        return float('inf')

    # 计算PSNR，假设像素值范围是0-255
    max_pixel = 255.0
    return 20 * log10(max_pixel / np.sqrt(mse))


def denoise_using_psnr(noisy_image, original_image=None, threshold=30):
    """
    使用基于PSNR的简单去噪方法
    这里采用中值滤波作为去噪算法示例
    """
    # 中值滤波去噪
    denoised_image = cv2.medianBlur(noisy_image, 3)

    # 如果提供了原始图像，可以根据PSNR阈值调整滤波强度
    if original_image is not None:
        current_psnr = psnr(original_image, denoised_image)
        if current_psnr < threshold:
            denoised_image = cv2.medianBlur(noisy_image, 5)

    return denoised_image


def main(image_path):
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("无法读取图像，请检查文件路径")
        return

    # 转换为RGB格式以便matplotlib显示
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 添加高斯噪声
    noisy_image = add_gaussian_noise(original_image)
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

    # 去噪处理
    denoised_image = denoise_using_psnr(noisy_image, original_image)
    denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

    # 计算PSNR值
    psnr_noisy = psnr(original_image, noisy_image)
    psnr_denoised = psnr(original_image, denoised_image)

    print(f"原始图像与噪声图像的PSNR: {psnr_noisy:.2f} dB")
    print(f"原始图像与去噪图像的PSNR: {psnr_denoised:.2f} dB")

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original_image_rgb)
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(noisy_image_rgb)
    plt.title(f'噪声图像 (PSNR: {psnr_noisy:.2f} dB)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(denoised_image_rgb)
    plt.title(f'去噪图像 (PSNR: {psnr_denoised:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存结果
    cv2.imwrite('noisy_image.jpg', noisy_image)
    cv2.imwrite('denoised_image.jpg', denoised_image)
    print("噪声图像和去噪图像已保存")


if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "Imgs/danxia1.png"
    main(image_path)
