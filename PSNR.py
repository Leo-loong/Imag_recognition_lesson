import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import io


# 设置matplotlib使用苹方字体显示中文
plt.rcParams["font.family"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 之后正常使用matplotlib绘图等操作

# 设置matplotlib支持中文显示（适配macOS）
plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def calculate_psnr(original, denoised):
    """计算两张图像的PSNR值"""
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)

    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def add_gaussian_noise(image, mean=0, var=0.001):
    """为图像添加高斯噪声"""
    sigma = var ** 0.5
    image = image.astype(np.float64) / 255.0
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 1)
    return (noisy_image * 255).astype(np.uint8)


def denoise_image(noisy_image, ksize=5):
    """使用高斯滤波对图像进行去噪"""
    return cv2.GaussianBlur(noisy_image, (ksize, ksize), 0)


def create_test_image():
    """创建一个简单的测试图像，避免依赖外部文件"""
    # 创建一个512x512的彩色图像
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制一些形状作为测试内容
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色矩形
    cv2.circle(img, (350, 150), 50, (0, 255, 0), -1)  # 绿色圆形
    cv2.line(img, (100, 300), (400, 300), (255, 0, 0), 5)  # 蓝色线条

    return img


def get_original_image():
    """获取原始图像，尝试多种方式确保成功"""
    # 尝试1: 查找当前目录下的图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            img = cv2.imread(file)
            if img is not None:
                print(f"使用当前目录下的图像: {file}")
                return img

    # 尝试2: 创建测试图像
    print("未找到现有图像，创建测试图像")
    return create_test_image()


def main():
    # 获取原始图像（多种方式确保成功）
    original_image = get_original_image()
    if original_image is None:
        print("无法获取图像，程序退出")
        return

    # 转换为RGB格式以便matplotlib显示
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 为原图添加高斯噪声
    noisy_image = add_gaussian_noise(original_image, var=0.005)

    # 对带噪声的图像进行去噪
    denoised_image = denoise_image(noisy_image, ksize=5)

    # 计算PSNR值（与原图比较更合理）
    psnr_value = calculate_psnr(original_image, denoised_image)
    print(f"原图与去噪图像的PSNR值: {psnr_value:.2f} dB")

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(noisy_image)
    plt.title('带高斯噪声的图像')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(denoised_image)
    plt.title(f'去噪后的图像 (PSNR: {psnr_value:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存去噪后的图像
    denoised_image_bgr = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('denoised_image.jpg', denoised_image_bgr)
    print("去噪后的图像已保存为 'denoised_image.jpg'")


if __name__ == "__main__":
    main()
