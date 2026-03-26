import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示，使用系统中已有的中文字体
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def add_gaussian_noise(image, mean=0, var=0.001):
    """
    给图像添加高斯噪声

    参数:
    image: 输入图像
    mean: 高斯噪声的均值
    var: 高斯噪声的方差

    返回:
    添加了高斯噪声的图像
    """
    # 将图像转换为浮点型，便于后续计算
    image = np.array(image / 255, dtype=float)

    # 生成高斯噪声
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)

    # 给图像添加噪声
    noisy_image = image + gauss

    # 将像素值限制在[0, 1]范围内
    noisy_image = np.clip(noisy_image, 0, 1)

    # 转换回uint8类型（0-255）
    noisy_image = np.uint8(noisy_image * 255)

    return noisy_image


def main():
    # 读取图像，可以替换为自己的图像路径
    image_path = 'Imgs/danxia1.png'  # 输入图像路径
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像，请检查路径: {image_path}")
        return

    # 转换为RGB格式（OpenCV默认读取为BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 添加高斯噪声，可调整mean和var参数控制噪声大小
    noisy_image = add_gaussian_noise(image, mean=0, var=0.005)
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

    # 显示原图和加噪后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(image_rgb)
    plt.title('原图')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(noisy_image_rgb)
    plt.title('添加高斯噪声后的图像')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存加噪后的图像
    output_path = 'noisy_image_1.jpg'
    cv2.imwrite(output_path, noisy_image)
    print(f"加噪后的图像已保存至: {output_path}")


if __name__ == "__main__":
    main()