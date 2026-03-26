import cv2
import random

def extract_and_save_image_block(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径。")
        return

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 确保图像足够大以截取64x64的块
    if height < 64 or width < 64:
        print("图像尺寸太小，无法截取64x64的块。")
        return

    # 随机选择一个中心像素点
    center_x = random.randint(32, width - 32)
    center_y = random.randint(32, height - 32)

    # 计算图像块的边界
    start_x = center_x - 32
    end_x = center_x + 32
    start_y = center_y - 32
    end_y = center_y + 32

    # 截取图像块
    image_block = image[start_y:end_y, start_x:end_x]

    # 保存图像块
    cv2.imwrite(output_path, image_block)
    print(f"图像块已保存到 {output_path}")

if __name__ == "__main__":
    # 请将此路径替换为你自己的图像路径
    input_image_path = "38.png"
    # 请指定输出图像块的保存路径
    output_image_path = "38_block.png"
    extract_and_save_image_block(input_image_path, output_image_path)