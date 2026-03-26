import cv2
import os

def downsample_image(image, scale_factor):
    # 获取图像的原始高度和宽度
    height, width = image.shape[:2]
    # 计算下采样后的高度和宽度
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    # 进行下采样
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return downsampled_image

def upsample_image(image, scale_factor):
    # 获取图像的原始高度和宽度
    height, width = image.shape[:2]
    # 计算上采样后的高度和宽度
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    # 进行上采样，使用 cv2.INTER_NEAREST 插值方法
    upsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return upsampled_image
def crop_image(image, target_size=120):
    height, width = image.shape[:2]
    # 确保图像足够大以进行裁剪
    if height < target_size or width < target_size:
        print("图像尺寸太小，无法裁剪成 150x150。")
        return None
    # 计算裁剪的起始位置，这里从图像中心裁剪
    start_x = (width - target_size) // 2
    start_y = (height - target_size) // 2
    end_x = start_x + target_size
    end_y = start_y + target_size
    # 裁剪图像
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image
if __name__ == "__main__":
    # 输入文件夹路径，包含待处理的图像
    input_folder = "hr/"
    # 输出文件夹路径，用于保存上采样后的图像
    output_folder = "lr_train/"
    output_folder_hr = "hr_train/"
    # 如果输出文件夹不存在，则创建它
    os.makedirs(output_folder,exist_ok=True)
    os.makedirs(output_folder_hr,exist_ok=True)

    # 下采样的缩放因子，这里设置为 0.25 表示将图像的尺寸缩小为原来的四分之一
    scale_factor = 1/4
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 构建完整的图像文件路径
            image_path = os.path.join(input_folder, filename)
            # 读取图像
            image = cv2.imread(image_path)
            if image is not None:
                # 进行下采样
                cropped_image = crop_image(image)
                down_img = downsample_image(cropped_image, scale_factor)
                # 进行上采样
                up_img = upsample_image(down_img, 1/scale_factor)
                # 构建输出图像的完整路径
                output_path = os.path.join(output_folder, filename)
                output_path_hr = os.path.join(output_folder_hr, filename)
                # 保存上采样后的图像
                cv2.imwrite(output_path, up_img)
                cv2.imwrite(output_path_hr, cropped_image)
                print(f"处理并保存了 {filename}")
            else:
                print(f"无法读取图像: {image_path}")