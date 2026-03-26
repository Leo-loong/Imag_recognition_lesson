import os
import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL.Image as Image
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import Tuple, Union
from CSC_test import CSC_SR

# 将 PyTorch 张量转换为 PIL 图像
def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())

# 自定义图像数据集类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_folder, d_lr_folder, d_hr_folder, transform=None):
        # RGB 图像文件夹路径
        self.rgb_folder = rgb_folder
        # D_lr 图像文件夹路径
        self.d_lr_folder = d_lr_folder
        # D_hr 图像文件夹路径
        self.d_hr_folder = d_hr_folder
        # 图像转换操作
        self.transform = transform
        # RGB 图像文件夹中的文件名列表
        self.file_names = sorted(os.listdir(rgb_folder))

    def __len__(self):
        # 返回数据集的长度
        return len(self.file_names)

    def __getitem__(self, idx):
        # 构建 RGB 图像的文件路径
        rgb_path = os.path.join(self.rgb_folder, self.file_names[idx])
        # 构建 D_lr 图像的文件路径
        d_lr_path = os.path.join(self.d_lr_folder, self.file_names[idx])
        # 构建 D_hr 图像的文件路径
        d_hr_path = os.path.join(self.d_hr_folder, self.file_names[idx])

        # 打开 RGB 图像
        rgb_image = Image.open(rgb_path).convert('L')
        # 打开 D_lr 图像并转换为灰度图
        d_lr_image = Image.open(d_lr_path).convert('L')
        # 打开 D_hr 图像并转换为灰度图
        d_hr_image = Image.open(d_hr_path).convert('L')

        # 若定义了转换操作，则对图像进行转换
        if self.transform:
            rgb_image = self.transform(rgb_image)
            d_lr_image = self.transform(d_lr_image)
            d_hr_image = self.transform(d_hr_image)

        return rgb_image, d_lr_image, d_hr_image

# 计算 PSNR 指标
def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.dim() == 3:
        a = a.unsqueeze(0)
    elif a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    elif b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    # 计算均方误差
    mse = torch.mean((a - b) ** 2).item()
    # 计算 PSNR
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return p

# 重构损失函数
def reconstruction_loss(rev_input, input):
    # 初始化均方误差损失函数
    loss_fn = torch.nn.MSELoss()
    # 计算重构损失
    loss = loss_fn(rev_input, input)
    return loss

# 平均指标计算类
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        # 当前值
        self.val = 0
        # 平均值
        self.avg = 0
        # 总和
        self.sum = 0
        # 数量
        self.count = 0

    def update(self, val, n=1):
        # 更新当前值
        self.val = val
        # 更新总和
        self.sum += val * n
        # 更新数量
        self.count += n
        # 计算平均值
        self.avg = self.sum / self.count

# 配置优化器
def configure_optimizers(net, args):
    # 使用 Adam 优化器
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    return optimizer

# 训练一个 epoch
def train_one_epoch(
        model, criterion_mse, train_dataloader, optimizer, epoch, logger_train, tb_logger, args
):
    # 将模型设置为训练模式
    model.train()
    # 获取模型所在的设备
    device = next(model.parameters()).device
    # 初始化总损失的平均指标
    total_loss = AverageMeter()
    # 初始化均方误差损失的平均指标
    total_mse = AverageMeter()

    for i, (rgb_image, d_lr_image, d_hr_image) in enumerate(train_dataloader):
        # 改成(rgb_image, d_lr_image, d_hr_image)
        # 将低质量图像移至设备
        rgb_image = rgb_image.to(device)
        d_lr_image = d_lr_image.to(device)
        d_hr_image = d_hr_image.to(device)

        # 清空优化器的梯度
        optimizer.zero_grad()
        # 前向传播计算输出
        outputs = model(rgb_image, d_lr_image)

        # 计算均方误差损失
        mse_loss = criterion_mse(outputs, d_hr_image)
        # 计算总损失
        loss = mse_loss * 255 ** 2

        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 更新总损失的平均指标
        total_loss.update(loss.item())
        # 更新均方误差损失的平均指标
        total_mse.update(mse_loss.item())

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i * len(rgb_image)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.3f} |'
                f'\tMSE loss: {mse_loss.item():.3f} |'
            )

    # 记录训练总损失到 TensorBoard
    tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.avg, epoch)
    # 记录训练均方误差损失到 TensorBoard
    tb_logger.add_scalar('{}'.format('[train]: mse'), total_mse.avg, epoch)

# 测试一个 epoch
def test_epoch(args, epoch, test_dataloader, model, logger_val, criterion_mse, tb_logger):
    # 将模型设置为评估模式
    model.eval()
    # 获取模型所在的设备
    device = next(model.parameters()).device
    # 初始化总损失的平均指标
    total_loss = AverageMeter()
    # 初始化均方误差损失的平均指标
    total_mse = AverageMeter()
    # 初始化 PSNR 的平均指标
    total_psnr = AverageMeter()
    # 初始化原始 PSNR 的平均指标
    ori_psnr = AverageMeter()
    i = 1

    with torch.no_grad():
        for rgb_image, d_lr_image, d_hr_image in tqdm(test_dataloader):
            # 将低质量图像移至设备
            rgb_image = rgb_image.to(device)
            d_lr_image = d_lr_image.to(device)
            d_hr_image = d_hr_image.to(device)

            # 前向传播计算输出
            outputs = model(rgb_image, d_lr_image)

            # 计算均方误差损失
            mse_loss = criterion_mse(outputs, d_hr_image)
            # 计算总损失
            loss = mse_loss * 255 ** 2

            # 更新总损失的平均指标
            total_loss.update(loss.item())
            # 更新均方误差损失的平均指标
            total_mse.update(mse_loss.item())

            # 构建保存图像的目录
            save_dir = os.path.join('experiments', args.experiment, 'images')
            ori_dir = os.path.join(save_dir, 'ori')
            if not os.path.exists(ori_dir):
                os.makedirs(ori_dir)
            hires_dir = os.path.join(save_dir, 'hires')
            if not os.path.exists(hires_dir):
                os.makedirs(hires_dir)
            lowres_dir = os.path.join(save_dir, 'lowres')
            if not os.path.exists(lowres_dir):
                os.makedirs(lowres_dir)
            enh_dir = os.path.join(save_dir, 'enhance')
            if not os.path.exists(enh_dir):
                os.makedirs(enh_dir)
            # 将高质量图像转换为 PIL 图像并保存
            oriimg = torch2img(rgb_image)
            oriimg.save(os.path.join(ori_dir, '%03d.png' % i))
            # 将低质量图像转换为 PIL 图像并保存
            hires_img = torch2img(d_hr_image)
            hires_img.save(os.path.join(hires_dir, '%03d.png' % i))
            # 将输出图像转换为 PIL 图像并保存
            enh_img = torch2img(outputs)
            enh_img.save(os.path.join(enh_dir, '%03d.png' % i))
            lowres_img = torch2img(d_lr_image)
            lowres_img.save(os.path.join(lowres_dir, '%03d.png' % i))
            # 计算增强图像的 PSNR
            p = compute_metrics(oriimg, enh_img)
            total_psnr.update(p)
            # 计算原始低质量图像的 PSNR
            pp = compute_metrics(enh_img, hires_img)
            ori_psnr.update(pp)
            i = i + 1

    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {total_loss.avg:.3f} |"
        f"\tMSE loss: {total_mse.avg:.3f} |"
        f"\tPSNR: {total_psnr.avg:.6f} |"
        f"\tPSNR_ori: {ori_psnr.avg:.6f} |"
    )

    # 记录测试总损失到 TensorBoard
    tb_logger.add_scalar('{}'.format('[val]: loss'), total_loss.avg, epoch)
    # 记录测试均方误差损失到 TensorBoard
    tb_logger.add_scalar('{}'.format('[val]: mse'), total_mse.avg, epoch)
    # 记录测试 PSNR 到 TensorBoard
    tb_logger.add_scalar('{}'.format('[val]: psnr'), total_psnr.avg, epoch)
    # 记录测试原始 PSNR 到 TensorBoard
    tb_logger.add_scalar('{}'.format('[val]: ori_psnr'), ori_psnr.avg, epoch)
    return total_loss.avg

# 保存模型检查点
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    # 保存当前检查点
    save_path = os.path.join(filename, "net_checkpoint.pth.tar")
    torch.save(state, save_path)
    if is_best:
        # 若为最佳检查点，则复制一份
        dest_filename = os.path.join(filename, "_checkpoint_best_loss.pth.tar")
        shutil.copyfile(save_path, dest_filename)

# 解析命令行参数
def parse_args(argv):
    parser = argparse.ArgumentParser(description="HybridUNet training script.")
    parser.add_argument(
        "-train_rgb", "--train_rgb_folder",
        type=str,
        required=True,
        help="Training RGB images folder"
    ),
    parser.add_argument(
        "-train_d_lr", "--train_d_lr_folder",
        type=str,
        required=True,
        help="Training D_lr images folder"
    ),
    parser.add_argument(
        "-train_d_hr", "--train_d_hr_folder",
        type=str,
        required=True,
        help="Training D_hr images folder"
    ),
    parser.add_argument(
        "-val_rgb", "--val_rgb_folder",
        type=str,
        required=True,
        help="Validation RGB images folder"
    ),
    parser.add_argument(
        "-val_d_lr", "--val_d_lr_folder",
        type=str,
        required=True,
        help="Validation D_lr images folder"
    ),
    parser.add_argument(
        "-val_d_hr", "--val_d_hr_folder",
        type=str,
        required=True,
        help="Validation D_hr images folder"
    ),
    parser.add_argument(
        "-e",
        "--epochs",
        default=100000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    ),
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    ),
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    ),
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    ),
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    ),
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        # default=(64, 64),
        default=(256, 256),
        help="Size of the training patches to be cropped (default: %(default)s)",
    ),
    parser.add_argument(
        "--val-patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the training patches to be cropped (default: %(default)s)",
    ),
    parser.add_argument("--cuda", action="store_true", help="Use cuda"),
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    ),
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint"),
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument("--val-freq", default=20, type=int),
    args = parser.parse_args(argv)
    return args

# 设置日志记录器
def setup_logger(name, log_dir, log_name, level=logging.INFO, screen=True, tofile=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if tofile:
        file_handler = logging.FileHandler(os.path.join(log_dir, f'{log_name}.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# 主函数
def main(argv):
    # 解析命令行参数
    args = parse_args(argv)
    # 创建实验目录
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    # 设置训练日志记录器
    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment,
                 level=logging.INFO,
                 screen=True, tofile=True)
    # 设置验证日志记录器
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment,
                 level=logging.INFO,
                 screen=True, tofile=True)

    # 获取训练日志记录器
    logger_train = logging.getLogger('train')
    # 获取验证日志记录器
    logger_val = logging.getLogger('val')

    # 初始化 TensorBoard 日志记录器
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    # 创建检查点保存目录
    if not os.path.exists(os.path.join('experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'checkpoints'))

    # 定义训练数据的转换操作
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.patch_size),
            transforms.ToTensor()
        ]
    )

    # 定义验证数据的转换操作
    val_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.val_patch_size),
            transforms.ToTensor()
        ]
    )

    # 创建训练数据集
    train_dataset = ImageDataset(args.train_rgb_folder, args.train_d_lr_folder, args.train_d_hr_folder,
                                 transform=train_transforms)
    # 创建验证数据集
    val_dataset = ImageDataset(args.val_rgb_folder, args.val_d_lr_folder, args.val_d_hr_folder,
                               transform=val_transforms)

    # 选择设备（GPU 或 CPU）
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 创建训练数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 初始化 HybridUNet 模型
    model = CSC_SR(num_iter=8, in_channels=1, num_filters=100, kernel_size=5, stride=1, alpha=0.1,soft_threshold=0.1)
    # 将模型移至设备
    model = model.to(device)

    # 初始化重构损失函数
    criterion_mse = reconstruction_loss
    # 配置优化器
    optimizer = configure_optimizers(model, args)

    # 初始化最后一个 epoch 的编号
    last_epoch = 0
    # 初始化当前损失
    loss = float("inf")
    # 初始化最佳损失
    best_loss = float("inf")
    if args.checkpoint:  # 若指定了检查点，则加载模型
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.param_groups[0]['lr'] = args.learning_rate
        best_loss = checkpoint["best_loss"]

    for epoch in range(last_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        # 训练一个 epoch
        train_one_epoch(
            model,
            criterion_mse,
            train_dataloader,
            optimizer,
            epoch,
            logger_train,
            tb_logger,
            args
        )

        if epoch % args.val_freq == 0:
            # 测试一个 epoch
            loss = test_epoch(args, epoch, val_dataloader, model, logger_val, criterion_mse, tb_logger)

        # 判断是否为最佳损失
        is_best = loss < best_loss
        # 更新最佳损失
        best_loss = min(loss, best_loss)

        if args.save:
            # 保存模型检查点
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                os.path.join('experiments', args.experiment, 'checkpoints')
            )
            if is_best:
                logger_val.info('best checkpoint saved.')
    else:
        # 最后进行一次测试
        loss = test_epoch(args, 0, val_dataloader, model, logger_val, criterion_mse, tb_logger)

if __name__ == "__main__":
    main(sys.argv[1:])