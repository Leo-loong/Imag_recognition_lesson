import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # 使用 Kaiming 正态分布初始化权重
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            # 将偏置初始化为零
            nn.init.zeros_(m.bias)


class CSC_layer(nn.Module):
    def __init__(self, num_iter, in_channels, num_filters, kernel_size, stride):
        super(CSC_layer, self).__init__()
        self.num_iter = num_iter
        self.in_channel = in_channels
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.num_filters = num_filters
        self.stride = stride
        self.down_conv = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                   kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        self.up_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                 kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        self.lam = nn.Parameter(0.01 * torch.ones(1, self.num_filters, 1, 1))
        self.restore_conv = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                      kernel_size=self.kernel_size, padding=self.padding, stride=self.stride,
                                      bias=False)
        # 应用kaiming初始化
        self.apply(kaiming_init)

    def forward(self, x):
        p1 = self.up_conv(x)  # F * x
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam))  # a^(0)
        for i in range(self.num_iter):
            p3 = self.down_conv(tensor)  # G * a^(0)
            p4 = self.up_conv(p3)  # F * G * a^(0)
            p5 = tensor - p4  # a^(0) - F * G * a^(0)
            p6 = torch.add(p1, p5)  # a^(0) - F * G * a^(0) + F * x_hat
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam))  # S(`)
        restore = self.restore_conv(tensor)
        return tensor, restore


class CSC_SR(nn.Module):
    def __init__(self, num_iter, in_channels, num_filters, kernel_size, stride, alpha=0.5,
                 soft_threshold=0.1):
        super(CSC_SR, self).__init__()
        self.csc_x = CSC_layer(num_iter, in_channels, num_filters, kernel_size, stride)
        self.csc_y = CSC_layer(num_iter, in_channels, num_filters, kernel_size, stride)
        self.alpha = alpha
        self.soft_threshold = nn.Parameter(torch.tensor(soft_threshold))
        self.final_conv = nn.Conv2d(in_channels=num_filters, out_channels=in_channels,
                                    kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                    stride=stride, bias=False)

    def forward(self, x, y):
        tensor_x, _ = self.csc_x(x)
        tensor_y, _ = self.csc_y(y)
        weighted_avg = self.alpha * tensor_x + (1 - self.alpha) * tensor_y
        soft_thresholded = torch.mul(torch.sign(weighted_avg),
                                     F.relu(torch.abs(weighted_avg) - self.soft_threshold))
        output = self.final_conv(soft_thresholded)
        return output
