import math
import numpy as np
import cv2
import os
from skimage.measure import _structural_similarity as ssim
from matplotlib import pyplot as plt
from scipy.stats import entropy
plt.style.use('classic')
def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    pixel_max = 1.
    return 20 * math.log10(pixel_max / math.sqrt(mse))
def batch_PSNR(img1, img2, data_range=255):
    Img1 = img1.data.cpu().numpy().astype(np.float32)
    Img2 = img2.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img1.shape[0]):
        PSNR += compare_psnr(Img1[i,:,:,:], Img1[i,:,:,:], data_range=data_range)
    return (PSNR/Img1.shape[0])
def RMSE(img1,img2):
    mse = np.mean((img1 - img2) ** 2)
    RMSE=math.sqrt(mse)
    return RMSE
def SSIM(img1,img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def normalize(x,times=256):
    x_min = x.min()
    x_max = x.max()
    x_norm = ((x-x_min)/(x_max-x_min)*2)*times
    return x_norm
def KL(img1,img2):
    # img1 = normalize(img1, times=1)
    # img2 = normalize(img2, times=1)
    kl = entropy(img1 / img1.sum(), img2 / img2.sum())
    return kl
def save_sub(img1,img2):
    # img1 = normalize(img1, times=128)
    # img2 = normalize(img2, times=128)
    print(psnr(img1, img2))
    color_img(img1-img2, cv2.COLORMAP_COOL)
def color_img(img, color):
    im_color = cv2.applyColorMap(img, color)
    cv2.imwrite('5_sub3.png',im_color)
a = cv2.imread('hires/001.png')
b = cv2.imread('lowres/001.png')
print(psnr(a, b))
