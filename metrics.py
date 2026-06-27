import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from PIL import Image

# Load saved result image
img = np.array(Image.open('interpolation_result.png'))

h, w = img.shape[0], img.shape[1]
each_w = w // 4

interp = img[:, each_w:each_w*2, 0].astype(np.float32) / 255.0
gt = img[:, each_w*2:each_w*3, 0].astype(np.float32) / 255.0

ssim_score = ssim(interp, gt, data_range=1.0)
psnr_score = psnr(gt, interp, data_range=1.0)
mse_score = np.mean((interp - gt) ** 2)

print(f"\n=== METRICS ===")
print(f"SSIM: {ssim_score:.4f}")
print(f"PSNR: {psnr_score:.2f} dB")
print(f"MSE:  {mse_score:.6f}")