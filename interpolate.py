import sys
import netCDF4 as nc
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, 'ECCV2022-RIFE')
from model.RIFE_HDv3 import Model

def read_nc(filepath):
    dataset = nc.Dataset(filepath, 'r')
    rad = np.array(dataset.variables['Rad'][:], dtype=np.float32)
    # Fill value replace karo
    fill_val = dataset.variables['Rad']._FillValue
    rad[rad == fill_val] = np.nan
    dataset.close()
    return rad

def norm255(frame):
    # NaN ignore karke normalize karo
    mn = np.nanmin(frame)
    mx = np.nanmax(frame)
    out = (frame - mn) / (mx - mn) * 255
    out = np.nan_to_num(out, nan=0)
    return out.astype(np.uint8)

def to_tensor_512(frame):
    mn = np.nanmin(frame)
    mx = np.nanmax(frame)
    norm = (frame - mn) / (mx - mn)
    norm = np.nan_to_num(norm, nan=0)
    small = cv2.resize(norm, (512, 512))
    t = np.stack([small, small, small], axis=0)
    return torch.from_numpy(t).unsqueeze(0).float()

print("Loading model...")
model = Model()
model.load_model('ECCV2022-RIFE/train_log', -1)
model.eval()
model.device()
print("Model loaded!")

raw_A = read_nc('data/Frame_A.nc')
raw_B = read_nc('data/Frame_B.nc')
raw_gt = read_nc('data/ground_truth.nc')

print(f"Raw A valid range: {np.nanmin(raw_A):.2f} to {np.nanmax(raw_A):.2f}")
print(f"NaN count: {np.isnan(raw_A).sum()}")

tensor_A = to_tensor_512(raw_A)
tensor_B = to_tensor_512(raw_B)

print("Running RIFE...")
with torch.no_grad():
    output = model.inference(tensor_A, tensor_B)

interp_512 = output[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]
interp_full = cv2.resize(interp_512, (raw_A.shape[1], raw_A.shape[0]))
interp_scaled = interp_full * (np.nanmax(raw_A) - np.nanmin(raw_A)) + np.nanmin(raw_A)

disp_A = norm255(raw_A)
disp_interp = norm255(interp_scaled)
disp_gt = norm255(raw_gt)
disp_B = norm255(raw_B)

print(f"NaN count in A: {np.isnan(raw_A).sum()}")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(disp_A, cmap='gray')
axes[0].set_title('Frame A - T=0')
axes[1].imshow(disp_interp, cmap='gray')
axes[1].set_title('AI Interpolated - T=10')
axes[2].imshow(disp_gt, cmap='gray')
axes[2].set_title('Ground Truth - T=10')
axes[3].imshow(disp_B, cmap='gray')
axes[3].set_title('Frame B - T=20')

plt.tight_layout()
plt.savefig('interpolation_result.png', dpi=100)
plt.show()


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

# Resize interp to match gt size for comparison
interp_resized = cv2.resize(interp_512, (512, 512))
gt_resized = cv2.resize(disp_gt, (512, 512))

ssim_score = ssim(interp_resized, gt_resized.astype(np.float32)/255, data_range=1.0)
psnr_score = psnr(gt_resized, (interp_resized*255).astype(np.uint8))
mse_score = np.mean((interp_resized - gt_resized.astype(np.float32)/255)**2)

print(f"SSIM: {ssim_score:.4f}")
print(f"PSNR: {psnr_score:.2f} dB")
print(f"MSE: {mse_score:.6f}")

print("Done!")