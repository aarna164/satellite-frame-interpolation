import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def read_goes_nc(filepath):
    dataset = nc.Dataset(filepath, 'r')
    
    # Channel 13 brightness temperature data
    rad = dataset.variables['Rad'][:]
    
    print(f"File: {filepath}")
    print(f"Shape: {rad.shape}")
    print(f"Min value: {rad.min():.2f}")
    print(f"Max value: {rad.max():.2f}")
    
    return rad

# Read frame A
frame_A = read_goes_nc('data/frame_A.nc')

# Normalize to 0-255 for visualization
frame_A_norm = (frame_A - frame_A.min()) / (frame_A.max() - frame_A.min()) * 255
frame_A_norm = frame_A_norm.astype(np.uint8)

# Show the image
plt.figure(figsize=(10, 8))
plt.imshow(frame_A_norm, cmap='gray')
plt.colorbar(label='Brightness Temperature')
plt.title('GOES-19 ABI Channel 13 - Frame A')
plt.savefig('frame_A.png')
plt.show()

print("Image saved as frame_A.png")

# Read Frame B
frame_B = read_goes_nc('data/Frame_B.nc')
frame_B_norm = (frame_B - frame_B.min()) / (frame_B.max() - frame_B.min()) * 255
frame_B_norm = frame_B_norm.astype(np.uint8)

# Read Ground Truth
gt = read_goes_nc('data/ground_truth.nc')
gt_norm = (gt - gt.min()) / (gt.max() - gt.min()) * 255
gt_norm = gt_norm.astype(np.uint8)

# Show all three side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(frame_A_norm, cmap='gray')
axes[0].set_title('Frame A - T=0')
axes[1].imshow(gt_norm, cmap='gray')
axes[1].set_title('Ground Truth - T=10')
axes[2].imshow(frame_B_norm, cmap='gray')
axes[2].set_title('Frame B - T=20')
plt.savefig('all_frames.png')
plt.show()
print("All frames saved!")

