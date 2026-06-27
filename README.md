# Satellite Frame Interpolation
## ISRO Hackathon 2026 — Problem Statement 12

AI-powered temporal resolution enhancement of satellite imagery using RIFE optical flow interpolation.

## What it does
Takes two consecutive GOES-19 satellite frames and generates the missing intermediate frame using deep learning — effectively doubling temporal resolution from 20 minutes to 10 minutes.

## Results
- SSIM: 0.5824
- PSNR: 16.07 dB
- MSE: 0.0247

## Tech Stack
- Python, PyTorch, RIFE HDv3
- netCDF4, numpy, OpenCV
- HTML/CSS/JS Dashboard

## Data Source
GOES-19 ABI Channel 13 — NOAA AWS Public Bucket

## How to run
```bash
pip install netCDF4 numpy opencv-python torch torchvision flask scikit-image matplotlib
python interpolate.py
```