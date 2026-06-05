#!/usr/bin/env python3
"""
Hartigan Dip Test — raw frequency values
Self-contained. Does not import from validate_iris_v3.
Run: python3 run_diptest.py
"""

import os, random, math
import numpy as np
from PIL import Image
from scipy import ndimage
import diptest

DATASET_PATH = "/Users/kaytfrazier/Downloads/CASIA-Iris-Thousand"
N_SUBJECTS   = 150
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── minimal pipeline (no imports from other files) ────────────────────────────

def load_gray(path):
    try:
        return np.array(Image.open(path).convert("L"), dtype=np.float64)
    except Exception:
        return None

def detect_iris(img):
    h, w = img.shape
    smooth = ndimage.gaussian_filter(img.astype(float), sigma=3)
    thresh = np.percentile(smooth, 5)
    dark   = smooth < thresh
    labeled, n = ndimage.label(dark)
    if n == 0:
        return None
    cy_c, cx_c = h / 2, w / 2
    best, cx, cy = -1, w // 2, h // 2
    for lbl in range(1, n + 1):
        sz = int(ndimage.sum(dark, labeled, lbl))
        if sz < 300:
            continue
        by, bx = ndimage.center_of_mass(dark, labeled, lbl)
        dist = math.hypot(bx - cx_c, by - cy_c)
        score = sz / (1.0 + dist)
        if score > best:
            best, cx, cy = score, int(bx), int(by)
    angles  = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    max_r   = min(cx, cy, w - cx, h - cy) - 5
    if max_r < 40:
        return None
    r_range = np.arange(5, min(max_r, 220))
    means   = []
    for r in r_range:
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, h - 1)
        means.append(img[ys, xs].mean())
    grad = np.gradient(ndimage.gaussian_filter1d(np.array(means), sigma=5))
    s1, e1 = int(np.searchsorted(r_range, 20)), int(np.searchsorted(r_range, 90))
    r_pupil = int(r_range[np.argmax(grad[s1:e1]) + s1]) if e1 > s1 else 45
    s2, e2  = int(np.searchsorted(r_range, r_pupil + 25)), int(np.searchsorted(r_range, min(len(r_range)-1, r_pupil + 130)))
    r_iris  = int(r_range[np.argmax(grad[s2:e2]) + s2]) if e2 > s2 else r_pupil + 90
    if r_iris - r_pupil < 40:
        r_iris = r_pupil + 90
    r_iris = min(r_iris, max_r)
    return cx, cy, r_pupil, r_iris

def get_frequency(img, cx, cy, r_pupil, r_iris, n_r=128, n_theta=360):
    radii  = np.linspace(r_pupil, r_iris, n_r)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rg, tg = np.meshgrid(radii, thetas, indexing="ij")
    xs = np.clip(cx + rg * np.cos(tg), 0, img.shape[1] - 1)
    ys = np.clip(cy + rg * np.sin(tg), 0, img.shape[0] - 1)
    strip = ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")
    profile  = strip.mean(axis=1)
    windowed = profile * np.hanning(n_r)
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[:2] = 0
    end = min(30, len(spectrum) - 1)
    seg = spectrum[2:end]
    thresh = seg.mean() + seg.std()
    peaks  = [i for i in range(2, end)
              if spectrum[i] > thresh
              and spectrum[i] >= spectrum[i-1]
              and spectrum[i] >= spectrum[i+1]]
    dom    = peaks[0] if peaks else int(np.argmax(seg)) + 2
    return dom * (2.0 * r_iris) / (r_iris - r_pupil)

def process_one(path):
    img = load_gray(path)
    if img is None:
        return None
    b = detect_iris(img)
    if b is None:
        return None
    cx, cy, rp, ri = b
    if ri - rp < 30:
        return None
    return get_frequency(img, cx, cy, rp, ri)

# ── collect frequencies ───────────────────────────────────────────────────────
exts = {".jpg", ".jpeg", ".png", ".bmp"}

print("Scanning dataset...")
all_right = []
for sid in sorted(os.listdir(DATASET_PATH)):
    sp = os.path.join(DATASET_PATH, sid)
    if not os.path.isdir(sp):
        continue
    ep = os.path.join(sp, "R")
    if not os.path.isdir(ep):
        continue
    imgs = sorted(f for f in os.listdir(ep)
                  if os.path.splitext(f)[1].lower() in exts)
    if imgs:
        all_right.append(os.path.join(ep, imgs[0]))

sample = random.sample(all_right, min(N_SUBJECTS, len(all_right)))
print(f"Processing {len(sample)} images...")

freqs = []
for i, path in enumerate(sample):
    f = process_one(path)
    if f is not None:
        freqs.append(f)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(sample)} done, {len(freqs)} valid so far")

freqs = np.array(freqs)
print(f"\nValid measurements : {len(freqs)}")
print(f"Range              : {freqs.min():.2f} – {freqs.max():.2f} cycles/width")
print(f"Mean               : {freqs.mean():.3f}")
print(f"Median             : {np.median(freqs):.3f}")
print(f"Std                : {freqs.std():.3f}")

# ── dip test ─────────────────────────────────────────────────────────────────
print("\nRunning Hartigan dip test on raw values...")
stat, pval = diptest.diptest(freqs)
print(f"  Dip statistic : {stat:.6f}")
print(f"  p-value       : {pval:.4f}")
if pval < 0.01:
    verdict = "STRONGLY MULTIMODAL (p<0.01) — eigenmode prediction CONFIRMED"
elif pval < 0.05:
    verdict = "MULTIMODAL (p<0.05) — eigenmode prediction supported"
elif pval < 0.10:
    verdict = "BORDERLINE (p<0.10) — weak evidence of multimodality"
else:
    verdict = "Cannot reject unimodality — eigenmode clustering not confirmed"
print(f"  Result        : {verdict}")
print("\nDone. Paste these results into the Claude chat.")
