# iris-frequency-framework
As Above, So Below — Frequency Iris Framework measurement pipeline
# Iris Frequency Framework

**As Above, So Below: Measuring the Prevalence and Constitutional 
Significance of Sacred Geometric Patterns in Human Iris Structure**

K. Frazier — June 2026

---

## What This Is

The measurement pipeline for the Iris Frequency Framework — a novel 
synthesis of spatial frequency analysis, sacred geometry, and 
constitutional reading of the human iris.

This repository contains the Python scripts used to validate the 
framework's measurement methodology on the CASIA-Iris-Thousand 
dataset (N=1,000 subjects, 20,000 images).

The full paper is available on OSF Preprints: [link once you have DOI]

---

## Scripts

| Script | Purpose |
|--------|---------|
| `validate_iris_v3.py` | Core measurement pipeline — iris segmentation, linear polar FFT, dominant frequency extraction, texture score, collarette ratio |
| `validate_extended.py` | Extended metrics — Eigenmode Dominance, HCI, Spectral Entropy, PSI, Symmetry Operator, collarette t-test |
| `validate_confirm.py` | Confirmation validation — collarette ratio at N=200, bilateral consistency test, reproducibility split test |
| `run_diptest.py` | Hartigan dip test on raw frequency values for eigenmode multimodality prediction |

---

## Dataset

CASIA-Iris-Thousand. 20,000 iris images from 1,000 subjects.  
Author: Professor Tieniu Tan, Institute of Automation, Chinese Academy of Sciences.  
Available via Kaggle (Sondos Aabed, uploader, MIT License).  
Original source: http://biometrics.idealtest.org

---

## Requirements
pip install Pillow scipy numpy diptest matplotlib
---

## Usage
python3 validate_iris_v3.py
python3 validate_extended.py
python3 validate_confirm.py
python3 run_diptest.py

Update the `DATASET_PATH` variable in each script to point to 
your local copy of the CASIA-Iris-Thousand dataset.

---

## Results (CASIA-Iris-Thousand, N=106 valid measurements)

| Metric | Result | Status |
|--------|--------|--------|
| Algorithm determinism | CV = 0.000 | ✓ Confirmed |
| Cross-image reproducibility | CV = 0.0494 | ✓ Confirmed |
| Collarette ratio (proxy) | 0.360 ± 0.081 | ✓ Consistent with φ |
| Eigenmode Dominance (E_d) | 0.632 ± 0.191 | ✓ Confirmed |
| Harmonic Coherence (HCI) | 0.860 ± 0.143 | ✓ Confirmed |
| Spectral Entropy (norm) | 0.341 ± 0.133 | ✓ Confirmed |
| Symmetry Operator (S) | 0.912 ± 0.096 | ✓ Confirmed |
| Eigenmode multimodality | D=0.022, p=0.99 | — Inconclusive |
| Bilateral consistency | r = −0.057 | — Not confirmed |

---

## License

MIT License. See LICENSE file.

---

## Citation

Frazier, K. (2026). As Above, So Below: Measuring the Prevalence 
and Constitutional Significance of Sacred Geometric Patterns in 
Human Iris Structure. OSF Preprints. [DOI once assigned]
