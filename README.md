---
title: LungScan — Nodule Analysis
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
---

# 🫁 LungScan — Lung Nodule Analysis

Automated lung nodule segmentation, juxta-pleural detection, and malignancy risk scoring from CT slices.

## Models
- **Segmentation**: ResUNet-34 (pretrained ResNet-34 encoder) trained on LIDC-IDRI
- **Malignancy**: ResNet-18 classifier trained on LIDC nodule crops

## Usage
1. Upload a CT slice (PNG/JPG or DICOM)
2. Adjust the segmentation threshold if needed (default 0.35)
3. Click **Analyze**

### Output
- Overlay image: nodules highlighted **red (malignant)** or **green (benign)**
- Per-nodule report: diameter, centroid coordinates, bounding box, juxta-pleural flag, malignancy probability

## Model Weights
Weights are loaded from this Space's model repository.  
To use your own weights, set the `HF_REPO` environment variable to your model repo ID
and upload `best_resunet34_v3.pth` and `best_malignancy_classifier.pth` there.

## Dataset
Trained on [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) — 
1018 CT scans with radiologist nodule annotations and malignancy scores.

## Performance
| Metric | Value |
|--------|-------|
| Mean Dice (all) | 0.63 |
| Juxta-pleural recall | 87% |
| Malignancy AUC | 0.83 |
