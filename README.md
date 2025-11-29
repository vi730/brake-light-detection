***

# Brake Light Detection using YOLOv7

Real-time vehicle brake light detection system using YOLOv7 for autonomous driving applications.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)

## Overview

Binary classification system that detects vehicle brake lights as **Braking** or **Normal** status using synthetic dataset generation and YOLOv7 framework.

Developed during internship at **Institute for Information Industry (III)**, Taiwan (2023) for **Autoware Challenge 2023**.

## Demo

### Real-world Inference
Real-time detection on road footage showing the model distinguishing between **Braking** and **Normal** states.

![Real-world Inference](demo/result_gif.gif)

### Synthetic Data Validation
Comparison between Ground Truth (Labels) and Model Predictions on the synthetic test batch.

| Ground Truth (Labels) | Model Prediction |
| :---: | :---: |
| ![Labels](demo/ground_truth.jpg) | ![Prediction](demo/prediction.jpg) |

## Performance

### Training Metrics
Training results over 20 epochs showing Precision, Recall, and mAP improvements.

![Training Metrics](demo/result_curve.png)

- **Training Environment:** Google Colab (T4 GPU)
- **Epochs:** 20 (Optimal for free-tier usage)
- **Model Size:** ~75MB
- **Speed:** Real-time on GPU

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 10GB+ free disk space

### Setup

```bash
# Clone this repository
git clone https://github.com/vi730/brake-light-detection.git
cd brake-light-detection

# Install dependencies
pip install -r requirements.txt

# Clone and setup YOLOv7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt

# Download pretrained weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
