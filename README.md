# Weapon Detection with YOLOv8

A computer vision project that detects weapons in images and videos using the state-of-the-art YOLOv8 object detection model.

## Overview

This project implements a weapon detection system using YOLOv8 (You Only Look Once version 8), one of the most advanced real-time object detection algorithms. The system can identify various types of weapons in images and video streams, making it suitable for security applications, surveillance systems, and safety monitoring.

## Features

- **Real-time Detection**: Fast and accurate weapon detection in images and videos
- **YOLOv8 Architecture**: Utilizes the latest YOLO model for superior performance
- **Customizable Training**: Easily configurable training parameters
- **Data Augmentation**: Built-in augmentation techniques for better generalization
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **GPU Acceleration**: Optimized for GPU training and inference
- **Flexible Model Sizes**: Support for multiple YOLOv8 variants (nano to extra-large)

## Model Variants

The project supports different YOLOv8 model sizes:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | Nano | Fastest | Good | Mobile/Edge devices |
| YOLOv8s | Small | Fast | Better | Real-time applications |
| YOLOv8m | Medium | Moderate | High | Balanced performance |
| YOLOv8l | Large | Slower | Higher | High accuracy needs |
| YOLOv8x | Extra Large | Slowest | Highest | Maximum accuracy |

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi
```

### Required Dependencies

```bash
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install pillow
pip install matplotlib
pip install numpy
```

### Quick Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── test/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### Data Configuration (data.yaml)

Create a `data.yaml` file with your dataset configuration:

```yaml
# Dataset configuration
path: /path/to/your/dataset  # Dataset root directory
train: images/train          # Train images (relative to 'path')
val: images/val             # Validation images (relative to 'path')
test: images/test           # Test images (optional)

# Class definitions
nc: 4  # Number of classes
names: ['pistol', 'rifle', 'knife', 'grenade']  # Class names
```

### Annotation Format

Annotations should be in YOLO format (normalized coordinates):

```
class_id center_x center_y width height
```

Example annotation file (`image1.txt`):
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.3
```

## Usage

### Training

1. **Prepare your dataset** following the structure above
2. **Update the data path** in `train.py`
3. **Run training**:

```bash
python train.py
```

### Training Configuration

The training script includes optimized hyperparameters:

```python
model.train(
    data="path/to/data.yaml",    # Dataset configuration
    epochs=200,                  # Training epochs
    batch=4,                     # Batch size
    imgsz=640,                   # Image size
    workers=4,                   # CPU workers
    lr0=0.001,                   # Initial learning rate
    cos_lr=True,                 # Cosine LR decay
    optimizer="SGD",             # Optimizer
    weight_decay=0.0005,         # L2 regularization
    augment=True,                # Data augmentation
    mosaic=1.0,                  # Mosaic augmentation
    mixup=0.2,                   # MixUp augmentation
    label_smoothing=0.1,         # Label smoothing
    patience=50                  # Early stopping
)
```

### Inference

After training, use the model for detection:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Predict on image
results = model('path/to/image.jpg')

# Predict on video
results = model('path/to/video.mp4')

# Real-time webcam detection
results = model(source=0, show=True)
```

## Project Structure

```
weapon-detection-yolov8/
├── train.py                 # Main training script
├── data.yaml               # Dataset configuration
├── requirements.txt        # Python dependencies
├── dataset/               # Dataset directory
│   ├── images/
│   └── labels/
├── runs/                  # Training outputs
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           └── confusion_matrix.png
├── models/                # Saved models
├── results/              # Detection results
└── README.md            # This file
```

## Training Parameters Explained

### Core Parameters
- **epochs**: Number of complete passes through the dataset
- **batch**: Number of images processed simultaneously
- **imgsz**: Input image size (640x640 recommended)
- **workers**: Number of CPU threads for data loading

### Learning Parameters
- **lr0**: Initial learning rate (0.001 works well)
- **cos_lr**: Cosine learning rate scheduler for smooth decay
- **optimizer**: SGD or AdamW (SGD often performs better)
- **weight_decay**: L2 regularization to prevent overfitting

### Augmentation Parameters
- **augment**: Enable/disable data augmentation
- **mosaic**: Probability of mosaic augmentation (combines 4 images)
- **mixup**: Probability of mixup augmentation (blends images)
- **label_smoothing**: Prevents overconfident predictions

### Regularization
- **patience**: Early stopping patience (stops if no improvement)

## Performance Monitoring

### Training Metrics
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Visualization
Training generates several useful plots:
- Loss curves (box, object, class losses)
- Precision-Recall curves
- Confusion matrix
- Training batch samples with predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation
- YOLO research community for continuous improvements
- Open-source computer vision community

**Disclaimer**: This project is for educational and research purposes. Users are responsible for ensuring ethical and legal use of this technology.
