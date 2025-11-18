# CNN Design Challenge - Tiny ImageNet Classifier

A high-performance custom CNN built from scratch for Tiny ImageNet classification with extensive regularization to prevent overfitting.

## Overview

This solution implements a deep Convolutional Neural Network (CNN) for the Tiny ImageNet classification challenge with the following key features:

### Architecture Highlights
- **Deep Sequential CNN** with 4 convolutional blocks
- **15 classes** classification
- **64x64 RGB** input images
- **~6.8M parameters** optimized for generalization

### Anti-Overfitting Strategies
1. **Data Augmentation**
   - Random horizontal flips
   - Random rotations (±15°)
   - Color jitter (brightness, contrast, saturation, hue)
   - Random affine transformations

2. **Regularization Techniques**
   - BatchNormalization after every convolutional and dense layer
   - Spatial Dropout (increasing from 0.2 to 0.4)
   - Dense layer Dropout (0.5)
   - Label Smoothing (0.1)
   - Weight Decay (1e-4)

3. **Training Optimizations**
   - Early stopping (patience: 15 epochs)
   - Learning rate scheduling (ReduceLROnPlateau)
   - Best model checkpointing
   - He initialization for weights

## Model Architecture

```
Input: 64x64x3 RGB Images

Block 1: 64x64 → 32x32
├── Conv2d(3, 64, 3x3) + BatchNorm + ReLU
├── Conv2d(64, 64, 3x3) + BatchNorm + ReLU
├── MaxPool2d(2x2)
└── Dropout2d(0.2)

Block 2: 32x32 → 16x16
├── Conv2d(64, 128, 3x3) + BatchNorm + ReLU
├── Conv2d(128, 128, 3x3) + BatchNorm + ReLU
├── MaxPool2d(2x2)
└── Dropout2d(0.3)

Block 3: 16x16 → 8x8
├── Conv2d(128, 256, 3x3) + BatchNorm + ReLU
├── Conv2d(256, 256, 3x3) + BatchNorm + ReLU
├── Conv2d(256, 256, 3x3) + BatchNorm + ReLU
├── MaxPool2d(2x2)
└── Dropout2d(0.4)

Block 4: 8x8 → 4x4
├── Conv2d(256, 512, 3x3) + BatchNorm + ReLU
├── Conv2d(512, 512, 3x3) + BatchNorm + ReLU
├── Conv2d(512, 512, 3x3) + BatchNorm + ReLU
├── MaxPool2d(2x2)
└── Dropout2d(0.4)

Classifier:
├── Flatten
├── Linear(8192, 1024) + BatchNorm + ReLU + Dropout(0.5)
├── Linear(1024, 512) + BatchNorm + ReLU + Dropout(0.5)
└── Linear(512, 15)

Output: 15 classes
```

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
Pillow>=8.0.0
tqdm>=4.60.0
```

## Usage

### For Google Colab

1. **Upload the notebook to Google Colab**
   ```python
   # Upload cnn_classifier.ipynb to Colab
   ```

2. **Enable GPU**
   - Runtime → Change runtime type → Hardware accelerator → GPU

3. **Download the dataset**
   ```python
   # Update TRAIN_DIR and VAL_DIR in the notebook with your dataset paths
   # Training data: https://drive.google.com/your-training-link
   # Validation data: https://drive.google.com/your-validation-link
   ```

4. **Run all cells**
   - The notebook will automatically train the model
   - Best model will be saved as `model.pth`
   - Training history will be plotted and saved

### For Local Training (with GPU)

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib Pillow tqdm

# Update paths in cnn_classifier.py
# TRAIN_DIR = 'path/to/train'
# VAL_DIR = 'path/to/val'

# Run training
python cnn_classifier.py
```

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Batch Size | 64 | Balanced memory usage and gradient stability |
| Learning Rate | 0.001 | Initial learning rate (adaptive) |
| Optimizer | Adam | Adaptive learning with weight decay (1e-4) |
| Max Epochs | 100 | Maximum training epochs |
| Early Stopping | 15 | Patience for early stopping |
| Label Smoothing | 0.1 | Prevent overconfident predictions |
| Dropout Rate | 0.2-0.5 | Progressive regularization |
| LR Scheduler | ReduceLROnPlateau | Reduce LR when val acc plateaus |

## Expected Performance

Based on the architecture and regularization techniques:

- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Overfitting Gap**: <5%
- **Training Time**: ~2-4 hours on GPU (depends on early stopping)

## Submission Files

The solution provides two formats:

### 1. Jupyter Notebook (cnn_classifier.ipynb)
- Recommended for Google Colab
- Interactive with visualizations
- Step-by-step execution

### 2. Python Script (cnn_classifier.py)
- For local/server training
- Command-line execution
- Same functionality as notebook

Both include:
- `TinyImageNetCNN` class (model architecture)
- `load_model()` function (load trained weights)
- `predict()` function (make predictions)

## Required Functions for Evaluation

### 1. Model Class
```python
model = TinyImageNetCNN(num_classes=15, dropout_rate=0.5)
```

### 2. Load Function
```python
model = load_model('model.pth', device='cuda')
# Returns: Loaded model ready for inference
```

### 3. Predict Function
```python
predictions = predict(model, test_loader, device='cuda')
# Returns: numpy array of predicted class labels
```

## How the Instructor Will Test

```python
# 1. Import your code
from cnn_classifier import TinyImageNetCNN, load_model, predict

# 2. Load your trained model
model = load_model('model.pth')

# 3. Prepare test data (hidden test set)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
test_dataset = ImageFolder(root='hidden_test_dir', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Get predictions
predictions = predict(model, test_loader)

# 5. Calculate accuracy
accuracy = calculate_accuracy(predictions, true_labels)
```

## Tips for High Accuracy

1. **Dataset Preparation**
   - Ensure training and validation data are properly organized
   - Verify image formats and sizes
   - Check class balance

2. **Training Monitoring**
   - Watch the overfitting gap (train_acc - val_acc)
   - If gap > 10%, increase regularization
   - If both accuracies are low, decrease regularization or increase model capacity

3. **Hyperparameter Tuning**
   - Try different learning rates (0.0001 - 0.01)
   - Adjust batch size based on GPU memory
   - Experiment with dropout rates (0.3 - 0.6)

4. **Data Augmentation**
   - The current augmentation is aggressive
   - If validation accuracy is too low, reduce augmentation intensity
   - If overfitting occurs, increase augmentation

5. **Early Stopping**
   - Current patience: 15 epochs
   - Increase if model hasn't converged
   - Decrease if training is too slow

## Submission Checklist

- [ ] Model uses only allowed layers (Conv2D, Pooling, BatchNorm, Dropout, Flatten, Dense)
- [ ] No pre-trained models used
- [ ] No transformers, attention, or residual connections
- [ ] No HPO or NAS algorithms used
- [ ] Built using PyTorch (GPU version)
- [ ] Data augmentation implemented
- [ ] Random seed set for reproducibility
- [ ] `load_model()` function works correctly
- [ ] `predict()` function works correctly
- [ ] `model.pth` file saved and tested
- [ ] Validation accuracy close to training accuracy (no overfitting)
- [ ] Code tested end-to-end

## File Structure for Submission

```
GroupX_Assignment.zip
├── cnn_classifier.ipynb  (or cnn_classifier.py)
└── model.pth
```

## Compliance with Requirements

### ✓ Allowed Components
- Conv2D layers
- MaxPooling layers
- BatchNormalization
- Dropout (Spatial and Regular)
- Flatten layer
- Dense (Linear) layers

### ✗ Not Used (As Per Rules)
- Pre-trained models (ResNet, VGG, EfficientNet)
- Transformers or attention mechanisms
- Residual/skip connections
- DenseBlocks
- HPO tools (Optuna, Ray Tune)
- NAS algorithms (DARTS, etc.)

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
BATCH_SIZE = 32  # or 16
```

### Low Validation Accuracy
```python
# Increase model capacity or reduce regularization
dropout_rate = 0.3  # instead of 0.5
```

### Overfitting (Val Acc << Train Acc)
```python
# Increase regularization
dropout_rate = 0.6
# Or increase data augmentation strength
```

### Model Not Learning
```python
# Check learning rate
LEARNING_RATE = 0.0001  # try lower
# Or check data normalization
```

## Contact & Support

For questions about the assignment, contact your instructor.

For technical issues with the code:
1. Check that all dependencies are installed
2. Verify GPU is available (`torch.cuda.is_available()`)
3. Ensure dataset paths are correct
4. Check that data is in ImageFolder format

## License

This code is provided for educational purposes as part of the CNN Design Challenge assignment.

---

**Good luck with your submission! 🚀**
