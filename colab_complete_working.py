# COMPLETE WORKING CODE FOR GOOGLE COLAB WITH PICKLE FILES
# Run these cells in order in your Colab notebook

# ============================================================================
# CELL 1: Install and Import
# ============================================================================
!pip install torch torchvision tqdm -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ============================================================================
# CELL 2: Load Data from Pickle Files
# ============================================================================
# Load data
with open('train-70_.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('validation-10_.pkl', 'rb') as f:
    val_data = pickle.load(f)

# Convert to numpy arrays
train_images = np.array(train_data['images'])
train_labels = np.array(train_data['labels'])
val_images = np.array(val_data['images'])
val_labels = np.array(val_data['labels'])

print(f"Train: {train_images.shape}, Val: {val_images.shape}")

# Remap labels to [0, num_classes-1]
unique_labels = np.unique(np.concatenate([train_labels, val_labels]))
label_mapping = {old: new for new, old in enumerate(unique_labels)}
train_labels = np.array([label_mapping[l] for l in train_labels])
val_labels = np.array([label_mapping[l] for l in val_labels])

print(f"Remapped labels to range: [{train_labels.min()}, {train_labels.max()}]")
print(f"Number of classes: {len(unique_labels)}")


# ============================================================================
# CELL 3: Custom Dataset Class
# ============================================================================
class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading images from pickle files
    """
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array of images (N, H, W, C)
            labels: numpy array of labels (N,)
            transform: torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        # Get unique classes
        self.classes = np.unique(labels).tolist()
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        image = self.images[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image for transforms
        # Handle different input formats
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            # If float, assume it's in [0, 1] and convert to uint8
            image = Image.fromarray((image * 255).astype(np.uint8))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# CELL 4: Model Architecture
# ============================================================================
class TinyImageNetCNN(nn.Module):
    """
    Custom CNN for Tiny ImageNet Classification
    Input: 64x64x3 RGB images
    Output: 15 classes
    """
    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(TinyImageNetCNN, self).__init__()

        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )

        # Block 4: 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Test model creation
model = TinyImageNetCNN(num_classes=15).to(device)
print(f'\nModel created successfully!')
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


# ============================================================================
# CELL 5: Data Transforms and DataLoaders
# ============================================================================
# Data augmentation for training (aggressive to prevent overfitting)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transform (no augmentation, only normalization)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets using the custom class
train_dataset = CustomImageDataset(
    images=train_images,
    labels=train_labels,
    transform=train_transform
)

val_dataset = CustomImageDataset(
    images=val_images,
    labels=val_labels,
    transform=val_transform
)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
print(f'Number of classes: {train_dataset.num_classes}')
print(f'Classes: {train_dataset.classes}')

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f'\nTraining batches: {len(train_loader)}')
print(f'Validation batches: {len(val_loader)}')


# ============================================================================
# CELL 6: Loss Function and Optimizer
# ============================================================================
# Loss function with label smoothing for better generalization
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target)
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes

# Initialize model, loss, optimizer
model = TinyImageNetCNN(num_classes=15, dropout_rate=0.5).to(device)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True,
    min_lr=1e-6
)

print('Training setup complete!')


# ============================================================================
# CELL 7: Training Functions
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# CELL 8: Training Loop
# ============================================================================
# Training loop with early stopping
best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
train_losses = []
train_accs = []
val_losses = []
val_accs = []

print('Starting training...\n')

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 60)

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_acc)

    # Print epoch summary
    print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'Overfitting gap: {abs(train_acc - val_acc):.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0

        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
        }, 'model.pth')

        print(f'✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
    else:
        patience_counter += 1
        print(f'No improvement. Patience: {patience_counter}/{PATIENCE}')

    print(f'Best Val Acc so far: {best_val_acc:.2f}% (Epoch {best_epoch})')
    print('=' * 60 + '\n')

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f'Early stopping triggered after {epoch+1} epochs!')
        print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')
        break

print('\nTraining completed!')
print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')


# ============================================================================
# CELL 9: Plot Training History
# ============================================================================
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(val_losses, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(train_accs, label='Train Acc', linewidth=2)
ax2.plot(val_accs, label='Val Acc', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'\nFinal Training Accuracy: {train_accs[-1]:.2f}%')
print(f'Final Validation Accuracy: {val_accs[-1]:.2f}%')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Overfitting Gap: {abs(train_accs[-1] - val_accs[-1]):.2f}%')


# ============================================================================
# CELL 10: Submission Functions
# ============================================================================
def load_model(model_path='model.pth', device='cuda'):
    """
    Load the trained model from file

    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model ready for inference
    """
    # Create model instance
    model = TinyImageNetCNN(num_classes=15, dropout_rate=0.5)

    # Load weights
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f'Model loaded successfully from {model_path}')
    print(f'Best validation accuracy during training: {checkpoint["val_acc"]:.2f}%')

    return model


def predict(model, test_loader, device='cuda'):
    """
    Make predictions on test data

    Args:
        model: Trained model
        test_loader: DataLoader containing test data
        device: Device to run inference on

    Returns:
        predictions: numpy array of predicted class labels
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_predictions = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_predictions)


# Test the load function
print('Testing load_model function...')
loaded_model = load_model('model.pth', device=device)
print('✓ Load function works correctly!')


# ============================================================================
# CELL 11: Verify Loaded Model
# ============================================================================
# Verify that loaded model works correctly
print('Testing predict function on validation set...')

# Get predictions
predictions = predict(loaded_model, val_loader, device=device)

# Calculate accuracy
true_labels = val_labels
accuracy = 100. * np.sum(predictions == true_labels) / len(true_labels)

print(f'\n✓ Predict function works correctly!')
print(f'Validation Accuracy with loaded model: {accuracy:.2f}%')
print(f'Number of correct predictions: {np.sum(predictions == true_labels)}/{len(true_labels)}')


# ============================================================================
# CELL 12: Final Summary
# ============================================================================
print('\n' + '='*70)
print('MODEL SUMMARY')
print('='*70)
print(f'Architecture: Custom Sequential CNN')
print(f'Total Parameters: {sum(p.numel() for p in loaded_model.parameters()):,}')
print(f'Trainable Parameters: {sum(p.numel() for p in loaded_model.parameters() if p.requires_grad):,}')
print(f'\nRegularization Techniques:')
print('  ✓ BatchNormalization (after each conv layer)')
print('  ✓ Dropout (spatial dropout in conv blocks, regular dropout in FC layers)')
print('  ✓ Data Augmentation (flip, rotation, color jitter, affine)')
print('  ✓ Label Smoothing (0.1)')
print('  ✓ Weight Decay (1e-4)')
print('  ✓ Early Stopping')
print('  ✓ Learning Rate Scheduling')
print(f'\nPerformance:')
print(f'  Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'  Achieved at Epoch: {best_epoch}')
print('='*70)

print('\n✅ READY FOR SUBMISSION!')
print('Download these files:')
print('1. This notebook (.ipynb)')
print('2. model.pth')
print('\nZip them as: GroupX_Assignment.zip')
