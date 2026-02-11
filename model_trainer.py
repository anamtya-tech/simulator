"""
Lightweight 1D CNN model for audio source classification.

This module implements an EfficientNet-inspired architecture optimized for:
1. Low-power devices (Raspberry Pi)
2. 1024-dimensional frequency bin inputs (spectrograms)
3. Incremental/continual learning
4. Fast inference

The model uses:
- Depthwise separable convolutions for efficiency
- Squeeze-and-Excitation blocks for feature recalibration
- Global average pooling to reduce parameters
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class AudioSpectrogramDataset(Dataset):
    """Dataset for frequency bin spectrograms (flexible size)"""
    
    def __init__(self, X, y, augment=False):
        """
        Args:
            X: Numpy array of shape (n_samples, num_bins) where num_bins is typically 128, 256, 512, or 1024
            y: Numpy array of integer labels (n_samples,)
            augment: Apply data augmentation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            # Simple augmentation: add small noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficiency"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Block (MBConv) with SE"""
    
    def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=3, stride=1, se_ratio=4):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.expand = None
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(hidden_dim, reduction=se_ratio)
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand is not None:
            out = self.expand(x)
        else:
            out = x
        
        # Depthwise
        out = self.depthwise(out)
        
        # SE (applied to hidden_dim features)
        out = self.se(out)
        
        # Projection
        out = self.project(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return out


class LightweightAudioClassifier(nn.Module):
    """
    Lightweight 1D CNN for audio source classification.
    Optimized for Raspberry Pi with ~100K parameters.
    Supports flexible input size (128, 256, 512, 1024 bins).
    """
    
    def __init__(self, num_classes, input_size=128, dropout=0.3):
        super().__init__()
        
        # Input: (batch, input_size) -> reshape to (batch, 1, input_size) for Conv1d
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks with progressively smaller spatial dimensions
        self.block1 = MBConvBlock(32, 32, expand_ratio=2, stride=1)
        self.block2 = MBConvBlock(32, 48, expand_ratio=4, stride=2)
        self.block3 = MBConvBlock(48, 48, expand_ratio=4, stride=1)
        self.block4 = MBConvBlock(48, 64, expand_ratio=4, stride=2)
        self.block5 = MBConvBlock(64, 64, expand_ratio=4, stride=1)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.num_classes = num_classes
        self.input_size = input_size
    
    def forward(self, x):
        # x: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.head(x)
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelTrainer:
    """Handles model training, evaluation, and incremental learning"""
    
    def __init__(self, model_save_dir, device=None):
        """
        Args:
            model_save_dir: Directory to save models and metadata
            device: torch device (auto-detected if None)
        """
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = None
        self.label_encoder = None
        self.training_history = []
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data from DataFrame.
        
        Args:
            df: DataFrame with columns [bin_0, ..., bin_N-1, label] where N is auto-detected
            test_size: Fraction for test set
            random_state: Random seed
        
        Returns:
            train_loader, val_loader, label_encoder
        """
        # Auto-detect number of bins from column names
        bin_cols = [col for col in df.columns if col.startswith('bin_')]
        bin_cols = sorted(bin_cols, key=lambda x: int(x.split('_')[1]))
        num_bins = len(bin_cols)
        print(f"INFO: Detected {num_bins} frequency bins in dataset")
        
        # Extract features and labels
        X = df[bin_cols].values
        y = df['label'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Create datasets
        train_dataset = AudioSpectrogramDataset(X_train, y_train, augment=True)
        val_dataset = AudioSpectrogramDataset(X_val, y_val, augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model(self, num_classes, input_size=128):
        """Create a new model"""
        self.model = LightweightAudioClassifier(num_classes=num_classes, input_size=input_size)
        self.model.to(self.device)
        
        print(f"Model created with {self.model.count_parameters():,} parameters (input_size={input_size})")
        return self.model
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, 
              patience=10, save_best=True):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            save_best: Save best model during training
        
        Returns:
            dict: Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        print(f"\nTraining on {self.device}")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('best_model.pth', epoch, val_loss, val_acc)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_val_acc': val_acc,
            'history': history
        })
        
        return history
    
    def predict(self, X, return_probs=False, batch_size=256):
        """
        Make predictions on new data with batching to prevent OOM.
        
        Args:
            X: Numpy array of shape (n_samples, input_size)
            return_probs: If True, return probabilities instead of class labels
            batch_size: Number of samples to process at once (default 256)
        
        Returns:
            predictions (labels or probabilities) and confidence scores
        """
        self.model.eval()
        
        # Handle input size mismatch by resizing
        if X.shape[1] != self.model.input_size:
            print(f"⚠️ Input size mismatch: model expects {self.model.input_size}, got {X.shape[1]}. Resizing...")
            if X.shape[1] > self.model.input_size:
                # Downsample: take every nth element
                step = X.shape[1] / self.model.input_size
                indices = [int(i * step) for i in range(self.model.input_size)]
                X = X[:, indices]
            else:
                # Upsample: interpolate
                old_x = np.linspace(0, 1, X.shape[1])
                new_x = np.linspace(0, 1, self.model.input_size)
                X_resized = np.zeros((X.shape[0], self.model.input_size))
                for i in range(X.shape[0]):
                    f = interp1d(old_x, X[i], kind='linear')
                    X_resized[i] = f(new_x)
                X = X_resized
        
        n_samples = X.shape[0]
        all_probs = []
        all_confidences = []
        
        # Process in batches to prevent OOM
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)
                
                outputs = self.model(X_tensor)
                probs = F.softmax(outputs, dim=1)
                confidences, predicted = probs.max(1)
                
                all_probs.append(probs.cpu())
                all_confidences.append(confidences.cpu())
                
                # Free GPU memory immediately
                del X_tensor, outputs, probs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        confidences = all_confidences.numpy()
        
        if return_probs:
            return all_probs.numpy(), confidences
        else:
            _, predicted = all_probs.max(1)
            predicted_labels = self.label_encoder.inverse_transform(predicted.numpy())
            return predicted_labels, confidences
    
    def save_checkpoint(self, filename, epoch=None, val_loss=None, val_acc=None):
        """Save model checkpoint with metadata"""
        checkpoint_path = self.model_save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder.classes_.tolist(),
            'num_classes': self.model.num_classes,
            'input_size': self.model.input_size,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        if val_acc is not None:
            checkpoint['val_acc'] = val_acc
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save metadata as JSON for easy inspection
        metadata = {
            'timestamp': checkpoint['timestamp'],
            'label_encoder': checkpoint['label_encoder'],
            'num_classes': checkpoint['num_classes'],
            'input_size': checkpoint['input_size'],
            'epoch': epoch,
            'val_loss': float(val_loss) if val_loss else None,
            'val_acc': float(val_acc) if val_acc else None,
            'parameter_count': self.model.count_parameters()
        }
        
        metadata_path = self.model_save_dir / f"{Path(filename).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = self.model_save_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(checkpoint['label_encoder'])
        
        # Recreate model
        num_classes = checkpoint['num_classes']
        input_size = checkpoint.get('input_size', 128)  # Default to 128 for old models
        self.create_model(num_classes, input_size=input_size)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return checkpoint
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train')
        axes[1].plot(history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
