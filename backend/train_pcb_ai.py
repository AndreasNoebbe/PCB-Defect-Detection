#!/usr/bin/env python3
"""
AI-Powered PCB Defect Detection Model Training (Fixed Paths)
Run from backend/ directory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import time
from pathlib import Path

class PCBDefectDataset(Dataset):
    """Custom dataset for PCB defect images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Define class mappings
        self.class_names = [
            'Missing_hole', 'Mouse_bite', 'Open_circuit', 
            'Short', 'Spur', 'Spurious_copper'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PCBDefectCNN(nn.Module):
    """CNN model for PCB defect classification"""
    
    def __init__(self, num_classes=6, pretrained=True):
        super(PCBDefectCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace final layer for our 6 classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(self.backbone.layer2(
                self.backbone.layer1(self.backbone.maxpool(
                    self.backbone.relu(self.backbone.bn1(
                        self.backbone.conv1(x)
                    ))
                ))
            ))
        ))
        
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        output = self.backbone.fc(features)
        
        return output, features

def main():
    """Main training function"""
    print("🔌 PCB Defect Detection AI Training")
    print("=" * 50)
    
    # Fixed dataset path (from backend/ directory)
    dataset_path = "PCB_DATASET/images"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Make sure you're running this from the backend/ directory")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Create model directory
    model_dir = Path("app/models/ai_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("📁 Loading PCB dataset...")
    
    image_paths = []
    labels = []
    
    defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    for label_idx, defect_type in enumerate(defect_types):
        defect_dir = os.path.join(dataset_path, defect_type)
        
        if os.path.exists(defect_dir):
            images = [f for f in os.listdir(defect_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {defect_type}: {len(images)} images")
            
            for img_file in images:
                img_path = os.path.join(defect_dir, img_file)
                image_paths.append(img_path)
                labels.append(label_idx)
        else:
            print(f"  ❌ Missing directory: {defect_dir}")
    
    print(f"📊 Total dataset: {len(image_paths)} images across {len(defect_types)} defect types")
    
    if len(image_paths) == 0:
        print("❌ No images found! Check your dataset structure.")
        return
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"🔄 Split: {len(train_paths)} train, {len(val_paths)} validation")
    
    # Create datasets
    train_dataset = PCBDefectDataset(train_paths, train_labels, train_transform)
    val_dataset = PCBDefectDataset(val_paths, val_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    print("🤖 Initializing AI model...")
    model = PCBDefectCNN(num_classes=len(defect_types), pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training loop
    print("🚀 Starting training...")
    num_epochs = 30
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  ⭐ New best validation accuracy: {best_acc:.2f}%")
            
            # Save model
            model_path = model_dir / "pcb_defect_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': defect_types,
                'best_acc': best_acc,
                'epoch': epoch,
                'model_architecture': 'resnet18'
            }, model_path)
            
            print(f"  💾 Model saved to {model_path}")
    
    print(f"\n🎯 Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"📁 Model saved at: {model_path}")
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    checkpoint = torch.load(model_path)
    test_model = PCBDefectCNN(num_classes=6)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loads successfully! Classes: {checkpoint['class_names']}")
    
    print("\n✅ Training complete! Next steps:")
    print("1. Replace app/services/cv_service.py with the new AI service")
    print("2. Restart your FastAPI server")
    print("3. Test with your React frontend!")

if __name__ == "__main__":
    main()
