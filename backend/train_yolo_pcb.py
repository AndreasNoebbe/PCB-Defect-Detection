#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import os

def main():
    print("🎯 YOLOv8 PCB Defect Detection Training")
    print("=" * 45)
    
    # Create model directory
    model_dir = Path("app/models/ai_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    print("🚀 Starting training...")
    print("📊 Training on your converted PCB dataset")
    
    # Train the model
    results = model.train(
        data='PCB_YOLO_Dataset/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=4,  # Smaller batch for large images
        save=True,
        project=str(model_dir),
        name="pcb_yolo",
        patience=15,
        device='0' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
        verbose=True
    )
    
    # Copy best model
    best_model = model_dir / "pcb_yolo" / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        deployment_model = model_dir / "pcb_yolo_model.pt"
        shutil.copy2(best_model, deployment_model)
        print(f"✅ Model saved: {deployment_model}")
        
        # Test the model
        print("🧪 Testing model...")
        test_model = YOLO(str(deployment_model))
        
        # Get a test image
        test_images = list(Path("PCB_YOLO_Dataset/images/val").glob("*.jpg"))
        if test_images:
            test_img = test_images[0]
            results = test_model(str(test_img))
            
            for r in results:
                print(f"📊 Test on {test_img.name}: {len(r.boxes)} defects detected")
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
