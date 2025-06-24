from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO('app/models/ai_models/pcb_yolo_model.pt')

# Test on multiple validation images with lower confidence
test_images = list(Path("PCB_YOLO_Dataset/images/val").glob("*.jpg"))[:5]

print("🧪 Testing YOLO model with different confidence thresholds...")

for conf_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"\n📊 Confidence threshold: {conf_thresh}")
    
    total_detections = 0
    for img_path in test_images:
        results = model(str(img_path), conf=conf_thresh, verbose=False)
        
        for r in results:
            detections = len(r.boxes)
            total_detections += detections
            if detections > 0:
                print(f"  {img_path.name}: {detections} defects")
    
    print(f"  Total detections across 5 images: {total_detections}")
