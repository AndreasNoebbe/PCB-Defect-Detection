import time
import os
from pathlib import Path
from typing import Dict

class YOLOPCBDefectDetector:
    def __init__(self):
        self.model = None
        self.class_names = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
        self.find_and_load_model()
        print("🎯 YOLO PCB Defect Detector initialized")
    
    def find_and_load_model(self):
        # Try multiple possible paths
        possible_paths = [
            "models/ai_models/pcb_yolo_model.pt",
            "app/models/ai_models/pcb_yolo_model.pt", 
            "../models/ai_models/pcb_yolo_model.pt",
            "./models/ai_models/pcb_yolo_model.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"📁 Found model at: {path}")
                break
        
        if not model_path:
            print("❌ Model not found at any expected location:")
            for path in possible_paths:
                print(f"  Checked: {path} - {os.path.exists(path)}")
            self.model = None
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_defects(self, image_path: str) -> Dict:
        start_time = time.time()
        
        if self.model is None:
            return self._fallback()
        
        try:
            results = self.model(image_path, conf=0.2, verbose=False)
            
            defects = []
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        if class_id < len(self.class_names):
                            defect_type = self.class_names[class_id]
                            
                            defects.append({
                                "type": defect_type,
                                "confidence": round(confidence, 3),
                                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                "severity": "critical" if defect_type in ["missing_hole", "open_circuit", "short"] else "major"
                            })
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "defect_detected": len(defects) > 0,
                "defects": defects,
                "defect_count": len(defects),
                "confidence_score": round(max([d["confidence"] for d in defects], default=0.0), 3),
                "processing_time_ms": processing_time,
                "ai_model": "YOLOv8-PCB-ObjectDetection"
            }
        
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return self._fallback()
    
    def _fallback(self):
        return {
            "defect_detected": False,
            "defects": [],
            "defect_count": 0,
            "confidence_score": 0.0,
            "processing_time_ms": 50,
            "ai_model": "fallback-no-model"
        }

PCBDefectDetector = YOLOPCBDefectDetector
DefectDetector = YOLOPCBDefectDetector
