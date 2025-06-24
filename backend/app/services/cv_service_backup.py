#!/usr/bin/env python3
"""
YOLO-based PCB Defect Detection Service
"""

import cv2
import numpy as np
from typing import List, Dict
import time
import os

class YOLOPCBDefectDetector:
    def __init__(self):
        self.model_path = "app/models/ai_models/pcb_yolo_model.pt"
        self.model = None
        self.class_names = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
        self.severity_map = {
            'missing_hole': 'critical',
            'mouse_bite': 'major', 
            'open_circuit': 'critical',
            'short': 'critical',
            'spur': 'minor',
            'spurious_copper': 'major'
        }
        self.load_model()
        print("🎯 YOLO PCB Defect Detector initialized")
    
    def load_model(self):
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print("✅ YOLO model loaded successfully!")
            else:
                print(f"⚠️ Model not found: {self.model_path}")
                self.model = None
        except ImportError:
            print("❌ ultralytics not installed")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_defects(self, image_path: str) -> Dict:
        start_time = time.time()
        
        try:
            if self.model is None:
                return self._fallback_detection(image_path)
            
            # Run YOLO inference with LOWER confidence threshold
            results = self.model(image_path, conf=0.15, verbose=False)  # Lower threshold
            
            defects = []
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        defect_type = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                        severity = self.severity_map.get(defect_type, 'major')
                        
                        defects.append({
                            "type": defect_type,
                            "confidence": round(confidence, 3),
                            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            "severity": severity
                        })
            
            # Sort by confidence
            defects.sort(key=lambda x: x['confidence'], reverse=True)
            
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
            print(f"❌ YOLO error: {e}")
            return self._fallback_detection(image_path)
    
    def _fallback_detection(self, image_path: str) -> Dict:
        return {
            "defect_detected": False,
            "defects": [],
            "defect_count": 0,
            "confidence_score": 0.0,
            "processing_time_ms": 50,
            "ai_model": "fallback-no-yolo"
        }

# Backward compatibility
PCBDefectDetector = YOLOPCBDefectDetector
DefectDetector = YOLOPCBDefectDetector
