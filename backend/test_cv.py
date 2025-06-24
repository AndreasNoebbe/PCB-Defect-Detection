import sys
sys.path.append('app')

from services.cv_service import DefectDetector

# Test CV processing
detector = DefectDetector()
results = detector.detect_defects("uploads/sample.jpg")
print("CV Results:", results)
