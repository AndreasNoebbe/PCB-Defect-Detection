#!/usr/bin/env python3
"""
Convert Pascal VOC PCB Dataset to YOLO Format for Object Detection
"""

import os
import xml.etree.ElementTree as ET
import shutil
import yaml
from pathlib import Path
import numpy as np

class PCBVOCtoYOLOConverter:
    def __init__(self, dataset_root="PCB_DATASET"):
        self.dataset_root = dataset_root
        self.annotations_dir = os.path.join(dataset_root, "Annotations")
        self.images_dir = os.path.join(dataset_root, "images")
        
        self.defect_classes = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.defect_classes)}
        
        self.output_dir = Path("PCB_YOLO_Dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔄 Converting PCB Dataset: VOC → YOLO")
    
    def parse_xml_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        filename = root.find('filename').text
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            center_x = (xmin + xmax) / 2.0 / img_width
            center_y = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            if name in self.class_to_id:
                objects.append({
                    'class_id': self.class_to_id[name],
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
        
        return {'filename': filename, 'objects': objects}
    
    def convert_dataset(self):
        print("🔄 Converting dataset...")
        
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        all_annotations = []
        defect_type_folders = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
        
        for defect_folder in defect_type_folders:
            annotation_folder = os.path.join(self.annotations_dir, defect_folder)
            image_folder = os.path.join(self.images_dir, defect_folder)
            
            if not os.path.exists(annotation_folder):
                continue
            
            print(f"📁 Processing {defect_folder}...")
            xml_files = [f for f in os.listdir(annotation_folder) if f.endswith('.xml')]
            
            for xml_file in xml_files:
                xml_path = os.path.join(annotation_folder, xml_file)
                
                try:
                    annotation = self.parse_xml_annotation(xml_path)
                    img_name = annotation['filename']
                    img_path = os.path.join(image_folder, img_name)
                    
                    if os.path.exists(img_path):
                        annotation['xml_path'] = xml_path
                        annotation['img_path'] = img_path
                        all_annotations.append(annotation)
                
                except Exception as e:
                    print(f"❌ Error processing {xml_file}: {e}")
        
        print(f"📊 Total annotations: {len(all_annotations)}")
        
        # Split 80/20
        np.random.seed(42)
        np.random.shuffle(all_annotations)
        split_idx = int(0.8 * len(all_annotations))
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
        
        print(f"🔄 Split: {len(train_annotations)} train, {len(val_annotations)} val")
        
        self._process_split(train_annotations, "train")
        self._process_split(val_annotations, "val")
        self._create_dataset_config()
        
        print("✅ Conversion completed!")
        return len(all_annotations)
    
    def _process_split(self, annotations, split_name):
        for i, annotation in enumerate(annotations):
            try:
                img_src = annotation['img_path']
                img_name = f"{split_name}_{i:04d}_{annotation['filename']}"
                img_dst = self.output_dir / "images" / split_name / img_name
                shutil.copy2(img_src, img_dst)
                
                label_name = img_name.replace('.jpg', '.txt')
                label_path = self.output_dir / "labels" / split_name / label_name
                
                with open(label_path, 'w') as f:
                    for obj in annotation['objects']:
                        f.write(f"{obj['class_id']} {obj['center_x']:.6f} {obj['center_y']:.6f} "
                               f"{obj['width']:.6f} {obj['height']:.6f}\n")
            
            except Exception as e:
                print(f"❌ Error processing {annotation['filename']}: {e}")
    
    def _create_dataset_config(self):
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.defect_classes),
            'names': self.defect_classes
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📄 Dataset config saved: {config_path}")

def main():
    print("🔌 PCB Dataset: Pascal VOC → YOLO Conversion")
    print("=" * 50)
    
    if not os.path.exists("PCB_DATASET"):
        print("❌ PCB_DATASET directory not found!")
        return
    
    converter = PCBVOCtoYOLOConverter()
    total_samples = converter.convert_dataset()
    
    print(f"\n✅ Conversion completed!")
    print(f"📁 YOLO dataset created in: PCB_YOLO_Dataset/")
    print(f"📊 Total samples: {total_samples}")
    print(f"🎯 Ready for YOLOv8 training!")

if __name__ == "__main__":
    main()