# 🔌 AI-Powered PCB Defect Detection System

An intelligent quality control system that uses computer vision and deep learning to automatically detect defects in Printed Circuit Boards (PCBs). The system achieves **88.4% mAP@0.5 accuracy** and can identify 6 different types of PCB defects in real-time.

![PCB Detection Demo](https://img.shields.io/badge/AI-YOLOv8-blue) ![Status](https://img.shields.io/badge/Status-Live-green) ![Accuracy](https://img.shields.io/badge/Accuracy-88.4%25-brightgreen)

## Live Demo

**Experience the system live at:** [andedam.dev](https://andedam.dev)

**Test Sample Image Input & Output:** ![test_pcb](https://github.com/user-attachments/assets/0ff10c75-c4b2-4419-a3dc-9eb16c57d871) ![image](https://github.com/user-attachments/assets/3e22c463-48a7-450d-b5ea-a107690a6d7b)



> **Note:** First-time loading may take 30-60 seconds due to free hosting tier limitations.

## Features

- **Real-time PCB defect detection** with bounding box localization
- **6 defect types classification**: Missing holes, Mouse bites, Open circuits, Short circuits, Spurs, Spurious copper
- **High accuracy**: 88.4% mAP@0.5 on validation dataset
- **Multiple defects per image**: Detects and localizes multiple defects simultaneously
- **Confidence scoring**: Provides confidence levels for each detection
- **Severity classification**: Categorizes defects as Critical, Major, or Minor
- **Interactive web interface** with drag-and-drop image upload
- **Batch processing** support for multiple PCB images
- **Inspection history** and analytics dashboard

## Architecture

### Frontend (React + Hostinger)
- **React.js** application with modern UI/UX
- **Hosted on Hostinger** as static files
- **Image upload** with preview and drag-and-drop
- **Real-time results** visualization with bounding boxes
- **Responsive design** for desktop and mobile

### Backend (FastAPI + Render)
- **FastAPI** RESTful API with automatic documentation
- **YOLOv8** object detection model
- **SQLite** database for inspection records
- **CORS-enabled** for cross-origin requests
- **Deployed on Render** cloud platform

## AI Model Performance

### Training Results
- **Model**: YOLOv8n (nano) - optimized for speed and accuracy
- **Dataset**: 711 PCB images with 6 defect categories
- **Training Split**: 80% training, 20% validation
- **Epochs**: 50 with early stopping
- **mAP@0.5**: 88.4%
- **Precision**: 94.4%
- **Recall**: 79.4%

### Defect Categories
| Defect Type | Severity | Description |
|-------------|----------|-------------|
| Missing Hole | Critical | Drill holes not present where expected |
| Mouse Bite | Major | Small notches at PCB edges |
| Open Circuit | Critical | Broken traces or connections |
| Short Circuit | Critical | Unintended connections between traces |
| Spur | Minor | Extra copper protrusions |
| Spurious Copper | Major | Unwanted copper patches |

## Dataset Information

- **Source**: PCB Defects Dataset from Kaggle
- **Total Images**: 711 high-resolution PCB images
- **Image Resolution**: 3034x1586 pixels
- **Annotation Format**: Pascal VOC XML format
- **Balanced Distribution**: ~118-119 images per defect class
- **Data Augmentation**: Random flips, rotations, and color jittering during training

## Technology Stack

### Frontend
- **React.js** - Component-based UI framework
- **Axios** - HTTP client for API calls
- **CSS3** - Modern styling with responsive design
- **JavaScript ES6+** - Modern JavaScript features

### Backend
- **FastAPI** - High-performance Python web framework
- **Ultralytics YOLOv8** - State-of-the-art object detection
- **OpenCV** - Computer vision preprocessing
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation and serialization
- **Python 3.11** - Runtime environment

### Machine Learning
- **PyTorch** - Deep learning framework
- **YOLOv8** - You Only Look Once object detection
- **Torchvision** - Computer vision transforms
- **NumPy** - Numerical computing
- **Pillow** - Image processing

### Deployment
- **Hostinger** - Frontend hosting (static files)
- **Render** - Backend hosting with auto-deployment
- **GitHub** - Version control and CI/CD triggers
- **Git** - Source code management

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 16+
- Git

### Complete Local Development Setup

```bash
# Step 1: Clone the repository
git clone https://github.com/AndreasNoebbe/PCB-Defect-Detection.git
cd PCB-Defect-Detection

# Step 2: Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 3: Train the Model (Optional)
# Convert dataset from Pascal VOC to YOLO format
python convert_voc_to_yolo.py

# Train the YOLOv8 model
python train_yolo_pcb.py

# Step 4: Start Backend Server
cd app
python main.py

# Step 5: Frontend Setup (Open new terminal)
cd ../frontend
npm install

# Step 6: Start Frontend Development Server
npm start

# Access Application:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
