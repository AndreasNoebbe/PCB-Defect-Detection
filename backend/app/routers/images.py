from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
import os
import uuid

from models.database import get_db, Inspection
from services.cv_service import DefectDetector

router = APIRouter(prefix="/api/v1/images", tags=["images"])

# Initialize the CV service
detector = DefectDetector()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process an image for defect detection"""
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"../uploads/{unique_filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Process image with CV
    results = detector.detect_defects(file_path)
    
    # Create inspection record in database
    inspection = Inspection(
        filename=file.filename,
        file_path=file_path,
        file_size=len(content),
        processing_status="completed",
        processing_time_ms=results["processing_time_ms"],
        confidence_score=results["confidence_score"],
        defect_detected=results["defect_detected"],
        defect_count=results["defect_count"]
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    
    return {
        "inspection_id": inspection.id,
        "filename": file.filename,
        "defect_detected": results["defect_detected"],
        "defect_count": results["defect_count"],
        "confidence_score": results["confidence_score"],
        "processing_time_ms": results["processing_time_ms"],
        "defects": results["defects"]
    }

@router.get("/{inspection_id}")
def get_inspection(inspection_id: int, db: Session = Depends(get_db)):
    """Get inspection results by ID"""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")
    
    return {
        "id": inspection.id,
        "filename": inspection.filename,
        "upload_timestamp": inspection.upload_timestamp,
        "processing_status": inspection.processing_status,
        "processing_time_ms": inspection.processing_time_ms,
        "defect_detected": inspection.defect_detected,
        "defect_count": inspection.defect_count,
        "confidence_score": inspection.confidence_score
    }

@router.get("/")
def list_inspections(db: Session = Depends(get_db)):
    """List all inspections"""
    inspections = db.query(Inspection).all()
    
    return [
        {
            "id": inspection.id,
            "filename": inspection.filename,
            "upload_timestamp": inspection.upload_timestamp,
            "defect_detected": inspection.defect_detected,
            "defect_count": inspection.defect_count,
            "confidence_score": inspection.confidence_score
        }
        for inspection in inspections
    ]

@router.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload and process multiple images for defect detection"""
    
    results = []
    total_processing_time = 0
    
    for file in files:
        try:
            # Generate unique filename
            file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = f"../uploads/{unique_filename}"
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process image with CV
            cv_results = detector.detect_defects(file_path)
            total_processing_time += cv_results["processing_time_ms"]
            
            # Create inspection record in database
            inspection = Inspection(
                filename=file.filename,
                file_path=file_path,
                file_size=len(content),
                processing_status="completed",
                processing_time_ms=cv_results["processing_time_ms"],
                confidence_score=cv_results["confidence_score"],
                defect_detected=cv_results["defect_detected"],
                defect_count=cv_results["defect_count"]
            )
            db.add(inspection)
            db.commit()
            db.refresh(inspection)
            
            results.append({
                "inspection_id": inspection.id,
                "filename": file.filename,
                "defect_detected": cv_results["defect_detected"],
                "defect_count": cv_results["defect_count"],
                "confidence_score": cv_results["confidence_score"],
                "processing_time_ms": cv_results["processing_time_ms"],
                "defects": cv_results["defects"],
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "total_processing_time_ms": total_processing_time,
        "results": results
    }
