from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import shutil
import uuid
from typing import List, Optional
from app.database.session import get_db
from app.database import models
from app.services.cad_service import CADProcessingService
from app.config import settings

router = APIRouter()

@router.post("/upload-cad")
async def upload_cad(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a DXF or DWG file and store it.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".dxf", ".dwg"]:
        raise HTTPException(status_code=400, detail="Only .DXF or .DWG files are supported.")

    # Unique filename for storage
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save to database
    cad_file = models.CADFile(
        filename=file.filename,
        file_path=file_path,
        original_format=file_ext[1:].upper(), # DXF/DWG
        status="uploaded"
    )
    db.add(cad_file)
    db.commit()
    db.refresh(cad_file)
    
    return {"message": "File uploaded successfully", "cad_file_id": cad_file.id}

@router.post("/process-cad/{cad_file_id}")
async def process_cad(cad_file_id: int, db: Session = Depends(get_db)):
    """
    Trigger the processing pipeline for an uploaded CAD file.
    """
    service = CADProcessingService(db)
    try:
        result = service.process_cad_file(cad_file_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Processing failed.")

@router.get("/floor/{floor_id}")
async def get_floor_data(floor_id: int, db: Session = Depends(get_db)):
    """
    Return extracted floor data.
    """
    floor = db.query(models.Floor).filter(models.Floor.id == floor_id).first()
    if not floor:
        raise HTTPException(status_code=404, detail="Floor not found")
        
    return {
        "floor_id": floor.id,
        "cad_file_id": floor.cad_file_id,
        "floor_number": floor.floor_number,
        "snapshot": floor.snapshot_path,
        "room_snapshots": floor.room_snapshots or [],
        "adjacency": floor.adjacency or [],
        "rooms": [
            {
                "id": room.id,
                "name": room.name,
                "area": room.area,
                "coordinates": room.coordinates,
                "centroid": room.centroid,
                "doors": room.door_count,
                "adjacency": room.adjacency or [],
                "confidence": room.confidence,
                "furniture": room.furniture or [],
                "snapshot": room.snapshot_path,
                "classification_method": room.classification_method,
            } for room in floor.rooms
        ],
        "doors": [
            {
                "id": d.id,
                "position": d.position,
                "width": d.width,
                "source": d.source,
                "block_name": d.block_name,
                "connected_rooms": d.connected_rooms or [],
            }
            for d in floor.doors
        ],
    }

@router.get("/snapshot/{floor_id}")
async def get_snapshot(floor_id: int, db: Session = Depends(get_db)):
    """
    Return the generated floor snapshot (PNG).
    """
    floor = db.query(models.Floor).filter(models.Floor.id == floor_id).first()
    if not floor or not floor.snapshot_path:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    snapshot_path = floor.snapshot_path
    if not os.path.isabs(snapshot_path):
        snapshot_path = os.path.join(settings.snapshot_dir, snapshot_path)
    if not os.path.exists(snapshot_path):
         raise HTTPException(status_code=404, detail="Snapshot file missing")
         
    return FileResponse(snapshot_path, media_type="image/png")

@router.get("/rooms/{floor_id}")
async def get_rooms(floor_id: int, db: Session = Depends(get_db)):
    """
    Return structured list of detected rooms for a given floor.
    """
    rooms = db.query(models.Room).filter(models.Room.floor_id == floor_id).all()
    if not rooms:
         raise HTTPException(status_code=404, detail="Rooms not found")
         
    return [
        {
            "id": room.id,
            "name": room.name,
            "area": room.area,
            "coordinates": room.coordinates,
            "centroid": room.centroid,
            "doors": room.door_count,
            "adjacency": room.adjacency or [],
            "confidence": room.confidence,
            "furniture": room.furniture or [],
            "snapshot": room.snapshot_path,
            "classification_method": room.classification_method,
        } for room in rooms
    ]


@router.get("/doors/{floor_id}")
async def get_doors(floor_id: int, db: Session = Depends(get_db)):
    """Return detected doors and connectivity for a floor."""
    doors = db.query(models.Door).filter(models.Door.floor_id == floor_id).all()
    if not doors:
        raise HTTPException(status_code=404, detail="Doors not found")

    return [
        {
            "id": d.id,
            "position": d.position,
            "width": d.width,
            "source": d.source,
            "block_name": d.block_name,
            "connected_rooms": d.connected_rooms or [],
        }
        for d in doors
    ]
