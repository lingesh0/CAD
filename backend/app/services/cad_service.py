import os
import shutil
import logging
from sqlalchemy.orm import Session
from app.database import models
from app.cad_parser.extractor import DXFExtractor
from app.cad_parser.converter import DWGConverter
from app.geometry_engine.room_detector import RoomDetector
from app.ai_processing.label_normalizer import LabelNormalizer
from app.snapshot_generator.renderer import SnapshotGenerator
from app.config import settings

logger = logging.getLogger(__name__)

class CADProcessingService:
    def __init__(self, db: Session):
        self.db = db
        self.converter = DWGConverter()
        self.normalizer = LabelNormalizer()
        self.snapshot_gen = SnapshotGenerator()

    def process_cad_file(self, cad_file_id: int):
        """
        Full processing pipeline for a CAD file.
        """
        cad_file = self.db.query(models.CADFile).filter(models.CADFile.id == cad_file_id).first()
        if not cad_file:
            raise ValueError(f"CAD File with id {cad_file_id} not found.")

        cad_file.status = "processing"
        self.db.commit()

        try:
            input_path = cad_file.file_path
            dxf_path = input_path

            # 1. Convert DWG to DXF if necessary
            if cad_file.original_format.lower() == "dwg":
                logger.info(f"Converting DWG to DXF: {input_path}")
                dxf_path = self.converter.convert_to_dxf(input_path, str(settings.output_dir))
            
            # 2. Extract Geometry
            extractor = DXFExtractor(dxf_path)
            geometry_data = extractor.extract_geometry()
            
            # 3. Create Floor Record (assume 1 floor per DXF for now)
            floor = models.Floor(cad_file_id=cad_file.id, floor_number=1)
            self.db.add(floor)
            self.db.flush() # Get floor.id
            
            # 4. Save Walls (segments from new extractor)
            walls_data = []
            for seg in geometry_data.get("segments", []):
                s, e = seg["start"], seg["end"]
                wall = models.Wall(
                    floor_id=floor.id,
                    start_point=s,
                    end_point=e,
                    length=((s[0] - e[0])**2 + (s[1] - e[1])**2)**0.5,
                )
                self.db.add(wall)
                walls_data.append({"start_point": s, "end_point": e})

            # 5. Room Detection
            detector = RoomDetector(geometry_data)
            detected_rooms = detector.detect_rooms()
            
            # 6. Normalize Labels
            raw_labels = [r["original_label"] for r in detected_rooms if r.get("original_label")]
            normalized_map = self.normalizer.normalize_labels(raw_labels)
            
            # 7. Save Rooms and Labels
            final_rooms_for_snapshot = []
            for room_data in detected_rooms:
                orig_label = room_data.get("original_label")
                norm_name = normalized_map.get(orig_label, room_data["name"])
                
                room = models.Room(
                    floor_id=floor.id,
                    name=norm_name,
                    original_label=orig_label,
                    area=room_data.get("area_sqft", room_data.get("area", 0.0)),
                    coordinates=room_data["coordinates"]
                )
                self.db.add(room)
                final_rooms_for_snapshot.append({
                    "name": norm_name,
                    "area_sqft": room_data.get("area_sqft", room_data.get("area", 0.0)),
                    "centroid": room_data.get("centroid"),
                    "coordinates": room_data["coordinates"],
                })

            # 8. Generate Snapshot
            snapshot_filename = self.snapshot_gen.generate_snapshot(
                walls_data, final_rooms_for_snapshot, floor_id=floor.id
            )
            floor.snapshot_path = snapshot_filename
            
            cad_file.status = "completed"
            self.db.commit()
            
            return {
                "floor_id": floor.id,
                "status": "completed",
                "snapshot": snapshot_filename,
                "room_count": len(detected_rooms)
            }

        except Exception as e:
            logger.exception(f"Error processing CAD file {cad_file_id}: {e}")
            cad_file.status = "failed"
            self.db.commit()
            raise
