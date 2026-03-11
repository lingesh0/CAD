import os
import shutil
import logging
from sqlalchemy.orm import Session
from app.database import models
from app.cad_parser.converter import DWGConverter
from app.pipeline.pipeline import CADPipeline
from app.config import settings

logger = logging.getLogger(__name__)

class CADProcessingService:
    def __init__(self, db: Session):
        self.db = db
        self.converter = DWGConverter()

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

            # Run the new high-accuracy pipeline
            pipeline = CADPipeline(
                output_dir=str(settings.output_dir),
                use_vision=True,
            )
            result = pipeline.run(input_path)

            # ── Persist to database ──────────────────────────────────────
            floor_data = result["floors"][0] if result.get("floors") else {}

            floor = models.Floor(cad_file_id=cad_file.id, floor_number=1)
            self.db.add(floor)
            self.db.flush()  # get floor.id

            floor.snapshot_path = floor_data.get("snapshot")
            floor.adjacency = floor_data.get("adjacency", [])

            rooms_payload = []
            room_snapshots = []
            for idx, room_data in enumerate(floor_data.get("rooms", [])):
                area_sqft = float(room_data.get("area_sqft", 0.0) or 0.0)
                room_snapshot = room_data.get("snapshot")
                room_snapshots.append(room_snapshot)

                room = models.Room(
                    floor_id=floor.id,
                    name=room_data.get("name", f"Room {idx + 1}"),
                    original_label=room_data.get("name"),
                    area=area_sqft,
                    coordinates=room_data.get("coordinates", []),
                )
                self.db.add(room)
                self.db.flush()

                room.centroid        = room_data.get("centroid", [0.0, 0.0])
                room.door_count      = int(room_data.get("doors", 0) or 0)
                room.adjacency       = room_data.get("adjacent", [])
                room.confidence      = float(room_data.get("confidence", 0.5) or 0.5)
                room.furniture       = room_data.get("furniture", [])
                room.snapshot_path   = room_snapshot
                room.classification_method = room_data.get("classification_method", "rules")

                rooms_payload.append({
                    "room_id": idx + 1,
                    "name":    room_data.get("name", f"Room {idx + 1}"),
                    "area_sqft":  round(area_sqft, 2),
                    "centroid":   room_data.get("centroid", [0.0, 0.0]),
                    "doors":      int(room_data.get("doors", 0) or 0),
                    "adjacency":  room_data.get("adjacent", []),
                    "confidence": float(room_data.get("confidence", 0.5) or 0.5),
                    "snapshot":   room_snapshot,
                    "furniture":  room_data.get("furniture", []),
                    "classification_method": room_data.get("classification_method", "rules"),
                })

            floor.room_snapshots = room_snapshots

            # Persist door records
            for door in floor_data.get("doors", []):
                drow = models.Door(
                    floor_id=floor.id,
                    position=door.get("center", [0.0, 0.0]),
                    width=float(door.get("width_m", 0.0) or 0.0),
                    source=door.get("source"),
                    block_name=door.get("block_name"),
                    connected_rooms=door.get("connected_rooms", []),
                )
                self.db.add(drow)

            cad_file.status = "completed"
            self.db.commit()

            return {
                "file": cad_file.filename,
                "status": "completed",
                "floors": [
                    {
                        "floor_id":  floor.id,
                        "snapshot":  floor.snapshot_path,
                        "doors":     floor_data.get("doors", []),
                        "adjacency": floor.adjacency,
                        "rooms":     rooms_payload,
                    }
                ],
            }

        except Exception as e:
            logger.exception(f"Error processing CAD file {cad_file_id}: {e}")
            cad_file.status = "failed"
            self.db.commit()
            raise
