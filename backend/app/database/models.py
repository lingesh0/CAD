from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base

class CADFile(Base):
    __tablename__ = "cad_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    original_format = Column(String, nullable=False) # DXF or DWG
    status = Column(String, default="uploaded") # uploaded, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    floors = relationship("Floor", back_populates="cad_file", cascade="all, delete-orphan")

class Floor(Base):
    __tablename__ = "floors"

    id = Column(Integer, primary_key=True, index=True)
    cad_file_id = Column(Integer, ForeignKey("cad_files.id"), nullable=False)
    floor_number = Column(Integer, default=1)
    snapshot_path = Column(String, nullable=True)
    room_snapshots = Column(JSON, nullable=True)
    adjacency = Column(JSON, nullable=True)  # list[[room_i, room_j], ...]
    created_at = Column(DateTime, default=datetime.utcnow)

    cad_file = relationship("CADFile", back_populates="floors")
    rooms = relationship("Room", back_populates="floor", cascade="all, delete-orphan")
    walls = relationship("Wall", back_populates="floor", cascade="all, delete-orphan")
    labels = relationship("Label", back_populates="floor", cascade="all, delete-orphan")
    doors = relationship("Door", back_populates="floor", cascade="all, delete-orphan")

class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    floor_id = Column(Integer, ForeignKey("floors.id"), nullable=False)
    name = Column(String, nullable=True) # Normalized name
    original_label = Column(String, nullable=True) # Text from CAD
    area = Column(Float, nullable=True)
    coordinates = Column(JSON, nullable=False) # List of [x, y]
    centroid = Column(JSON, nullable=True)  # [x, y]
    door_count = Column(Integer, nullable=True, default=0)
    adjacency = Column(JSON, nullable=True)  # list[int]
    confidence = Column(Float, nullable=True)
    furniture = Column(JSON, nullable=True)  # list[str]
    snapshot_path = Column(String, nullable=True)
    classification_method = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    floor = relationship("Floor", back_populates="rooms")

class Wall(Base):
    __tablename__ = "walls"

    id = Column(Integer, primary_key=True, index=True)
    floor_id = Column(Integer, ForeignKey("floors.id"), nullable=False)
    start_point = Column(JSON, nullable=False) # [x, y]
    end_point = Column(JSON, nullable=False) # [x, y]
    length = Column(Float, nullable=True)

    floor = relationship("Floor", back_populates="walls")

class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, index=True)
    floor_id = Column(Integer, ForeignKey("floors.id"), nullable=False)
    text = Column(String, nullable=False)
    position = Column(JSON, nullable=False) # [x, y]
    
    floor = relationship("Floor", back_populates="labels")


class Door(Base):
    __tablename__ = "doors"

    id = Column(Integer, primary_key=True, index=True)
    floor_id = Column(Integer, ForeignKey("floors.id"), nullable=False)
    position = Column(JSON, nullable=False)  # [x, y]
    width = Column(Float, nullable=True)
    source = Column(String, nullable=True)
    block_name = Column(String, nullable=True)
    connected_rooms = Column(JSON, nullable=True)  # list[int]

    floor = relationship("Floor", back_populates="doors")
