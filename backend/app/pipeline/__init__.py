"""
CAD Pipeline Package
====================
High-accuracy DXF/DWG floorplan extraction pipeline.

Modules
-------
cad_parser       – DXF geometry extraction with layer filtering
wall_graph       – Wall graph construction with gap closing  
door_detector    – Door detection from ARC entities and INSERT blocks
block_detector   – Furniture/fixture detection from INSERT blocks
room_classifier  – Multi-signal scoring engine (geometry+blocks+adjacency+vision)
adjacency_graph  – Room connectivity graph
vision_validator – Gemini Vision validation for ambiguous rooms
snapshot_renderer– Floor and per-room PNG snapshot generation
pipeline         – Top-level orchestrator
"""
from app.pipeline.pipeline import CADPipeline, run_pipeline

__all__ = ["CADPipeline", "run_pipeline"]
