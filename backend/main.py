import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

from app.api import cad_router

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Backend API for CAD floor-plan processing engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.project_name}

# Include routers
app.include_router(cad_router.router, prefix="/api", tags=["CAD"])
