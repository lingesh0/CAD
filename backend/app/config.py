from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path

class Settings(BaseSettings):
    project_name: str = "CAD Processing Engine API"
    version: str = "1.0.0"
    
    # Database
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/cad_engine")
    
    # External APIs
    gemini_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "uploads")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")
    
    @property
    def snapshot_dir(self) -> Path:
        return self.output_dir / "snapshots"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.snapshot_dir, exist_ok=True)
