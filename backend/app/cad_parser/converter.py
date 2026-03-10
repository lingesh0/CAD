import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DWGConverter:
    def __init__(self, oda_path: str = "ODAFileConverter"):
        """
        oda_path should be the executable name/path for Teigha/ODA File Converter.
        e.g., 'C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe' on Windows
        or 'ODAFileConverter' if in PATH.
        """
        self.oda_path = oda_path

    def convert_to_dxf(self, input_dwg_path: str, output_dir: str) -> str:
        """
        Converts DWG to DXF.
        Returns the path to the output DXF file.
        ODA Format: ODAFileConverter <InputFolder> <OutputFolder> <Version> <OutputFormat> <Recurse> <Audit> <InputFileFilter>
        """
        input_path = Path(input_dwg_path).resolve()
        out_dir = Path(output_dir).resolve()
        os.makedirs(out_dir, exist_ok=True)
        
        in_dir = input_path.parent
        filename = input_path.name

        # Teigha requires an input and output directory, not file paths directly.
        # We process a specific file by setting the filter.
        
        cmd = [
            self.oda_path,
            str(in_dir),
            str(out_dir),
            "ACAD2018", # target version
            "DXF",      # target format
            "0",        # recurse
            "0",        # audit
            filename
        ]
        
        try:
            logger.info(f"Running DWG->DXF conversion: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Conversion complete.")
            
            # The output dxf will have the same stem name
            out_dxf = out_dir / (input_path.stem + ".dxf")
            if out_dxf.exists():
                return str(out_dxf)
            else:
                raise FileNotFoundError(f"Expected output DXF {out_dxf} not found.")

        except subprocess.CalledProcessError as e:
            logger.error(f"ODA Converter failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert DWG to DXF: {e.stderr}")
        except FileNotFoundError:
            logger.error("ODAFileConverter executable not found. Ensure it is installed and in PATH.")
            raise
