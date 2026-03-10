import ezdxf
from typing import List, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from ezdxf.document import Drawing
    from ezdxf.layouts import Modelspace
    from ezdxf.entities import LWPolyline, Polyline, Insert, Text, MText

def extract_entities(file_path):
    """
    Extract relevant entities from a DXF file.

    Args:
        file_path (str): Path to the DXF file.

    Returns:
        dict: Extracted entities categorized by type.
    """
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    entities = {
        "LINE": [],
        "LWPOLYLINE": [],
        "POLYLINE": [],
        "TEXT": [],
        "MTEXT": [],
        "INSERT": [],
        "ATTRIB": []
    }

    # Extract LINE entities
    for line in msp.query("LINE"):
        entities["LINE"].append({
            "start": line.dxf.start,
            "end": line.dxf.end,
            "layer": line.dxf.layer
        })

    # Extract LWPOLYLINE entities
    for lwpolyline in msp.query("LWPOLYLINE"):
        lwpolyline = cast("LWPolyline", lwpolyline)
        entities["LWPOLYLINE"].append({
            "points": list(lwpolyline.get_points()),  # Use get_points() for LWPolyline
            "is_closed": lwpolyline.closed,  # Use closed attribute
            "layer": lwpolyline.dxf.layer
        })

    # Extract POLYLINE entities
    for polyline in msp.query("POLYLINE"):
        polyline = cast("Polyline", polyline)
        entities["POLYLINE"].append({
            "points": [vertex.dxf.location for vertex in polyline.vertices],
            "layer": polyline.dxf.layer
        })

    # Extract TEXT entities
    for text in msp.query("TEXT"):
        text = cast("Text", text)
        entities["TEXT"].append({
            "text": text.dxf.text,
            "position": text.dxf.insert,
            "layer": text.dxf.layer
        })

    # Extract MTEXT entities
    for mtext in msp.query("MTEXT"):
        mtext = cast("MText", mtext)
        entities["MTEXT"].append({
            "text": mtext.dxf.text,
            "position": mtext.dxf.insert,
            "layer": mtext.dxf.layer
        })

    # Extract INSERT blocks and their attributes
    for block in msp.query("INSERT"):
        block = cast("Insert", block)
        block_data = {
            "name": block.dxf.name,
            "position": block.dxf.insert,
            "layer": block.dxf.layer,
            "attributes": []
        }
        for attrib in block.attribs:
            block_data["attributes"].append({
                "tag": attrib.dxf.tag,
                "text": attrib.dxf.text
            })
        entities["INSERT"].append(block_data)

    return entities

if __name__ == "__main__":
    # Example usage
    file_path = "../files/FF.dxf"  # Replace with your DXF file path
    extracted_entities = extract_entities(file_path)
    for entity_type, items in extracted_entities.items():
        print(f"{entity_type}: {len(items)} items")