import ezdxf
import sys

def inspect_dxf_deep(file_path):
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        print(f"--- Deep Inspecting {file_path} ---")
        
        def count_types(layout, counts):
            for entity in layout:
                t = entity.dxftype()
                counts[t] = counts.get(t, 0) + 1
        
        # 1. Modelspace counts
        msp_counts = {}
        count_types(msp, msp_counts)
        print("Modelspace Entity counts:")
        for t, count in msp_counts.items():
            print(f"  {t}: {count}")
            
        # 2. Check Blocks
        print(f"Total Blocks: {len(doc.blocks)}")
        block_counts = {}
        for block in doc.blocks:
            count_types(block, block_counts)
            
        print("All Blocks Entity counts (combined):")
        for t, count in block_counts.items():
            print(f"  {t}: {count}")

        # 3. Sample some POLYLINEs
        print("Sample POLYLINE layers:")
        polylines = msp.query("POLYLINE")
        if polylines:
            layers = set(p.dxf.layer for p in polylines[:100])
            print(f"  Found polylines in layers: {layers}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_dxf_deep(sys.argv[1] if len(sys.argv) > 1 else r"e:\civil\files\FF.dxf")
