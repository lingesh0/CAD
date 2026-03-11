from sqlalchemy import text

from app.database.session import engine


SQL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS doors (
        id SERIAL PRIMARY KEY,
        floor_id INTEGER NOT NULL REFERENCES floors(id) ON DELETE CASCADE,
        position JSON NOT NULL,
        width DOUBLE PRECISION,
        source VARCHAR,
        block_name VARCHAR,
        connected_rooms JSON
    )
    """,
    "ALTER TABLE floors ADD COLUMN IF NOT EXISTS room_snapshots JSON",
    "ALTER TABLE floors ADD COLUMN IF NOT EXISTS adjacency JSON",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS centroid JSON",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS door_count INTEGER DEFAULT 0",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS adjacency JSON",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS furniture JSON",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS snapshot_path VARCHAR",
    "ALTER TABLE rooms ADD COLUMN IF NOT EXISTS classification_method VARCHAR",
]


def main() -> None:
    with engine.begin() as conn:
        for stmt in SQL_STATEMENTS:
            conn.execute(text(stmt))
    print("Schema upgrade complete: persistence fields for doors/adjacency/confidence/furniture/snapshots are ready.")


if __name__ == "__main__":
    main()
