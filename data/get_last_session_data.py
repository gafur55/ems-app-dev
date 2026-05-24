"""
Export the most recent session's stimulation data to CSV.

Usage:
    python get_last_session.py
    python get_last_session.py output.csv
"""

import sys
from database import EMSDatabase


def export_last_session(output_path: str = None) -> str:
    db = EMSDatabase()

    # Find the most recent session
    row = db.conn.execute("""
        SELECT session_id, participant_id, started_at
        FROM sessions
        ORDER BY session_id DESC
        LIMIT 1
    """).fetchone()

    if row is None:
        print("✗ No sessions found in database")
        db.close()
        sys.exit(1)

    session_id     = row["session_id"]
    participant_id = row["participant_id"]
    started_at     = row["started_at"]

    print(f"Last session: ID={session_id}  participant={participant_id}  "
          f"started={started_at}")

    # Default filename includes the session ID so files don't overwrite each other
    if output_path is None:
        output_path = f"session_{session_id}_{participant_id}.csv"

    # Filter the full flat export down to this session
    df = db.export_flat_dataframe()
    df = df[df["session_id"] == session_id]

    if df.empty:
        print(f"⚠ Session {session_id} has no stimulations recorded yet")
    else:
        print(f"  {len(df)} stimulations, {df.shape[1]} columns")

    df.to_csv(output_path, index=False)
    print(f"✓ Exported to {output_path}")

    db.close()
    return output_path


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    export_last_session(output)