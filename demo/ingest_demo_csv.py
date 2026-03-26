"""Ingest the sample flow cytometry CSV for the demo.

Uses the real ingest_lab_file tool logic to ingest demo/sample_flow_cytometry.csv.
Reports stats: rows ingested, markers detected, file_id.

Usage:
    python demo/ingest_demo_csv.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncocontext.tools.ingest_lab_file import ingest_lab_file


async def ingest_demo_csv() -> dict:
    """Ingest the demo sample_flow_cytometry.csv.

    Returns:
        Result dict from ingest_lab_file.
    """
    csv_path = Path(__file__).resolve().parent / "sample_flow_cytometry.csv"

    if not csv_path.exists():
        print(f"❌ Demo CSV not found at: {csv_path}")
        return {"error": "File not found"}

    print(f"\n🧪 OncoContext Demo CSV Ingestion")
    print(f"{'=' * 50}")
    print(f"File: {csv_path}")
    print()

    result = await ingest_lab_file(
        file_path=str(csv_path),
        file_type="csv",
        experiment_label="CD8 exhaustion panel - organoid co-culture 2024-01",
        metadata={
            "cell_line": "patient-derived tumor organoids",
            "treatment": "Control / Organoid co-culture / Anti-PD1",
            "timepoints": "24h, 48h, 72h",
            "assay": "flow cytometry",
        },
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result

    print(f"✅ CSV Ingested Successfully!")
    print(f"{'=' * 50}")
    print(f"   File ID: {result['file_id']}")
    print(f"   File Name: {result['file_name']}")
    print(f"   Rows: {result['row_count']}")
    print(f"   Columns: {result['column_count']}")
    print(f"   Chunks indexed: {result['chunk_count']}")
    print(f"   Detected markers: {', '.join(result['detected_markers'])}")
    print()
    print(f"📝 Summary:")
    print(f"   {result['summary']}")
    print()
    print(f"💡 Save this file_id for the cross_reference demo:")
    print(f"   file_id = \"{result['file_id']}\"")
    print()
    print(f"🔒 Your data stays LOCAL — never sent to any external service.")

    return result


if __name__ == "__main__":
    asyncio.run(ingest_demo_csv())
