from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiskit_ibm_runtime.fake_provider import FakeManilaV2


# Export the fake backend properties to a JSON file.
def export_backend_properties(output_path: Path | None = None) -> Path:
    backend = FakeManilaV2()
    properties = backend.properties().to_dict()

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "fake_manila_v2_backend_properties.json"

    output_path.write_text(json.dumps(properties, indent=2, default=str))
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fake backend properties to JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to fake_manila_v2_backend_properties.json beside this script.",
    )
    args = parser.parse_args()

    output_path = export_backend_properties(args.output)
    print(f"Wrote backend properties to {output_path}")


if __name__ == "__main__":
    main()
