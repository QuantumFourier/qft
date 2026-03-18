from __future__ import annotations

import json
from pathlib import Path

from qiskit_ibm_runtime.fake_provider import FakeManilaV2


# Export the fake backend properties to a JSON file.
def main() -> None:
    backend = FakeManilaV2()
    properties = backend.properties().to_dict()

    output_path = Path(__file__).resolve().parent / "fake_manila_v2_backend_properties.json"
    output_path.write_text(json.dumps(properties, indent=2, default=str))
    print(f"Wrote backend properties to {output_path.resolve()}")


if __name__ == "__main__":
    main()
