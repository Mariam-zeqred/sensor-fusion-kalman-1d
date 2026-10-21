"""``python -m kalman1d``: filter the real day, write figures/ and figures/results.json."""

import json

from .report import main

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
