"""Convert FACETpy corrected outputs into a BIDS dataset.

This is a thin wrapper around ``facetpy-run to-bids`` for users who prefer a
script in the repository:

    python scripts/convert_outputs_to_bids.py --input-dir output/chunks --recursive --output-dir output/bids
"""

from facet.cli import bids_main

if __name__ == "__main__":
    raise SystemExit(bids_main())
