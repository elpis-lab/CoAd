"""Print planning success, successful-query time and path length; optionally export CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.reporting import parser, table

if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--output", type=Path, help="Optional CSV file")
    table(cli.parse_args())
