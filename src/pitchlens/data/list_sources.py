"""Display the source registry without downloading any data."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

DEFAULT_REGISTRY = Path("configs/data_sources.toml")


def load_sources(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return document["sources"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--status", help="Only show sources with this status")
    args = parser.parse_args()

    sources = load_sources(args.registry)
    if args.status:
        sources = [source for source in sources if source["status"] == args.status]

    if not sources:
        print("No matching data sources.")
        return

    columns = ("id", "status", "data_license", "domain")
    widths = {
        column: max(len(column), *(len(str(source[column])) for source in sources))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for source in sources:
        print(
            "  ".join(str(source[column]).ljust(widths[column]) for column in columns)
        )


if __name__ == "__main__":
    main()
