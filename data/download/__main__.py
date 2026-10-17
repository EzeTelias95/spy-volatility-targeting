import argparse
import logging
import sys

from . import __doc__ as usage
from .config import DataError
from .dataset import build
from .snapshot import freeze


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="download fresh data and write a new snapshot + manifest before building",
    )
    args = parser.parse_args()
    try:
        if args.freeze:
            freeze()
        build()
    except DataError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
