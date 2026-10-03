import sys
from pathlib import Path

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import amazon_converter


convert = amazon_converter("Movies_and_TV", "amazon-movies-and-tv")


if __name__ == "__main__":
    convert()
