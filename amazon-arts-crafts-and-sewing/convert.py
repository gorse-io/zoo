import sys
from pathlib import Path

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import amazon_converter


convert = amazon_converter("Arts_Crafts_and_Sewing", "amazon-arts-crafts-and-sewing")


if __name__ == "__main__":
    convert()
