import sys
from pathlib import Path

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import amazon_converter


convert = amazon_converter("Patio_Lawn_and_Garden", "amazon-patio-lawn-and-garden")


if __name__ == "__main__":
    convert()
