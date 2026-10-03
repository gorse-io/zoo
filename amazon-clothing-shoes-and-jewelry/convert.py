import sys
from pathlib import Path

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import amazon_converter


convert = amazon_converter("Clothing_Shoes_and_Jewelry", "amazon-clothing-shoes-and-jewelry")


if __name__ == "__main__":
    convert()
