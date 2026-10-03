import sys
from pathlib import Path

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import amazon_converter


convert = amazon_converter("Grocery_and_Gourmet_Food", "amazon-grocery-and-gourmet-food")


if __name__ == "__main__":
    convert()
