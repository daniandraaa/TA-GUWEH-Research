"""Runner tahap augment; lihat README_REVISI.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import main

if __name__ == "__main__":
    main(["augment", *sys.argv[1:]])
