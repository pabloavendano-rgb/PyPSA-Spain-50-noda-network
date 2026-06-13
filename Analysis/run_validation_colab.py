"""
Colab entry point for run_validation.py.
Uses config_colab.py instead of config.py — everything else is identical.

Run from repo root:
    !python Analysis/run_validation_colab.py
"""
import sys
from pathlib import Path

# Swap in the Colab config before run_validation imports 'config'
sys.path.insert(0, str(Path(__file__).parent))
import config_colab
sys.modules["config"] = config_colab

# Run the full validation pipeline
from run_validation import main
main()
