"""
prepare_raw_data.py
===================
Standalone script to convert raw wine data to standardized CSV format.

Purpose
-------
The raw wine dataset (data/raw/wine.data) has no headers, making it difficult
to understand what each column represents. This script reads the raw data and
writes it to a properly formatted CSV file with column headers.

Input:  data/raw/wine.data (no headers, 178 samples × 14 columns)
Output: data/processed/wine.csv (with headers, ready for analysis)

Usage
-----
From project root directory, run:
    python src/preprocessing/prepare_raw_data.py

This is a one-time operation. After running, use load_wine() to load the
standardized CSV file.

Dependencies
------------
- None (uses only standard library: os, sys, csv, math)
- No pandas, numpy, or scikit-learn required
"""

import os
import sys

# ── Add project root to Python path so we can import modules ──────────────
# This allows us to import from src.preprocessing even when running this script
# from the src/preprocessing directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import the prepare_data function from data_loader module
from src.preprocessing.data_loader import prepare_data


if __name__ == "__main__":
    """
    Main execution block — runs when script is executed directly.
    
    This converts data/raw/wine.data to data/processed/wine.csv by calling
    the prepare_data() function with default paths.
    """
    print("\n" + "="*60)
    print("  Wine Data Preparation")
    print("="*60)
    
    # ── Call prepare_data with default paths ──────────────────────────────
    success = prepare_data(
        raw_path="data/raw/wine.data",      # Input: raw data
        output_path="data/processed/wine.csv"  # Output: processed CSV
    )
    
    # ── Print result ──────────────────────────────────────────────────────
    if success:
        print("\n✓ Data preparation completed successfully!")
        print("  Use load_wine('data/processed/wine.csv') to load the data.")
    else:
        print("\n✗ Data preparation failed!")
        print("  Make sure data/raw/wine.data exists.")
        sys.exit(1)  # Exit with error code
