#!/usr/bin/env python
"""
Download and save panel datasets to local storage

This script downloads the following datasets:
1. Grunfeld investment data (statsmodels)
2. Wooldridge wage panel data (linearmodels)
3. US state crime rate data (statsmodels)
4. Lalonde dataset for DID analysis (dowhy)

Usage:
    python download_datasets.py
"""

import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_grunfeld():
    """Download Grunfeld investment data"""
    print("\n[1/4] Downloading Grunfeld investment data...")
    
    try:
        from statsmodels.datasets import grunfeld
        ds = grunfeld.load_pandas()
        data = getattr(ds, 'data', ds)
        df = data.reset_index()
        
        filepath = os.path.join(DATASETS_DIR, 'grunfeld.csv')
        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved to: {filepath}")
        print(f"  ✓ Shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_wage_panel():
    """Download Wooldridge wage panel data"""
    print("\n[2/4] Downloading Wooldridge wage panel data...")
    
    try:
        from linearmodels.datasets import wage_panel
        ds = wage_panel.load()
        data = getattr(ds, 'data', ds)
        df = data.reset_index()
        
        filepath = os.path.join(DATASETS_DIR, 'wage_panel.csv')
        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved to: {filepath}")
        print(f"  ✓ Shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_statecrime():
    """Download US state crime rate data"""
    print("\n[3/4] Downloading US state crime rate data...")
    
    try:
        import statsmodels.api as sm
        crime_data = sm.datasets.statecrime.load_pandas().data
        df = crime_data.reset_index()
        
        filepath = os.path.join(DATASETS_DIR, 'statecrime.csv')
        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved to: {filepath}")
        print(f"  ✓ Shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
        return True
    except AttributeError:
        print("  ⚠ statecrime dataset not available in statsmodels")
        print("  → Skipping (will use synthetic data in benchmark)")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_lalonde():
    """Download Lalonde dataset for DID analysis"""
    print("\n[4/4] Downloading Lalonde dataset for DID analysis...")
    
    try:
        from dowhy import datasets
        
        # Generate Lalonde-style dataset
        lalonde = datasets.lalonde_dataset()
        
        if isinstance(lalonde, dict):
            # If returned as dict with 'df' key
            if 'df' in lalonde:
                df = lalonde['df']
            else:
                df = pd.DataFrame(lalonde)
        else:
            df = lalonde
        
        filepath = os.path.join(DATASETS_DIR, 'lalonde.csv')
        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved to: {filepath}")
        print(f"  ✓ Shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
        
        # Save metadata if available
        if isinstance(lalonde, dict):
            meta = {k: v for k, v in lalonde.items() if k != 'df'}
            if meta:
                import json
                meta_path = os.path.join(DATASETS_DIR, 'lalonde_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2, default=str)
                print(f"  ✓ Metadata saved to: {meta_path}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  → Note: dowhy package required for Lalonde dataset")
        return False


def main():
    """Download all datasets"""
    print("=" * 60)
    print("Panel Ensemble XGBoost - Dataset Downloader")
    print("=" * 60)
    print(f"Target directory: {DATASETS_DIR}")
    
    results = {
        'grunfeld': download_grunfeld(),
        'wage_panel': download_wage_panel(),
        'statecrime': download_statecrime(),
        'lalonde': download_lalonde(),
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓ Downloaded" if success else "✗ Failed/Skipped"
        print(f"  {name}: {status}")
    
    success_count = sum(results.values())
    print(f"\nTotal: {success_count}/{len(results)} datasets downloaded")
    
    # List all CSV files in datasets folder
    csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    if csv_files:
        print(f"\nAvailable datasets:")
        for f in csv_files:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
