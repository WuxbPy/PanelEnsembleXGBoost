Classic Panel Benchmark
=======================

This example demonstrates how to benchmark PanelEnsembleXGBoost against
traditional panel models using three classic datasets.

Workflow
--------

1. **Load data**: Grunfeld, WagePanel, or CrimeData
2. **Split**: Time-based (70% train / 30% test)
3. **Fit models**: PooledOLS, Fixed Effects, Random Effects, MixedLM,
   GPBoost, PanelEnsembleXGBoost
4. **Evaluate**: Compare R², MSE, RMSE, MAE
5. **Report**: Generate markdown report and visualization

.. code-block:: bash

   cd examples
   python classic_panel_benchmark.py

Output
------

::

   benchmark_results/
   ├── classic_panel_benchmark.md    # Comprehensive report
   ├── classic_panel_benchmark.csv   # Results table
   ├── benchmark_grunfeld.png        # Grunfeld performance
   ├── benchmark_wagepanel.png       # WagePanel performance
   ├── benchmark_crimedata.png       # CrimeData performance
   └── benchmark_cross_dataset.png   # Cross-dataset comparison

Key Functions
-------------

.. code-block:: python

   from examples.classic_panel_benchmark import (
       train_test_split_by_time,
       load_grunfeld_data,
       load_wage_panel_data,
       load_crime_data,
       run_benchmark_for_dataset,
   )
