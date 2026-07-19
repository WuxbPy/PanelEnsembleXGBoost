.. _benchmarks:

Benchmark Comparison
====================

The package includes a comprehensive benchmark suite to compare
PanelEnsembleXGBoost against traditional panel models and modern
ensemble methods.

Datasets
--------

Three classic panel datasets are used:

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 10 15 35

   * - Dataset
     - N
     - Groups
     - Periods
     - Dependent
     - Features
   * - Grunfeld
     - 220
     - 11 firms
     - 20 years
     - invest
     - value, capital
   * - WagePanel
     - 3815
     - 545 workers
     - 7 years
     - lwage
     - educ, exper, union, married
   * - CrimeData
     - 800
     - 50 states
     - 16 years
     - crmrte
     - polpc, unem

Baseline Models
---------------

- **PooledOLS**: Ordinary least squares ignoring panel structure
- **PanelOLS_FE**: Entity fixed effects
- **RandomEffects**: Random effects model
- **MixedLM**: Linear mixed model (statsmodels)
- **GPBoost**: Gradient boosting with random effects

Data Split
----------

The benchmark uses a **time-based** split: the first 70% of time periods
are used for training, the remaining 30% for testing. This avoids the
forward-looking bias that would result from random splitting of panel data.

.. code-block:: python

   from examples.classic_panel_benchmark import train_test_split_by_time
   train, test = train_test_split_by_time(df, time_col='year', train_frac=0.7)

Running the Benchmark
---------------------

.. code-block:: bash

   cd examples
   python classic_panel_benchmark.py
