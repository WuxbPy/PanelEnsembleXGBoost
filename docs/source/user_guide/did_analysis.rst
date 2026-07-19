.. _did_analysis:

DID Causal Analysis
===================

The :mod:`did_analysis` module provides a comprehensive Difference-in-Differences
(DID) causal inference framework for panel data.

Key Components
--------------

:class:`DIDAnalyzer`
    Main class for DID analysis with built-in parallel trends testing,
    placebo tests, and heterogeneity analysis.

Functional API
    ``estimate_att()``, ``bootstrap_att()``, ``test_parallel_trends()``,
    ``run_placebo_test()``, ``analyze_heterogeneity()``

Basic Usage
-----------

.. code-block:: python

   from did_analysis import DIDAnalyzer, estimate_att

   # Quick ATT estimation
   att = estimate_att(
       data=df,
       outcome_col='outcome',
       treatment_col='treated',
       unit_col='unit_id',
       time_col='year',
       post_period=2012
   )
   print(f"ATT = {att:.4f}")

   # Full analysis with diagnostics
   from did_analysis import test_parallel_trends, run_placebo_test

   ppt = test_parallel_trends(df, 'outcome', 'treated', 'unit_id', 'year')
   placebo = run_placebo_test(df, 'outcome', 'treated', 'unit_id', 'year')
