DID Analysis Example
====================

.. code-block:: python

   from did_analysis import (
       DIDAnalyzer,
       estimate_att,
       test_parallel_trends,
       run_placebo_test,
       analyze_heterogeneity
   )

   # --- Basic ATT estimation ---
   att = estimate_att(data, 'outcome', 'treated', 'unit_id', 'year', post_period=2012)
   print(f"ATT = {att:.4f}")

   # --- Parallel trends test ---
   test_result = test_parallel_trends(
       data, 'outcome', 'treated', 'unit_id', 'year', pre_periods=3
   )

   # --- Placebo test ---
   placebo = run_placebo_test(data, 'outcome', 'treated', 'unit_id', 'year')

   # --- Heterogeneity analysis ---
   het_results = analyze_heterogeneity(
       data, 'outcome', 'treated', 'unit_id', 'year',
       heterogeneity_var='size'  # e.g., firm size
   )
