.. _theory:

Theoretical Foundations
=======================

The package includes a :class:`TheoreticalPanelEnsembleXGBoost` class that
extends the core model with theoretical analysis tools.

Consistency
-----------

**Theorem 1 (Fixed-T Consistency):** When *T* is fixed and *N → ∞*,
the three-stage estimator converges to the true function:

.. math::

   \lim_{N \to \infty} \|\hat{f} - f\| = O_p\left(\frac{1}{\sqrt{N}}\right)

Small Sample Bias
-----------------

**Theorem 2 (Bias Order):** When *N* is fixed and *T → ∞*,

.. math::

   \text{Bias}(\hat{\alpha}_i) = O\left(\frac{1}{T}\right) + O\left(\frac{1}{N}\right)

Error Bounds
------------

:meth:`TheoreticalPanelEnsembleXGBoost.derive_error_bounds` derives
theoretical upper bounds for each stage's error contribution.

.. code-block:: python

   from panel_ensemble_xgboost import TheoreticalPanelEnsembleXGBoost
   
   theory = TheoreticalPanelEnsembleXGBoost()
   proof = theory.prove_consistency(n_units=300, n_periods=10)
   print(proof['consistency_rate'])
   # Output: O_p(1/√N) for fixed T

Panel Diagnostics
-----------------

:class:`PanelDiagnostics` provides assumption-testing tools:

.. code-block:: python

   from panel_ensemble_xgboost import PanelDiagnostics
   
   results = PanelDiagnostics.test_panel_assumptions(
       X, y, id_col='region', time_col='year'
   )
   print(results['intraclass_correlation'])

Uncertainty Quantification
--------------------------

:class:`StagewiseUncertaintyQuantification` provides bootstrap-based
standard errors and confidence intervals for each stage's contribution.

.. code-block:: python

   from panel_ensemble_xgboost import StagewiseUncertaintyQuantification
   
   uq = StagewiseUncertaintyQuantification()
   boot_results = uq.bootstrap_stage_contributions(
       model, X, y, n_bootstrap=500
   )
