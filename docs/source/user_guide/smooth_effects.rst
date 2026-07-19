.. _smooth_effects:

Stage 2: GAM Smooth Effects
============================

Overview
--------

Stage 2 captures smooth nonlinear relationships for continuous covariates
that remain after removing group-specific heterogeneity. Two smoothing
methods are available: **GAM** (via pyGAM) and **B-spline**.

GAM Smoothing (method='gam')
----------------------------

Uses :mod:`pygam` to fit Generalized Additive Models:

.. code-block:: python

   model = PanelEnsembleXGBoost(
       smooth_terms={'time': 'gam', 'income': 'gam'},
       ...
   )

- Each smooth term is modeled using spline basis functions
- The number of splines is adaptive: ``max(5, min(15, n_samples // 40))``
- Falls back to ``LinearRegression`` if pyGAM is not installed

B-spline Smoothing (method='bspline')
-------------------------------------

Uses a custom B-spline implementation based on :func:`scipy.interpolate.BSpline`:

.. code-block:: python

   model = PanelEnsembleXGBoost(
       smooth_terms={'income': 'bspline', 'gdp': 'bspline'},
       ...
   )

- Degree 3 (cubic) B-splines
- Internal knots placed at quantiles of the covariate
- Knot positions are saved during training and reused during prediction
  to ensure consistency

Important Design Detail
-----------------------

For **non-Gaussian families**, Stage 2 works in the **link function space**
rather than the response space. This is critical because:

1. All three stages must produce additive outputs in the same space
2. The final inverse-link transformation only works correctly if the sum
   is in link space
3. Linear regression (not logistic/pyGAM family models) is used as the
   backend to ensure link-space outputs

Residual Handling
-----------------

Since Stage 2 models residuals from Stage 1 (:math:`r_1`), its output
represents the **smooth nonlinear pattern** not captured by group
heterogeneity alone.

.. code-block:: python

   # Access stage predictions
   stage_preds = model.get_stage_predictions(X)
   y_smooth = stage_preds['stage2_smooth_effects']
