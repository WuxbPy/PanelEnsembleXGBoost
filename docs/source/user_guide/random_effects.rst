.. _random_effects:

Stage 1: Random/Mixed Effects
=============================

Overview
--------

Stage 1 captures unobserved heterogeneity across panel entities (individuals,
firms, regions, etc.) by estimating a mixed-effects model. For Gaussian data,
we use a **Linear Mixed Model (LMM)**; for non-Gaussian families, a
**Generalized Linear Mixed Model (GLMM)** via Penalized Quasi-Likelihood (PQL).

How It Works (Gaussian LMM)
---------------------------

The model assumes:

.. math::

   y_{it} = X_{it}\beta + \alpha_i + \varepsilon_{it}

where :math:`\alpha_i \sim (0, \sigma^2_\alpha)` are group-specific random
intercepts and :math:`\varepsilon_{it} \sim (0, \sigma^2_\varepsilon)` is
the error term.

Estimation proceeds in five steps:

1. **OLS fixed effects** — Fit :math:`\hat{\beta} = (X'X)^{-1}X'y`
2. **Residuals** — :math:`r_{it} = y_{it} - X_{it}\hat{\beta}`
3. **Variance components** — Estimate :math:`\hat{\sigma}^2_\varepsilon`
   (within-group) and :math:`\hat{\sigma}^2_\alpha` (between-group)
4. **Shrinkage factors** — :math:`\lambda_i = \frac{\hat{\sigma}^2_\alpha}
   {\hat{\sigma}^2_\alpha + \hat{\sigma}^2_\varepsilon / T_i}`
5. **Shrunken random effects** — :math:`\hat{\alpha}_i = \lambda_i \cdot
   \bar{r}_i` where :math:`\bar{r}_i` is the mean residual for group *i*

This is known as **empirical Bayes shrinkage**: groups with fewer observations
are shrunk more strongly toward zero.

.. code-block:: python

   # Configuration
   model = PanelEnsembleXGBoost(
       random_effects_groups=['region'],  # Grouping variable(s)
       ...
   )

   # Groups with < 2 observations are excluded
   # Minimum 3 groups required; < 5 groups triggers a caution

Output
------

- ``self.y_pred_random_effects``: The combined prediction
  :math:`X\hat{\beta} + \hat{\alpha}_i`
- ``self.residuals_stage1``: :math:`r_1 = y - \hat{y}_1`
- ``self.mixed_lm_model``: Dictionary containing all fitted parameters

Non-Gaussian (GLMM) Extension
-----------------------------

For binomial/multinomial families, the PQL algorithm iterates between:

- **E-step:** Estimate random effects given current fixed effects
- **M-step:** Estimate fixed effects given current random effects

All predictions are in the **link function space** (log-odds for binomial).

Diagnostic Output
-----------------

When ``verbose=True``, Stage 1 prints:

- Primary group variable name and number of groups
- Random effects variance (:math:`\sigma^2_\alpha`)
- Error variance (:math:`\sigma^2_\varepsilon`)
- Average shrinkage factor
- Stage MSE (for Gaussian) or deviance (for non-Gaussian)
