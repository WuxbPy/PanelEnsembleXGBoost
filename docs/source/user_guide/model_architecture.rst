.. _model_architecture:

Three-Stage Ensemble Architecture
=================================

PanelEnsembleXGBoost processes panel data through three sequential stages,
each capturing a distinct aspect of the data generating process:

.. raw:: html

   <div class="architecture-diagram">
   <pre style="font-size:1.1em; line-height:1.8;">
   <strong>Input</strong> (X, y, groups)
         │
         ▼
   ┌──────────────────────────────────────┐
   │ <strong>Stage 1: Random/Mixed Effects</strong>       │
   │ LMM / GLMM (PQL Approximation)       │
   │ Captures between-group heterogeneity │
   │ Output: y₁ = fixed effects + random effects │
   │ Residual: r₁ = y − y₁               │
   └──────────────────────────────────────┘
         │
         ▼
   ┌──────────────────────────────────────┐
   │ <strong>Stage 2: GAM Smooth Effects</strong>        │
   │ pyGAM or B-spline with             │
   │ distribution-appropriate link       │
   │ Models smooth nonlinear patterns    │
   │ Output: y₂ = s₁(r₁) + ... + sₖ(r₁) │
   │ Residual: r₂ = r₁ − y₂             │
   └──────────────────────────────────────┘
         │
         ▼
   ┌──────────────────────────────────────┐
   │ <strong>Stage 3: XGBoost Interaction Effects</strong> │
   │ XGBRegressor with adaptive objective │
   │ Captures high-order interactions     │
   │ Output: y₃ = f(r₂, y₂)             │
   └──────────────────────────────────────┘
         │
         ▼
   <strong>Final Prediction</strong> = ŷ = y₁ + y₂ + y₃
   (in link function space, then transformed via inverse link)
   </pre>
   </div>

Stage-by-Stage Details
----------------------

Stage 1: Random/Mixed Effects (:math:`\hat{y}_1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Account for unobserved heterogeneity across groups (individuals,
firms, regions, etc.).

**Method:**
- **Gaussian family:** Linear Mixed Model (LMM) using Penalized
  Quasi-Likelihood (PQL) approximation. Estimates fixed effects
  coefficients via OLS, then shrinks group residuals toward zero
  using empirical Bayes shrinkage factors.
- **Non-Gaussian families:** Generalized Linear Mixed Model (GLMM)
  via iterative PQL, working in the link function space.

**Output:** :math:`\hat{y}_1 = X\beta + Zu` where :math:`u` are shrunken
group random effects.

**Residual (passed to Stage 2):** :math:`r_1 = y - \hat{y}_1`

**Key parameters:**
- ``random_effects_groups``: List of column names identifying grouping variables
- Minimum 3 groups required; groups with only 1 observation are excluded

Stage 2: GAM Smooth Effects (:math:`\hat{y}_2`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Model smooth nonlinear relationships for continuous covariates.

**Methods:**
- **'gam' (default):** Uses :mod:`pygam` with appropriate family:
  * Gaussian → ``LinearGAM`` with spline terms
  * Binomial → ``LogisticGAM`` with spline terms
  * Fallback: ``sklearn.linear_model.LinearRegression`` (when pygam unavailable)
- **'bspline':** Custom B-spline basis expansion (degree 3, adaptive knots)
  combined with ``LinearRegression``.

**Critical design:** For non-Gaussian families, Stage 2 works in the
**link function space** to ensure additivity when combining all three stages.

**Output:** :math:`\hat{y}_2 = \sum_{j} s_j(r_1)`, sum of smooth terms
**Residual (passed to Stage 3):** :math:`r_2 = r_1 - \hat{y}_2`

Stage 3: XGBoost Nonlinear Interaction Effects (:math:`\hat{y}_3`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Capture complex interaction patterns and high-order nonlinearities
that remain after removing group heterogeneity and smooth trends.

**Method:**
- Uses ``xgboost.XGBRegressor`` with ``'reg:squarederror'`` objective
  (for all families, operating in link space)
- Includes Stage 2 predictions as an additional feature (``gam_smooth_pred``)
  to allow the boosting stage to adjust for systematic smooth effects

**Output:** :math:`\hat{y}_3 = f(r_2; X_{numeric}, \hat{y}_2)`

Final Prediction
----------------

.. math::

   \eta = \hat{y}_1 + \hat{y}_2 + \hat{y}_3
   
   \hat{y} = 
   \begin{cases}
   \eta & \text{(Gaussian, identity link)} \\
   \sigma(\eta) = \frac{1}{1 + e^{-\eta}} & \text{(Binomial, logit link)}
   \end{cases}

Design Rationale
----------------

1. **Sequential residual learning:** Each stage models what the previous stages
   could not capture, preventing duplication of effort and ensuring
   complementary contributions.

2. **Link function space additivity:** All three stages produce outputs in the
   linear predictor (link) space, ensuring their sum corresponds to a
   valid prediction after applying the inverse link function.

3. **Graceful degradation:** If a stage fails (e.g., insufficient groups for
   random effects, pygam unavailable for GAM), the model falls back to
   simpler alternatives or skips the stage entirely.

4. **Modularity:** Each stage can be configured, extended, or replaced
   independently.
