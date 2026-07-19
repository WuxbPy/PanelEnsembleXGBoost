.. _xgboost_effects:

Stage 3: XGBoost Interaction Effects
=====================================

Overview
--------

Stage 3 applies XGBoost gradient boosting to capture complex interaction
patterns and high-order nonlinearities that persist after accounting for
group heterogeneity (Stage 1) and smooth trends (Stage 2).

Input Features
--------------

The XGBoost model receives:

1. **All numeric feature columns** from the original ``X``
2. **Stage 2 prediction** as an additional feature ``gam_smooth_pred``

The inclusion of ``gam_smooth_pred`` allows the boosting stage to:
- Detect and correct systematic biases in Stage 2 predictions
- Model interactions between smooth effects and other features

Objective Function
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 40

   * - Family
     - Objective
     - Rationale
   * - ``gaussian``
     - ``reg:squarederror``
     - Standard L2 loss in original space
   * - ``binomial``
     - ``reg:squarederror`` (link space)
     - Maintains additivity in link space
   * - ``multinomial``
     - ``multi:softprob``
     - Probability-decomposition via XGBoost directly

Parameter Control
-----------------

.. code-block:: python

   model = PanelEnsembleXGBoost(
       xgb_params={
           'n_estimators': 100,     # Number of boosting rounds
           'max_depth': 3,           # Tree depth (controls interaction order)
           'learning_rate': 0.1,     # Step size shrinkage
           'subsample': 0.8,         # Row subsampling (prevents overfitting)
           'colsample_bytree': 0.8,  # Column subsampling
           'reg_alpha': 0.1,         # L1 regularization
           'reg_lambda': 0.1,        # L2 regularization
           'min_child_weight': 5,    # Minimum leaf weight
           'random_state': 42
       },
       ...
   )

Default parameters (if not specified):

.. code-block:: python

   {
       'n_estimators': 100,
       'max_depth': 3,
       'learning_rate': 0.1,
       'verbosity': 0,
       'random_state': 42
   }

Feature Column Ordering
-----------------------

**Important:** The model saves the exact column order used during training
(``self.xgb_feature_columns``). At prediction time, the same column order
is enforced. Missing columns are filled with 0 and a warning is issued.
