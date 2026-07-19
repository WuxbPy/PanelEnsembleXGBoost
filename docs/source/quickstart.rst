Quickstart
==========

This guide walks you through the basic usage of PanelEnsembleXGBoost for
both regression and binary classification tasks.

Regression Task (Gaussian Family)
---------------------------------

.. code-block:: python

   from panel_ensemble_xgboost import PanelEnsembleXGBoost
   import numpy as np
   import pandas as pd

   # --- 1. Prepare panel data ---
   n_groups = 50
   n_time = 10
   n_obs = n_groups * n_time

   X = pd.DataFrame({
       'x1': np.random.randn(n_obs),
       'x2': np.random.randn(n_obs),
       'group': np.repeat(range(n_groups), n_time),
       'time': np.tile(range(n_time), n_groups)
   })
   y = 0.5 * X['x1'] + 0.3 * X['x2'] + np.random.randn(n_obs)

   # Grouping data (can include time column as well)
   groups = X[['group']]

   # --- 2. Initialize the model ---
   model = PanelEnsembleXGBoost(
       random_effects_groups=['group'],       # Column name(s) for random effects
       smooth_terms={'time': 'gam', 'x1': 'bspline'},  # Smooth terms config
       xgb_params={
           'n_estimators': 100,
           'max_depth': 3,
           'learning_rate': 0.1,
           'random_state': 42
       },
       family='gaussian',                     # Distribution family
       verbose=True
   )

   # --- 3. Fit the model ---
   model.fit(X, y, groups_for_random_effects=groups)

   # --- 4. Make predictions ---
   y_pred = model.predict(X)

   # --- 5. Inspect stage-wise contributions ---
   stage_preds = model.get_stage_predictions(X)
   print("Stage 1 (Random Effects):", np.mean(stage_preds['stage1_random_effects']))
   print("Stage 2 (Smooth Effects):", np.mean(stage_preds['stage2_smooth_effects']))
   print("Stage 3 (XGBoost):",       np.mean(stage_preds['stage3_nonlinear_effects']))

   # --- 6. Model summary ---
   model.summary()

Binary Classification Task (Binomial Family)
--------------------------------------------

.. code-block:: python

   from panel_ensemble_xgboost import PanelEnsembleXGBoost
   import numpy as np
   import pandas as pd

   # --- 1. Prepare binary classification panel data ---
   n_groups = 30
   n_time = 8
   n_obs = n_groups * n_time

   X = pd.DataFrame({
       'x1': np.random.randn(n_obs),
       'x2': np.random.randn(n_obs),
       'group': np.repeat(range(n_groups), n_time),
       'time': np.tile(range(n_time), n_groups)
   })
   # Generate binary outcome
   logit = 0.5 * X['x1'] - 0.3 * X['x2'] + np.random.randn(n_obs) * 0.5
   y_binary = (logit > 0).astype(int)

   groups = X[['group']]

   # --- 2. Initialize with Binomial family ---
   model = PanelEnsembleXGBoost(
       random_effects_groups=['group'],
       smooth_terms={'time': 'gam'},
       family='binomial',   # <-- Binary classification
       verbose=True
   )

   # --- 3. Fit ---
   model.fit(X, y_binary, groups_for_random_effects=groups)

   # --- 4. Predict class labels ---
   y_pred = model.predict(X)

   # --- 5. Predict probabilities ---
   y_proba = model.predict_proba(X)

SHAP Interpretability
---------------------

.. code-block:: python

   from shap_analyzer import SHAPAnalyzer

   # Create analyzer with the model's prediction function
   analyzer = SHAPAnalyzer(
       predict_fn=model.predict,
       X_background=X,
       feature_names=X.columns.tolist()
   )

   # Compute SHAP values
   shap_values = analyzer.compute_shap_values(X)

   # Generate visualizations
   analyzer.plot_summary(plot_type='bar')
   analyzer.plot_summary(plot_type='beeswarm')
   analyzer.plot_waterfall(instance_idx=0)
   analyzer.plot_partial_dependence()

   # Get feature importance table
   importance_df = analyzer.get_feature_importance()
   print(importance_df.head(10))

DID Causal Analysis
-------------------

.. code-block:: python

   from did_analysis import DIDAnalyzer, estimate_att

   # Estimate Average Treatment Effect on the Treated
   att_result = estimate_att(
       data=df,
       outcome_col='outcome',
       treatment_col='treated',
       unit_col='unit_id',
       time_col='year',
       post_period=2012  # Treatment year
   )

   # Parallel trends test
   from did_analysis import test_parallel_trends
   test_result = test_parallel_trends(
       data=df,
       outcome_col='outcome',
       treatment_col='treated',
       unit_col='unit_id',
       time_col='year',
       pre_periods=3
   )

What's Next?
------------

* :doc:`user_guide/model_architecture` — Understand the three-stage architecture in detail
* :doc:`user_guide/distribution_families` — Explore supported distribution families
* :doc:`user_guide/input_output_spec` — Detailed input/output specifications
* :doc:`api/PanelEnsembleXGBoost` — Complete API reference
* :doc:`examples/classic_benchmark` — Full benchmark examples
