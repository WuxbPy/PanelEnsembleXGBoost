SHAP Analysis Example
=====================

.. code-block:: python

   from panel_ensemble_xgboost import PanelEnsembleXGBoost
   from shap_analyzer import SHAPAnalyzer
   import pandas as pd
   import numpy as np

   # --- Train model ---
   model = PanelEnsembleXGBoost(
       random_effects_groups=['group'],
       smooth_terms={'time': 'gam'},
       family='gaussian'
   )
   model.fit(X_train, y_train, groups_for_random_effects=groups_train)

   # --- SHAP analysis ---
   analyzer = SHAPAnalyzer(
       predict_fn=model.predict,
       X_background=X_train,
       feature_names=X_train.columns.tolist()
   )

   # Compute SHAP values
   analyzer.compute_shap_values(X_test)

   # Generate all standard plots
   analyzer.generate_all_plots(output_dir='./shap_output')

   # Feature importance
   importance = analyzer.get_feature_importance()
   print(importance.head(10))
