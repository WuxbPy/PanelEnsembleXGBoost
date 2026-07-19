API Reference
=============

Package Structure
-----------------

.. code-block:: text

   PanelEnsembleXGBoost/
   ├── panel_ensemble_xgboost.py   # Core model
   ├── family_strategies.py        # Distribution families
   ├── family_strategies_simplified.py
   ├── shap_analyzer.py            # SHAP interpretability
   ├── did_analysis.py             # DID causal inference
   ├── differentiation_analysis.py # Innovation analysis
   ├── utils.py                    # Utility functions
   └── __init__.py                 # Package entry point

.. toctree::
   :maxdepth: 2
   
   PanelEnsembleXGBoost
   SHAPAnalyzer
   DIDAnalyzer
   family_strategies
   utils
