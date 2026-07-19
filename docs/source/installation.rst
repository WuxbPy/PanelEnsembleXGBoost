Installation
============

PanelEnsembleXGBoost requires Python 3.8 or later. The recommended way to install
is via pip:

.. code-block:: bash

   pip install panel-ensemble-xgboost

Alternatively, you can install directly from the source repository:

.. code-block:: bash

   git clone https://github.com/WuxbPy/PanelEnsembleXGBoost.git
   cd PanelEnsembleXGBoost
   pip install -r requirements.txt

Core Dependencies
-----------------

The following core dependencies are required:

.. list-table::
   :header-rows: 1
   :widths: 20 15 60

   * - Package
     - Minimum Version
     - Purpose
   * - ``numpy``
     - >=1.19.0
     - Numerical computation
   * - ``pandas``
     - >=1.3.0
     - Data manipulation
   * - ``scikit-learn``
     - >=1.0.0
     - ML utilities and metrics
   * - ``statsmodels``
     - >=0.13.0
     - Statistical models and tests
   * - ``xgboost``
     - >=1.6.0
     - Gradient boosting engine
   * - ``matplotlib``
     - >=3.3.0
     - Plotting and visualization
   * - ``scipy``
     - >=1.5.0
     - Scientific computing
   * - ``shap``
     - >=0.41.0
     - Model interpretability

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 55

   * - Package
     - Install Command
     - Purpose
   * - ``pygam``
     - ``pip install pygam``
     - GAM smoothing (Stage 2, enhanced mode)
   * - ``linearmodels``
     - ``pip install linearmodels``
     - Traditional panel model benchmarks
   * - ``gpboost``
     - ``pip install gpboost``
     - GPBoost comparison benchmarks

For documentation build dependencies, see :ref:`docs/requirements.txt <docs/requirements.txt>`.
