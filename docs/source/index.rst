.. PanelEnsembleXGBoost documentation master file
   ============================================================================

Welcome to PanelEnsembleXGBoost's documentation!
=================================================

.. rst-class:: lead

   **PanelEnsembleXGBoost** is an academic-grade Python package for complex panel data modeling,
   featuring a three-stage ensemble architecture that combines Random/Mixed Effects,
   GAM Smooth Effects, and XGBoost Nonlinear Interaction Effects.

.. raw:: html

   <div class="architecture-diagram">
   <strong>Three-Stage Ensemble Architecture</strong><br><br>
   <span style="color:#2980b9;">Stage 1:</span> Random/Mixed Effects (LMM/GLMM) &rarr;
   <span style="color:#27ae60;">Stage 2:</span> GAM Smooth Effects &rarr;
   <span style="color:#c0392b;">Stage 3:</span> XGBoost Interaction Effects<br>
   <span style="font-size:0.85em; color:#7f8c8d;">
   [Linear Predictor] &rarr; [Nonlinear Residuals] &rarr; [Complex Interactions]
   </span>
   </div>

.. grid:: 2

   .. grid-item-card:: 🏗️ Three-Stage Architecture
      :link: user_guide/model_architecture
      :link-type: ref

      Stage 1: Random/Mixed Effects (LMM/GLMM) + 
      Stage 2: GAM Smooth Effects (pyGAM/B-spline) + 
      Stage 3: XGBoost Interaction Effects

   .. grid-item-card:: 📊 Multi-Family Support
      :link: user_guide/distribution_families
      :link-type: ref

      Gaussian (regression), Binomial (binary classification),
      Multinomial (multiclass classification) through unified strategy pattern

   .. grid-item-card:: 🔍 SHAP Interpretability
      :link: api/SHAPAnalyzer
      :link-type: ref

      Model-agnostic SHAP analysis with summary, waterfall, force, and
      partial dependence plots

   .. grid-item-card:: ⚖️ DID Causal Inference
      :link: user_guide/did_analysis
      :link-type: ref

      Difference-in-Differences estimator with parallel-trends testing,
      placebo tests, and heterogeneity analysis

   .. grid-item-card:: 🧪 Empirical Validation
      :link: user_guide/benchmarks
      :link-type: ref

      Benchmark against GPBoost, MixedLM, Fixed Effects, Random Effects,
      and Pooled OLS on classic panel datasets

   .. grid-item-card:: 🛠️ Modular Design
      :link: api/modules
      :link-type: ref

      Clean separation of core modeling, interpretability, causal analysis,
      and utility functions

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/model_architecture
   user_guide/distribution_families
   user_guide/input_output_spec
   user_guide/random_effects
   user_guide/smooth_effects
   user_guide/xgboost_effects
   user_guide/did_analysis
   user_guide/benchmarks
   user_guide/theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/modules
   api/PanelEnsembleXGBoost
   api/SHAPAnalyzer
   api/DIDAnalyzer
   api/family_strategies
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/classic_benchmark
   examples/shap_analysis
   examples/did_analysis

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
