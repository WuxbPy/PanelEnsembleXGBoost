# PanelEnsembleXGBoost: Modular Panel Data Ensemble Learning Framework

**PanelEnsembleXGBoost** is an academic-grade Python package for complex panel data modeling, featuring a three-stage ensemble architecture that combines Random/Mixed Effects, GAM Smooth Effects, and XGBoost Nonlinear Interaction Effects. The framework supports multiple distribution families (Gaussian, Binomial, Multinomial) and includes independent modules for SHAP interpretability analysis and Difference-in-Differences (DID) causal inference.

-   **Documentation**: [https://github.com/WuxbPy/PanelEnsembleXGBoost](https://github.com/WuxbPy/PanelEnsembleXGBoost)
-   **Free software**: [MIT License](https://opensource.org/licenses/MIT)

## Introduction

Panel data, also known as longitudinal or cross-sectional time-series data, is ubiquitous in economics, social sciences, and public health research. Traditional panel models (e.g., fixed effects, random effects) often struggle to capture complex nonlinear relationships and interaction effects, while modern machine learning methods may ignore the inherent panel structure, leading to biased inferences.

**PanelEnsembleXGBoost** addresses these limitations by introducing a three-stage ensemble learning framework specifically designed for panel data:

1.  **Stage 1: Random/Mixed Effects** – Captures unobserved heterogeneity across groups using Linear Mixed Models (LMM) or Generalized Linear Mixed Models (GLMM).
2.  **Stage 2: GAM Smooth Effects** – Models smooth nonlinear effects of continuous covariates via Generalized Additive Models (GAM) or B‑splines.
3.  **Stage 3: XGBoost Nonlinear Interaction Effects** – Learns complex interaction patterns and high-order nonlinearities using gradient‑boosted trees, with an adaptive objective function that respects the chosen distribution family.

The package employs a **strategy pattern** to support multiple distribution families (`gaussian` for regression, `binomial` for binary classification, `multinomial` for multiclass classification), making it suitable for a wide range of panel‑data tasks. Additionally, the framework is enhanced with theoretical depth (small‑sample asymptotic properties, hypothesis testing, uncertainty quantification) and comes with comprehensive empirical validation tools.

## Features

-   🏗️ **Three‑Stage Ensemble Architecture** – Seamlessly integrates Random/Mixed Effects, GAM Smoothing, and XGBoost Interaction Effects.
-   📊 **Multi‑Distribution Family Support** – Handles Gaussian (regression), Binomial (binary classification), and Multinomial (multiclass classification) tasks through a unified strategy pattern.
-   🔍 **Independent SHAP Interpretability** – `SHAPAnalyzer` module provides model‑agnostic explanation tools for any prediction function.
-   ⚖️ **Difference‑in‑Differences (DID) Causal Inference** – A standalone `DIDAnalyzer` module with ATT estimation, parallel‑trends testing, placebo tests, and heterogeneity analysis.
-   🧪 **Empirical Validation Suite** – Benchmark comparisons against classical panel models (GPBoost, linear panel models) and comprehensive evaluation metrics.
-   🛠️ **Modular & Extensible Design** – Clean separation of core modeling, interpretability, causal analysis, and utility functions.

### Dependencies

Core dependencies are listed in `requirements.txt`:

```
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
matplotlib>=3.3.0
scipy>=1.5.0
xgboost>=1.6.0
lightgbm>=3.3.0
shap>=0.41.0
```

Optional dependencies for extended functionality include `pygam` (for GAM smoothing), `gpboost`, and `linearmodels` (for benchmark comparisons).

## Quick Start

### Basic Regression Task

```python
from PanelEnsembleXGBoost import PanelEnsembleXGBoost
import numpy as np
import pandas as pd

# Simulate panel data
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
groups = X['group'].values

# Initialize model (default Gaussian family for regression)
model = PanelEnsembleXGBoost(
    random_effects_groups=['group'],
    smooth_terms={'time': 'gam', 'x1': 'bspline'}
)

# Fit
model.fit(X, y, groups_for_random_effects=groups)

# Predict
y_pred = model.predict(X)
```

### Binary Classification Task

```python
# Binary classification with Binomial family
model = PanelEnsembleXGBoost(
    family='binomial',
    random_effects_groups=['group']
)

# Generate binary labels
y_binary = (y > 0).astype(int)

model.fit(X, y_binary, groups_for_random_effects=groups)

# Predict probabilities
y_proba = model.predict_proba(X)
```

### SHAP Analysis

```python
from PanelEnsembleXGBoost import SHAPAnalyzer

# Initialize analyzer with the model's prediction function
analyzer = SHAPAnalyzer(predict_func=model.predict, feature_names=X.columns.tolist())

# Compute SHAP values
shap_values = analyzer.compute_shap(X)

# Generate summary plot
analyzer.summary_plot(shap_values, X)
```

### DID Causal Analysis

```python
from PanelEnsembleXGBoost import DIDAnalyzer, estimate_att

# Load panel data with treatment assignment
data = pd.read_csv('your_panel_data.csv')

# Estimate Average Treatment Effect on the Treated (ATT)
att_result = estimate_att(
    data,
    outcome_col='outcome',
    treatment_col='treated',
    unit_col='unit_id',
    time_col='year'
)

# Run parallel-trends test
parallel_test = test_parallel_trends(data, outcome_col='outcome', treatment_col='treated',
                                     unit_col='unit_id', time_col='year', pre_periods=3)
```



## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/WuxbPy/PanelEnsembleXGBoost).

## Acknowledgments

-   The development of this package was supported by the Guangzhou Academy of Social Sciences (GZASS).
-   Thanks to the open‑source communities of `statsmodels`, `xgboost`, `shap`, and `pygam` for their invaluable tools.