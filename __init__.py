"""
=============================================================================
PanelEnsembleXGBoost: Modular Panel Data Ensemble Learning Framework
=============================================================================

An academic-grade Python toolkit for complex panel data modeling, supporting:
- Three-stage ensemble modeling (Random Effects + GAM Smoothing + XGBoost Interactions)
- Independent SHAP interpretability analysis
- Difference-in-Differences (DID) causal inference
- Theoretical depth enhancement (small-sample asymptotic properties, hypothesis testing, uncertainty quantification)
- Empirical validation (benchmark comparison, Chinese city panel data, comprehensive evaluation framework)
- Innovation analysis (differentiated contributions, methodology comparison)

Author: wuxibo@GZASS
Version: 2.1.0 
Date: February 2026
=============================================================================
"""

__version__ = "2.1.0"
__author__ = "wuxibo@GZASS"

# Core models
from panel_ensemble_xgboost import PanelEnsembleXGBoost, TheoreticalPanelEnsembleXGBoost, PanelDiagnostics, StagewiseUncertaintyQuantification


# Empirical validation module
from benchmark_comparison import ChineseCityPanelLoader, BenchmarkComparator, ModelEvaluationFramework

# Independent analysis modules
from shap_analyzer import SHAPAnalyzer

from did_analysis import (
    DIDAnalyzer,
    estimate_att,
    bootstrap_att,
    test_parallel_trends,
    run_placebo_test,
    analyze_heterogeneity
)
# Innovation analysis module
from differentiation_analysis import (
    InnovationDifferentiator,
    MethodologyTaxonomy,
    StageContributionAnalyzer,
    MethodologyCategory,
    MethodCharacteristics
)

# Distribution family strategies
from family_strategies_simplified import (
    get_family_strategy,
    FamilyStrategy,
    GaussianStrategy,
    BinomialStrategy,
    PoissonStrategy,
    GammaStrategy,
    MultinomialStrategy
)

# Utility functions
from utils import get_bspline_design_matrix, check_pygam_available

# Check dependencies
PYGAM_AVAILABLE = check_pygam_available()

__all__ = [
    # Core models
    'PanelEnsembleXGBoost',
    'TheoreticalPanelEnsembleXGBoost',
    'PanelDiagnostics',
    'StagewiseUncertaintyQuantification',
    # Family strategies
    'get_family_strategy',
    'FamilyStrategy',
    'GaussianStrategy',
    'BinomialStrategy',
    'PoissonStrategy',
    'GammaStrategy',
    'MultinomialStrategy',
    # Empirical validation module
    'ChineseCityPanelLoader',
    'BenchmarkComparator',
    'ModelEvaluationFramework',
    # Independent analysis modules
    'SHAPAnalyzer',
    'DIDAnalyzer',
    # Innovation analysis module
    'InnovationDifferentiator',
    'MethodologyTaxonomy',
    'StageContributionAnalyzer',
    'MethodologyCategory',
    'MethodCharacteristics',
    # Functional DID analysis
    'estimate_att',
    'bootstrap_att',
    'test_parallel_trends', 
    'run_placebo_test',
    'analyze_heterogeneity',
    # Utilities
    'get_bspline_design_matrix',
    'check_pygam_available',
    'PYGAM_AVAILABLE',
]
