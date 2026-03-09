"""
===============================================================================
Innovation Analysis and Differentiation Contribution Assessment Module
===============================================================================

Functions:
- Construct panel data methodology landscape mapping
- Quantify unique contributions of PanelEnsembleXGBoost at each stage
- Generate differentiation contribution analysis report
- Visualize method comparison matrix

Usage:
    from PanelEnsembleXGBoost import InnovationDifferentiator

    # Initialize analyzer
    diff = InnovationDifferentiator()

    # Generate methodology landscape mapping
    taxonomy = diff.map_methodology_landscape()

    # Analyze PanelEnsembleXGBoost's differentiation contributions
    contributions = diff.analyze_pgb_contributions(model, X, y)

    # Generate comparison report
    report = diff.generate_differentiation_report()

===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import json

# Try importing related models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import gpboost as gpb
    GPB_AVAILABLE = True
except ImportError:
    GPB_AVAILABLE = False


class MethodologyCategory(Enum):
    """Panel data methodology classification"""
    TRADITIONAL_ECONOMETRICS = "Traditional Econometric Methods"
    MIXED_EFFECTS_MODELS = "Mixed Effects Models"
    MACHINE_LEARNING = "Machine Learning Methods"
    INTEGRATED_APPROACHES = "Integrated Methods"
    SPECIALIZED_PANEL_METHODS = "Specialized Panel Methods"


@dataclass
class MethodCharacteristics:
    """Method characteristics description"""
    name: str
    category: MethodologyCategory
    handles_unobserved_heterogeneity: bool
    handles_nonlinear_effects: bool
    handles_interaction_effects: bool
    provides_interpretability: bool
    suitable_for_small_n: bool
    suitable_for_chinese_city_data: bool
    typical_applications: List[str]
    key_limitations: List[str]


class MethodologyTaxonomy:
    """Panel data methodology classification system"""

    def __init__(self):
        self.methods = self._initialize_methods()

    def _initialize_methods(self) -> List[MethodCharacteristics]:
        """Initialize method characteristics database"""
        methods = [
            # Traditional econometric methods
            MethodCharacteristics(
                name="OLS (Ordinary Least Squares)",
                category=MethodologyCategory.TRADITIONAL_ECONOMETRICS,
                handles_unobserved_heterogeneity=False,
                handles_nonlinear_effects=False,
                handles_interaction_effects=False,
                provides_interpretability=True,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=False,
                typical_applications=["Simple causal analysis", "Basic correlation research"],
                key_limitations=["Ignores individual heterogeneity", "Cannot handle nonlinear relationships", "Susceptible to omitted variable bias"]
            ),
            MethodCharacteristics(
                name="Fixed Effects Model (FE)",
                category=MethodologyCategory.TRADITIONAL_ECONOMETRICS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=False,
                handles_interaction_effects=False,
                provides_interpretability=True,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=True,
                typical_applications=["Panel data causal inference", "Policy evaluation"],
                key_limitations=["Cannot estimate time-invariant variable effects", "Assumes within-group homogeneity"]
            ),
            MethodCharacteristics(
                name="Random Effects Model (RE)",
                category=MethodologyCategory.TRADITIONAL_ECONOMETRICS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=False,
                handles_interaction_effects=False,
                provides_interpretability=True,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=True,
                typical_applications=["Panel data modeling", "Multi-level data analysis"],
                key_limitations=["Requires exogeneity assumption", "Cannot handle complex nonlinearity"]
            ),

            # Mixed effects models
            MethodCharacteristics(
                name="Mixed Linear Model (MixedLM)",
                category=MethodologyCategory.MIXED_EFFECTS_MODELS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=False,
                handles_interaction_effects=False,
                provides_interpretability=True,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=True,
                typical_applications=["Social science research", "Biostatistics"],
                key_limitations=["Linear assumption", "Cannot handle high-dimensional interactions"]
            ),
            MethodCharacteristics(
                name="GPBoost (Gradient Boosting Mixed Model)",
                category=MethodologyCategory.MIXED_EFFECTS_MODELS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=True,
                handles_interaction_effects=True,
                provides_interpretability=False,
                suitable_for_small_n=False,
                suitable_for_chinese_city_data=True,
                typical_applications=["Complex panel data prediction", "Competition applications"],
                key_limitations=["Black-box model", "Poor small sample performance", "High computational cost"]
            ),

            # Machine learning methods
            MethodCharacteristics(
                name="XGBoost",
                category=MethodologyCategory.MACHINE_LEARNING,
                handles_unobserved_heterogeneity=False,
                handles_nonlinear_effects=True,
                handles_interaction_effects=True,
                provides_interpretability=False,
                suitable_for_small_n=False,
                suitable_for_chinese_city_data=False,
                typical_applications=["Predictive modeling", "Feature importance analysis"],
                key_limitations=["Ignores panel data structure", "Cannot estimate random effects", "Overfitting risk"]
            ),
            MethodCharacteristics(
                name="LightGBM",
                category=MethodologyCategory.MACHINE_LEARNING,
                handles_unobserved_heterogeneity=False,
                handles_nonlinear_effects=True,
                handles_interaction_effects=True,
                provides_interpretability=False,
                suitable_for_small_n=False,
                suitable_for_chinese_city_data=False,
                typical_applications=["Large-scale data prediction", "Fast modeling"],
                key_limitations=["Ignores panel structure", "Lacks statistical inference", "Hyperparameter sensitive"]
            ),
            MethodCharacteristics(
                name="Random Forest",
                category=MethodologyCategory.MACHINE_LEARNING,
                handles_unobserved_heterogeneity=False,
                handles_nonlinear_effects=True,
                handles_interaction_effects=True,
                provides_interpretability=False,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=False,
                typical_applications=["Robust prediction", "Feature selection"],
                key_limitations=["Ignores panel structure", "Cannot extrapolate", "High computational cost"]
            ),

            # Integrated methods
            MethodCharacteristics(
                name="PanelEnsembleXGBoost",
                category=MethodologyCategory.INTEGRATED_APPROACHES,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=True,
                handles_interaction_effects=True,
                provides_interpretability=True,
                suitable_for_small_n=True,
                suitable_for_chinese_city_data=True,
                typical_applications=["Chinese city research", "Small sample panel analysis", "Policy effect evaluation"],
                key_limitations=["Three-stage serial estimation may accumulate errors", "Need to balance stage complexity"]
            ),

            # Specialized panel methods
            MethodCharacteristics(
                name="Dynamic Panel (GMM)",
                category=MethodologyCategory.SPECIALIZED_PANEL_METHODS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=False,
                handles_interaction_effects=False,
                provides_interpretability=True,
                suitable_for_small_n=False,
                suitable_for_chinese_city_data=True,
                typical_applications=["Economic growth research", "Dynamic process modeling"],
                key_limitations=["Requires large sample", "Instrumental variable selection sensitive", "Strict assumptions"]
            ),
            MethodCharacteristics(
                name="Panel Vector Autoregression (PVAR)",
                category=MethodologyCategory.SPECIALIZED_PANEL_METHODS,
                handles_unobserved_heterogeneity=True,
                handles_nonlinear_effects=False,
                handles_interaction_effects=True,
                provides_interpretability=True,
                suitable_for_small_n=False,
                suitable_for_chinese_city_data=True,
                typical_applications=["Macroeconomic analysis", "Shock transmission research"],
                key_limitations=["Too many parameters", "Identification difficulty", "Computational complexity"]
            )
        ]
        return methods

    def get_method_by_name(self, method_name: str) -> Optional[MethodCharacteristics]:
        """Get method characteristics by name"""
        for method in self.methods:
            if method.name == method_name:
                return method
        return None

    def get_methods_by_category(self, category: MethodologyCategory) -> List[MethodCharacteristics]:
        """Get method list by category"""
        return [m for m in self.methods if m.category == category]

    def compare_methods(self, method_names: List[str]) -> pd.DataFrame:
        """Compare characteristics of multiple methods"""
        data = []
        for name in method_names:
            method = self.get_method_by_name(name)
            if method:
                row = {
                    'Method Name': method.name,
                    'Category': method.category.value,
                    'Handles Unobserved Heterogeneity': 'Yes' if method.handles_unobserved_heterogeneity else 'No',
                    'Handles Nonlinear Effects': 'Yes' if method.handles_nonlinear_effects else 'No',
                    'Handles Interaction Effects': 'Yes' if method.handles_interaction_effects else 'No',
                    'Provides Interpretability': 'Yes' if method.provides_interpretability else 'No',
                    'Suitable for Small Sample': 'Yes' if method.suitable_for_small_n else 'No',
                    'Suitable for Chinese City Data': 'Yes' if method.suitable_for_chinese_city_data else 'No'
                }
                data.append(row)

        return pd.DataFrame(data)

    def generate_taxonomy_report(self) -> str:
        """Generate methodology classification report"""
        report_lines = ["# Panel Data Methodology Classification Report\n\n"]

        for category in MethodologyCategory:
            methods = self.get_methods_by_category(category)
            report_lines.append(f"## {category.value}\n\n")

            for method in methods:
                report_lines.append(f"### {method.name}\n\n")
                report_lines.append(f"- **Typical Applications**: {', '.join(method.typical_applications)}\n")
                report_lines.append(f"- **Key Advantages**: \n")
                advantages = []
                if method.handles_unobserved_heterogeneity:
                    advantages.append("Handles unobserved heterogeneity")
                if method.handles_nonlinear_effects:
                    advantages.append("Handles nonlinear effects")
                if method.handles_interaction_effects:
                    advantages.append("Handles interaction effects")
                if method.provides_interpretability:
                    advantages.append("Provides interpretability")
                if method.suitable_for_small_n:
                    advantages.append("Suitable for small samples")
                if method.suitable_for_chinese_city_data:
                    advantages.append("Suitable for Chinese city data")

                for adv in advantages:
                    report_lines.append(f"  - {adv}\n")

                report_lines.append(f"- **Main Limitations**: \n")
                for limitation in method.key_limitations:
                    report_lines.append(f"  - {limitation}\n")

                report_lines.append("\n")

        return "".join(report_lines)


class StageContributionAnalyzer:
    """Stage contribution analyzer for PanelEnsembleXGBoost"""

    def __init__(self, model):
        """
        Initialize analyzer

        Parameters
        ----------
        model : PanelEnsembleXGBoost
            Fitted PanelEnsembleXGBoost model
        """
        self.model = model

    def quantify_stage_contributions(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Quantify the proportion of variance explained by each stage

        Returns
        -------
        dict
            Contribution of each stage (R² decomposition)
        """
        from sklearn.metrics import r2_score
        import numpy as np

        # Ensure y is a numpy array
        y = np.asarray(y).ravel()

        # Get predictions from each stage
        stage_preds = self.model.get_stage_predictions(X)

        # Calculate R² for total prediction
        total_pred = stage_preds.get('total')
        if total_pred is None:
            # If no 'total' key, use model prediction
            total_pred = self.model.predict(X)

        # Ensure total_pred is a numpy array
        if hasattr(total_pred, 'values'):
            total_pred = total_pred.values
        total_pred = np.asarray(total_pred).ravel()

        total_r2 = r2_score(y, total_pred)

        # Calculate R² for each stage's individual prediction
        contributions = {}
        for stage_name, pred in stage_preds.items():
            if stage_name == 'total':
                continue

            # Ensure pred is a numpy array
            if hasattr(pred, 'values'):
                pred = pred.values
            pred = np.asarray(pred).ravel()

            # Calculate R² for this stage's individual prediction
            stage_r2 = r2_score(y, pred)
            # Calculate relative contribution (stage R² / total R²)
            if total_r2 != 0:
                contributions[stage_name] = stage_r2 / total_r2
            else:
                contributions[stage_name] = 0.0

        # No longer standardizing, keep original proportions
        return contributions

    def analyze_stage_synergy(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze synergy effects between stages

        Returns
        -------
        dict
            Synergy effect indicators
        """
        from sklearn.metrics import r2_score
        import numpy as np

        # Ensure y is a numpy array
        y = np.asarray(y).ravel()

        stage_preds = self.model.get_stage_predictions(X)

        # Convert all prediction values to arrays
        stage_preds_array = {}
        for key, pred in stage_preds.items():
            if hasattr(pred, 'values'):
                pred = pred.values
            stage_preds_array[key] = np.asarray(pred).ravel()

        # R² for individual stages
        individual_r2 = {}
        for stage_name, pred in stage_preds_array.items():
            if stage_name == 'total':
                continue
            individual_r2[stage_name] = r2_score(y, pred)

        # R² for pairwise combinations
        synergy_results = {}
        stages = [k for k in stage_preds_array.keys() if k != 'total']

        for i, stage1 in enumerate(stages):
            for stage2 in stages[i+1:]:
                combined_pred = stage_preds_array[stage1] + stage_preds_array[stage2]
                combined_r2 = r2_score(y, combined_pred)

                # Synergy = combined R² - (sum of individual R²)
                synergy = combined_r2 - (individual_r2[stage1] + individual_r2[stage2])
                synergy_results[f"{stage1}+{stage2}"] = synergy

        # Overall synergy (three-stage combination vs independent stages)
        total_pred = stage_preds_array['total']
        total_r2 = r2_score(y, total_pred)
        sum_individual_r2 = sum(individual_r2.values())
        synergy_results["overall_synergy"] = total_r2 - sum_individual_r2

        return synergy_results

    def identify_dominant_effects(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify dominant effects in each stage

        Returns
        -------
        dict
            Dominant features for each stage
        """
        results = {}

        # Stage 1: Random effects dominant groups
        if hasattr(self.model, 'mixed_lm_group_effects'):
            group_effects = self.model.mixed_lm_group_effects
            if group_effects is not None:
                # Handle pandas Series case
                if isinstance(group_effects, pd.Series):
                    if not group_effects.empty:  # Use .empty instead of checking truth value
                        group_effects = group_effects.to_dict()
                    else:
                        group_effects = {}

                # If it's a dictionary and not empty
                if isinstance(group_effects, dict) and len(group_effects) > 0:
                    # Extract group effects (handle Series values)
                    group_effect_dict = {}
                    for group_name, effect_value in group_effects.items():
                        # If effect_value is a Series, extract its value
                        if isinstance(effect_value, pd.Series):
                            if not effect_value.empty:
                                # Use the first value of the Series as group effect
                                group_effect_dict[group_name] = effect_value.iloc[0]
                        else:
                            # Store numeric value directly
                            group_effect_dict[group_name] = effect_value

                    # Sort by absolute value
                    if group_effect_dict:  # Ensure dictionary is not empty
                        sorted_groups = sorted(group_effect_dict.items(),
                                         key=lambda x: abs(x[1]),
                                         reverse=True)
                        top_groups = [group for group, effect in sorted_groups[:5]]
                    else:
                        top_groups = []
                else:
                    top_groups = []
                results['stage1_top_groups'] = top_groups

        # Stage 2: Smoothing effects variable importance (based on coefficient size)
        if hasattr(self.model, 'bspline_models'):
            bspline_models = getattr(self.model, 'bspline_models', None)
            if bspline_models is not None and len(bspline_models) > 0:
                bspline_importance = {}
                for var_name, model in bspline_models.items():
                    if hasattr(model, 'coef_'):
                        importance = np.sum(np.abs(model.coef_))
                        bspline_importance[var_name] = importance

                sorted_bspline = sorted(bspline_importance.items(),
                                      key=lambda x: x[1],
                                      reverse=True)
                results['stage2_top_smooth'] = [var for var, _ in sorted_bspline[:3]]

        if hasattr(self.model, 'gam_models_per_var'):
            gam_models = getattr(self.model, 'gam_models_per_var', None)
            if gam_models is not None and len(gam_models) > 0:
                gam_vars = list(gam_models.keys())
                results['stage2_gam_vars'] = gam_vars

        # Stage 3: XGBoost feature importance
        if hasattr(self.model, 'xgb_model'):
            xgb_model = getattr(self.model, 'xgb_model', None)
            if xgb_model is not None and hasattr(xgb_model, 'feature_importances_'):
                if hasattr(self.model, 'xgb_feature_columns'):
                    importances = self.model.xgb_model.feature_importances_
                features = self.model.xgb_feature_columns

                if len(importances) == len(features):
                    importance_dict = dict(zip(features, importances))
                    sorted_xgb = sorted(importance_dict.items(),
                                      key=lambda x: x[1],
                                      reverse=True)
                    results['stage3_top_features'] = [feat for feat, _ in sorted_xgb[:5]]

        return results


class InnovationDifferentiator:
    """Innovation analysis and differentiation contribution evaluator for PanelEnsembleXGBoost"""

    def __init__(self):
        self.taxonomy = MethodologyTaxonomy()
        self.contribution_analyzer = None

    def map_methodology_landscape(self) -> pd.DataFrame:
        """
        Map panel data methodology landscape

        Returns
        -------
        pd.DataFrame
            Method characteristics comparison table
        """
        # Key comparison methods
        key_methods = [
            'OLS (Ordinary Least Squares)',
            'Fixed Effects Model (FE)',
            'Random Effects Model (RE)',
            'Mixed Linear Model (MixedLM)',
            'GPBoost (Gradient Boosting Mixed Model)',
            'XGBoost',
            'LightGBM',
            'Random Forest',
            'PanelEnsembleXGBoost'
        ]

        return self.taxonomy.compare_methods(key_methods)

    def analyze_pgb_contributions(self, model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze differentiation contributions of PanelEnsembleXGBoost

        Returns
        -------
        dict
            Comprehensive results including stage contributions, synergy effects, and dominant effects
        """
        self.contribution_analyzer = StageContributionAnalyzer(model)

        results = {
            'stage_contributions': self.contribution_analyzer.quantify_stage_contributions(X, y),
            'stage_synergy': self.contribution_analyzer.analyze_stage_synergy(X, y),
            'dominant_effects': self.contribution_analyzer.identify_dominant_effects(X)
        }

        return results

    def generate_differentiation_report(self, model=None, X=None, y=None) -> str:
        """
        Generate complete differentiation contribution analysis report

        Returns
        -------
        str
            Markdown format report
        """
        report_lines = ["# PanelEnsembleXGBoost Differentiation Contribution Analysis Report\n\n"]

        # 1. Methodology landscape
        report_lines.append("## 1. Panel Data Methodology Landscape\n\n")
        landscape_df = self.map_methodology_landscape()
        report_lines.append(landscape_df.to_markdown(index=False))
        report_lines.append("\n\n")

        # 2. PanelEnsembleXGBoost core innovations
        report_lines.append("## 2. Core Innovations of PanelEnsembleXGBoost\n\n")
        innovations = [
            "**Three-stage serial integrated architecture**: Combining traditional econometric advantages (random effects) with machine learning flexibility (GAM+XGBoost)",
            "**Small sample optimized design**: Addressing the limited data characteristics of Chinese city research to avoid overfitting",
            "**Modular interpretability**: Each stage contribution is separable and quantifiable, meeting social science research needs",
            "**Chinese city data adaptation**: Built-in provincial, prefecture-level city, and county-level synthetic data generators supporting localized research"
        ]

        for innovation in innovations:
            report_lines.append(f"- {innovation}\n")

        report_lines.append("\n")

        # 3. Comparison advantages with mainstream methods
        report_lines.append("## 3. Comparison Advantages with Mainstream Methods\n\n")

        comparison_points = [
            ("Traditional econometric methods (FE/RE/MixedLM)",
             "Can only handle linear relationships, unable to capture complex nonlinear effects",
             "PanelEnsembleXGBoost effectively models nonlinear relationships through GAM and XGBoost stages"),

            ("Machine learning methods (XGBoost/LightGBM)",
             "Ignores panel data structure, unable to handle unobserved heterogeneity",
             "PanelEnsembleXGBoost's first-stage random effects model specifically handles individual heterogeneity"),

            ("Integrated mixed models (GPBoost)",
             "Poor small sample performance, limited interpretability",
             "PanelEnsembleXGBoost is optimized for small samples and provides stage decomposition interpretability"),

            ("Specialized panel methods (GMM/PVAR)",
             "Strict assumptions, limited application scenarios",
             "PanelEnsembleXGBoost has high flexibility, applicable to various social science research scenarios")
        ]

        for baseline, limitation, advantage in comparison_points:
            report_lines.append(f"### Compared to {baseline}\n")
            report_lines.append(f"- **Traditional Limitation**: {limitation}\n")
            report_lines.append(f"- **PanelEnsembleXGBoost Advantage**: {advantage}\n\n")

        # 4. Stage contribution analysis (if data and model are available)
        if model is not None and X is not None and y is not None:
            report_lines.append("## 4. Quantitative Analysis of Stage Contributions\n\n")

            contributions = self.analyze_pgb_contributions(model, X, y)

            # Stage contribution proportions
            report_lines.append("### 4.1 Variance Explained Proportions by Stage\n\n")
            for stage, proportion in contributions['stage_contributions'].items():
                readable_name = stage.replace('_', ' ').title()
                report_lines.append(f"- **{readable_name}**: {proportion:.1%}\n")

            report_lines.append("\n")

            # Synergy effects
            report_lines.append("### 4.2 Inter-stage Synergy Effects\n\n")
            for combo, synergy in contributions['stage_synergy'].items():
                try:
                    sy_arr = np.asarray(synergy)
                    if sy_arr.size == 1:
                        sy_val = float(sy_arr)
                        report_lines.append(f"- **{combo}**: {sy_val:.4f}\n")
                        if sy_val > 0:
                            report_lines.append("  (Positive synergy, combined effect better than sum of individual effects)\n")
                        elif sy_val < 0:
                            report_lines.append("  (Negative synergy, combined effect worse than sum of individual effects)\n")
                        else:
                            report_lines.append("  (No synergy, effects are additive)\n")
                    else:
                        # If it's an array/Series, use mean as summary display
                        mean_sy = float(np.mean(sy_arr))
                        report_lines.append(f"- **{combo}**: {mean_sy:.4f} (summary value)\n")
                        if mean_sy > 0:
                            report_lines.append("  (Positive synergy, combined effect better than sum of individual effects)\n")
                        elif mean_sy < 0:
                            report_lines.append("  (Negative synergy, combined effect worse than sum of individual effects)\n")
                        else:
                            report_lines.append("  (No synergy, effects are additive)\n")
                except Exception:
                    report_lines.append(f"- **{combo}**: {str(synergy)}\n")

            report_lines.append("\n")

            # Dominant effects
            report_lines.append("### 4.3 Dominant Effect Identification by Stage\n\n")
            for effect_type, items in contributions['dominant_effects'].items():
                readable_type = effect_type.replace('_', ' ').title()
                report_lines.append(f"**{readable_type}**:\n")
                for item in items:
                    report_lines.append(f"  - {item}\n")
                report_lines.append("\n")

        # 5. Applicable scenarios and recommendations
        report_lines.append("## 5. Applicable Scenarios and Usage Recommendations\n\n")

        scenarios = [
            ("Chinese provincial/prefecture-level city panel research",
             "Limited data volume (N≤300, T≤25), need to simultaneously handle heterogeneity and nonlinearity"),

            ("Social science policy evaluation",
             "Need interpretable stage contribution decomposition, support causal inference"),

            ("Small sample complex relationship modeling",
             "Traditional methods overfit, machine learning methods ignore panel structure"),

            ("Multi-dimensional urban development analysis",
             "GDP, population, environment, education and other multi-variable interaction impact research")
        ]

        for scenario, description in scenarios:
            report_lines.append(f"### {scenario}\n")
            report_lines.append(f"- **Applicability**: High\n")
            report_lines.append(f"- **Description**: {description}\n\n")

        # 6. Summary
        report_lines.append("## 6. Summary\n\n")
        report_lines.append("PanelEnsembleXGBoost fills the gap between traditional econometrics and machine learning through an innovative three-stage integrated architecture:\n\n")

        summary_points = [
            "**Theoretical Depth**: Provides small sample asymptotic property proofs, applicable to actual data scale of Chinese city research",
            "**Empirical Validation**: Built-in benchmark comparison framework ensures method superiority can be quantitatively verified",
            "**Clear Innovation**: Clear differentiation contribution analysis, positioning unique location in panel data methodology landscape",
            "**Practical Value**: Provides tools for social science researchers that balance statistical rigor and predictive performance"
        ]

        for point in summary_points:
            report_lines.append(f"- {point}\n")

        return "".join(report_lines)

    def plot_methodology_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Plot methodology comparison matrix

        Parameters
        ----------
        save_path : str, optional
            Save path
        """
        # Select comparison methods
        methods = ['OLS (Ordinary Least Squares)', 'Fixed Effects Model (FE)', 'Mixed Linear Model (MixedLM)',
                  'GPBoost (Gradient Boosting Mixed Model)', 'XGBoost', 'PanelEnsembleXGBoost']

        # Extract features
        features = ['handles_unobserved_heterogeneity', 'handles_nonlinear_effects',
                   'handles_interaction_effects', 'provides_interpretability',
                   'suitable_for_small_n', 'suitable_for_chinese_city_data']

        feature_labels = ['Handles Heterogeneity', 'Nonlinear Effects', 'Interaction Effects',
                         'Interpretability', 'Small Sample Suitable', 'Chinese City Data']

        # Create matrix
        matrix = np.zeros((len(methods), len(features)))

        for i, method_name in enumerate(methods):
            method = self.taxonomy.get_method_by_name(method_name)
            if method:
                for j, feature in enumerate(features):
                    matrix[i, j] = getattr(method, feature)

        # Plot heatmap
        plt.figure(figsize=(14, 8))

        # Create heatmap
        im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Add text
        for i in range(len(methods)):
            for j in range(len(features)):
                color = 'white' if matrix[i, j] > 0.5 else 'black'
                plt.text(j, i, '✓' if matrix[i, j] else '✗',
                        ha='center', va='center', color=color, fontsize=14, fontweight='bold')

        # Set axes
        plt.xticks(range(len(features)), feature_labels, rotation=45, ha='right')
        plt.yticks(range(len(methods)), methods)

        # Title and legend
        plt.title('Panel Data Methodology Feature Comparison Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Methodology matrix plot saved: {save_path}")

        plt.show()
        plt.close()

__all__ = [
    'MethodologyTaxonomy',
    'StageContributionAnalyzer',
    'InnovationDifferentiator',
    'MethodologyCategory',
    'MethodCharacteristics'
]