"""
=============================================================================
SHAP Interpretability Analysis Module
=============================================================================

Features:
- SHAP analysis independent of main model
- SHAP value computation for arbitrary prediction functions
- Multiple visualization methods (summary plot, waterfall plot, partial dependence plot, etc.)
- Stage contribution analysis

Usage:
    from PanelGAMBoost import SHAPAnalyzer
    
    # Create analyzer
    analyzer = SHAPAnalyzer(model.predict, X_train)
    
    # Compute SHAP values
    analyzer.compute_shap_values(X_test)
    
    # Generate visualizations
    analyzer.plot_summary()
    analyzer.plot_partial_dependence()

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from typing import Callable, Optional, Union, List, Dict, Any

from utils import check_shap_available

SHAP_AVAILABLE = check_shap_available()


class SHAPAnalyzer:
    """
    Standalone SHAP analyzer that can be used with any prediction function
    
    Characteristics:
    - Does not depend on PanelGAMBoost class structure
    - Supports arbitrary prediction functions (must accept numpy arrays and return predictions)
    - Provides complete SHAP visualization and analysis functionality
    
    Parameters
    ----------
    predict_fn : callable
        Prediction function that accepts (n_samples, n_features) array and returns (n_samples,) predictions
    X_background : pd.DataFrame or np.ndarray
        Background data for initializing SHAP explainer
    feature_names : list, optional
        List of feature names, defaults to column names of X_background
    """
    
    def __init__(self, 
                 predict_fn: Callable,
                 X_background: Union[pd.DataFrame, np.ndarray],
                 feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer
        
        Parameters
        ----------
        predict_fn : callable
            Prediction function that accepts array input and returns predictions
        X_background : pd.DataFrame or np.ndarray
            Background data for SHAP explainer initialization
        feature_names : list, optional
            List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP package is required. Install with: pip install shap"
            )
        
        self.predict_fn = predict_fn
        self.X_background = self._to_dataframe(X_background, feature_names)
        self.feature_names = feature_names or self.X_background.columns.tolist()
        
        # SHAP-related attributes
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.X_shap_processed = None
        
    def _to_dataframe(self, X, feature_names=None):
        """Convert input to DataFrame"""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            X = np.asarray(X)
            if feature_names and len(feature_names) == X.shape[1]:
                return pd.DataFrame(X, columns=feature_names)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    def compute_shap_values(self, 
                           X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                           background_samples: int = 100,
                           approximate: bool = False) -> np.ndarray:
        """
        Compute SHAP values
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, optional
            Data to explain, defaults to background data
        background_samples : int, default=100
            Number of background samples for KernelExplainer
        approximate : bool, default=False
            Whether to use approximation methods to speed up computation
            
        Returns
        -------
        np.ndarray
            SHAP values array with shape (n_samples, n_features)
        """
        import shap
        
        if X is None:
            X = self.X_background
        else:
            X = self._to_dataframe(X, self.feature_names)
        
        self.X_shap_processed = X.copy()
        
        # Check for low-variance features and warn
        feature_vars = X.var(axis=0)
        low_var_features = feature_vars[feature_vars < 1e-10]
        if len(low_var_features) > 0:
            warnings.warn(
                f"Found {len(low_var_features)} feature(s) with very low variance: "
                f"{list(low_var_features.index[:5])}... These may cause visualization issues."
            )
        
        # Prepare background data
        n_bg = min(background_samples, len(self.X_background))
        background_data = self.X_background.sample(n_bg, random_state=42)
        
        print(f"Initializing KernelExplainer with {n_bg} background samples...")
        
        # Create prediction function wrapper
        def predict_wrapper(X_input):
            """Wrap prediction function to handle input/output formatting"""
            X_df = self._to_dataframe(X_input, self.feature_names)
            pred = self.predict_fn(X_df)
            return np.asarray(pred).ravel()
        
        # Initialize explainer
        try:
            self.explainer = shap.KernelExplainer(
                predict_wrapper, 
                background_data,
                link="identity"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize KernelExplainer: {e}")
        
        # Compute SHAP values
        print(f"Computing SHAP values for {len(X)} samples...")
        
        if approximate and len(X) > 100:
            # Use sample for large datasets
            sample_idx = np.random.choice(len(X), 100, replace=False)
            X_compute = X.iloc[sample_idx]
        else:
            X_compute = X
        
        self.shap_values = self.explainer.shap_values(X_compute)
        self.expected_value = self.explainer.expected_value
        
        # If using sample, expand back to full data
        if approximate and len(X) > 100:
            self.shap_values = self.explainer.shap_values(X)
        
        print(f"SHAP values computed. Shape: {self.shap_values.shape}")
        
        # Check for features with zero SHAP variance
        shap_vars = np.var(self.shap_values, axis=0)
        zero_shap_var_count = np.sum(shap_vars < 1e-10)
        if zero_shap_var_count > 0:
            warnings.warn(
                f"{zero_shap_var_count} feature(s) have zero variance in SHAP values. "
                f"These will be excluded from KDE-based visualizations."
            )
        
        return self.shap_values
    
    def plot_summary(self,
                    plot_type: str = 'bar',
                    max_display: int = 20,
                    save_path: Optional[str] = None,
                    **kwargs) -> None:
        """
        Plot SHAP summary plot

        Parameters
        ----------
        plot_type : str, default='bar'
            Plot type: 'bar', 'beeswarm', or 'violin'
        max_display : int, default=20
            Maximum number of features to display
        save_path : str, optional
            Save path
        """
        import shap
        
        if self.shap_values is None:
            self.compute_shap_values()
        
        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background
        
        # Filter out features with zero variance in SHAP values to avoid KDE errors
        valid_mask = np.abs(self.shap_values).std(axis=0) > 1e-10
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            warnings.warn(f"Filtering {n_invalid} feature(s) with zero variance in SHAP values")
            
        shap_values_filtered = self.shap_values[:, valid_mask]
        X_filtered = X_for_plot.iloc[:, valid_mask] if hasattr(X_for_plot, 'iloc') else X_for_plot[:, valid_mask]
        feature_names_filtered = [f for f, m in zip(self.feature_names, valid_mask) if m]
        
        if len(feature_names_filtered) == 0:
            raise ValueError("All features have zero variance in SHAP values - cannot create visualization")
        
        plt.figure(figsize=(10, 8))
        
        try:
            if plot_type == 'bar':
                shap.summary_plot(
                    shap_values_filtered, 
                    X_filtered, 
                    plot_type='bar',
                    feature_names=feature_names_filtered, 
                    max_display=max_display, 
                    show=False,
                    **kwargs
                )
                title = 'SHAP Feature Importance (Bar)'
            elif plot_type == 'beeswarm':
                shap.summary_plot(
                    shap_values_filtered, 
                    X_filtered, 
                    plot_type='dot',
                    feature_names=feature_names_filtered, 
                    max_display=max_display, 
                    show=False,
                    **kwargs
                )
                title = 'SHAP Feature Impact (Beeswarm)'
            elif plot_type == 'violin':
                shap.summary_plot(
                    shap_values_filtered, 
                    X_filtered, 
                    plot_type='violin',
                    feature_names=feature_names_filtered, 
                    max_display=max_display, 
                    show=False,
                    **kwargs
                )
                title = 'SHAP Feature Impact (Violin)'
            else:
                raise ValueError("plot_type must be 'bar', 'beeswarm', or 'violin'")
        except Exception as e:
            plt.close()
            if 'violin' in plot_type.lower() or 'kde' in str(e).lower() or 'singular' in str(e).lower():
                # Fallback to dot plot if violin fails
                warnings.warn(f"Failed to create {plot_type} plot due to KDE error, falling back to dot plot")
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values_filtered, 
                    X_filtered, 
                    plot_type='dot',
                    feature_names=feature_names_filtered, 
                    max_display=max_display, 
                    show=False,
                    **kwargs
                )
                title = 'SHAP Feature Impact (Beeswarm - fallback)'
            else:
                raise
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved {plot_type} plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_waterfall(self,
                      instance_idx: int = 0,
                      max_display: int = 20,
                      save_path: Optional[str] = None) -> None:
        """
        Plot SHAP waterfall plot

        Parameters
        ----------
        instance_idx : int, default=0
            Index of instance to explain
        max_display : int, default=20
            Maximum number of features to display
        save_path : str, optional
            Save path
        """
        import shap
        
        if self.shap_values is None:
            self.compute_shap_values()
        
        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background
        
        if instance_idx >= len(X_for_plot):
            raise ValueError(f"instance_idx {instance_idx} exceeds data size {len(X_for_plot)}")
        
        X = X_for_plot.iloc[[instance_idx]]
        shap_val = self.shap_values[[instance_idx]]
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_val[0],
                base_values=self.expected_value,
                data=X.iloc[0],
                feature_names=self.feature_names
            ), 
            max_display=max_display, 
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot (Instance {instance_idx})',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved waterfall plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_force(self,
                  instance_idx: int = 0,
                  save_path: Optional[str] = None) -> None:
        """
        Plot SHAP force plot

        Parameters
        ----------
        instance_idx : int, default=0
            Index of instance to explain
        save_path : str, optional
            Save path
        """
        import shap
        
        if self.shap_values is None:
            self.compute_shap_values()
        
        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background
        
        X = X_for_plot.iloc[[instance_idx]]
        shap_val = self.shap_values[[instance_idx]]
        
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            self.expected_value, 
            shap_val[0], 
            X.iloc[0],
            feature_names=self.feature_names, 
            matplotlib=True, 
            show=False
        )
        
        plt.title(f'SHAP Force Plot (Instance {instance_idx})',
                 fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved force plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_partial_dependence(self,
                                features: Optional[List[str]] = None,
                                n_points: int = 200,
                                save_path: Optional[str] = None) -> None:
        """
        Plot partial dependence plots (based on SHAP values)

        Parameters
        ----------
        features : list, optional
            List of features to plot, defaults to all features
        n_points : int, default=100
            Number of points for smoothing curve
        save_path : str, optional
            Save path
        """
        if self.shap_values is None:
            self.compute_shap_values()

        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background

        # Determine features to plot
        if features is None:
            features = self.feature_names
        else:
            features = [f for f in features if f in self.feature_names]

        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature_name in enumerate(features):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            feature_idx = self.feature_names.index(feature_name)
            
            # Get feature values and corresponding SHAP values
            X_feature = X_for_plot[feature_name].values
            shap_values_feature = self.shap_values[:, feature_idx]

            # Draw scatter points
            ax.scatter(X_feature, shap_values_feature, alpha=0.4, s=20,
                      edgecolors='none', color='steelblue')

            # Fit smooth curve and confidence interval
            try:
                # Sort
                sort_idx = np.argsort(X_feature)
                X_sorted = X_feature[sort_idx]
                shap_sorted = shap_values_feature[sort_idx]

                # Use moving average smoothing
                from scipy.ndimage import uniform_filter1d
                window = max(10, len(X_sorted) // 20)
                shap_smooth = uniform_filter1d(shap_sorted, size=window)

                # Calculate moving standard deviation (for confidence interval)
                shap_squared = uniform_filter1d(shap_sorted**2, size=window)
                shap_std = np.sqrt(np.maximum(shap_squared - shap_smooth**2, 0))

                # 95% confidence interval (using 1.96 * standard deviation)
                ci_upper = shap_smooth + 1.96 * shap_std / np.sqrt(window)
                ci_lower = shap_smooth - 1.96 * shap_std / np.sqrt(window)

                # Draw confidence interval (light blue)
                ax.fill_between(X_sorted, ci_lower, ci_upper,
                               color='lightblue', alpha=0.5, label='95% CI')

                # Draw smooth curve
                ax.plot(X_sorted, shap_smooth, color='red', linewidth=2,
                       label='Trend', alpha=0.8)
            except Exception as e:
                warnings.warn(f"Smoothing failed for {feature_name}: {e}")
            
            ax.set_title(feature_name, fontsize=11, fontweight='bold')
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('SHAP Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('SHAP Partial Dependence Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved partial dependence plots to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_dependence(self,
                       feature: str,
                       interaction_index: Optional[Union[str, int]] = 'auto',
                       save_path: Optional[str] = None) -> None:
        """
        Plot SHAP dependence plot for a single feature
        
        Parameters
        ----------
        feature : str
            Feature name to plot
        interaction_index : str, int, or 'auto', optional
            Feature to use for coloring (interaction effect).
            If 'auto', automatically selects the feature with strongest interaction.
            If None, no coloring is applied.
        save_path : str, optional
            Save path
        """
        import shap
        
        if self.shap_values is None:
            self.compute_shap_values()
        
        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found. Available: {self.feature_names}")
        
        feature_idx = self.feature_names.index(feature)
        
        # Check if SHAP values for this feature have variance
        shap_var = np.var(self.shap_values[:, feature_idx])
        if shap_var < 1e-10:
            warnings.warn(f"Feature '{feature}' has zero variance in SHAP values, skipping dependence plot")
            # Create a simple message plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No variation in SHAP values for '{feature}'",
                    ha='center', va='center', fontsize=14)
            plt.title(f'SHAP Dependence Plot: {feature}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()
            return
        
        # Handle interaction index
        if interaction_index == 'auto':
            interaction_idx = 'auto'
        elif interaction_index is None:
            interaction_idx = None
        elif isinstance(interaction_index, str):
            if interaction_index not in self.feature_names:
                raise ValueError(f"Interaction feature '{interaction_index}' not found")
            interaction_idx = self.feature_names.index(interaction_index)
        else:
            interaction_idx = interaction_index
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        try:
            # Use SHAP's dependence plot
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X_for_plot,
                interaction_index=interaction_idx,
                feature_names=self.feature_names,
                show=False
            )
        except Exception as e:
            plt.close()
            if 'kde' in str(e).lower() or 'singular' in str(e).lower():
                # Fallback: create scatter plot without KDE
                warnings.warn(f"KDE failed for dependence plot, using simple scatter plot")
                plt.figure(figsize=(10, 6))
                
                X_feature = X_for_plot.iloc[:, feature_idx].values if hasattr(X_for_plot, 'iloc') else X_for_plot[:, feature_idx]
                shap_feature = self.shap_values[:, feature_idx]
                
                plt.scatter(X_feature, shap_feature, alpha=0.5, s=20, edgecolors='none')
                
                if interaction_idx is not None and interaction_idx != 'auto':
                    if isinstance(interaction_idx, int):
                        interaction_feature = self.feature_names[interaction_idx]
                    else:
                        interaction_feature = interaction_idx
                    
                    plt.title(f'SHAP Dependence: {feature} (colored by {interaction_feature})',
                             fontsize=12, fontweight='bold')
                else:
                    plt.title(f'SHAP Dependence Plot: {feature}', fontsize=12, fontweight='bold')
            else:
                raise
        
        plt.xlabel(feature, fontsize=11)
        plt.ylabel('SHAP Value', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved dependence plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (based on mean absolute SHAP values)

        Returns
        -------
        pd.DataFrame
            Sorted feature importance table
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self,
                                top_n: int = 20,
                                save_path: Optional[str] = None) -> None:
        """
        Plot feature importance chart

        Parameters
        ----------
        top_n : int, default=20
            Display top N features
        save_path : str, optional
            Save path
        """
        importance_df = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(importance_df)))
        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                       color=colors, edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def explain_instance(self,
                        instance_idx: int = 0,
                        return_plot: bool = False) -> Dict[str, Any]:
        """
        Detailed explanation for a single instance

        Parameters
        ----------
        instance_idx : int, default=0
            Instance index
        return_plot : bool, default=False
            Whether to return waterfall plot

        Returns
        -------
        dict
            Dictionary containing detailed explanation
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        X_for_plot = self.X_shap_processed if self.X_shap_processed is not None else self.X_background
        
        if instance_idx >= len(X_for_plot):
            raise ValueError(f"instance_idx {instance_idx} exceeds data size {len(X_for_plot)}")
        
        X_instance = X_for_plot.iloc[[instance_idx]]
        shap_instance = self.shap_values[[instance_idx]]
        
        # Compute prediction
        y_pred = self.predict_fn(X_instance)[0]

        # Compute contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_instance.iloc[0].values,
            'shap_value': shap_instance[0]
        })
        contributions['abs_shap'] = np.abs(contributions['shap_value'])
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        explanation = {
            'instance_idx': instance_idx,
            'predicted_value': y_pred,
            'base_value': self.expected_value,
            'contributions': contributions
        }
        
        if return_plot:
            self.plot_waterfall(instance_idx)
        
        return explanation
    
    def generate_all_plots(self,
                          output_dir: str = './shap_plots',
                          sample_instances: Optional[List[int]] = None) -> None:
        """
        Generate all SHAP visualizations

        Parameters
        ----------
        output_dir : str, default='./shap_plots'
            Output directory
        sample_instances : list, optional
            List of instance indices to explain in detail
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("Generating All SHAP Visualizations")
        print("=" * 70)
        
        # 1. Feature importance (bar plot)
        print("\n1. Generating feature importance bar plot...")
        self.plot_feature_importance(
            save_path=f'{output_dir}/feature_importance.png'
        )

        # 2. Summary plot (beeswarm plot)
        print("2. Generating beeswarm plot...")
        self.plot_summary(
            plot_type='beeswarm',
            save_path=f'{output_dir}/beeswarm.png'
        )

        # 3. Partial dependence plots
        print("3. Generating partial dependence plots...")
        self.plot_partial_dependence(
            save_path=f'{output_dir}/partial_dependence.png'
        )

        # 4. Instance explanations
        print("4. Generating instance explanations...")
        if sample_instances is None:
            sample_instances = [0, len(self.X_background)//2, len(self.X_background)-1]

        for idx in sample_instances:
            if 0 <= idx < len(self.X_background):
                self.plot_waterfall(
                    instance_idx=idx,
                    save_path=f'{output_dir}/waterfall_{idx}.png'
                )
        
        print(f"\nAll plots saved to: {output_dir}")
        print("=" * 70)
