"""
=============================================================================
Basic Utility Functions Module
=============================================================================

Contains:
- B-spline design matrix generation
- Dependency checking utilities
- Data preprocessing helper functions
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import warnings


def check_pygam_available():
    """Check if pygam is available"""
    try:
        import pygam
        return True
    except ImportError:
        return False


def check_shap_available():
    """Check if shap is available"""
    try:
        import shap
        return True
    except ImportError:
        return False




def get_bspline_design_matrix(X, degree=3, n_knots=10, knots=None, return_knots=False):
    """
    Generate B-spline design matrix for given covariates.
    
    Parameters
    ----------
    X : pd.Series or np.ndarray
        Input covariates
    degree : int, default=3
        Degree of B-spline
    n_knots : int, default=10
        Number of internal knots (used only if knots is None)
    knots : np.ndarray, optional
        Pre-defined knot vector. If provided, these knots are used instead of computing new ones.
        This is critical for prediction to use the same knots as training.
    return_knots : bool, default=False
        If True, return the knot vector along with the design matrix.
        
    Returns
    -------
    np.ndarray or tuple
        B-spline design matrix with shape (n_samples, n_basis_funcs)
        If return_knots=True, also returns the knot vector
    """
    # Convert to numpy array if needed and ensure 1D
    X = np.asarray(X).ravel()
    n_samples = len(X)
    
    # If pre-defined knots are provided, use them directly
    if knots is not None:
        t = knots
        degree_used = degree
    else:
        # Get valid (non-NaN) values for knot placement
        X_clean = X[~np.isnan(X)]
        
        # Handle case with too few valid points
        if len(X_clean) < 2:
            # Return a matrix of zeros with appropriate shape
            n_basis = degree + 1
            if return_knots:
                return np.zeros((n_samples, n_basis)), None
            return np.zeros((n_samples, n_basis))
        
        # Create knots
        internal_knots = np.quantile(X_clean, np.linspace(0, 1, n_knots + 2)[1:-1])
        
        # Full knot vector with boundary knots repeated degree+1 times at each end
        t = np.concatenate(([X_clean.min()] * (degree + 1), 
                          internal_knots, 
                          [X_clean.max()] * (degree + 1)))
        
        # Handle potential edge case where knots are too close
        unique_t = np.unique(t)
        if len(unique_t) < degree + 2:
            # Fallback: use simpler spline if knots are insufficient
            n_knots_fallback = min(len(X_clean) // 4, 5)
            if n_knots_fallback > 0:
                internal_knots = np.quantile(X_clean, np.linspace(0, 1, n_knots_fallback + 2)[1:-1])
                t = np.concatenate(([X_clean.min()] * (degree + 1), 
                                  internal_knots, 
                                  [X_clean.max()] * (degree + 1)))
            else:
                # If still not enough, return a constant or linear term (degree 1 with no internal knots)
                if degree > 1:
                    t = np.array([X_clean.min(), X_clean.min(), X_clean.max(), X_clean.max()])
                    degree = 1
        
        degree_used = degree
    
    # Number of basis functions
    n_basis = len(t) - degree_used - 1
    
    # Handle edge case where n_basis is invalid
    if n_basis <= 0:
        if return_knots:
            return np.zeros((n_samples, max(1, degree + 1))), t
        return np.zeros((n_samples, max(1, degree + 1)))
    
    # Create design matrix by evaluating each basis function separately
    # Each column is a basis function, each row is a sample
    basis_matrix = np.zeros((n_samples, n_basis))
    
    for i in range(n_basis):
        # Create unit coefficient vector for this basis function
        c_i = np.zeros(n_basis)
        c_i[i] = 1.0
        
        # Create B-spline for this basis function
        spl_i = BSpline(t, c_i, degree_used)
        
        # Evaluate basis function at all X values
        basis_matrix[:, i] = spl_i(X)
    
    if return_knots:
        return basis_matrix, t
    return basis_matrix


def clean_dataframe(df, required_cols=None, drop_na_subset=None):
    """
    Clean DataFrame, handle missing values and type conversion
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    required_cols : list, optional
        Required columns
    drop_na_subset : list, optional
        Subset of columns to check for missing values
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    df = df.copy()
    
    # Check required columns
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Drop missing values
    if drop_na_subset:
        df = df.dropna(subset=drop_na_subset)
    
    return df


def validate_panel_data(X, id_col='ID', time_col='year'):
    """
    Validate panel data structure
    
    Parameters
    ----------
    X : pd.DataFrame
        Panel data
    id_col : str, default='ID'
        Individual identifier column
    time_col : str, default='year'
        Time identifier column
        
    Returns
    -------
    dict
        Validation result information
    """
    info = {
        'n_observations': len(X),
        'n_units': X[id_col].nunique() if id_col in X.columns else None,
        'n_periods': X[time_col].nunique() if time_col in X.columns else None,
        'is_balanced': False
    }
    
    if id_col in X.columns and time_col in X.columns:
        # Check if the panel is balanced
        obs_per_unit = X.groupby(id_col).size()
        info['is_balanced'] = obs_per_unit.nunique() == 1
        info['min_obs_per_unit'] = obs_per_unit.min()
        info['max_obs_per_unit'] = obs_per_unit.max()
    
    return info


def format_formula(y_name, x_names, include_intercept=True):
    """
    Format regression formula
    
    Parameters
    ----------
    y_name : str
        Dependent variable name
    x_names : list
        List of independent variable names
    include_intercept : bool, default=True
        Whether to include intercept
        
    Returns
    -------
    str
        R-style formula string
    """
    if x_names:
        rhs = " + ".join(x_names)
        if not include_intercept:
            rhs += " - 1"
    else:
        rhs = "1" if include_intercept else "0"
    
    return f"{y_name} ~ {rhs}"


def safe_predict(model, X, fallback_value=0.0):
    """
    Safe model prediction, returns default value on failure
    
    Parameters
    ----------
    model : object
        Object with predict method
    X : array-like
        Prediction data
    fallback_value : float, default=0.0
        Default value on failure
        
    Returns
    -------
    np.ndarray
        Prediction results
    """
    try:
        return np.asarray(model.predict(X)).ravel()
    except Exception as e:
        warnings.warn(f"Prediction failed: {e}. Returning fallback values.")
        return np.full(len(X) if hasattr(X, '__len__') else 1, fallback_value)
