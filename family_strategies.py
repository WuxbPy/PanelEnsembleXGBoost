"""
Family Strategies for PanelEnsembleXGBoost

This module implements the strategy pattern for different distribution families
supported by PanelEnsembleXGBoost:
- 'gaussian': Gaussian distribution (continuous regression)
- 'binomial': Binomial distribution (binary classification)
- 'poisson': Poisson distribution (count data)
- 'gamma': Gamma distribution (positive continuous data)
- 'multinomial': Multinomial distribution (multiclass classification)

Author: PanelEnsembleXGBoost Research Team
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type
from sklearn.preprocessing import LabelEncoder


class FamilyStrategy(ABC):
    """
    Abstract base class for distribution family strategies
    
    Each family strategy must implement methods for:
    - Processing target variables
    - Validating target format
    - Link functions (connecting linear predictor to mean)
    - Inverse link functions
    - Residual computation
    - Deviance computation (for model evaluation)
    - Model class selection for each stage
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.task_type = 'regression'  # 'regression' or 'classification'
        self.classes_ = None  # For classification tasks
    
    @abstractmethod
    def process_y(self, y: Any) -> np.ndarray:
        """Process target variable to appropriate format"""
        pass
    
    @abstractmethod
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate target variable format"""
        pass
    
    @abstractmethod
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """
        Link function: g(mu) = eta
        Transforms mean parameter to linear predictor
        """
        pass
    
    @abstractmethod
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Inverse link function: mu = g^(-1)(eta)
        Transforms linear predictor to mean parameter
        """
        pass
    
    @abstractmethod
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Compute working residuals"""
        pass
    
    @abstractmethod
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Compute deviance (goodness of fit measure)"""
        pass
    
    @abstractmethod
    def get_stage2_model_class(self):
        """Return appropriate model class for Stage 2 (GAM smoothing)"""
        pass
    
    @abstractmethod
    def get_stage2_fallback_model(self):
        """Return fallback model class when GAM is unavailable"""
        pass
    
    @abstractmethod
    def get_xgb_model_class(self):
        """Return appropriate XGBoost model class for Stage 3"""
        pass
    
    @abstractmethod
    def get_xgb_objective(self) -> str:
        """Return XGBoost objective function name"""
        pass
    
    def predict_to_label(self, mu: np.ndarray) -> np.ndarray:
        """
        Convert predicted probabilities to class labels
        For classification tasks only
        """
        return mu


class GaussianStrategy(FamilyStrategy):
    """Gaussian distribution strategy (continuous regression)"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.task_type = 'regression'
    
    def process_y(self, y: Any) -> np.ndarray:
        """Process target variable - ensure numeric array"""
        y_arr = np.asarray(y).ravel()
        return y_arr.astype(np.float64)
    
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate - Gaussian accepts any real values"""
        if not np.all(np.isfinite(y)):
            raise ValueError("Target variable contains non-finite values")
        return True
    
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """Identity link: g(mu) = mu"""
        return mu
    
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Identity inverse link: mu = eta"""
        return eta
    
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Raw residuals for Gaussian"""
        return y - mu
    
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Gaussian deviance (sum of squared residuals)"""
        return np.sum((y - mu) ** 2)
    
    def get_stage2_model_class(self):
        """Use LinearGAM for Gaussian"""
        try:
            from pygam import LinearGAM
            return LinearGAM
        except ImportError:
            from sklearn.linear_model import LinearRegression
            return LinearRegression
    
    def get_stage2_fallback_model(self):
        """Fallback to LinearRegression"""
        from sklearn.linear_model import LinearRegression
        return LinearRegression
    
    def get_xgb_model_class(self):
        """Use XGBRegressor for Gaussian"""
        import xgboost as xgb
        return xgb.XGBRegressor
    
    def get_xgb_objective(self) -> str:
        """Squared error objective"""
        return 'reg:squarederror'


class BinomialStrategy(FamilyStrategy):
    """Binomial distribution strategy (binary classification)"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.task_type = 'classification'
        self.positive_class = self.params.get('positive_class', 1)
        self.label_encoder = LabelEncoder()
    
    def process_y(self, y: Any) -> np.ndarray:
        """Process target - encode as 0/1"""
        y_arr = np.asarray(y).ravel()
        
        # Handle string labels
        if y_arr.dtype == object or y_arr.dtype.kind == 'U':
            self.label_encoder.fit(y_arr)
            y_processed = self.label_encoder.transform(y_arr)
            self.classes_ = self.label_encoder.classes_
        else:
            # Assume already numeric
            y_processed = y_arr.astype(int)
            self.classes_ = np.unique(y_processed)
        
        return y_processed.astype(np.float64)
    
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate - binary values only"""
        unique_vals = np.unique(y)
        if len(unique_vals) > 2:
            raise ValueError(f"Binomial family requires binary targets, got {len(unique_vals)} classes")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Binomial targets must be 0 or 1")
        return True
    
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """Logit link: g(mu) = log(mu/(1-mu))"""
        mu = np.clip(mu, 1e-7, 1 - 1e-7)
        return np.log(mu / (1 - mu))
    
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Logistic inverse link: mu = 1/(1+exp(-eta))"""
        return 1.0 / (1.0 + np.exp(-eta))
    
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Working residuals for binomial"""
        return y - mu
    
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Binomial deviance"""
        mu = np.clip(mu, 1e-7, 1 - 1e-7)
        y01 = np.clip(y, 1e-7, 1 - 1e-7)
        dev = 2 * np.sum(y01 * np.log(y01 / mu) + (1 - y01) * np.log((1 - y01) / (1 - mu)))
        return dev
    
    def get_stage2_model_class(self):
        """Use LogisticGAM for binomial"""
        try:
            from pygam import LogisticGAM
            return LogisticGAM
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression
    
    def get_stage2_fallback_model(self):
        """Fallback to LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    
    def get_xgb_model_class(self):
        """Use XGBClassifier for binomial"""
        import xgboost as xgb
        return xgb.XGBClassifier
    
    def get_xgb_objective(self) -> str:
        """Logistic regression objective"""
        return 'binary:logistic'
    
    def predict_to_label(self, mu: np.ndarray) -> np.ndarray:
        """Convert probabilities to binary labels"""
        return (mu >= 0.5).astype(int)


class PoissonStrategy(FamilyStrategy):
    """Poisson distribution strategy (count data)"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.task_type = 'regression'
        self.exposure = self.params.get('exposure', None)
    
    def process_y(self, y: Any) -> np.ndarray:
        """Process target - ensure non-negative integers"""
        y_arr = np.asarray(y).ravel()
        
        if np.any(y_arr < 0):
            raise ValueError("Poisson targets must be non-negative")
        if np.any(y_arr != y_arr.astype(int)):
            # Allow floats but warn
            pass
        
        return y_arr.astype(np.float64)
    
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate - non-negative values"""
        if np.any(y < 0):
            raise ValueError("Poisson targets must be non-negative")
        return True
    
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """Log link: g(mu) = log(mu)"""
        mu = np.maximum(mu, 1e-7)
        return np.log(mu)
    
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Exponential inverse link: mu = exp(eta)"""
        # Clip eta to prevent overflow in exp
        eta_clipped = np.clip(eta, -20, 20)
        return np.exp(eta_clipped)
    
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Working residuals for Poisson"""
        return y - mu
    
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Poisson deviance"""
        mu = np.maximum(mu, 1e-7)
        y_pos = y > 0
        dev = 2 * np.sum(
            y[y_pos] * np.log(y[y_pos] / mu[y_pos]) - (y[y_pos] - mu[y_pos])
        )
        dev += 2 * np.sum(mu[~y_pos])  # For y=0 observations
        return dev
    
    def get_stage2_model_class(self):
        """Use PoissonGAM for Poisson"""
        try:
            from pygam import PoissonGAM
            return PoissonGAM
        except ImportError:
            from sklearn.linear_model import PoissonRegressor
            return PoissonRegressor
    
    def get_stage2_fallback_model(self):
        """Fallback to PoissonRegressor"""
        from sklearn.linear_model import PoissonRegressor
        return PoissonRegressor
    
    def get_xgb_model_class(self):
        """Use XGBRegressor with count objective"""
        import xgboost as xgb
        return xgb.XGBRegressor
    
    def get_xgb_objective(self) -> str:
        """Poisson regression objective"""
        return 'count:poisson'


class GammaStrategy(FamilyStrategy):
    """Gamma distribution strategy (positive continuous data)"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.task_type = 'regression'
        self.link = self.params.get('link', 'log')
    
    def process_y(self, y: Any) -> np.ndarray:
        """Process target - ensure positive values"""
        y_arr = np.asarray(y).ravel()
        
        if np.any(y <= 0):
            raise ValueError("Gamma targets must be strictly positive")
        
        return y_arr.astype(np.float64)
    
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate - positive values only"""
        if np.any(y <= 0):
            raise ValueError("Gamma targets must be strictly positive")
        return True
    
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """Log link (default): g(mu) = log(mu)"""
        mu = np.maximum(mu, 1e-7)
        return np.log(mu)
    
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Exponential inverse link: mu = exp(eta)"""
        # Clip eta to prevent overflow in exp
        eta_clipped = np.clip(eta, -20, 20)
        return np.exp(eta_clipped)
    
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Working residuals for Gamma"""
        return (y - mu) / mu  # Pearson residuals
    
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Gamma deviance"""
        mu = np.maximum(mu, 1e-7)
        dev = 2 * np.sum(-np.log(y / mu) + (y - mu) / mu)
        return dev
    
    def get_stage2_model_class(self):
        """Use GammaGAM for Gamma"""
        try:
            from pygam import GammaGAM
            return GammaGAM
        except ImportError:
            # Fallback to GammaRegressor or custom implementation
            try:
                from sklearn.linear_model import GammaRegressor
                return GammaRegressor
            except ImportError:
                from sklearn.linear_model import LinearRegression
                return LinearRegression
    
    def get_stage2_fallback_model(self):
        """Fallback to GammaRegressor"""
        try:
            from sklearn.linear_model import GammaRegressor
            return GammaRegressor
        except ImportError:
            from sklearn.linear_model import LinearRegression
            return LinearRegression
    
    def get_xgb_model_class(self):
        """Use XGBRegressor with gamma objective"""
        import xgboost as xdb
        return xdb.XGBRegressor
    
    def get_xgb_objective(self) -> str:
        """Gamma regression objective"""
        return 'reg:gamma'


class MultinomialStrategy(FamilyStrategy):
    """Multinomial distribution strategy (multiclass classification)"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.task_type = 'classification'
        self.num_class = None
        self.label_encoder = LabelEncoder()
    
    def process_y(self, y: Any) -> np.ndarray:
        """Process target - encode classes as integers"""
        y_arr = np.asarray(y).ravel()
        
        # Encode labels
        self.label_encoder.fit(y_arr)
        y_processed = self.label_encoder.transform(y_arr)
        self.classes_ = self.label_encoder.classes_
        self.num_class = len(self.classes_)
        
        return y_processed.astype(np.float64)
    
    def validate_y(self, y: np.ndarray) -> bool:
        """Validate - ensure we have multiple classes"""
        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError("Multinomial requires at least 2 classes")
        return True
    
    def link_function(self, mu: np.ndarray) -> np.ndarray:
        """
        Multinomial logit link (softmax)
        mu is expected to be probability matrix (n_samples, n_classes)
        """
        # For multinomial, we typically work in probability space
        # Link function is more complex and handled in XGBoost
        return mu
    
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Softmax inverse link
        """
        if eta.ndim == 1:
            # Binary case handled by binomial
            return 1.0 / (1.0 + np.exp(-eta))
        else:
            # Multiclass softmax
            exp_eta = np.exp(eta - np.max(eta, axis=1, keepdims=True))
            return exp_eta / np.sum(exp_eta, axis=1, keepdims=True)
    
    def compute_residual(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Working residuals for multinomial"""
        # One-hot encode y if needed
        if mu.ndim > 1 and y.ndim == 1:
            y_onehot = np.zeros_like(mu)
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
            return y_onehot - mu
        return y - mu
    
    def compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Multinomial deviance (cross-entropy)"""
        if mu.ndim > 1:
            # Multiclass
            y_int = y.astype(int)
            log_probs = np.log(np.maximum(mu[np.arange(len(y)), y_int], 1e-7))
            return -2 * np.sum(log_probs)
        else:
            # Binary fallback
            mu = np.clip(mu, 1e-7, 1 - 1e-7)
            y01 = np.clip(y, 1e-7, 1 - 1e-7)
            dev = 2 * np.sum(y01 * np.log(y01 / mu) + (1 - y01) * np.log((1 - y01) / (1 - mu)))
            return dev
    
    def get_stage2_model_class(self):
        """Use LogisticGAM for multinomial (simplified)"""
        try:
            from pygam import LogisticGAM
            return LogisticGAM
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression
    
    def get_stage2_fallback_model(self):
        """Fallback to LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    
    def get_xgb_model_class(self):
        """Use XGBClassifier for multinomial"""
        import xgboost as xdb
        return xdb.XGBClassifier
    
    def get_xgb_objective(self) -> str:
        """Multiclass softmax objective"""
        return 'multi:softprob'
    
    def predict_to_label(self, mu: np.ndarray) -> np.ndarray:
        """Convert probabilities to class labels"""
        if mu.ndim > 1:
            return np.argmax(mu, axis=1)
        else:
            return (mu >= 0.5).astype(int)


def get_family_strategy(family: str, params: Optional[Dict[str, Any]] = None) -> FamilyStrategy:
    """
    Factory function to create appropriate family strategy
    
    Parameters
    ----------
    family : str
        Distribution family name: 'gaussian', 'binomial', 'poisson', 'gamma', 'multinomial'
    params : dict, optional
        Family-specific parameters
        
    Returns
    -------
    FamilyStrategy
        Appropriate strategy instance
        
    Raises
    ------
    ValueError
        If family name is not recognized
    """
    family = family.lower()
    
    strategies = {
        'gaussian': GaussianStrategy,
        'binomial': BinomialStrategy,
        'poisson': PoissonStrategy,
        'gamma': GammaStrategy,
        'multinomial': MultinomialStrategy
    }
    
    if family not in strategies:
        raise ValueError(
            f"Unknown family '{family}'. "
            f"Supported families: {list(strategies.keys())}"
        )
    
    return strategies[family](params)
