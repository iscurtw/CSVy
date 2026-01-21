"""
Random Forest Model - Production-Ready Hockey Goal Prediction

This module provides Random Forest-based models for predicting hockey game outcomes.
Includes OOB scoring, feature importance, and hyperparameter search utilities.

Classes:
    - RandomForestModel: Single-target Random Forest regression
    - RandomForestGoalPredictor: Dual model for home/away goal prediction

Functions:
    - grid_search_rf: Exhaustive hyperparameter search
    - random_search_rf: Randomized hyperparameter search

Usage:
    from utils.random_forest_model import RandomForestModel, RandomForestGoalPredictor
    
    # Single target prediction
    model = RandomForestModel(n_estimators=200, max_depth=15)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Goal prediction (both home and away)
    predictor = RandomForestGoalPredictor()
    predictor.fit(games_df)
    home_pred, away_pred = predictor.predict_goals(game)
"""

import json
import logging
import pickle
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.inspection import permutation_importance

# Configure module logger
logger = logging.getLogger(__name__)


# Column name mappings - consistent with other models
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
}


def get_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Find the correct column name in a DataFrame using aliases."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


class RandomForestModel:
    """
    Random Forest regression model for hockey goal prediction.
    
    This model uses an ensemble of decision trees with bootstrap aggregation (bagging)
    to predict game outcomes. Provides OOB scoring, feature importance, and 
    permutation importance analysis.
    
    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : int or None, default=15
        Maximum depth of trees. None for unlimited.
    min_samples_split : int, default=5
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum samples required in a leaf node.
    max_features : str or float, default='sqrt'
        Number of features to consider per split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    oob_score : bool, default=True
        Whether to compute out-of-bag R² score.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    random_state : int, default=42
        Random seed for reproducibility.
    name : str, optional
        Model name for identification.
    
    Attributes
    ----------
    model : RandomForestRegressor
        The underlying sklearn model.
    feature_names : list
        Names of features used in training.
    feature_importances_ : pd.Series
        Impurity-based feature importance scores.
    oob_score_ : float
        Out-of-bag R² score (if oob_score=True).
    is_fitted : bool
        Whether the model has been fitted.
    
    Examples
    --------
    >>> model = RandomForestModel(n_estimators=200, max_depth=15)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> print(f"OOB R²: {model.oob_score_:.4f}")
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 42,
    }
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Union[str, float] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        name: Optional[str] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.name = name or f"RF_{n_estimators}trees"
        
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None
        self.oob_score_ = None
        self.is_fitted = False
        self.training_info = {}
    
    def _create_model(self) -> RandomForestRegressor:
        """Create the sklearn RandomForestRegressor."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score and self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'RandomForestModel':
        """
        Fit the Random Forest model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Target values.
        feature_names : list, optional
            Feature names (inferred from DataFrame if not provided).
        
        Returns
        -------
        self
            Fitted model instance.
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X, y)
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        # Store OOB score if available
        if self.oob_score and self.bootstrap:
            self.oob_score_ = self.model.oob_score_
        
        # Store training info
        self.training_info = {
            'n_samples': len(y),
            'n_features': X.shape[1],
            'timestamp': datetime.now().isoformat(),
            'oob_score': self.oob_score_,
        }
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} on {len(y)} samples. OOB R²: {self.oob_score_:.4f}" 
                    if self.oob_score_ else f"Fitted {self.name} on {len(y)} samples")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features.
        y : pd.Series or np.ndarray
            True target values.
        
        Returns
        -------
        dict
            Dictionary with RMSE, MAE, and R² metrics.
        """
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
        }
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features.
        y : pd.Series or np.ndarray
            Target values.
        cv : int, default=5
            Number of folds.
        
        Returns
        -------
        dict
            Mean and std of RMSE across folds.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        model = self._create_model()
        scores = cross_val_score(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=self.n_jobs
        )
        
        return {
            'cv_rmse_mean': -scores.mean(),
            'cv_rmse_std': scores.std(),
            'cv_scores': -scores,
        }
    
    def get_feature_importance(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        method: str = 'impurity'
    ) -> pd.Series:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, optional
            Required for permutation importance.
        y : pd.Series or np.ndarray, optional
            Required for permutation importance.
        method : str, default='impurity'
            'impurity' for built-in importance, 'permutation' for permutation importance.
        
        Returns
        -------
        pd.Series
            Feature importance scores, sorted descending.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if method == 'impurity':
            return self.feature_importances_
        
        elif method == 'permutation':
            if X is None or y is None:
                raise ValueError("X and y required for permutation importance")
            
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            perm_importance = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            return pd.Series(
                perm_importance.importances_mean,
                index=self.feature_names
            ).sort_values(ascending=False)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'impurity' or 'permutation'.")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str or Path
            File path for saving.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importances_': self.feature_importances_,
            'oob_score_': self.oob_score_,
            'params': self.get_params(),
            'training_info': self.training_info,
            'name': self.name,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'RandomForestModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str or Path
            File path to load from.
        
        Returns
        -------
        RandomForestModel
            Loaded model instance.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved params
        instance = cls(**model_data['params'], name=model_data.get('name'))
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.feature_importances_ = model_data['feature_importances_']
        instance.oob_score_ = model_data['oob_score_']
        instance.training_info = model_data.get('training_info', {})
        instance.is_fitted = True
        
        logger.info(f"Loaded model from {path}")
        return instance


class RandomForestGoalPredictor:
    """
    Dual Random Forest model for predicting both home and away goals.
    
    Uses two separate RandomForestModel instances, one for home goals
    and one for away goals.
    
    Parameters
    ----------
    feature_columns : list, optional
        Column names to use as features.
    **rf_params
        Parameters passed to RandomForestModel.
    
    Examples
    --------
    >>> predictor = RandomForestGoalPredictor(n_estimators=200)
    >>> predictor.fit(games_df)
    >>> home_pred, away_pred = predictor.predict_goals(new_game)
    """
    
    DEFAULT_FEATURES = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_win_pct', 'away_win_pct',
        'home_goals_avg', 'away_goals_avg',
        'home_goals_against_avg', 'away_goals_against_avg',
        'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct',
        'home_rest_days', 'away_rest_days',
    ]
    
    def __init__(self, feature_columns: Optional[List[str]] = None, **rf_params):
        self.feature_columns = feature_columns
        self.rf_params = rf_params
        
        self.home_model = RandomForestModel(name='RF_home_goals', **rf_params)
        self.away_model = RandomForestModel(name='RF_away_goals', **rf_params)
        
        self.is_fitted = False
        self.training_info = {}
    
    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from dataframe."""
        if self.feature_columns:
            available = [c for c in self.feature_columns if c in df.columns]
            return df[available]
        
        # Auto-detect features
        available = [c for c in self.DEFAULT_FEATURES if c in df.columns]
        if available:
            return df[available]
        
        # Fallback: use all numeric columns except targets
        exclude = ['home_goals', 'away_goals', 'total_goals']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[[c for c in numeric_cols if c not in exclude]]
    
    def fit(
        self,
        df: pd.DataFrame,
        home_goals_col: str = 'home_goals',
        away_goals_col: str = 'away_goals'
    ) -> 'RandomForestGoalPredictor':
        """
        Fit both home and away goal prediction models.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and goal columns.
        home_goals_col : str, default='home_goals'
            Column name for home goals.
        away_goals_col : str, default='away_goals'
            Column name for away goals.
        
        Returns
        -------
        self
            Fitted predictor instance.
        """
        # Get feature columns
        X = self._get_features(df)
        self.feature_columns = list(X.columns)
        
        # Get targets
        home_col = get_column(df, 'home_goals') or home_goals_col
        away_col = get_column(df, 'away_goals') or away_goals_col
        
        y_home = df[home_col]
        y_away = df[away_col]
        
        # Fit models
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        
        self.is_fitted = True
        self.training_info = {
            'n_samples': len(df),
            'n_features': len(self.feature_columns),
            'home_oob_r2': self.home_model.oob_score_,
            'away_oob_r2': self.away_model.oob_score_,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Fitted dual RF predictor. Home OOB R²: {self.home_model.oob_score_:.4f}, "
                    f"Away OOB R²: {self.away_model.oob_score_:.4f}")
        
        return self
    
    def predict_goals(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict home and away goals.
        
        Parameters
        ----------
        df : pd.DataFrame
            Features for prediction.
        
        Returns
        -------
        tuple
            (home_goals_predictions, away_goals_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._get_features(df)
        
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        
        return home_pred, away_pred
    
    def predict_winner(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict game winners.
        
        Returns
        -------
        pd.Series
            'home', 'away', or 'tie' for each game.
        """
        home_pred, away_pred = self.predict_goals(df)
        
        results = []
        for h, a in zip(home_pred, away_pred):
            if h > a + 0.5:
                results.append('home')
            elif a > h + 0.5:
                results.append('away')
            else:
                results.append('tie')
        
        return pd.Series(results, index=df.index)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        home_goals_col: str = 'home_goals',
        away_goals_col: str = 'away_goals'
    ) -> Dict[str, float]:
        """
        Evaluate prediction performance.
        
        Returns
        -------
        dict
            RMSE, MAE for home, away, and combined.
        """
        home_col = get_column(df, 'home_goals') or home_goals_col
        away_col = get_column(df, 'away_goals') or away_goals_col
        
        home_pred, away_pred = self.predict_goals(df)
        
        y_home = df[home_col].values
        y_away = df[away_col].values
        
        return {
            'home_rmse': np.sqrt(mean_squared_error(y_home, home_pred)),
            'home_mae': mean_absolute_error(y_home, home_pred),
            'away_rmse': np.sqrt(mean_squared_error(y_away, away_pred)),
            'away_mae': mean_absolute_error(y_away, away_pred),
            'combined_rmse': np.sqrt(mean_squared_error(
                np.concatenate([y_home, y_away]),
                np.concatenate([home_pred, away_pred])
            )),
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save predictor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'home_model': self.home_model,
            'away_model': self.away_model,
            'feature_columns': self.feature_columns,
            'rf_params': self.rf_params,
            'training_info': self.training_info,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved predictor to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'RandomForestGoalPredictor':
        """Load predictor from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            feature_columns=model_data['feature_columns'],
            **model_data['rf_params']
        )
        instance.home_model = model_data['home_model']
        instance.away_model = model_data['away_model']
        instance.training_info = model_data.get('training_info', {})
        instance.is_fitted = True
        
        return instance


def grid_search_rf(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[RandomForestModel, Dict]:
    """
    Perform grid search for Random Forest hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Training features.
    y : pd.Series or np.ndarray
        Target values.
    param_grid : dict, optional
        Parameter grid. Uses defaults if not provided.
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Parallel jobs.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    tuple
        (best_model, search_results)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    
    search = GridSearchCV(
        rf, param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    else:
        feature_names = None
    
    if isinstance(y, pd.Series):
        y = y.values
    
    search.fit(X, y)
    
    # Create model with best params
    best_model = RandomForestModel(**search.best_params_)
    best_model.fit(X, y, feature_names=feature_names)
    
    results = {
        'best_params': search.best_params_,
        'best_score': -search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
    }
    
    return best_model, results


def random_search_rf(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    param_distributions: Optional[Dict] = None,
    n_iter: int = 50,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[RandomForestModel, Dict]:
    """
    Perform randomized search for Random Forest hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Training features.
    y : pd.Series or np.ndarray
        Target values.
    param_distributions : dict, optional
        Parameter distributions. Uses defaults if not provided.
    n_iter : int, default=50
        Number of parameter combinations to try.
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Parallel jobs.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    tuple
        (best_model, search_results)
    """
    from scipy.stats import randint, uniform
    
    if param_distributions is None:
        param_distributions = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', 0.5, 0.8],
        }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    
    search = RandomizedSearchCV(
        rf, param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42,
        return_train_score=True
    )
    
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    else:
        feature_names = None
    
    if isinstance(y, pd.Series):
        y = y.values
    
    search.fit(X, y)
    
    # Create model with best params
    best_params = {k: v for k, v in search.best_params_.items() 
                   if k in RandomForestModel.DEFAULT_PARAMS}
    best_model = RandomForestModel(**best_params)
    best_model.fit(X, y, feature_names=feature_names)
    
    results = {
        'best_params': search.best_params_,
        'best_score': -search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
    }
    
    return best_model, results
