import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
from itertools import product

from sklearn.base import BaseEstimator, RegressorMixin

from xgboost import XGBRegressor
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ZeroModel:
    def predict(self, X):
        return 0

def create_hyper_params(grid_params):
    """
    Generate hyperparameter combinations for each model type.

    Parameters:
        grid_params (dict): A dictionary where keys are model types, and values are dictionaries
                            of parameter names and their corresponding lists of values.

    Returns:
        dict: A dictionary containing lists of hyperparameter combinations for each model type.
    """
    hyper_params_all = {}
    for model_type, param_grid in grid_params.items():
        # Generate all combinations of parameters for this model type
        param_combinations = list(product(*param_grid.values()))
        # Convert each combination to a dictionary
        hyper_params_all[model_type] = [
            dict(zip(param_grid.keys(), combination)) for combination in param_combinations
        ]
    return hyper_params_all

# Define a scaler that applies group-wise scaling for prediction
class GroupScaler:
    def __init__(self, scalers, sets_train):
        self.scalers = scalers
        self.sets_train_unique = np.unique(sets_train)

    def transform(self, X, sets, z):
        if sets is None:
            sets = np.array(['noclass'] * X.shape[0], dtype=object)
        elif np.any(~np.isin(sets, self.sets_train_unique)):
            raise Exception("Not implemented")

        X_scaled = np.zeros_like(X)
        for group in np.unique(sets):
            group_idx = np.where(sets == group)[0]
            scaler_group = self.scalers[group]
            X_scaled[group_idx, :] = scaler_group.transform(X[group_idx, :])
        return X_scaled

class XGBWithES(BaseEstimator, RegressorMixin):
    """Minimal XGBoost wrapper with:
        - early stopping via xgb.cv
        - refit on full dataset with best iteration
        - only a few tuned hyperparameters
    """

    def __init__(
        self,
        max_depth=6,
        min_child_weight=1,
        learning_rate=0.1,
        n_estimators=2000,
        n_splits=5,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=1,
    ):
        self.max_depth = max_depth
        self.colsample_bytree = 1.0
        self.min_child_weight = min_child_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.best_n_estimators = None

    def fit(self, X, y):
        y = np.ravel(y)

        # ---- 1) Use xgb.cv to find best number of boosting rounds ----
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "colsample_bytree": self.colsample_bytree,
            "eta": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "seed": self.random_state,
            "nthread": self.n_jobs,
        }

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            nfold=self.n_splits,
            early_stopping_rounds=self.early_stopping_rounds,
            shuffle=True,
            verbose_eval=False,
        )

        # Length of cv_results equals the best boosting round when early stopping is used
        self.best_n_estimators = int(cv_results.shape[0])

        # ---- 2) Refit sklearn-style model on full data with best_n_estimators ----
        self.model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            n_estimators=self.best_n_estimators,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class FittedModel:
    """
    A wrapper for the fitted model, scaler, PCA, and confounder regressor pipeline with a predict function.
    """

    def __init__(self, model, scaler, selection_fun, confounder_regressor_X, confounder_regressor_y):
        self.model = model
        self.scaler = scaler
        self.selection_fun = selection_fun
        self.confounder_regressor_X = confounder_regressor_X
        self.confounder_regressor_y = confounder_regressor_y


    def predict_inner(self, X, z, sets=None):
        """
        Apply scaling, PCA, and model prediction to new data after regressing out confounders.

        Parameters:
            X (numpy.ndarray): Input data.
            z (numpy.ndarray): Confounder variables.

        Returns:
            numpy.ndarray: Predicted values.
        """

        # Scale
        X_scaled = self.scaler.transform(X, sets, z)

        # Regress-out confounders from X using the pre-fitted regressor
        residual_X = X_scaled - self.confounder_regressor_X.predict(z)

        # reduce dimensionality
        X_select = self.selection_fun(residual_X)

        return self.model.predict(X_select).reshape(-1, 1)
    
    def predict(self, X, z, sets=None):
        if not hasattr(self.scaler, 'sets_train_unique') or (hasattr(self.scaler, 'sets_train_unique') and np.all(np.isin(sets, self.scaler.sets_train_unique))):
            return self.predict_inner(X, z, sets)
        elif not np.any(np.isin(sets, self.scaler.sets_train_unique)): 
            # in case sets occurring in 
            predictions_per_set = []
            for train_set in self.scaler.sets_train_unique:
                dummy_sets = np.array([train_set] * X.shape[0], dtype=object)
                predictions_per_set.append(self.predict_inner(X, z, dummy_sets))
            return np.mean(np.stack(predictions_per_set, axis=1), axis=1)
        else:
            raise Exception('not implemented')


def fit_model(X_train, y_train, z_train, sets_train, n_comp, model_type, hyper_params, numIter):
        """
        Train a single model with scaling and PCA, adjusting for confounding effects if necessary.

        Parameters:
            X_train (array): Feature matrix for training.
            y_train (array): Target variable for training.
            z_train (array): Confounder data for adjustment.
            n_comp (int): Number of PCA components to use for dimensionality reduction.
            model_type (str): Type of model to train ('linear' or 'xgboost').
            hyper_params (dict): Hyperparameters for the model being trained.

        Returns:
            FittedModel: A wrapper object containing the trained model, scaler, PCA, and confounder regressors.
        """

        # Scale the training set independently within groups defined by sets_train ================
        unique_groups = np.unique(sets_train)
        X_train_scaled = np.zeros_like(X_train)
        scalers = {}
        for group in unique_groups:
            group_idx = np.where(sets_train == group)[0]
            scaler_group = StandardScaler()
            X_train_scaled[group_idx, :] = scaler_group.fit_transform(X_train[group_idx, :])
            scalers[group] = scaler_group

        scaler = GroupScaler(scalers, sets_train)


        # remove confounders ================
        if(len(z_train)>0):
            # Fit confounder regressors on training data
            confounder_regressor_X = LinearRegression().fit(z_train, X_train_scaled)
            confounder_regressor_y = LinearRegression().fit(z_train, y_train)
        else:
            confounder_regressor_X = ZeroModel()
            confounder_regressor_y = ZeroModel()

        # compute residuals for X and y
        X_train_scaled = X_train_scaled - confounder_regressor_X.predict(z_train)
        y_train = y_train - confounder_regressor_y.predict(z_train)


        # Define the model ================
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'xgboost':
            model = XGBWithES(
                max_depth=int(hyper_params["max_depth"]),
                min_child_weight=float(hyper_params["min_child_weight"]),
                learning_rate=float(hyper_params["learning_rate"]),
                n_estimators=numIter  # numIter defines the *max* boosting rounds
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")


        # input space dimensionality reduction ================
        n_comp = min(n_comp, min(X_train.shape)) # Ensure n_comp is not larger than the number of columns or rows in X_train
        if(n_comp > 0):
            pca = PCA(n_components=n_comp, svd_solver='full')
            pca.fit(X_train_scaled)
            selection_mat = pca.components_.T
        else:
            selection_mat = np.eye(X_train.shape[1])
        selection_fun = lambda X: np.dot(X, selection_mat).reshape(-1, 1) if selection_mat.shape[1] == 1 else np.dot(X, selection_mat)

        X_train_select = selection_fun(X_train_scaled)


        # fit the model ================
        model.fit(X_train_select, y_train)


        # Return the FittedModel object ================
        return FittedModel(model, scaler, selection_fun, confounder_regressor_X, confounder_regressor_y)


def evaluate_params(X_train, y_train, z_train, subIDs_train, sets_train, scorer, n_comp, model_type, hyper_params, numIter):
        # # Perform cross-validation with parallel processing

        # Use GroupKFold to ensure individuals do not appear in both train and test splits
        gkf = GroupKFold(n_splits=10, shuffle=True, random_state=42)

        scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=subIDs_train):
            # Create train and validation splits
            X_train_split = X_train[train_idx,:]
            X_val_split = X_train[val_idx,:]
            y_train_split = y_train[train_idx]
            y_val_split = y_train[val_idx]
            sets_train_split = sets_train[train_idx]
            sets_val_split = sets_train[val_idx]
            if(len(z_train) > 0):
                z_train_split = z_train[train_idx]
                z_val_split = z_train[val_idx]
            else:
                z_train_split = []
                z_val_split = []

            # Train the model
            model_object = fit_model(X_train_split, y_train_split, z_train_split, sets_train_split, n_comp, model_type, hyper_params, numIter)

            # Evaluate on the validation set
            y_val_pred = model_object.predict(X_val_split, z_val_split, sets_val_split)
            y_val_split_corrected = y_val_split - model_object.confounder_regressor_y.predict(z_val_split)
            score = scorer._score_func(y_val_split_corrected, y_val_pred)
            scores.append(score)

        # Return the mean score and the model object
        return np.mean(scores), {
            'n_components': n_comp,
            'model_type': model_type,
            'hyper_params': hyper_params
        }, model_object

def print_best_params(best_params):
    print("Best Parameters:")
    for key, value in best_params.items():
        if isinstance(value, dict):  # Handle nested dictionaries (e.g., hyper_params)
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

def train_model(X_train, y_train, z_train, subIDs_train, n_components, model_type_all, grid_params, ncores, numIter, scorer, sets_train=None):
    """
    Train the best model (LinearRegression or XGBoost) with PCA using explicit hyperparameter
    and cross-validation loops, along with proper data scaling and parallel processing.

    Parameters:
        X_train (numpy.ndarray): The feature matrix for training.
        y_train (numpy.ndarray): The target variable for training.
        z_train (array): Confounder data used to adjust for confounding effects.
        n_components (list): A list of PCA components to test for dimensionality reduction.
        model_type_all (list): List of model types to test ('linear', 'xgboost').
        grid_params (dict): Dictionary of hyperparameters to search for each model type.
        ncores (int): Number of CPU cores to use for parallel processing during cross-validation.
        numIter (int): Maximum number of boosting rounds for XGBoost.
        scorer (callable): Custom scoring function to evaluate model performance.

    Returns:
        FittedModel: A wrapper object containing the trained model, scaler, PCA, and confounder regressors.
    """
    # Initialize variables to store the best model and hyperparameters
    best_score = -np.inf
    best_params = {}
    best_model_object = None

    if(sets_train is None):
        sets_train = np.array(['noclass'] * len(y_train), dtype=object)

    # Generate hyperparameter combinations
    hyper_params_all = create_hyper_params(grid_params)

    # Use Parallel for hyperparameter search
    results = Parallel(n_jobs=ncores)(
        delayed(evaluate_params)(X_train, y_train, z_train, subIDs_train, sets_train, scorer, n_comp, model_type, hyper_params, numIter)
        for n_comp in n_components
        for model_type in model_type_all
        for hyper_params in hyper_params_all[model_type]
    )

    # Find the best parameters
    for score, params, model_object in results:
        if score > best_score:
            best_score = score
            best_params = params
            best_model_object = model_object

    # # Print the best parameters and score
    print_best_params(best_params)

    # Refit the best model on the full training dataset
    best_model_object = fit_model(
        X_train, y_train, z_train, sets_train, best_params['n_components'], best_params['model_type'], best_params['hyper_params'], numIter
    )

    return best_model_object, best_score, best_params
