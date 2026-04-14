"""
SVR Subsample Algorithm with Residual-based Neighbor Selection

This module implements an optimized Support Vector Regression (SVR) training algorithm using 
intelligent subsample selection combined with Bayesian hyperparameter optimization.

The algorithm is based on the paper "Nearest neighbors methods for support vector machines" 
by Camelo, S. A., González-Lima, M. D., Quiroz, A. J., and has been modified to support 
residual-based neighbor selection as an alternative to spatial distance-based selection.

Key Features:
    - Efficient training on large datasets through intelligent subsampling
    - Bayesian optimization for hyperparameter tuning
    - Support for both spatial and residual-based neighbor selection
    - Iterative refinement process for improved model performance
    - Comprehensive logging and performance metrics

Example:
    >>> config = SVRConfig(subsample_fraction=0.01, use_residual_criterion=True)
    >>> optimizer = SVRSubsampleOptimizer(config)
    >>> model, info = optimizer.train(train_data, test_data, 'target', 'rbf', params)

Author: Adapted from original implementation
Version: 2.0
"""

import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import BayesSearchCV


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SVRConfig:
    """
    Configuration parameters for SVR subsample algorithm.
    
    This dataclass encapsulates all configurable parameters for the SVR subsample
    optimization algorithm, providing sensible defaults while allowing customization.
    
    Attributes:
        subsample_fraction (float): Initial subsample size as fraction of training data (δ parameter).
            Default: 0.01 (1% of data).
        r_set_fraction (float): Size of random set R as fraction of remaining data (ε parameter).
            Default: 0.1 (10% of remaining data).
        num_neighbors (int): Number of nearest neighbors to find for each support vector (k parameter).
            Default: 5.
        random_state (int): Random seed for reproducibility.
            Default: 45.
        cv_folds (int): Number of cross-validation folds for Bayesian optimization.
            Default: 3.
        bayes_n_points (int): Number of points to evaluate in each Bayesian optimization iteration.
            Default: 10.
        bayes_n_iter (int): Maximum number of iterations for Bayesian optimization search.
            Limits total evaluations to n_points × n_iter. Default: 20.
        convergence_threshold (float): R² difference threshold to trigger iterative refinement.
            Default: 0.02.
        iteration_threshold (float): R² difference threshold to stop iterative refinement.
            Default: 0.05.
        use_residual_criterion (bool): Whether to use residual-based (True) or spatial (False) neighbor selection.
            Default: True.
    """
    subsample_fraction: float = 0.01  # δ (delta) parameter
    r_set_fraction: float = 0.1       # ε (epsilon) parameter
    num_neighbors: int = 5            # k parameter
    random_state: int = 45
    cv_folds: int = 3
    bayes_n_points: int = 5
    bayes_n_iter: int = 20
    convergence_threshold: float = 0.02
    iteration_threshold: float = 0.05
    use_residual_criterion: bool = True  # New parameter for professor's requirement


class SVRSubsampleOptimizer:
    """
    Optimized SVR training using intelligent subsample selection with Bayesian optimization.
    
    This class implements an efficient algorithm for training Support Vector Regression models
    on large datasets by intelligently selecting subsamples of the data. The algorithm combines
    the benefits of reduced computational complexity with maintained or improved model performance.
    
    The algorithm supports two neighbor selection criteria:
        1. Spatial: Traditional k-nearest neighbors based on Euclidean distance
        2. Residual: Neighbors selected based on similarity of prediction residuals
    
    Attributes:
        config (SVRConfig): Configuration object containing all algorithm parameters
        training_time (float): Total time spent in training
        iterations (int): Number of iterations performed in refinement process
    
    Methods:
        train: Main method to train the SVR model using the subsample algorithm
    
    Example:
        >>> config = SVRConfig(subsample_fraction=0.01, num_neighbors=5)
        >>> optimizer = SVRSubsampleOptimizer(config)
        >>> model, info = optimizer.train(train_df, test_df, 'target', 'rbf', kernel_params)
    """
    
    def __init__(self, config: SVRConfig = SVRConfig()):
        """
        Initialize the SVR subsample optimizer.
        
        Args:
            config (SVRConfig): Configuration object with algorithm parameters.
                               If not provided, uses default configuration.
        """
        self.config = config
        self.training_time = 0
        self.iterations = 0
        
    def _calculate_residuals(self, model: SVR, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Calculate residuals for each data point using the SVR model.
        
        The residual is computed as: r_i = |f(X_i) - y_i| - epsilon
        where f(X_i) is the model prediction and epsilon is the SVR's epsilon parameter.
        
        Args:
            model (SVR): Trained SVR model
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            np.ndarray: Array of residuals for each data point
        """
        predictions = model.predict(X)
        epsilon = model.epsilon
        residuals = np.abs(predictions - y.values) - epsilon
        return residuals
    
    def _find_residual_based_neighbors(
        self, 
        model: SVR,
        support_vectors_df: pd.DataFrame,
        candidate_set: pd.DataFrame,
        X_column_names: List[str],
        y_column_name: str,
        k: int
    ) -> List[int]:
        """
        Find k neighbors based on residual similarity instead of spatial distance.
        
        This method implements the professor's proposed approach where neighbors are selected
        based on having similar prediction residuals rather than being spatially close.
        For each support vector, it finds k points from the candidate set with the most
        similar residual values.
        
        Args:
            model (SVR): Trained SVR model used to calculate residuals
            support_vectors_df (pd.DataFrame): DataFrame containing support vectors
            candidate_set (pd.DataFrame): DataFrame of candidate points to search for neighbors
            X_column_names (List[str]): Names of feature columns
            y_column_name (str): Name of target column
            k (int): Number of neighbors to find per support vector
            
        Returns:
            List[int]: Indices of selected neighbors (duplicates removed)
        """
        # Calculate residuals for support vectors
        sv_residuals = self._calculate_residuals(
            model, 
            support_vectors_df[X_column_names], 
            support_vectors_df[y_column_name]
        )
        
        # Calculate residuals for candidate set
        candidate_residuals = self._calculate_residuals(
            model,
            candidate_set[X_column_names],
            candidate_set[y_column_name]
        )
        
        neighbor_indices = []
        
        # For each support vector, find k points with most similar residuals
        for sv_residual in sv_residuals:
            # Calculate absolute differences in residuals
            residual_differences = np.abs(candidate_residuals - sv_residual)
            
            # Get indices of k smallest differences
            k_nearest_indices = np.argpartition(residual_differences, min(k, len(residual_differences)-1))[:k]
            neighbor_indices.extend(k_nearest_indices.tolist())
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(neighbor_indices))
    
    def _find_spatial_neighbors(
        self,
        support_vectors_df: pd.DataFrame,
        candidate_set: pd.DataFrame,
        X_column_names: List[str],
        k: int
    ) -> List[int]:
        """
        Find k nearest neighbors based on spatial (Euclidean) distance.
        
        This is the original neighbor selection method that finds the k spatially
        closest points to each support vector.
        
        Args:
            support_vectors_df (pd.DataFrame): DataFrame containing support vectors
            candidate_set (pd.DataFrame): DataFrame of candidate points to search for neighbors
            X_column_names (List[str]): Names of feature columns
            k (int): Number of neighbors to find per support vector
            
        Returns:
            List[int]: Indices of selected neighbors (duplicates removed)
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nbrs.fit(candidate_set[X_column_names].values)
        
        distances, indices = nbrs.kneighbors(
            support_vectors_df[X_column_names].values,
            return_distance=True
        )
        
        # Flatten and remove duplicates
        neighbor_indices = [item for sublist in indices for item in sublist]
        return list(dict.fromkeys(neighbor_indices))
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        kernel_params: Dict
    ) -> BayesSearchCV:
        """
        Train SVR model using Bayesian optimization for hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            kernel_params (Dict): Hyperparameter search space for Bayesian optimization
            
        Returns:
            BayesSearchCV: Trained model with optimized hyperparameters
        """
        svr = BayesSearchCV(
            SVR(),
            kernel_params,
            cv=self.config.cv_folds,
            n_jobs=-1,
            n_points=self.config.bayes_n_points,
            n_iter=self.config.bayes_n_iter,
            verbose=0  # Reduce verbosity, use logging instead
        )
        
        start_time = time.time()
        svr.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
        logger.info(f"Best parameters: {svr.best_params_}")
        
        return svr
    
    def _evaluate_model(
        self,
        model: BayesSearchCV,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Computes multiple regression metrics including R² score, MAE, and RMSE.
        
        Args:
            model (BayesSearchCV): Trained model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target values
            
        Returns:
            Dict[str, float]: Dictionary containing performance metrics:
                - r2_score: Coefficient of determination
                - mae: Mean Absolute Error
                - rmse: Root Mean Squared Error
        """
        score = model.score(X_test, y_test)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        metrics = {
            'r2_score': score,
            'mae': mae,
            'rmse': rmse
        }
        
        logger.info(f"Model performance - R²: {score:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return metrics
    
    def train(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        y_label_name: str,
        kernel_type: str,
        kernel_params: Dict,
        original_support_vectors: Optional[np.ndarray] = None,
        original_sv_indices: Optional[List[int]] = None
    ) -> Tuple[BayesSearchCV, Dict[str, any]]:
        """
        Main training method implementing the subsample algorithm for SVR.
        
        This method implements the complete training pipeline:
        1. Select initial random subsample
        2. Train initial SVR model
        3. Identify support vectors
        4. Find neighbors (using residual or spatial criterion)
        5. Create new subsample and retrain
        6. Iterate if improvement is significant
        
        Args:
            train_data (pd.DataFrame): Complete training dataset including features and target
            test_data (pd.DataFrame): Test dataset for evaluation
            y_label_name (str): Name of the target column in the dataframes
            kernel_type (str): Type of kernel to use ('linear', 'rbf', 'poly')
            kernel_params (Dict): Hyperparameter search spaces for each kernel type
            original_support_vectors (Optional[np.ndarray]): Support vectors from original model for comparison
            original_sv_indices (Optional[List[int]]): Indices of original support vectors
            
        Returns:
            Tuple[BayesSearchCV, Dict[str, any]]: Tuple containing:
                - best_model: The optimized SVR model
                - training_info: Dictionary with training statistics and metrics
                
        Example:
            >>> params = {'rbf': {'C': (1e-3, 1e3, 'log-uniform'), 'gamma': (1e-6, 1e1, 'log-uniform')}}
            >>> model, info = optimizer.train(train_df, test_df, 'target', 'rbf', params)
        """
        logger.info(f"Starting SVR subsample training with {'residual' if self.config.use_residual_criterion else 'spatial'} criterion")
        start_time = time.time()
        
        # Get feature column names
        X_columns = [col for col in train_data.columns if col != y_label_name]
        
        # Step 1: Initial random subsample
        subsample_T0 = train_data.sample(
            frac=self.config.subsample_fraction,
            random_state=self.config.random_state
        )
        logger.info(f"Initial subsample size: {subsample_T0.shape}")
        
        # Step 2: Initialize S(0) = D \ T(0)
        set_S = train_data.drop(subsample_T0.index)
        
        # Step 3: Train initial model
        logger.info("Training initial model...")
        model_0 = self._train_model(
            subsample_T0[X_columns],
            subsample_T0[y_label_name],
            kernel_params[kernel_type]
        )
        
        metrics_0 = self._evaluate_model(
            model_0,
            test_data[X_columns],
            test_data[y_label_name]
        )
        
        # Step 4: Get support vectors
        support_indices = model_0.best_estimator_.support_
        support_vectors_df = subsample_T0.iloc[support_indices]
        
        # Step 5: Find neighbors (using residual or spatial criterion)
        if self.config.use_residual_criterion:
            logger.info("Using residual-based neighbor selection")
            neighbor_indices = self._find_residual_based_neighbors(
                model_0.best_estimator_,
                support_vectors_df,
                set_S,
                X_columns,
                y_label_name,
                self.config.num_neighbors
            )
        else:
            logger.info("Using spatial-based neighbor selection")
            neighbor_indices = self._find_spatial_neighbors(
                support_vectors_df,
                set_S,
                X_columns,
                self.config.num_neighbors
            )
        
        neighbors_df = set_S.iloc[neighbor_indices]
        
        # Step 6: Create R set and new subsample
        remaining_S = set_S.drop(neighbors_df.index)
        set_R = remaining_S.sample(
            frac=self.config.r_set_fraction,
            random_state=self.config.random_state
        )
        
        subsample_T1 = pd.concat([neighbors_df, support_vectors_df, set_R], axis=0)
        subsample_T1 = shuffle(subsample_T1, random_state=self.config.random_state)
        
        # Update S for next iteration
        set_S1 = set_S.drop(list(set_R.index) + list(neighbors_df.index))
        
        # Step 7: Train model on new subsample
        logger.info(f"Training model 1 with subsample size: {subsample_T1.shape}")
        model_1 = self._train_model(
            subsample_T1[X_columns],
            subsample_T1[y_label_name],
            kernel_params[kernel_type]
        )
        
        metrics_1 = self._evaluate_model(
            model_1,
            test_data[X_columns],
            test_data[y_label_name]
        )
        
        # Check improvement
        improvement = metrics_0['r2_score'] - metrics_1['r2_score']
        logger.info(f"R² improvement: {improvement:.4f}")
        
        # Iterative refinement if needed
        if improvement >= self.config.convergence_threshold:
            logger.info("Starting iterative refinement...")
            best_model = self._iterative_refinement(
                model_1,
                subsample_T1,
                set_R,
                set_S1,
                test_data,
                X_columns,
                y_label_name,
                kernel_params[kernel_type],
                metrics_1['r2_score']
            )
        else:
            best_model = model_1
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        training_info = {
            'total_time': total_time,
            'iterations': self.iterations,
            'final_subsample_size': subsample_T1.shape[0] if improvement < self.config.convergence_threshold else None,
            'initial_metrics': metrics_0,
            'final_metrics': self._evaluate_model(best_model, test_data[X_columns], test_data[y_label_name])
        }
        
        # Compare with original support vectors if provided
        if original_support_vectors is not None:
            self._compare_with_original(
                best_model.best_estimator_,
                original_support_vectors,
                original_sv_indices,
                train_data,
                y_label_name
            )
        
        logger.info(f"Training completed in {total_time:.2f} seconds after {self.iterations} iterations")
        
        return best_model, training_info
    
    def _iterative_refinement(
        self,
        current_model: BayesSearchCV,
        current_subsample: pd.DataFrame,
        previous_R: pd.DataFrame,
        current_S: pd.DataFrame,
        test_data: pd.DataFrame,
        X_columns: List[str],
        y_label_name: str,
        kernel_params: Dict,
        previous_score: float
    ) -> BayesSearchCV:
        """
        Perform iterative refinement of the model by repeatedly updating the subsample.
        
        This method continues to refine the model by:
        1. Finding new support vectors in the R set
        2. Finding their neighbors
        3. Creating new subsample
        4. Retraining and evaluating
        5. Stopping when improvement is below threshold
        
        Args:
            current_model (BayesSearchCV): Current best model
            current_subsample (pd.DataFrame): Current training subsample
            previous_R (pd.DataFrame): Previous R set
            current_S (pd.DataFrame): Current remaining data set S
            test_data (pd.DataFrame): Test data for evaluation
            X_columns (List[str]): Feature column names
            y_label_name (str): Target column name
            kernel_params (Dict): Hyperparameter search space
            previous_score (float): Previous iteration's R² score
            
        Returns:
            BayesSearchCV: Final optimized model after refinement
        """
        iteration = 0
        
        while True:
            iteration += 1
            logger.info(f"Iteration {iteration}")
            
            # Get new support vectors from R set
            support_indices = current_model.best_estimator_.support_
            support_vectors_df = current_subsample.iloc[support_indices]
            new_support_vectors = pd.merge(support_vectors_df, previous_R, how='inner')
            
            if new_support_vectors.empty:
                logger.warning("No new support vectors found in R set")
                break
            
            # Find neighbors for new support vectors
            if self.config.use_residual_criterion:
                neighbor_indices = self._find_residual_based_neighbors(
                    current_model.best_estimator_,
                    new_support_vectors,
                    current_S,
                    X_columns,
                    y_label_name,
                    self.config.num_neighbors
                )
            else:
                neighbor_indices = self._find_spatial_neighbors(
                    new_support_vectors,
                    current_S,
                    X_columns,
                    self.config.num_neighbors
                )
            
            if not neighbor_indices:
                logger.warning("No neighbors found")
                break
                
            neighbors_df = current_S.iloc[neighbor_indices]
            
            # Create new R set
            remaining_S = current_S.drop(neighbors_df.index)
            if remaining_S.empty:
                logger.warning("No more data in S set")
                break
                
            new_R = remaining_S.sample(
                frac=min(self.config.r_set_fraction, 1.0),
                random_state=self.config.random_state
            )
            
            # Create new subsample
            new_subsample = pd.concat([neighbors_df, new_support_vectors, new_R], axis=0)
            new_subsample = shuffle(new_subsample, random_state=self.config.random_state)
            
            # Update sets
            current_S = current_S.drop(list(new_R.index) + list(neighbors_df.index))
            
            # Train new model
            new_model = self._train_model(
                new_subsample[X_columns],
                new_subsample[y_label_name],
                kernel_params
            )
            
            # Evaluate
            metrics = self._evaluate_model(
                new_model,
                test_data[X_columns],
                test_data[y_label_name]
            )
            
            # Check improvement
            improvement = previous_score - metrics['r2_score']
            logger.info(f"Iteration {iteration} - R² improvement: {improvement:.4f}")
            
            if improvement < self.config.iteration_threshold:
                logger.info("Convergence reached")
                self.iterations = iteration
                return new_model
            
            # Update for next iteration
            current_model = new_model
            current_subsample = new_subsample
            previous_R = new_R
            previous_score = metrics['r2_score']
            
            # Safety check
            if iteration >= 10 or current_S.shape[0] < 10:
                logger.warning("Maximum iterations reached or insufficient data")
                self.iterations = iteration
                return current_model
    
    def _compare_with_original(
        self,
        model: SVR,
        original_sv: np.ndarray,
        original_indices: List[int],
        train_data: pd.DataFrame,
        y_label_name: str
    ):
        """
        Compare current support vectors with original ones.
        
        This method analyzes how many of the original support vectors are retained
        in the final model and examines the characteristics of new support vectors.
        
        Args:
            model (SVR): Trained SVR model
            original_sv (np.ndarray): Original support vectors for comparison
            original_indices (List[int]): Indices of original support vectors
            train_data (pd.DataFrame): Complete training dataset
            y_label_name (str): Name of target column
        """
        current_sv = model.support_vectors_
        
        # Find intersection
        intersection = self._multidim_intersect(original_sv, current_sv)
        ratio = len(intersection) / len(original_sv) * 100
        
        logger.info(f"Percentage of original support vectors retained: {ratio:.2f}%")
        
        # Find different support vectors
        diff_sv = self._multidim_diff(current_sv, original_sv)
        
        if diff_sv.shape[0] > 0:
            self._analyze_new_support_vectors(
                diff_sv,
                train_data,
                y_label_name,
                original_indices
            )
    
    def _multidim_intersect(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """
        Find intersection of two multidimensional arrays.
        
        This method finds common rows between two 2D arrays, useful for comparing
        support vector sets.
        
        Args:
            arr1 (np.ndarray): First array
            arr2 (np.ndarray): Second array
            
        Returns:
            np.ndarray: Array containing rows present in both input arrays
        """
        arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
        arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
        intersected = np.intersect1d(arr1_view, arr2_view)
        return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
    
    def _multidim_diff(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """
        Find difference between two multidimensional arrays.
        
        This method finds rows present in arr1 but not in arr2.
        
        Args:
            arr1 (np.ndarray): First array
            arr2 (np.ndarray): Second array
            
        Returns:
            np.ndarray: Array containing rows in arr1 that are not in arr2
        """
        arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
        arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
        diff = np.setdiff1d(arr1_view, arr2_view)
        return diff.view(arr1.dtype).reshape(-1, arr1.shape[1])
    
    def _analyze_new_support_vectors(
        self,
        new_sv: np.ndarray,
        train_data: pd.DataFrame,
        y_label_name: str,
        original_indices: List[int]
    ):
        """
        Analyze characteristics of new support vectors not present in original model.
        
        This method examines how the new support vectors relate to the original ones
        by checking if their nearest neighbors overlap with original support vectors.
        
        Args:
            new_sv (np.ndarray): New support vectors not in original set
            train_data (pd.DataFrame): Complete training dataset
            y_label_name (str): Name of target column
            original_indices (List[int]): Indices of original support vectors
        """
        X_columns = [col for col in train_data.columns if col != y_label_name]
        
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto')
        nbrs.fit(train_data[X_columns].values)
        
        _, indices = nbrs.kneighbors(new_sv)
        original_sv_df = train_data.iloc[original_indices]
        
        overlap_percentages = []
        for neighbor_indices in indices:
            neighbors_df = train_data.iloc[neighbor_indices]
            overlap = pd.merge(original_sv_df, neighbors_df, how='inner')
            overlap_percentages.append(len(overlap) / 5 * 100)
        
        avg_overlap = np.mean(overlap_percentages)
        logger.info(f"Average overlap of new support vectors' neighbors with original SVs: {avg_overlap:.2f}%")


# Example usage function
def example_usage():
    """
    Example of how to use the improved SVR subsample optimizer.
    
    This function demonstrates the typical workflow for using the SVRSubsampleOptimizer
    including configuration setup, parameter definition, and model training.
    
    Returns:
        SVRSubsampleOptimizer: Configured optimizer instance
        
    Note:
        This is a demonstration function. In practice, you would load your actual
        data and adjust parameters according to your specific needs.
    """
    
    # Configuration
    config = SVRConfig(
        subsample_fraction=0.01,
        r_set_fraction=0.1,
        num_neighbors=5,
        use_residual_criterion=True,  # Set to True for professor's requirement
        convergence_threshold=0.02
    )
    
    # Kernel parameters for Bayesian optimization
    kernel_params = {
        'rbf': {
            'C': (1e-3, 1e3, 'log-uniform'),
            'gamma': (1e-6, 1e1, 'log-uniform'),
            'epsilon': (1e-3, 1e1, 'log-uniform')
        },
        'linear': {
            'C': (1e-3, 1e3, 'log-uniform'),
            'epsilon': (1e-3, 1e1, 'log-uniform')
        },
        'poly': {
            'C': (1e-3, 1e3, 'log-uniform'),
            'gamma': (1e-6, 1e1, 'log-uniform'),
            'epsilon': (1e-3, 1e1, 'log-uniform'),
            'degree': (2, 5)
        }
    }
    
    # Create optimizer
    optimizer = SVRSubsampleOptimizer(config)
    
    # Assuming you have train_data, test_data loaded
    # best_model, info = optimizer.train(
    #     train_data=train_df,
    #     test_data=test_df,
    #     y_label_name='target',
    #     kernel_type='rbf',
    #     kernel_params=kernel_params
    # )
    
    return optimizer


if __name__ == "__main__":
    # Example of how to use the code
    optimizer = example_usage()
    print("SVR Subsample Optimizer initialized successfully!")