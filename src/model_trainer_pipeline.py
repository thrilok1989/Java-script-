"""
Model Training Pipeline
Trains XGBoost model on real trading data collected from TrainingDataCollector
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainerPipeline:
    """
    Complete pipeline for training XGBoost models on real trading data

    Features:
    - Loads data from TrainingDataCollector
    - Preprocesses and cleans data
    - Trains multiple models with hyperparameter tuning
    - Evaluates performance
    - Saves best model
    - Tracks model versions
    """

    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        """Initialize Model Trainer Pipeline"""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.training_file = os.path.join(data_dir, "training_data.csv")

        # Create directories
        os.makedirs(model_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.model_metadata = {}

    def load_training_data(self, min_samples: int = 50) -> pd.DataFrame:
        """
        Load training data

        Args:
            min_samples: Minimum number of samples required

        Returns:
            DataFrame with training data
        """
        if not os.path.exists(self.training_file):
            raise FileNotFoundError(f"Training data file not found: {self.training_file}")

        df = pd.read_csv(self.training_file)

        if len(df) < min_samples:
            raise ValueError(f"Insufficient training data. Need at least {min_samples} samples, found {len(df)}")

        logger.info(f"📊 Loaded {len(df)} training samples")

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training

        Args:
            df: Raw training data

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Define outcome column
        target_col = 'actual_direction'

        # Define feature columns (exclude timestamp and outcome columns)
        exclude_cols = ['timestamp', 'actual_direction', 'actual_move_1h', 'actual_move_1d', 'profitable', 'pnl_percent']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle missing values
        df = df.fillna(0)

        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values

        self.feature_names = feature_cols

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"✅ Prepared data: Train={len(X_train)}, Test={len(X_test)}, Features={len(feature_cols)}")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparameter_tuning: bool = False
    ) -> Dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results
        """
        logger.info("🤖 Training XGBoost model...")

        if hyperparameter_tuning:
            logger.info("🔍 Performing hyperparameter tuning...")
            self.model = self._hyperparameter_tuning(X_train, y_train)
        else:
            # Use default optimized parameters
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,  # BUY, SELL, HOLD
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            }

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': self._get_feature_importance()
        }

        logger.info(f"✅ Training complete!")
        logger.info(f"   Train Accuracy: {train_score:.4f}")
        logger.info(f"   Test Accuracy: {test_score:.4f}")
        logger.info(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return results

    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Best XGBoost model
        """
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }

        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"🏆 Best parameters: {grid_search.best_params_}")
        logger.info(f"🏆 Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}

        importance_values = self.model.feature_importances_
        importance_dict = {}

        # Sort by importance
        sorted_indices = np.argsort(importance_values)[::-1]

        for idx in sorted_indices:
            feature_name = self.feature_names[idx]
            importance_dict[feature_name] = float(importance_values[idx])

        return importance_dict

    def save_model(self, model_name: str = None, results: Dict = None):
        """
        Save trained model and metadata

        Args:
            model_name: Optional custom model name
            results: Training results to save as metadata
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if model_name is None:
            model_name = f"xgboost_model_{timestamp}"

        # Save model
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        joblib.dump(self.model, model_path)

        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

        # Save feature names
        feature_path = os.path.join(self.model_dir, f"{model_name}_features.json")
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'results': results
        }

        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update latest model symlink
        latest_model_path = os.path.join(self.model_dir, "latest_model.pkl")
        latest_scaler_path = os.path.join(self.model_dir, "latest_scaler.pkl")
        latest_features_path = os.path.join(self.model_dir, "latest_features.json")

        # Copy to latest (overwrite)
        joblib.dump(self.model, latest_model_path)
        joblib.dump(self.scaler, latest_scaler_path)
        with open(latest_features_path, 'w') as f:
            json.dump(self.feature_names, f)

        logger.info(f"💾 Model saved:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Scaler: {scaler_path}")
        logger.info(f"   Features: {feature_path}")
        logger.info(f"   Metadata: {metadata_path}")

        return model_path

    def load_model(self, model_name: str = "latest"):
        """
        Load a saved model

        Args:
            model_name: Name of model to load (default: "latest")
        """
        if model_name == "latest":
            model_path = os.path.join(self.model_dir, "latest_model.pkl")
            scaler_path = os.path.join(self.model_dir, "latest_scaler.pkl")
            feature_path = os.path.join(self.model_dir, "latest_features.json")
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            feature_path = os.path.join(self.model_dir, f"{model_name}_features.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)

        logger.info(f"✅ Model loaded: {model_path}")

        return self.model

    def evaluate_on_new_data(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate loaded model on new data

        Args:
            df: New data to evaluate

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")

        # Prepare data
        target_col = 'actual_direction'
        exclude_cols = ['timestamp', 'actual_direction', 'actual_move_1h', 'actual_move_1d', 'profitable', 'pnl_percent']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred.tolist(),
            'actual': y.tolist()
        }


def run_training_pipeline(
    data_dir: str = "data",
    model_dir: str = "models",
    hyperparameter_tuning: bool = False,
    min_samples: int = 50
):
    """
    Complete end-to-end training pipeline

    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save models
        hyperparameter_tuning: Whether to tune hyperparameters
        min_samples: Minimum samples required

    Returns:
        Training results
    """
    print("="*60)
    print("🤖 XGBoost Model Training Pipeline")
    print("="*60)

    # Initialize
    trainer = ModelTrainerPipeline(data_dir, model_dir)

    # Load data
    print("\n📊 Loading training data...")
    try:
        df = trainer.load_training_data(min_samples=min_samples)
        print(f"   ✅ Loaded {len(df)} samples")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    # Prepare data
    print("\n🔧 Preparing features...")
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(df)

    # Train
    print(f"\n🤖 Training model...")
    results = trainer.train_model(X_train, y_train, X_test, y_test, hyperparameter_tuning)

    # Display results
    print("\n" + "="*60)
    print("📊 Training Results")
    print("="*60)
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"CV Score:       {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")

    print("\n📈 Top 10 Important Features:")
    importance = results['feature_importance']
    for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"   {i}. {feat}: {imp:.4f}")

    # Save
    print("\n💾 Saving model...")
    model_path = trainer.save_model(results=results)
    print(f"   ✅ Saved to: {model_path}")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)

    return results


if __name__ == "__main__":
    # Run training pipeline
    run_training_pipeline(
        hyperparameter_tuning=False,  # Set to True for better results (slower)
        min_samples=50
    )
