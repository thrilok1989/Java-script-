"""
Enhanced XGBoost ML Analyzer with Real Training Support
Integrates with TrainingDataCollector and ModelTrainerPipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import os
import joblib
import json

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.training_data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


@dataclass
class MLPredictionResult:
    """ML Prediction Result"""
    prediction: str  # "BUY", "SELL", "HOLD"
    probability: float  # 0-1
    confidence: float  # 0-100
    feature_importance: Dict[str, float]
    all_probabilities: Dict[str, float]  # Probabilities for all classes
    expected_return: float  # Expected % return
    risk_score: float  # 0-100
    recommendation: str
    model_version: str
    using_real_model: bool  # NEW: Indicates if using trained model or simulated


class XGBoostMLAnalyzerEnhanced:
    """
    Enhanced XGBoost ML Analyzer with Real Training Support

    New Features:
    - Automatically loads pre-trained models from models/ directory
    - Integrates with TrainingDataCollector to log predictions
    - Falls back to simulated training if no model exists (first-time users)
    - Supports model versioning and updates
    """

    def __init__(self, model_dir: str = "models", data_dir: str = "data", auto_load_model: bool = True):
        """
        Initialize Enhanced XGBoost ML Analyzer

        Args:
            model_dir: Directory containing saved models
            data_dir: Directory for training data
            auto_load_model: Automatically load latest model if available
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost scikit-learn")

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.using_real_model = False
        self.model_version = "simulated_v1.0"

        # Initialize training data collector
        try:
            self.training_collector = TrainingDataCollector(data_dir=data_dir)
        except Exception as e:
            logger.warning(f"Could not initialize TrainingDataCollector: {e}")
            self.training_collector = None

        # XGBoost parameters (optimized for trading)
        self.params = {
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
            'n_jobs': -1
        }

        # Try to load pre-trained model
        if auto_load_model:
            self._auto_load_model()

    def _auto_load_model(self):
        """Automatically load latest trained model if available"""
        latest_model_path = os.path.join(self.model_dir, "latest_model.pkl")

        if os.path.exists(latest_model_path):
            try:
                self.load_trained_model()
                logger.info(f"✅ Loaded pre-trained model from {latest_model_path}")
                logger.info(f"   Model trained on YOUR real trading data!")
                logger.info(f"   Features: {len(self.feature_names)}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
                logger.info("Will use simulated training instead")
        else:
            logger.info(f"ℹ️ No pre-trained model found at {latest_model_path}")
            logger.info("   Will use simulated training (for first-time users)")
            logger.info("   Collect 50+ trade outcomes, then run training pipeline to create real model")

    def load_trained_model(self, model_name: str = "latest"):
        """
        Load a pre-trained model

        Args:
            model_name: Name of model to load (default: "latest")
        """
        if model_name == "latest":
            model_path = os.path.join(self.model_dir, "latest_model.pkl")
            scaler_path = os.path.join(self.model_dir, "latest_scaler.pkl")
            feature_path = os.path.join(self.model_dir, "latest_features.json")
            metadata_path = os.path.join(self.model_dir, "latest_metadata.json")
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            feature_path = os.path.join(self.model_dir, f"{model_name}_features.json")
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        # Load model
        self.model = joblib.load(model_path)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load feature names
        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)

        # Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('model_name', 'unknown')

        self.is_trained = True
        self.using_real_model = True

        return True

    def predict(
        self,
        features_df: pd.DataFrame,
        log_prediction: bool = True,
        current_market_data: Optional[Dict] = None
    ) -> MLPredictionResult:
        """
        Make prediction using XGBoost model

        Args:
            features_df: DataFrame with extracted features
            log_prediction: Whether to log prediction for future training
            current_market_data: Current market data for logging

        Returns:
            MLPredictionResult with prediction and probabilities
        """
        # Ensure model is trained
        if not self.is_trained:
            if self.using_real_model:
                # Try to load again
                try:
                    self.load_trained_model()
                except:
                    pass

            if not self.is_trained:
                # Fall back to simulated training
                logger.info("No model loaded. Training with simulated data...")
                self.train_model_with_simulated_data()

        # Prepare features
        if self.using_real_model:
            # Match features to trained model
            missing_features = set(self.feature_names) - set(features_df.columns)
            for feat in missing_features:
                features_df[feat] = 0

            # Reorder to match training
            features_df = features_df[self.feature_names]

            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=self.feature_names)
        else:
            # Simulated model - use as-is
            # Ensure feature count matches
            if len(features_df.columns) > 50:
                features_df = features_df.iloc[:, :50]
            elif len(features_df.columns) < 50:
                # Pad with zeros
                for i in range(len(features_df.columns), 50):
                    features_df[f'feature_{i}'] = 0

        # Make prediction
        y_pred = self.model.predict(features_df)[0]
        y_proba = self.model.predict_proba(features_df)[0]

        # Map prediction to label
        label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        prediction = label_map[y_pred]
        probability = y_proba[y_pred]

        # Get all probabilities
        all_probs = {
            "BUY": float(y_proba[0]),
            "SELL": float(y_proba[1]),
            "HOLD": float(y_proba[2])
        }

        # Calculate confidence (0-100)
        confidence = probability * 100

        # Calculate expected return (based on prediction probabilities)
        expected_return = (y_proba[0] * 2.0) + (y_proba[1] * -2.0) + (y_proba[2] * 0.0)

        # Calculate risk score
        risk_score = (1 - probability) * 100

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            # Get top 10 important features
            if len(importance_values) <= len(self.feature_names):
                top_indices = np.argsort(importance_values)[-10:]
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        feature_importance[self.feature_names[idx]] = float(importance_values[idx])

        # Generate recommendation
        recommendation = self._generate_ml_recommendation(
            prediction, confidence, expected_return, risk_score
        )

        # Create result
        result = MLPredictionResult(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            feature_importance=feature_importance,
            all_probabilities=all_probs,
            expected_return=expected_return,
            risk_score=risk_score,
            recommendation=recommendation,
            model_version=self.model_version,
            using_real_model=self.using_real_model
        )

        # Log prediction for future training
        if log_prediction and self.training_collector and current_market_data:
            try:
                prediction_id = self.training_collector.record_prediction(
                    prediction_data=current_market_data,
                    feature_values=features_df.iloc[0].to_dict() if len(features_df) > 0 else {},
                    ml_result=result
                )
                logger.info(f"📝 Logged prediction: {prediction_id}")
            except Exception as e:
                logger.warning(f"Failed to log prediction: {e}")

        return result

    def _generate_ml_recommendation(
        self,
        prediction: str,
        confidence: float,
        expected_return: float,
        risk_score: float
    ) -> str:
        """Generate trading recommendation from ML prediction"""
        model_status = "🎯 REAL MODEL" if self.using_real_model else "🤖 SIMULATED"

        if prediction == "BUY":
            if confidence > 80:
                return f"{model_status} 🚀 STRONG BUY - High confidence ({confidence:.1f}%), Expected: +{expected_return:.2f}%"
            elif confidence > 65:
                return f"{model_status} ✅ BUY - Good confidence ({confidence:.1f}%)"
            else:
                return f"{model_status} ⚠️ WEAK BUY - Low confidence ({confidence:.1f}%), be cautious"

        elif prediction == "SELL":
            if confidence > 80:
                return f"{model_status} 🔻 STRONG SELL - High confidence ({confidence:.1f}%), Expected: {expected_return:.2f}%"
            elif confidence > 65:
                return f"{model_status} ❌ SELL - Good confidence ({confidence:.1f}%)"
            else:
                return f"{model_status} ⚠️ WEAK SELL - Low confidence ({confidence:.1f}%), be cautious"

        else:  # HOLD
            return f"{model_status} ⏸️ HOLD - Stay in cash ({confidence:.1f}% confidence), Wait for better setup"

    def train_model_with_simulated_data(self, n_samples: int = 1000):
        """
        Train XGBoost model with simulated training data
        (Fallback for when no real trained model exists)

        In production, replace this by running the training pipeline:
        python -m src.model_trainer_pipeline
        """
        logger.info("⚠️ Using SIMULATED training data (no real model found)")
        logger.info("   To use REAL model: Collect 50+ trade outcomes and run training pipeline")

        # Generate random features (simulate historical data)
        np.random.seed(42)

        n_features = 50
        X = np.random.randn(n_samples, n_features)

        # Generate labels based on feature combinations (simulate profitable patterns)
        buy_score = (
            X[:, 0] +  # Price momentum
            X[:, 10] +  # Institutional confidence
            X[:, 20] -  # Inverse trap probability
            X[:, 5]    # Volatility factor
        )

        sell_score = -(
            X[:, 0] +
            X[:, 11] +  # Retail activity
            X[:, 21]    # Trap detection
        )

        # Create labels
        y = np.zeros(n_samples)
        y[buy_score > 1.5] = 0  # BUY
        y[sell_score > 1.5] = 1  # SELL
        y[(buy_score <= 1.5) & (sell_score <= 1.5)] = 2  # HOLD

        # Train XGBoost
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)

        # Save feature names
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        self.is_trained = True
        self.using_real_model = False
        self.model_version = "simulated_v1.0"

        logger.info("✅ Simulated model training complete!")
        logger.info("   This is NOT trained on your real data.")
        logger.info("   For best results: Run python -m src.model_trainer_pipeline after collecting data")

        return self.model

    def get_training_stats(self) -> Dict:
        """Get statistics about training data collection"""
        if self.training_collector:
            return self.training_collector.get_stats()
        else:
            return {
                'total_samples': 0,
                'message': 'TrainingDataCollector not initialized'
            }

    def record_outcome(self, prediction_id: str, outcome: Dict) -> bool:
        """
        Record the outcome of a prediction for future training

        Args:
            prediction_id: ID from the prediction log
            outcome: Dict with keys: actual_direction, profitable, pnl_percent, etc.

        Returns:
            True if successful
        """
        if self.training_collector:
            # Get feature values from prediction log
            log_df = self.training_collector.get_prediction_log()
            pred_row = log_df[log_df['prediction_id'] == prediction_id]

            if len(pred_row) == 0:
                logger.error(f"Prediction ID not found: {prediction_id}")
                return False

            # Extract feature values (you'd need to store these with the prediction)
            # For now, we'll create a placeholder
            feature_values = {}  # This should be loaded from somewhere

            return self.training_collector.record_outcome(
                prediction_id=prediction_id,
                feature_values=feature_values,
                outcome=outcome
            )
        else:
            logger.warning("TrainingDataCollector not available")
            return False


# Maintain backward compatibility - alias to original class name
XGBoostMLAnalyzer = XGBoostMLAnalyzerEnhanced
