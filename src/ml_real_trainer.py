"""
ML Real Trainer - Train on ACTUAL Trade Data
=============================================
Trains XGBoost model on real recorded signals with outcomes

Features:
1. Trains on actual historical trades (not simulated)
2. Uses cross-validation for reliability
3. Calculates feature importance
4. Provides model performance metrics
5. Auto-retrains when enough new data available
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import pickle

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = "ml_models"
MODEL_FILE = "xgboost_real_model.pkl"
SCALER_FILE = "feature_scaler.pkl"
METRICS_FILE = "model_metrics.json"


@dataclass
class TrainingResult:
    """Result of model training"""
    success: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_score: float
    cv_std: float
    feature_importance: Dict[str, float]
    training_samples: int
    model_version: str
    trained_at: str
    message: str


@dataclass
class ModelMetrics:
    """Stored model metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_score: float
    training_samples: int
    last_trained: str
    version: str


class MLRealTrainer:
    """
    Trains ML model on REAL historical trade data
    """

    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not installed. Run: pip install xgboost scikit-learn")

        self.model_dir = MODEL_DIR
        self.model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        self.scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        self.metrics_path = os.path.join(MODEL_DIR, METRICS_FILE)

        os.makedirs(self.model_dir, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_version = "v2.0_real_data"

        # Feature columns for training
        self.feature_columns = [
            # Core scores
            'regime_score', 'volatility_score', 'oi_trap_score',
            'cvd_score', 'liquidity_score', 'institutional_score', 'spike_score',

            # Diamond features
            'cvd_diamond_score', 'gamma_flip_score',
            'black_order_score', 'block_trade_score',

            # Option chain
            'pcr', 'atm_iv_ce', 'atm_iv_pe',

            # Technical
            'rsi', 'macd_signal', 'vwap_position', 'atr',

            # Market state
            'vix', 'time_to_expiry_days',

            # Confidence
            'confidence'
        ]

        # XGBoost parameters (optimized)
        self.xgb_params = {
            'objective': 'binary:logistic',  # WIN vs LOSS
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

        # Load existing model if available
        self._load_model()

    def _load_model(self):
        """Load previously trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded existing trained model")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        """Save trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Could not save model: {e}")

    def _save_metrics(self, metrics: ModelMetrics):
        """Save model metrics"""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump({
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1,
                    'cv_score': metrics.cv_score,
                    'training_samples': metrics.training_samples,
                    'last_trained': metrics.last_trained,
                    'version': metrics.version
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from recorded signals DataFrame

        Args:
            df: DataFrame with recorded signals

        Returns:
            (X, y) - Features and labels
        """
        # Filter only WIN and LOSS (exclude BREAKEVEN for clearer signal)
        df_filtered = df[df['outcome'].isin(['WIN', 'LOSS'])].copy()

        if len(df_filtered) < 10:
            raise ValueError(f"Not enough training samples. Need at least 10, got {len(df_filtered)}")

        # Select features
        available_features = [c for c in self.feature_columns if c in df_filtered.columns]
        self.feature_names = available_features

        X = df_filtered[available_features].fillna(0).values

        # Labels: 1 = WIN, 0 = LOSS
        y = (df_filtered['outcome'] == 'WIN').astype(int).values

        return X, y

    def train(self, training_df: pd.DataFrame) -> TrainingResult:
        """
        Train XGBoost model on real data

        Args:
            training_df: DataFrame with recorded signals and outcomes

        Returns:
            TrainingResult with metrics
        """
        try:
            logger.info(f"Starting training with {len(training_df)} samples...")

            # Prepare data
            X, y = self.prepare_features(training_df)

            logger.info(f"Features: {len(self.feature_names)}, Samples: {len(X)}")
            logger.info(f"Class distribution: WIN={sum(y)}, LOSS={len(y)-sum(y)}")

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train XGBoost
            self.model = xgb.XGBClassifier(**self.xgb_params)
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Feature importance
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            # Save model
            self.is_trained = True
            self._save_model()

            # Save metrics
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                cv_score=cv_mean,
                training_samples=len(X),
                last_trained=datetime.now().isoformat(),
                version=self.model_version
            )
            self._save_metrics(metrics)

            result = TrainingResult(
                success=True,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                cv_score=cv_mean,
                cv_std=cv_std,
                feature_importance=importance,
                training_samples=len(X),
                model_version=self.model_version,
                trained_at=datetime.now().isoformat(),
                message=f"Model trained successfully! Accuracy: {accuracy:.2%}, CV Score: {cv_mean:.2%}"
            )

            logger.info(result.message)
            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                accuracy=0, precision=0, recall=0, f1=0,
                cv_score=0, cv_std=0,
                feature_importance={},
                training_samples=0,
                model_version=self.model_version,
                trained_at=datetime.now().isoformat(),
                message=f"Training failed: {str(e)}"
            )

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction using trained model

        Args:
            features: Dictionary of feature values

        Returns:
            Prediction result with probability
        """
        if not self.is_trained or self.model is None:
            return {
                'prediction': 'HOLD',
                'probability': 0.5,
                'confidence': 0,
                'message': 'Model not trained'
            }

        try:
            # Prepare features in correct order
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            X_scaled = self.scaler.transform(X)

            # Predict
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]

            win_prob = proba[1] if len(proba) > 1 else proba[0]
            loss_prob = proba[0] if len(proba) > 1 else 1 - proba[0]

            # Determine signal
            if win_prob > 0.6:
                prediction = 'BUY' if features.get('regime_score', 0) >= 0 else 'SELL'
                confidence = win_prob * 100
            elif loss_prob > 0.6:
                prediction = 'HOLD'
                confidence = loss_prob * 100
            else:
                prediction = 'HOLD'
                confidence = 50

            return {
                'prediction': prediction,
                'probability': float(win_prob),
                'confidence': float(confidence),
                'win_probability': float(win_prob),
                'loss_probability': float(loss_prob),
                'message': f'{prediction} signal with {confidence:.1f}% confidence'
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'prediction': 'HOLD',
                'probability': 0.5,
                'confidence': 0,
                'message': f'Prediction error: {str(e)}'
            }

    def get_model_info(self) -> Dict:
        """Get current model information"""
        info = {
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'feature_count': len(self.feature_names),
            'features': self.feature_names
        }

        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
                    info.update(metrics)
            except:
                pass

        return info


# Singleton instance
_trainer = None

def get_trainer() -> MLRealTrainer:
    """Get singleton trainer instance"""
    global _trainer
    if _trainer is None:
        _trainer = MLRealTrainer()
    return _trainer


def train_on_real_data() -> TrainingResult:
    """
    Train model on collected real data

    Usage:
        from src.ml_real_trainer import train_on_real_data
        result = train_on_real_data()
        print(f"Accuracy: {result.accuracy:.2%}")
    """
    from src.ml_data_collector import get_data_collector

    collector = get_data_collector()
    training_df = collector.get_training_data()

    if len(training_df) < 10:
        return TrainingResult(
            success=False,
            accuracy=0, precision=0, recall=0, f1=0,
            cv_score=0, cv_std=0,
            feature_importance={},
            training_samples=len(training_df),
            model_version="v2.0_real_data",
            trained_at=datetime.now().isoformat(),
            message=f"Need at least 10 completed signals. Currently have {len(training_df)}."
        )

    trainer = get_trainer()
    return trainer.train(training_df)


def predict_with_real_model(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Make prediction using real-data trained model

    Usage:
        from src.ml_real_trainer import predict_with_real_model

        result = predict_with_real_model({
            'regime_score': 50,
            'cvd_diamond_score': 65,
            ...
        })
    """
    trainer = get_trainer()
    return trainer.predict(features)
