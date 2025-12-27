"""
Training Data Collector
Collects market predictions and actual outcomes for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects trading predictions and actual outcomes to build training dataset

    This enables your AI to learn from YOUR actual trading patterns and improve over time.
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize Training Data Collector"""
        self.data_dir = data_dir
        self.training_file = os.path.join(data_dir, "training_data.csv")
        self.prediction_log = os.path.join(data_dir, "prediction_log.csv")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Initialize files if they don't exist
        self._initialize_files()

    def _initialize_files(self):
        """Initialize CSV files with headers"""

        # Training data file
        if not os.path.exists(self.training_file):
            # Define all feature columns based on XGBoost analyzer
            feature_columns = [
                'timestamp',
                'nifty_price',
                'nifty_change',

                # Volatility features
                'vix',
                'atr',
                'atr_pct',
                'volatility_regime',
                'vix_percentile',
                'compression_score',

                # Option chain features
                'pcr',
                'total_ce_oi',
                'total_pe_oi',
                'oi_imbalance',
                'max_pain',
                'max_pain_distance',

                # Bias indicators (13 indicators)
                'bias_oi', 'bias_chgoi', 'bias_volume', 'bias_delta',
                'bias_iv', 'bias_atm_iv', 'bias_pcr', 'bias_buildup',
                'bias_unwinding', 'bias_max_pain', 'bias_gamma',
                'bias_vanna', 'bias_charm',

                # OI Trap features
                'trap_detected',
                'trap_probability',
                'retail_trap_score',

                # CVD features
                'cvd_value',
                'delta_imbalance',
                'orderflow_strength',
                'cvd_bias',

                # Institutional/Retail features
                'institutional_confidence',
                'retail_confidence',
                'smart_money',

                # Liquidity features
                'gravity_strength',
                'num_support_zones',
                'num_resistance_zones',

                # Market regime
                'ml_regime',
                'regime_confidence',

                # Time features
                'is_expiry_day',
                'minutes_to_close',
                'hour_of_day',

                # OUTCOME LABELS (what we're trying to predict)
                'actual_direction',  # 0=SELL, 1=HOLD, 2=BUY
                'actual_move_1h',    # % move in next 1 hour
                'actual_move_1d',    # % move in next 1 day
                'profitable',        # Was the trade profitable?
                'pnl_percent'        # Actual P&L %
            ]

            df = pd.DataFrame(columns=feature_columns)
            df.to_csv(self.training_file, index=False)
            logger.info(f"✅ Created training data file: {self.training_file}")

        # Prediction log file
        if not os.path.exists(self.prediction_log):
            log_columns = [
                'timestamp',
                'prediction_id',
                'ml_prediction',
                'ml_confidence',
                'nifty_price_at_prediction',
                'vix_at_prediction',
                'pcr_at_prediction',
                'final_verdict',
                'trade_taken',
                'entry_price',
                'exit_price',
                'outcome_recorded'
            ]

            df = pd.DataFrame(columns=log_columns)
            df.to_csv(self.prediction_log, index=False)
            logger.info(f"✅ Created prediction log file: {self.prediction_log}")

    def record_prediction(
        self,
        prediction_data: Dict,
        feature_values: Dict,
        ml_result: any
    ) -> str:
        """
        Record a prediction when it's made

        Args:
            prediction_data: Current market data
            feature_values: All extracted features
            ml_result: ML prediction result

        Returns:
            prediction_id: Unique ID for this prediction
        """
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_entry = {
            'timestamp': datetime.now(),
            'prediction_id': prediction_id,
            'ml_prediction': ml_result.prediction if hasattr(ml_result, 'prediction') else 'UNKNOWN',
            'ml_confidence': ml_result.confidence if hasattr(ml_result, 'confidence') else 0.0,
            'nifty_price_at_prediction': prediction_data.get('nifty_price', 0),
            'vix_at_prediction': prediction_data.get('vix', 0),
            'pcr_at_prediction': prediction_data.get('pcr', 0),
            'final_verdict': prediction_data.get('final_verdict', 'UNKNOWN'),
            'trade_taken': False,
            'entry_price': None,
            'exit_price': None,
            'outcome_recorded': False
        }

        # Append to prediction log
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv(self.prediction_log, mode='a', header=False, index=False)

        logger.info(f"📝 Recorded prediction: {prediction_id}")
        return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        feature_values: Dict,
        outcome: Dict
    ):
        """
        Record the actual outcome of a prediction

        Args:
            prediction_id: ID from record_prediction()
            feature_values: Original feature values
            outcome: Dict with actual_direction, actual_move, profitable, pnl_percent
        """
        try:
            # Create training data row
            training_row = {
                **feature_values,
                'timestamp': datetime.now(),
                'actual_direction': outcome.get('actual_direction', 1),  # 0=SELL, 1=HOLD, 2=BUY
                'actual_move_1h': outcome.get('actual_move_1h', 0.0),
                'actual_move_1d': outcome.get('actual_move_1d', 0.0),
                'profitable': outcome.get('profitable', False),
                'pnl_percent': outcome.get('pnl_percent', 0.0)
            }

            # Append to training data
            df = pd.DataFrame([training_row])
            df.to_csv(self.training_file, mode='a', header=False, index=False)

            # Update prediction log
            log_df = pd.read_csv(self.prediction_log)
            if prediction_id in log_df['prediction_id'].values:
                log_df.loc[log_df['prediction_id'] == prediction_id, 'outcome_recorded'] = True
                log_df.loc[log_df['prediction_id'] == prediction_id, 'exit_price'] = outcome.get('exit_price')
                log_df.to_csv(self.prediction_log, index=False)

            logger.info(f"✅ Recorded outcome for {prediction_id}: Profitable={outcome.get('profitable')}, P&L={outcome.get('pnl_percent'):.2f}%")

            return True

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return False

    def get_training_data(self) -> pd.DataFrame:
        """Load all training data"""
        if os.path.exists(self.training_file):
            df = pd.read_csv(self.training_file)
            logger.info(f"📊 Loaded {len(df)} training samples")
            return df
        else:
            logger.warning("No training data found")
            return pd.DataFrame()

    def get_prediction_log(self) -> pd.DataFrame:
        """Load prediction log"""
        if os.path.exists(self.prediction_log):
            return pd.read_csv(self.prediction_log)
        else:
            return pd.DataFrame()

    def get_stats(self) -> Dict:
        """Get statistics about collected data"""
        training_df = self.get_training_data()
        log_df = self.get_prediction_log()

        if len(training_df) == 0:
            return {
                'total_samples': 0,
                'predictions_with_outcomes': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0
            }

        # Calculate stats
        profitable_trades = training_df[training_df['profitable'] == True]

        stats = {
            'total_samples': len(training_df),
            'predictions_with_outcomes': len(log_df[log_df['outcome_recorded'] == True]),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(training_df[training_df['profitable'] == False]),
            'win_rate': (len(profitable_trades) / len(training_df) * 100) if len(training_df) > 0 else 0.0,
            'avg_pnl': training_df['pnl_percent'].mean() if 'pnl_percent' in training_df.columns else 0.0,
            'total_pnl': training_df['pnl_percent'].sum() if 'pnl_percent' in training_df.columns else 0.0,
            'best_trade': training_df['pnl_percent'].max() if 'pnl_percent' in training_df.columns else 0.0,
            'worst_trade': training_df['pnl_percent'].min() if 'pnl_percent' in training_df.columns else 0.0,
        }

        return stats

    def export_for_analysis(self, output_file: str = None):
        """Export training data for external analysis"""
        if output_file is None:
            output_file = os.path.join(self.data_dir, f"training_export_{datetime.now().strftime('%Y%m%d')}.csv")

        df = self.get_training_data()
        df.to_csv(output_file, index=False)

        logger.info(f"📁 Exported {len(df)} samples to {output_file}")
        return output_file
