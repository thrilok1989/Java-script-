"""
Training Dataset Generator
==========================
Generates labeled training data for the structure-based ML system.

Key principles:
1. Label STRUCTURES, not signals
2. Label BEFORE-move patterns, not after
3. Include MAE/MFE for outcome quality
4. Separate expiry vs non-expiry data
5. Include neutral outcomes

Output format:
- Features: Market structure snapshot
- Labels: What happened in next N candles
- Metadata: Regime, expiry, time, outcome quality
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os

from .market_structure_features import (
    MarketStructureFeatureExtractor,
    MarketStructureSnapshot,
    MarketStructure,
    snapshot_to_dict
)

logger = logging.getLogger(__name__)


# =============================================================================
# OUTCOME LABELS
# =============================================================================

class OutcomeLabel(Enum):
    """Outcome labels for training"""
    EXPANSION_UP = "EXPANSION_UP"
    EXPANSION_DOWN = "EXPANSION_DOWN"
    FAKE_BREAK_UP = "FAKE_BREAK_UP"
    FAKE_BREAK_DOWN = "FAKE_BREAK_DOWN"
    SL_HUNT_UP = "SL_HUNT_UP"
    SL_HUNT_DOWN = "SL_HUNT_DOWN"
    CONTINUED_RANGE = "CONTINUED_RANGE"
    REVERSAL_UP = "REVERSAL_UP"
    REVERSAL_DOWN = "REVERSAL_DOWN"
    NO_SIGNIFICANT_MOVE = "NO_SIGNIFICANT_MOVE"


@dataclass
class OutcomeMetrics:
    """Metrics for outcome quality assessment"""
    mae: float = 0.0                # Maximum Adverse Excursion (against)
    mfe: float = 0.0                # Maximum Favorable Excursion (for)
    final_move: float = 0.0         # Final price change %
    bars_to_peak: int = 0           # Bars until MFE achieved
    bars_to_trough: int = 0         # Bars until MAE achieved
    move_quality: float = 0.0       # MFE / MAE ratio
    hit_target: bool = False        # Did it hit target?
    hit_stop: bool = False          # Did it hit stop?


@dataclass
class LabeledPattern:
    """A labeled pattern for training"""
    # Timestamp and context
    timestamp: datetime
    symbol: str
    regime: str
    is_expiry_day: bool
    is_expiry_week: bool
    vix_level: float

    # Structure at pattern time
    structure_type: str
    structure_confidence: float
    structure_phase: str

    # Features (flat dict for ML)
    features: Dict[str, float]

    # Outcome
    outcome_label: str
    outcome_direction: str          # UP/DOWN/NEUTRAL
    outcome_magnitude: float        # % move
    outcome_bars: int               # Bars until outcome
    outcome_metrics: OutcomeMetrics

    # Quality indicators
    pattern_quality: float          # 0-100 how clean is pattern
    label_confidence: float         # 0-100 how confident in label


# =============================================================================
# DATASET GENERATOR
# =============================================================================

class TrainingDatasetGenerator:
    """
    Generates labeled training data from historical market data

    Process:
    1. Slide through historical data
    2. At each point, extract structure features
    3. Look forward to determine outcome
    4. Label the pattern with outcome
    5. Store for ML training
    """

    def __init__(
        self,
        forward_bars: int = 15,         # How many bars to look forward
        move_threshold: float = 0.5,    # % threshold for significant move
        fake_threshold: float = 0.3,    # MAE threshold for fake break
        sl_hunt_threshold: float = 0.4  # Threshold for SL hunt detection
    ):
        """Initialize generator"""
        self.forward_bars = forward_bars
        self.move_threshold = move_threshold
        self.fake_threshold = fake_threshold
        self.sl_hunt_threshold = sl_hunt_threshold

        self.feature_extractor = MarketStructureFeatureExtractor()

    def generate_from_df(
        self,
        df: pd.DataFrame,
        option_chain_history: Optional[List[Dict]] = None,
        vix_history: Optional[List[float]] = None,
        expiry_dates: Optional[List[datetime]] = None,
        symbol: str = "NIFTY"
    ) -> List[LabeledPattern]:
        """
        Generate labeled patterns from historical DataFrame

        Args:
            df: Historical OHLCV DataFrame (must have enough history)
            option_chain_history: List of option chain dicts by date
            vix_history: List of VIX values by date
            expiry_dates: List of expiry dates
            symbol: Symbol name

        Returns:
            List of LabeledPattern for training
        """
        if df is None or len(df) < self.forward_bars + 50:
            logger.warning("Insufficient data for training generation")
            return []

        labeled_patterns = []

        # Ensure we have required columns
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        # Slide through data
        for i in range(50, len(df) - self.forward_bars):
            try:
                # Get historical window for feature extraction
                window = df.iloc[max(0, i-50):i+1].copy()

                # Current spot price
                spot_price = float(window[close_col].iloc[-1])

                # Get timestamp
                if 'datetime' in window.columns:
                    timestamp = window['datetime'].iloc[-1]
                elif window.index.name == 'datetime':
                    timestamp = window.index[-1]
                else:
                    timestamp = datetime.now() - timedelta(days=len(df)-i)

                # Get VIX if available
                vix = vix_history[i] if vix_history and i < len(vix_history) else 15.0

                # Check if expiry day/week
                is_expiry_day = False
                is_expiry_week = False
                if expiry_dates:
                    for exp_date in expiry_dates:
                        if isinstance(timestamp, datetime):
                            if timestamp.date() == exp_date.date():
                                is_expiry_day = True
                            if 0 <= (exp_date.date() - timestamp.date()).days <= 5:
                                is_expiry_week = True

                # Extract structure features
                snapshot = self.feature_extractor.extract_features(
                    df=window,
                    spot_price=spot_price,
                    symbol=symbol,
                    vix_data={'value': vix}
                )

                # Look forward to determine outcome
                forward_window = df.iloc[i:i+self.forward_bars+1]
                outcome_label, outcome_metrics, outcome_direction = self._determine_outcome(
                    forward_window, spot_price, close_col, high_col, low_col
                )

                # Calculate pattern quality
                pattern_quality = self._calculate_pattern_quality(snapshot)

                # Calculate label confidence
                label_confidence = self._calculate_label_confidence(
                    outcome_metrics, outcome_label
                )

                # Flatten features for ML
                flat_features = self._flatten_snapshot(snapshot)

                # Create labeled pattern
                pattern = LabeledPattern(
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
                    symbol=symbol,
                    regime=snapshot.primary_structure.value,
                    is_expiry_day=is_expiry_day,
                    is_expiry_week=is_expiry_week,
                    vix_level=vix,
                    structure_type=snapshot.primary_structure.value,
                    structure_confidence=snapshot.structure_confidence,
                    structure_phase=snapshot.structure_phase.value,
                    features=flat_features,
                    outcome_label=outcome_label.value,
                    outcome_direction=outcome_direction,
                    outcome_magnitude=outcome_metrics.final_move,
                    outcome_bars=outcome_metrics.bars_to_peak,
                    outcome_metrics=outcome_metrics,
                    pattern_quality=pattern_quality,
                    label_confidence=label_confidence
                )

                labeled_patterns.append(pattern)

            except Exception as e:
                logger.debug(f"Error at index {i}: {e}")
                continue

        logger.info(f"Generated {len(labeled_patterns)} labeled patterns")
        return labeled_patterns

    def _determine_outcome(
        self,
        forward_df: pd.DataFrame,
        entry_price: float,
        close_col: str,
        high_col: str,
        low_col: str
    ) -> Tuple[OutcomeLabel, OutcomeMetrics, str]:
        """
        Determine what outcome occurred after the pattern

        Uses MAE/MFE analysis to classify outcome quality
        """
        if len(forward_df) < 2:
            return OutcomeLabel.NO_SIGNIFICANT_MOVE, OutcomeMetrics(), "NEUTRAL"

        # Calculate price extremes
        highs = forward_df[high_col].values
        lows = forward_df[low_col].values
        closes = forward_df[close_col].values

        # Max move up and down from entry
        max_high = max(highs)
        min_low = min(lows)
        final_close = closes[-1]

        move_up_pct = (max_high - entry_price) / entry_price * 100
        move_down_pct = (entry_price - min_low) / entry_price * 100
        final_move_pct = (final_close - entry_price) / entry_price * 100

        # Find bars to peaks
        bars_to_high = np.argmax(highs)
        bars_to_low = np.argmin(lows)

        # Determine MFE/MAE based on final direction
        if final_move_pct > 0:  # Ended up
            mfe = move_up_pct
            mae = move_down_pct
            bars_to_peak = bars_to_high
            bars_to_trough = bars_to_low
            direction = "UP"
        elif final_move_pct < 0:  # Ended down
            mfe = move_down_pct
            mae = move_up_pct
            bars_to_peak = bars_to_low
            bars_to_trough = bars_to_high
            direction = "DOWN"
        else:
            mfe = max(move_up_pct, move_down_pct)
            mae = min(move_up_pct, move_down_pct)
            bars_to_peak = max(bars_to_high, bars_to_low)
            bars_to_trough = min(bars_to_high, bars_to_low)
            direction = "NEUTRAL"

        # Move quality = MFE / MAE (higher = cleaner move)
        move_quality = mfe / mae if mae > 0 else mfe

        metrics = OutcomeMetrics(
            mae=mae,
            mfe=mfe,
            final_move=final_move_pct,
            bars_to_peak=int(bars_to_peak),
            bars_to_trough=int(bars_to_trough),
            move_quality=move_quality,
            hit_target=mfe > self.move_threshold * 2,
            hit_stop=mae > self.move_threshold * 2
        )

        # Classify outcome
        label = self._classify_outcome(
            move_up_pct, move_down_pct, final_move_pct,
            bars_to_high, bars_to_low, mae, mfe, move_quality
        )

        return label, metrics, direction

    def _classify_outcome(
        self,
        move_up: float,
        move_down: float,
        final_move: float,
        bars_to_high: int,
        bars_to_low: int,
        mae: float,
        mfe: float,
        move_quality: float
    ) -> OutcomeLabel:
        """Classify the outcome into a label"""
        # Check for significant move
        if abs(final_move) < self.move_threshold:
            if max(move_up, move_down) < self.move_threshold:
                return OutcomeLabel.NO_SIGNIFICANT_MOVE
            else:
                return OutcomeLabel.CONTINUED_RANGE

        # Determine primary direction
        if final_move > 0:  # Ended up
            # Check for fake break down first
            if move_down > self.fake_threshold and bars_to_low < bars_to_high:
                if move_down > self.sl_hunt_threshold:
                    return OutcomeLabel.SL_HUNT_DOWN
                return OutcomeLabel.FAKE_BREAK_DOWN

            # Check for reversal (was going down, then up)
            if bars_to_low < 3 and move_down > self.move_threshold * 0.5:
                return OutcomeLabel.REVERSAL_UP

            # Clean expansion up
            return OutcomeLabel.EXPANSION_UP

        else:  # Ended down
            # Check for fake break up first
            if move_up > self.fake_threshold and bars_to_high < bars_to_low:
                if move_up > self.sl_hunt_threshold:
                    return OutcomeLabel.SL_HUNT_UP
                return OutcomeLabel.FAKE_BREAK_UP

            # Check for reversal
            if bars_to_high < 3 and move_up > self.move_threshold * 0.5:
                return OutcomeLabel.REVERSAL_DOWN

            # Clean expansion down
            return OutcomeLabel.EXPANSION_DOWN

    def _calculate_pattern_quality(self, snapshot: MarketStructureSnapshot) -> float:
        """Calculate quality score for the pattern (0-100)"""
        quality = 50.0  # Base

        # Higher structure confidence = higher quality
        quality += snapshot.structure_confidence * 0.3

        # Clear structure (not NEUTRAL or TRANSITION) = higher quality
        if snapshot.primary_structure not in [MarketStructure.NEUTRAL, MarketStructure.TRANSITION]:
            quality += 15

        # Mature structures = higher quality
        if hasattr(snapshot, 'derived_indicators'):
            maturity = snapshot.derived_indicators.structure_maturity
            quality += maturity * 0.2

        return min(100, max(0, quality))

    def _calculate_label_confidence(
        self,
        metrics: OutcomeMetrics,
        label: OutcomeLabel
    ) -> float:
        """Calculate confidence in the label (0-100)"""
        confidence = 50.0

        # Higher move quality = higher confidence
        if metrics.move_quality > 2:
            confidence += 20
        elif metrics.move_quality > 1:
            confidence += 10

        # Larger moves = higher confidence
        if abs(metrics.final_move) > 1:
            confidence += 15
        elif abs(metrics.final_move) > 0.5:
            confidence += 8

        # Hit target = higher confidence
        if metrics.hit_target:
            confidence += 10

        # Low MAE = higher confidence
        if metrics.mae < 0.3:
            confidence += 10

        return min(100, max(0, confidence))

    def _flatten_snapshot(self, snapshot: MarketStructureSnapshot) -> Dict[str, float]:
        """Flatten snapshot to dict for ML"""
        features = {}

        # Price features
        if hasattr(snapshot, 'price_features'):
            pf = snapshot.price_features
            features['range_atr_ratio'] = pf.price_range_atr_ratio
            features['clv'] = pf.clv
            features['clv_trend'] = pf.clv_trend
            features['upper_wick_ratio'] = pf.upper_wick_ratio
            features['lower_wick_ratio'] = pf.lower_wick_ratio
            features['wick_imbalance'] = pf.wick_imbalance
            features['rejection_score'] = pf.rejection_score
            features['higher_highs_count'] = float(pf.higher_highs_count)
            features['lower_lows_count'] = float(pf.lower_lows_count)
            features['equal_highs_count'] = float(pf.equal_highs_count)
            features['equal_lows_count'] = float(pf.equal_lows_count)
            features['momentum_5'] = pf.price_momentum_5
            features['momentum_20'] = pf.price_momentum_20
            features['momentum_divergence'] = 1.0 if pf.momentum_divergence else 0.0

        # Volume/OI features
        if hasattr(snapshot, 'volume_oi_features'):
            vf = snapshot.volume_oi_features
            features['volume_ratio'] = vf.volume_ratio
            features['volume_trend'] = vf.volume_trend
            features['volume_price_div'] = 1.0 if vf.volume_price_divergence else 0.0
            features['volume_climax'] = 1.0 if vf.volume_climax else 0.0
            features['volume_dry_up'] = 1.0 if vf.volume_dry_up else 0.0
            features['oi_change_pct'] = vf.oi_change_pct
            features['oi_price_div'] = 1.0 if vf.oi_price_divergence else 0.0
            features['oi_slope'] = vf.oi_slope
            features['oi_buildup_score'] = vf.oi_buildup_score
            features['oi_pcr'] = vf.oi_pcr

        # Delta/Flow features
        if hasattr(snapshot, 'delta_flow_features'):
            df = snapshot.delta_flow_features
            features['cvd_value'] = df.cvd_value
            features['cvd_slope'] = df.cvd_slope
            features['cvd_price_div'] = 1.0 if df.cvd_price_divergence else 0.0
            features['delta_imbalance'] = df.delta_imbalance
            features['delta_absorption'] = 1.0 if df.delta_absorption else 0.0
            features['delta_exhaustion'] = 1.0 if df.delta_exhaustion else 0.0

        # Volatility features
        if hasattr(snapshot, 'volatility_features'):
            volf = snapshot.volatility_features
            features['atr_ratio'] = volf.atr_ratio
            features['atr_percentile'] = volf.atr_percentile
            features['compression_score'] = volf.compression_score
            features['compression_duration'] = float(volf.compression_duration)
            features['bollinger_squeeze'] = 1.0 if volf.bollinger_squeeze else 0.0
            features['vix_level'] = volf.vix_level
            features['vix_percentile'] = volf.vix_percentile

        # Gamma features
        if hasattr(snapshot, 'gamma_features'):
            gf = snapshot.gamma_features
            features['gex_flip_distance'] = gf.gex_flip_distance
            features['pin_probability'] = gf.pin_probability
            features['days_to_expiry'] = gf.days_to_expiry
            features['is_expiry_day'] = 1.0 if gf.is_expiry_day else 0.0
            features['is_expiry_week'] = 1.0 if gf.is_expiry_week else 0.0

        # Derived indicators
        if hasattr(snapshot, 'derived_indicators'):
            di = snapshot.derived_indicators
            features['accumulation_score'] = di.accumulation_score
            features['distribution_score'] = di.distribution_score
            features['compression_derived'] = di.compression_score
            features['manipulation_score'] = di.manipulation_score
            features['structure_maturity'] = di.structure_maturity
            features['breakout_imminence'] = di.breakout_imminence
            features['sl_hunt_prob_above'] = di.sl_hunt_probability_above
            features['sl_hunt_prob_below'] = di.sl_hunt_probability_below
            features['structure_clarity'] = di.structure_clarity

        return features

    def to_dataframe(self, patterns: List[LabeledPattern]) -> pd.DataFrame:
        """Convert labeled patterns to DataFrame for ML training"""
        if not patterns:
            return pd.DataFrame()

        rows = []
        for p in patterns:
            row = {
                'timestamp': p.timestamp,
                'symbol': p.symbol,
                'regime': p.regime,
                'is_expiry_day': p.is_expiry_day,
                'is_expiry_week': p.is_expiry_week,
                'vix_level': p.vix_level,
                'structure_type': p.structure_type,
                'structure_confidence': p.structure_confidence,
                'structure_phase': p.structure_phase,
                'outcome_label': p.outcome_label,
                'outcome_direction': p.outcome_direction,
                'outcome_magnitude': p.outcome_magnitude,
                'outcome_bars': p.outcome_bars,
                'mae': p.outcome_metrics.mae,
                'mfe': p.outcome_metrics.mfe,
                'move_quality': p.outcome_metrics.move_quality,
                'pattern_quality': p.pattern_quality,
                'label_confidence': p.label_confidence,
            }
            # Add all features
            row.update(p.features)
            rows.append(row)

        return pd.DataFrame(rows)

    def save_dataset(
        self,
        patterns: List[LabeledPattern],
        filepath: str,
        format: str = 'csv'
    ) -> bool:
        """Save dataset to file"""
        try:
            df = self.to_dataframe(patterns)

            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'parquet':
                df.to_parquet(filepath, index=False)
            elif format == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                logger.warning(f"Unknown format: {format}")
                return False

            logger.info(f"Saved {len(patterns)} patterns to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return False

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load dataset from file"""
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith('.parquet'):
                return pd.read_parquet(filepath)
            elif filepath.endswith('.json'):
                return pd.read_json(filepath)
            else:
                logger.warning(f"Unknown file format: {filepath}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()


# =============================================================================
# DATASET STATISTICS
# =============================================================================

def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """Get statistics about a training dataset"""
    if df.empty:
        return {}

    stats = {
        'total_samples': len(df),
        'date_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max() if 'timestamp' in df.columns else None,
        },
        'outcome_distribution': df['outcome_label'].value_counts().to_dict() if 'outcome_label' in df.columns else {},
        'regime_distribution': df['regime'].value_counts().to_dict() if 'regime' in df.columns else {},
        'expiry_breakdown': {
            'expiry_day_samples': len(df[df['is_expiry_day'] == True]) if 'is_expiry_day' in df.columns else 0,
            'expiry_week_samples': len(df[df['is_expiry_week'] == True]) if 'is_expiry_week' in df.columns else 0,
            'non_expiry_samples': len(df[(df['is_expiry_day'] == False) & (df['is_expiry_week'] == False)]) if 'is_expiry_day' in df.columns else 0,
        },
        'quality_metrics': {
            'avg_pattern_quality': df['pattern_quality'].mean() if 'pattern_quality' in df.columns else 0,
            'avg_label_confidence': df['label_confidence'].mean() if 'label_confidence' in df.columns else 0,
            'avg_move_quality': df['move_quality'].mean() if 'move_quality' in df.columns else 0,
        },
        'feature_count': len([c for c in df.columns if c not in [
            'timestamp', 'symbol', 'regime', 'is_expiry_day', 'is_expiry_week',
            'structure_type', 'structure_confidence', 'structure_phase',
            'outcome_label', 'outcome_direction', 'outcome_magnitude', 'outcome_bars',
            'mae', 'mfe', 'move_quality', 'pattern_quality', 'label_confidence', 'vix_level'
        ]]),
    }

    return stats


def print_dataset_statistics(stats: Dict) -> None:
    """Print dataset statistics in readable format"""
    print("=" * 50)
    print("TRAINING DATASET STATISTICS")
    print("=" * 50)
    print(f"\nTotal Samples: {stats.get('total_samples', 0)}")

    date_range = stats.get('date_range', {})
    print(f"Date Range: {date_range.get('start')} to {date_range.get('end')}")

    print(f"\nFeature Count: {stats.get('feature_count', 0)}")

    print("\n📊 OUTCOME DISTRIBUTION:")
    for outcome, count in stats.get('outcome_distribution', {}).items():
        pct = count / stats.get('total_samples', 1) * 100
        print(f"  {outcome:25} {count:6} ({pct:5.1f}%)")

    print("\n📈 REGIME DISTRIBUTION:")
    for regime, count in stats.get('regime_distribution', {}).items():
        pct = count / stats.get('total_samples', 1) * 100
        print(f"  {regime:25} {count:6} ({pct:5.1f}%)")

    expiry = stats.get('expiry_breakdown', {})
    print("\n📅 EXPIRY BREAKDOWN:")
    print(f"  Expiry Day Samples:   {expiry.get('expiry_day_samples', 0)}")
    print(f"  Expiry Week Samples:  {expiry.get('expiry_week_samples', 0)}")
    print(f"  Non-Expiry Samples:   {expiry.get('non_expiry_samples', 0)}")

    quality = stats.get('quality_metrics', {})
    print("\n⭐ QUALITY METRICS:")
    print(f"  Avg Pattern Quality:   {quality.get('avg_pattern_quality', 0):.1f}")
    print(f"  Avg Label Confidence:  {quality.get('avg_label_confidence', 0):.1f}")
    print(f"  Avg Move Quality:      {quality.get('avg_move_quality', 0):.2f}")

    print("=" * 50)
