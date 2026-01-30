"""
Indicators 2026 Module
======================
Streamlined indicators for 2026 trading.

REMOVE (2025):
- MACD
- RSI divergence
- Multi EMA clouds
- Supertrend in chop

KEEP & MODIFY (2026):
- EMA 20 & 50 (distance only) - Compression / Sudden expansion
- ATR (core engine) - Stop size, target expectation, regime classification
- Volume (context only) - Volume without price movement = absorption

Philosophy:
- Fewer indicators, used correctly
- Focus on actionable signals
- ATR is the core engine for everything
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pytz

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# EMA DISTANCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class EMADistanceAnalyzer:
    """
    EMA 20 & 50 - Focus on DISTANCE only.

    NOT for crossover signals.
    Use for:
    - Compression detection (EMAs close together)
    - Sudden expansion (EMAs spreading rapidly)
    - Trend strength (distance magnitude)
    """

    @staticmethod
    def calculate_emas(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate EMA 20 and EMA 50"""
        close = df['close'] if 'close' in df.columns else df['Close']
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        return {'ema20': ema20, 'ema50': ema50}

    @staticmethod
    def analyze_ema_distance(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze EMA distance for compression/expansion signals.
        """
        if df is None or len(df) < 50:
            return {
                'current_distance': 0,
                'state': 'INSUFFICIENT_DATA',
                'valid': False
            }

        close = df['close'] if 'close' in df.columns else df['Close']
        emas = EMADistanceAnalyzer.calculate_emas(df)
        ema20 = emas['ema20']
        ema50 = emas['ema50']

        # Current distance (as % of price)
        current_distance = ((ema20.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1]) * 100

        # Historical distance for comparison
        distance_series = ((ema20 - ema50) / close) * 100
        avg_distance = abs(distance_series.iloc[-lookback:]).mean()
        current_abs_distance = abs(current_distance)

        # Distance velocity (how fast is it changing?)
        if len(distance_series) >= 5:
            distance_velocity = (abs(distance_series.iloc[-1]) - abs(distance_series.iloc[-5])) / 5
        else:
            distance_velocity = 0

        # Determine state
        compression_threshold = avg_distance * 0.5
        expansion_threshold = avg_distance * 1.5

        if current_abs_distance < compression_threshold:
            state = 'COMPRESSED'
            description = 'EMAs squeezed together. Breakout setup forming.'
        elif current_abs_distance > expansion_threshold:
            state = 'EXPANDED'
            description = 'EMAs spread wide. Strong trend or overextended.'
        else:
            state = 'NORMAL'
            description = 'Normal EMA spacing.'

        # Velocity state
        if distance_velocity > 0.02:
            velocity_state = 'EXPANDING_FAST'
        elif distance_velocity < -0.02:
            velocity_state = 'COMPRESSING_FAST'
        else:
            velocity_state = 'STABLE'

        # Direction (which EMA is above)
        if current_distance > 0:
            direction = 'BULLISH'  # EMA20 above EMA50
        elif current_distance < 0:
            direction = 'BEARISH'  # EMA20 below EMA50
        else:
            direction = 'NEUTRAL'

        return {
            'current_distance': round(current_distance, 3),
            'current_distance_pct': round(current_abs_distance, 3),
            'avg_distance': round(avg_distance, 3),
            'distance_ratio': round(current_abs_distance / avg_distance, 2) if avg_distance > 0 else 1,
            'distance_velocity': round(distance_velocity, 4),
            'state': state,
            'velocity_state': velocity_state,
            'direction': direction,
            'description': description,
            'ema20': round(ema20.iloc[-1], 2),
            'ema50': round(ema50.iloc[-1], 2),
            'valid': True
        }

    @staticmethod
    def get_compression_score(df: pd.DataFrame) -> int:
        """
        Get compression score (0-100).
        Higher = more compressed = bigger breakout potential.
        """
        analysis = EMADistanceAnalyzer.analyze_ema_distance(df)
        if not analysis.get('valid'):
            return 50

        ratio = analysis.get('distance_ratio', 1)
        # Inverse relationship: lower ratio = higher compression score
        score = max(0, min(100, int((2 - ratio) * 50)))
        return score


# ═══════════════════════════════════════════════════════════════════════════════
# ATR CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ATRCoreEngine:
    """
    ATR - The Core Engine for 2026 Trading.

    Use ATR for:
    - Stop size calculation
    - Target expectation
    - Regime classification
    - Position sizing guidance
    """

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR series"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def get_stop_size(
        df: pd.DataFrame,
        multiplier: float = 1.5,
        atr_period: int = 14
    ) -> Dict[str, Any]:
        """
        Calculate recommended stop size based on ATR.

        Standard: 1.5x ATR (conservative)
        Aggressive: 1.0x ATR
        Wide: 2.0x ATR
        """
        atr = ATRCoreEngine.calculate_atr(df, atr_period)
        current_atr = atr.iloc[-1]

        close = df['close'] if 'close' in df.columns else df['Close']
        current_price = close.iloc[-1]

        # Calculate different stop levels
        stops = {
            'tight': round(current_atr * 1.0, 2),
            'standard': round(current_atr * 1.5, 2),
            'wide': round(current_atr * 2.0, 2),
            'recommended': round(current_atr * multiplier, 2)
        }

        # As percentage of price
        stops_pct = {
            'tight_pct': round(stops['tight'] / current_price * 100, 2),
            'standard_pct': round(stops['standard'] / current_price * 100, 2),
            'wide_pct': round(stops['wide'] / current_price * 100, 2),
            'recommended_pct': round(stops['recommended'] / current_price * 100, 2)
        }

        return {
            'current_atr': round(current_atr, 2),
            'current_price': round(current_price, 2),
            'stops': stops,
            'stops_pct': stops_pct,
            'recommendation': f"Use {multiplier}x ATR = {stops['recommended']} points as stop"
        }

    @staticmethod
    def get_target_expectation(
        df: pd.DataFrame,
        risk_reward: float = 2.0,
        atr_period: int = 14
    ) -> Dict[str, Any]:
        """
        Calculate expected targets based on ATR and risk-reward.
        """
        atr = ATRCoreEngine.calculate_atr(df, atr_period)
        current_atr = atr.iloc[-1]

        close = df['close'] if 'close' in df.columns else df['Close']
        current_price = close.iloc[-1]

        # Standard 1.5x ATR stop
        stop_distance = current_atr * 1.5

        # Targets based on R:R
        targets = {
            '1R': round(stop_distance * 1.0, 2),
            '2R': round(stop_distance * 2.0, 2),
            '3R': round(stop_distance * 3.0, 2),
            'recommended': round(stop_distance * risk_reward, 2)
        }

        # In terms of price levels (for longs)
        target_levels_long = {
            '1R': round(current_price + targets['1R'], 2),
            '2R': round(current_price + targets['2R'], 2),
            '3R': round(current_price + targets['3R'], 2)
        }

        # In terms of price levels (for shorts)
        target_levels_short = {
            '1R': round(current_price - targets['1R'], 2),
            '2R': round(current_price - targets['2R'], 2),
            '3R': round(current_price - targets['3R'], 2)
        }

        return {
            'current_atr': round(current_atr, 2),
            'stop_distance': round(stop_distance, 2),
            'targets': targets,
            'long_targets': target_levels_long,
            'short_targets': target_levels_short,
            'recommendation': f"Target {risk_reward}R = {targets['recommended']} points from entry"
        }

    @staticmethod
    def classify_regime(
        df: pd.DataFrame,
        atr_period: int = 14,
        avg_period: int = 20
    ) -> Dict[str, Any]:
        """
        Classify market regime using ATR.

        Regimes:
        - LOW_VOL: ATR < 70% of average
        - NORMAL: ATR 70-130% of average
        - HIGH_VOL: ATR > 130% of average
        """
        atr = ATRCoreEngine.calculate_atr(df, atr_period)
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-avg_period:].mean()

        if avg_atr == 0:
            return {'regime': 'UNKNOWN', 'valid': False}

        atr_ratio = current_atr / avg_atr

        if atr_ratio < 0.7:
            regime = 'LOW_VOL'
            characteristics = {
                'stop_size': 'Use tighter stops (1x ATR)',
                'target': 'Lower targets, faster exits',
                'position': 'Can use larger position size',
                'strategy': 'Range strategies, mean reversion'
            }
        elif atr_ratio > 1.3:
            regime = 'HIGH_VOL'
            characteristics = {
                'stop_size': 'Use wider stops (2x ATR)',
                'target': 'Trail aggressively',
                'position': 'Reduce position size',
                'strategy': 'Momentum, breakouts'
            }
        else:
            regime = 'NORMAL'
            characteristics = {
                'stop_size': 'Standard stops (1.5x ATR)',
                'target': 'Normal R:R targets',
                'position': 'Standard position size',
                'strategy': 'Trend following'
            }

        return {
            'regime': regime,
            'atr_ratio': round(atr_ratio, 2),
            'current_atr': round(current_atr, 2),
            'avg_atr': round(avg_atr, 2),
            'characteristics': characteristics,
            'valid': True
        }

    @staticmethod
    def get_position_size_guide(
        df: pd.DataFrame,
        account_risk_pct: float = 1.0,
        account_size: float = 100000
    ) -> Dict[str, Any]:
        """
        Get position sizing guidance based on ATR.
        """
        regime = ATRCoreEngine.classify_regime(df)
        stop_info = ATRCoreEngine.get_stop_size(df)

        if not regime.get('valid'):
            return {'valid': False}

        # Adjust risk based on regime
        regime_multiplier = {
            'LOW_VOL': 1.2,   # Can risk slightly more in low vol
            'NORMAL': 1.0,
            'HIGH_VOL': 0.7  # Risk less in high vol
        }.get(regime['regime'], 1.0)

        adjusted_risk_pct = account_risk_pct * regime_multiplier
        risk_amount = account_size * (adjusted_risk_pct / 100)

        stop_points = stop_info['stops']['recommended']
        if stop_points > 0:
            max_quantity = int(risk_amount / stop_points)
        else:
            max_quantity = 0

        return {
            'regime': regime['regime'],
            'base_risk_pct': account_risk_pct,
            'adjusted_risk_pct': round(adjusted_risk_pct, 2),
            'risk_amount': round(risk_amount, 2),
            'stop_points': stop_points,
            'suggested_quantity': max_quantity,
            'recommendation': f"Based on {regime['regime']} regime, risk {adjusted_risk_pct:.1f}% = {max_quantity} quantity",
            'valid': True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME CONTEXT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class VolumeContextAnalyzer:
    """
    Volume - Use for CONTEXT only.

    Key insight: Volume without price movement = absorption.

    NOT for:
    - Volume breakout signals alone
    - Volume divergence

    USE for:
    - Confirming price moves (high vol + price move = real)
    - Detecting absorption (high vol + no move = absorption)
    - Identifying exhaustion (high vol + reversal = exhaustion)
    """

    @staticmethod
    def analyze_volume_context(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze volume in context of price movement.
        """
        if df is None or len(df) < lookback:
            return {'valid': False}

        close = df['close'] if 'close' in df.columns else df['Close']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        # Current vs average volume
        current_vol = volume.iloc[-1]
        avg_vol = volume.iloc[-lookback:].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

        # Current price movement
        current_range = high.iloc[-1] - low.iloc[-1]
        avg_range = (high.iloc[-lookback:] - low.iloc[-lookback:]).mean()
        range_ratio = current_range / avg_range if avg_range > 0 else 1

        # Volume-Price relationship
        if vol_ratio > 1.5 and range_ratio > 1.5:
            vp_relationship = 'CONFIRMED_MOVE'
            interpretation = 'High volume with price expansion. Real move.'
        elif vol_ratio > 1.5 and range_ratio < 0.7:
            vp_relationship = 'ABSORPTION'
            interpretation = 'High volume but tight range. Absorption happening.'
        elif vol_ratio < 0.5 and range_ratio > 1.5:
            vp_relationship = 'SUSPICIOUS_MOVE'
            interpretation = 'Big move on low volume. May not sustain.'
        elif vol_ratio < 0.7 and range_ratio < 0.7:
            vp_relationship = 'QUIET'
            interpretation = 'Low activity. Wait for catalyst.'
        else:
            vp_relationship = 'NORMAL'
            interpretation = 'Normal volume-price relationship.'

        return {
            'current_volume': int(current_vol),
            'avg_volume': int(avg_vol),
            'vol_ratio': round(vol_ratio, 2),
            'range_ratio': round(range_ratio, 2),
            'relationship': vp_relationship,
            'interpretation': interpretation,
            'valid': True
        }

    @staticmethod
    def detect_exhaustion(
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Dict[str, Any]:
        """
        Detect potential exhaustion (high vol + reversal pattern).
        """
        if df is None or len(df) < lookback + 5:
            return {'exhaustion_detected': False, 'valid': False}

        close = df['close'] if 'close' in df.columns else df['Close']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        # Check for volume spike
        recent_vol = volume.iloc[-lookback:].mean()
        prev_vol = volume.iloc[-lookback*2:-lookback].mean()
        vol_spike = recent_vol / prev_vol if prev_vol > 0 else 1

        # Check for price reversal sign
        recent_high = high.iloc[-lookback:].max()
        recent_low = low.iloc[-lookback:].min()
        current_close = close.iloc[-1]

        # Bullish exhaustion: New highs but closing weak
        bullish_exhaustion = (
            vol_spike > 1.5 and
            current_close < recent_high * 0.995 and
            current_close < close.iloc[-2]
        )

        # Bearish exhaustion: New lows but closing strong
        bearish_exhaustion = (
            vol_spike > 1.5 and
            current_close > recent_low * 1.005 and
            current_close > close.iloc[-2]
        )

        exhaustion_detected = bullish_exhaustion or bearish_exhaustion
        exhaustion_type = None
        if bullish_exhaustion:
            exhaustion_type = 'BULLISH_EXHAUSTION'
        elif bearish_exhaustion:
            exhaustion_type = 'BEARISH_EXHAUSTION'

        return {
            'exhaustion_detected': exhaustion_detected,
            'exhaustion_type': exhaustion_type,
            'vol_spike_ratio': round(vol_spike, 2),
            'warning': f'{exhaustion_type} detected. Reversal possible.' if exhaustion_detected else None,
            'valid': True
        }

    @staticmethod
    def get_volume_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess volume quality for the current session.
        """
        context = VolumeContextAnalyzer.analyze_volume_context(df)
        exhaustion = VolumeContextAnalyzer.detect_exhaustion(df)

        if not context.get('valid'):
            return {'quality': 'UNKNOWN', 'valid': False}

        vol_ratio = context['vol_ratio']
        relationship = context['relationship']

        # Quality assessment
        if relationship == 'CONFIRMED_MOVE' and not exhaustion.get('exhaustion_detected'):
            quality = 'EXCELLENT'
            trading_implication = 'Strong volume confirms move. Trade with trend.'
        elif relationship == 'CONFIRMED_MOVE' and exhaustion.get('exhaustion_detected'):
            quality = 'CAUTION'
            trading_implication = 'Volume spike with exhaustion signs. Take profits.'
        elif relationship == 'ABSORPTION':
            quality = 'GOOD_FOR_REVERSAL'
            trading_implication = 'Absorption detected. Watch for reversal setup.'
        elif relationship == 'SUSPICIOUS_MOVE':
            quality = 'POOR'
            trading_implication = 'Move lacks volume confirmation. Be skeptical.'
        elif relationship == 'QUIET':
            quality = 'WAIT'
            trading_implication = 'Low activity. Wait for volume increase.'
        else:
            quality = 'NEUTRAL'
            trading_implication = 'Normal conditions.'

        return {
            'quality': quality,
            'vol_ratio': vol_ratio,
            'relationship': relationship,
            'trading_implication': trading_implication,
            'exhaustion': exhaustion,
            'valid': True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INDICATORS 2026
# ═══════════════════════════════════════════════════════════════════════════════

class Indicators2026:
    """
    Unified streamlined indicators for 2026.

    Only 3 core indicators:
    1. EMA Distance (compression/expansion)
    2. ATR (core engine)
    3. Volume (context)
    """

    def __init__(self):
        self.ema_analyzer = EMADistanceAnalyzer()
        self.atr_engine = ATRCoreEngine()
        self.volume_analyzer = VolumeContextAnalyzer()

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive indicator analysis.
        """
        if df is None or df.empty:
            return {'valid': False}

        # EMA Analysis
        ema_analysis = self.ema_analyzer.analyze_ema_distance(df)

        # ATR Analysis
        atr_regime = self.atr_engine.classify_regime(df)
        atr_stops = self.atr_engine.get_stop_size(df)
        atr_targets = self.atr_engine.get_target_expectation(df)

        # Volume Analysis
        volume_context = self.volume_analyzer.analyze_volume_context(df)
        volume_quality = self.volume_analyzer.get_volume_quality(df)

        # Generate overall assessment
        overall = self._generate_overall_assessment(ema_analysis, atr_regime, volume_quality)

        return {
            'timestamp': datetime.now(IST).isoformat(),
            'ema': ema_analysis,
            'atr': {
                'regime': atr_regime,
                'stops': atr_stops,
                'targets': atr_targets
            },
            'volume': {
                'context': volume_context,
                'quality': volume_quality
            },
            'overall': overall,
            'valid': True
        }

    def _generate_overall_assessment(
        self,
        ema: Dict,
        atr: Dict,
        volume: Dict
    ) -> Dict[str, Any]:
        """Generate overall indicator assessment"""
        signals = []

        # EMA signals
        ema_state = ema.get('state', 'NORMAL')
        if ema_state == 'COMPRESSED':
            signals.append(('EMA_COMPRESSED', 'Breakout setup forming'))
        elif ema_state == 'EXPANDED' and ema.get('velocity_state') == 'EXPANDING_FAST':
            signals.append(('TREND_STRONG', 'Strong trend momentum'))

        # ATR signals
        atr_regime = atr.get('regime', 'NORMAL')
        if atr_regime == 'LOW_VOL':
            signals.append(('LOW_VOL', 'Tight stops, larger size OK'))
        elif atr_regime == 'HIGH_VOL':
            signals.append(('HIGH_VOL', 'Wide stops, reduce size'))

        # Volume signals
        vol_quality = volume.get('quality', 'NEUTRAL')
        if vol_quality == 'EXCELLENT':
            signals.append(('VOL_CONFIRMS', 'Volume confirms move'))
        elif vol_quality == 'CAUTION':
            signals.append(('VOL_EXHAUSTION', 'Exhaustion signs, take profits'))
        elif vol_quality == 'GOOD_FOR_REVERSAL':
            signals.append(('VOL_ABSORPTION', 'Absorption, watch for reversal'))

        # Overall bias
        bullish_count = sum(1 for s, _ in signals if 'STRONG' in s or 'CONFIRMS' in s)
        bearish_count = sum(1 for s, _ in signals if 'EXHAUSTION' in s or 'REVERSAL' in s)

        if bullish_count > bearish_count:
            overall_bias = 'FAVORABLE'
        elif bearish_count > bullish_count:
            overall_bias = 'CAUTIOUS'
        else:
            overall_bias = 'NEUTRAL'

        return {
            'bias': overall_bias,
            'signals': signals,
            'summary': ' | '.join([f"{s[0]}" for s in signals]) if signals else 'No special signals'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_indicators_2026(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick indicator analysis"""
    analyzer = Indicators2026()
    return analyzer.analyze(df)


def get_ema_compression_score(df: pd.DataFrame) -> int:
    """Get EMA compression score (0-100)"""
    return EMADistanceAnalyzer.get_compression_score(df)


def get_atr_stops(df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, Any]:
    """Get ATR-based stop levels"""
    return ATRCoreEngine.get_stop_size(df, multiplier)


def get_atr_targets(df: pd.DataFrame, rr: float = 2.0) -> Dict[str, Any]:
    """Get ATR-based target levels"""
    return ATRCoreEngine.get_target_expectation(df, rr)


def get_volume_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Get volume context analysis"""
    return VolumeContextAnalyzer.get_volume_quality(df)
