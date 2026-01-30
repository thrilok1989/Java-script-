"""
Chart Analysis 2026 Module
==========================
Modern chart analysis replacing outdated pattern recognition.

STOP LOOKING FOR (2025):
- Head & Shoulder
- Double Top/Bottom
- Textbook triangles
- They form, but don't resolve cleanly now

FOCUS ON (2026):
- Range Structure (PDH/PDL, Opening Range)
- Volatility Patterns (ATR compression/expansion)
- VWAP Behavior (Acceptance, Rejection)

Output:
- State: Balance / Expansion / Exhaustion
- Risk: Low / Medium / High
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pytz

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# RANGE STRUCTURE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class RangeStructureAnalyzer:
    """
    Analyze key range structures.

    Key Levels:
    - Previous Day High/Low (PDH/PDL)
    - Opening Range (15-30 min)

    Range breaks decide the day.
    """

    @staticmethod
    def calculate_previous_day_levels(
        df: pd.DataFrame,
        current_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Calculate Previous Day High/Low.

        Args:
            df: DataFrame with OHLCV data (must have datetime index or 'datetime' column)
            current_date: Current trading date
        """
        if df is None or df.empty:
            return {
                'pdh': None, 'pdl': None, 'pd_range': None,
                'pd_midpoint': None, 'valid': False
            }

        try:
            # Ensure datetime index
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            elif not isinstance(df.index, pd.DatetimeIndex):
                return {'pdh': None, 'pdl': None, 'pd_range': None, 'pd_midpoint': None, 'valid': False}

            # Get current date
            if current_date is None:
                current_date = datetime.now(IST).date()
            elif isinstance(current_date, datetime):
                current_date = current_date.date()

            # Get previous day's data
            df['date'] = df.index.date
            dates = sorted(df['date'].unique())

            if len(dates) < 2:
                return {'pdh': None, 'pdl': None, 'pd_range': None, 'pd_midpoint': None, 'valid': False}

            # Find previous day
            prev_date = None
            for i, d in enumerate(dates):
                if d >= current_date and i > 0:
                    prev_date = dates[i-1]
                    break
            if prev_date is None and len(dates) >= 2:
                prev_date = dates[-2]

            if prev_date is None:
                return {'pdh': None, 'pdl': None, 'pd_range': None, 'pd_midpoint': None, 'valid': False}

            # Calculate levels
            prev_data = df[df['date'] == prev_date]

            high_col = 'high' if 'high' in prev_data.columns else 'High'
            low_col = 'low' if 'low' in prev_data.columns else 'Low'

            pdh = prev_data[high_col].max()
            pdl = prev_data[low_col].min()
            pd_range = pdh - pdl
            pd_midpoint = (pdh + pdl) / 2

            return {
                'pdh': round(pdh, 2),
                'pdl': round(pdl, 2),
                'pd_range': round(pd_range, 2),
                'pd_midpoint': round(pd_midpoint, 2),
                'prev_date': prev_date,
                'valid': True
            }
        except Exception as e:
            return {'pdh': None, 'pdl': None, 'pd_range': None, 'pd_midpoint': None, 'valid': False, 'error': str(e)}

    @staticmethod
    def calculate_opening_range(
        df: pd.DataFrame,
        or_minutes: int = 15
    ) -> Dict[str, Any]:
        """
        Calculate Opening Range (first N minutes).

        Args:
            df: Intraday DataFrame
            or_minutes: Opening range period (15 or 30 minutes)
        """
        if df is None or df.empty:
            return {
                'or_high': None, 'or_low': None, 'or_range': None,
                'or_midpoint': None, 'valid': False
            }

        try:
            # Ensure datetime index
            if 'datetime' in df.columns:
                df = df.set_index('datetime')

            # Get today's data
            today = datetime.now(IST).date()
            df['date'] = df.index.date
            today_data = df[df['date'] == today]

            if today_data.empty:
                # Use latest day
                today_data = df[df['date'] == df['date'].max()]

            if today_data.empty:
                return {'or_high': None, 'or_low': None, 'or_range': None, 'or_midpoint': None, 'valid': False}

            # Market open time (9:15 IST)
            market_open = time(9, 15)
            or_end = time(9, 15 + or_minutes)

            # Filter to opening range
            or_data = today_data[
                (today_data.index.time >= market_open) &
                (today_data.index.time <= or_end)
            ]

            if or_data.empty:
                # Fallback: use first N candles
                or_data = today_data.head(or_minutes)

            high_col = 'high' if 'high' in or_data.columns else 'High'
            low_col = 'low' if 'low' in or_data.columns else 'Low'

            or_high = or_data[high_col].max()
            or_low = or_data[low_col].min()
            or_range = or_high - or_low
            or_midpoint = (or_high + or_low) / 2

            return {
                'or_high': round(or_high, 2),
                'or_low': round(or_low, 2),
                'or_range': round(or_range, 2),
                'or_midpoint': round(or_midpoint, 2),
                'or_minutes': or_minutes,
                'valid': True
            }
        except Exception as e:
            return {'or_high': None, 'or_low': None, 'or_range': None, 'or_midpoint': None, 'valid': False, 'error': str(e)}

    @staticmethod
    def analyze_range_position(
        current_price: float,
        pd_levels: Dict,
        or_levels: Dict
    ) -> Dict[str, Any]:
        """
        Analyze current price position relative to ranges.
        """
        result = {
            'price': current_price,
            'pd_position': None,
            'or_position': None,
            'range_state': None,
            'key_level': None,
            'distance_to_key': None
        }

        # Previous day range position
        if pd_levels.get('valid'):
            pdh = pd_levels['pdh']
            pdl = pd_levels['pdl']
            pd_range = pd_levels['pd_range']

            if current_price > pdh:
                result['pd_position'] = 'ABOVE_PDH'
                result['distance_to_key'] = round(current_price - pdh, 2)
                result['key_level'] = pdh
            elif current_price < pdl:
                result['pd_position'] = 'BELOW_PDL'
                result['distance_to_key'] = round(pdl - current_price, 2)
                result['key_level'] = pdl
            else:
                pd_pct = (current_price - pdl) / pd_range * 100 if pd_range > 0 else 50
                if pd_pct > 75:
                    result['pd_position'] = 'UPPER_PD_RANGE'
                elif pd_pct < 25:
                    result['pd_position'] = 'LOWER_PD_RANGE'
                else:
                    result['pd_position'] = 'MID_PD_RANGE'
                result['distance_to_key'] = min(abs(current_price - pdh), abs(current_price - pdl))
                result['key_level'] = pdh if abs(current_price - pdh) < abs(current_price - pdl) else pdl

        # Opening range position
        if or_levels.get('valid'):
            or_high = or_levels['or_high']
            or_low = or_levels['or_low']
            or_range = or_levels['or_range']

            if current_price > or_high:
                result['or_position'] = 'ABOVE_OR'
            elif current_price < or_low:
                result['or_position'] = 'BELOW_OR'
            else:
                result['or_position'] = 'INSIDE_OR'

        # Determine range state
        if result['pd_position'] == 'ABOVE_PDH' or result['pd_position'] == 'BELOW_PDL':
            if result['or_position'] == 'ABOVE_OR' or result['or_position'] == 'BELOW_OR':
                result['range_state'] = 'EXPANSION'
            else:
                result['range_state'] = 'TESTING_PD'
        elif result['or_position'] == 'INSIDE_OR':
            result['range_state'] = 'BALANCE'
        else:
            result['range_state'] = 'RANGING'

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY PATTERN ANALYZER (ATR-BASED)
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityPatternAnalyzer:
    """
    Analyze volatility patterns using ATR.

    This REPLACES classical chart patterns.

    Key Patterns:
    - ATR compression zones (breakout coming)
    - Sudden ATR expansion candles (move started)
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
    def detect_compression_zones(
        df: pd.DataFrame,
        atr_period: int = 14,
        compression_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect ATR compression zones.

        Compression = ATR < threshold * ATR_avg
        """
        if df is None or len(df) < atr_period * 2:
            return {
                'in_compression': False,
                'compression_ratio': None,
                'valid': False
            }

        atr = VolatilityPatternAnalyzer.calculate_atr(df, atr_period)
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-atr_period*2:-atr_period].mean()

        if avg_atr == 0:
            return {'in_compression': False, 'compression_ratio': None, 'valid': False}

        compression_ratio = current_atr / avg_atr
        in_compression = compression_ratio < compression_threshold

        # Calculate compression duration
        compression_duration = 0
        if in_compression:
            for i in range(len(atr) - 1, -1, -1):
                if atr.iloc[i] / avg_atr < compression_threshold:
                    compression_duration += 1
                else:
                    break

        return {
            'in_compression': in_compression,
            'compression_ratio': round(compression_ratio, 3),
            'current_atr': round(current_atr, 2),
            'avg_atr': round(avg_atr, 2),
            'compression_duration': compression_duration,
            'breakout_imminent': compression_duration >= 5 and in_compression,
            'valid': True
        }

    @staticmethod
    def detect_expansion_candles(
        df: pd.DataFrame,
        atr_period: int = 14,
        expansion_threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect sudden ATR expansion candles.

        Expansion candle = TR > threshold * ATR
        """
        if df is None or len(df) < atr_period + 1:
            return {
                'expansion_detected': False,
                'expansion_candles': [],
                'valid': False
            }

        atr = VolatilityPatternAnalyzer.calculate_atr(df, atr_period)

        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        # Current candle TR
        current_tr = high.iloc[-1] - low.iloc[-1]
        current_atr = atr.iloc[-1]

        if current_atr == 0:
            return {'expansion_detected': False, 'expansion_candles': [], 'valid': False}

        expansion_ratio = current_tr / current_atr
        expansion_detected = expansion_ratio > expansion_threshold

        # Find recent expansion candles
        expansion_candles = []
        lookback = min(10, len(df))
        for i in range(-lookback, 0):
            tr = high.iloc[i] - low.iloc[i]
            if atr.iloc[i] > 0:
                ratio = tr / atr.iloc[i]
                if ratio > expansion_threshold:
                    expansion_candles.append({
                        'index': i,
                        'ratio': round(ratio, 2),
                        'direction': 'UP' if close.iloc[i] > close.iloc[i-1] else 'DOWN'
                    })

        return {
            'expansion_detected': expansion_detected,
            'current_expansion_ratio': round(expansion_ratio, 2),
            'expansion_candles': expansion_candles,
            'current_direction': 'UP' if close.iloc[-1] > close.iloc[-2] else 'DOWN' if len(close) > 1 else 'FLAT',
            'valid': True
        }

    @staticmethod
    def classify_volatility_regime(
        df: pd.DataFrame,
        atr_period: int = 14
    ) -> Dict[str, Any]:
        """
        Classify current volatility regime.

        Regimes:
        - COMPRESSED: Breakout imminent
        - NORMAL: Regular trading
        - ELEVATED: Wide ranges
        - EXTREME: Massive volatility
        """
        compression = VolatilityPatternAnalyzer.detect_compression_zones(df, atr_period)
        expansion = VolatilityPatternAnalyzer.detect_expansion_candles(df, atr_period)

        if not compression.get('valid'):
            return {'regime': 'UNKNOWN', 'valid': False}

        ratio = compression['compression_ratio']

        if ratio < 0.6:
            regime = 'COMPRESSED'
            description = 'Volatility squeezed. Breakout imminent.'
            trading_action = 'Prepare for breakout. Set tight alerts.'
        elif ratio < 0.85:
            regime = 'LOW'
            description = 'Below average volatility. Range likely.'
            trading_action = 'Range strategies. Small positions.'
        elif ratio < 1.15:
            regime = 'NORMAL'
            description = 'Normal volatility. Follow trend.'
            trading_action = 'Standard strategies. Normal sizing.'
        elif ratio < 1.5:
            regime = 'ELEVATED'
            description = 'Above average volatility. Widen stops.'
            trading_action = 'Trail stops. Take profits early.'
        else:
            regime = 'EXTREME'
            description = 'Extreme volatility. High risk.'
            trading_action = 'Reduce size or wait. High uncertainty.'

        return {
            'regime': regime,
            'ratio': ratio,
            'description': description,
            'trading_action': trading_action,
            'compression_data': compression,
            'expansion_data': expansion,
            'valid': True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VWAP BEHAVIOR ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class VWAPBehaviorAnalyzer:
    """
    Analyze VWAP behavior.

    VWAP > Pattern in 2026.

    Key Behaviors:
    - Acceptance above/below VWAP
    - VWAP rejections
    - VWAP as dynamic support/resistance
    """

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap

    @staticmethod
    def analyze_vwap_behavior(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze price behavior around VWAP.
        """
        if df is None or len(df) < lookback:
            return {
                'vwap_position': None,
                'acceptance': None,
                'rejection_count': 0,
                'valid': False
            }

        vwap = VWAPBehaviorAnalyzer.calculate_vwap(df)
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        current_vwap = vwap.iloc[-1]
        current_price = close.iloc[-1]

        # VWAP position
        vwap_distance = (current_price - current_vwap) / current_vwap * 100
        if vwap_distance > 0.3:
            vwap_position = 'ABOVE'
        elif vwap_distance < -0.3:
            vwap_position = 'BELOW'
        else:
            vwap_position = 'AT_VWAP'

        # Acceptance (sustained position above/below)
        recent_closes = close.iloc[-lookback:]
        recent_vwap = vwap.iloc[-lookback:]
        above_count = sum(recent_closes > recent_vwap)
        below_count = sum(recent_closes < recent_vwap)

        if above_count >= lookback * 0.8:
            acceptance = 'ACCEPTED_ABOVE'
        elif below_count >= lookback * 0.8:
            acceptance = 'ACCEPTED_BELOW'
        else:
            acceptance = 'ROTATING'

        # Rejection detection (wick through VWAP but close on other side)
        rejection_count = 0
        rejections = []
        for i in range(-lookback, 0):
            vwap_val = vwap.iloc[i]
            candle_high = high.iloc[i]
            candle_low = low.iloc[i]
            candle_close = close.iloc[i]

            # Bullish rejection (low touched VWAP, close above)
            if candle_low <= vwap_val <= candle_close:
                if candle_close > vwap_val * 1.001:  # Close meaningfully above
                    rejection_count += 1
                    rejections.append({'type': 'BULLISH', 'index': i})

            # Bearish rejection (high touched VWAP, close below)
            if candle_close <= vwap_val <= candle_high:
                if candle_close < vwap_val * 0.999:  # Close meaningfully below
                    rejection_count += 1
                    rejections.append({'type': 'BEARISH', 'index': i})

        # Recent rejection bias
        recent_rejections = [r for r in rejections if r['index'] >= -5]
        bullish_rejections = sum(1 for r in recent_rejections if r['type'] == 'BULLISH')
        bearish_rejections = sum(1 for r in recent_rejections if r['type'] == 'BEARISH')

        if bullish_rejections > bearish_rejections:
            recent_bias = 'BULLISH_REJECTIONS'
        elif bearish_rejections > bullish_rejections:
            recent_bias = 'BEARISH_REJECTIONS'
        else:
            recent_bias = 'NEUTRAL'

        return {
            'current_vwap': round(current_vwap, 2),
            'current_price': round(current_price, 2),
            'vwap_distance_pct': round(vwap_distance, 3),
            'vwap_position': vwap_position,
            'acceptance': acceptance,
            'rejection_count': rejection_count,
            'recent_bias': recent_bias,
            'signal': VWAPBehaviorAnalyzer._get_vwap_signal(vwap_position, acceptance, recent_bias),
            'valid': True
        }

    @staticmethod
    def _get_vwap_signal(position: str, acceptance: str, bias: str) -> Dict[str, Any]:
        """Generate VWAP-based signal"""
        if acceptance == 'ACCEPTED_ABOVE' and position == 'ABOVE':
            return {
                'direction': 'BULLISH',
                'strength': 'STRONG',
                'action': 'Look for long entries on VWAP pullbacks'
            }
        elif acceptance == 'ACCEPTED_BELOW' and position == 'BELOW':
            return {
                'direction': 'BEARISH',
                'strength': 'STRONG',
                'action': 'Look for short entries on VWAP rallies'
            }
        elif acceptance == 'ROTATING':
            if bias == 'BULLISH_REJECTIONS':
                return {
                    'direction': 'BULLISH',
                    'strength': 'MODERATE',
                    'action': 'VWAP acting as support. Watch for acceptance above.'
                }
            elif bias == 'BEARISH_REJECTIONS':
                return {
                    'direction': 'BEARISH',
                    'strength': 'MODERATE',
                    'action': 'VWAP acting as resistance. Watch for acceptance below.'
                }
            else:
                return {
                    'direction': 'NEUTRAL',
                    'strength': 'WEAK',
                    'action': 'Price rotating around VWAP. Wait for clear direction.'
                }
        else:
            return {
                'direction': 'MIXED',
                'strength': 'WEAK',
                'action': 'Mixed signals. Wait for clarity.'
            }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED CHART ANALYSIS 2026
# ═══════════════════════════════════════════════════════════════════════════════

class ChartAnalysis2026:
    """
    Unified chart analysis for 2026.

    Combines:
    - Range Structure
    - Volatility Patterns
    - VWAP Behavior

    Output:
    - State: Balance / Expansion / Exhaustion
    - Risk: Low / Medium / High
    """

    def __init__(self):
        self.range_analyzer = RangeStructureAnalyzer()
        self.vol_analyzer = VolatilityPatternAnalyzer()
        self.vwap_analyzer = VWAPBehaviorAnalyzer()

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive chart analysis.
        """
        if df is None or df.empty:
            return {
                'state': 'UNKNOWN',
                'risk': 'UNKNOWN',
                'valid': False
            }

        # Get current price
        close_col = 'close' if 'close' in df.columns else 'Close'
        current_price = df[close_col].iloc[-1]

        # Range analysis
        pd_levels = self.range_analyzer.calculate_previous_day_levels(df)
        or_levels = self.range_analyzer.calculate_opening_range(df, 15)
        range_position = self.range_analyzer.analyze_range_position(current_price, pd_levels, or_levels)

        # Volatility analysis
        vol_regime = self.vol_analyzer.classify_volatility_regime(df)

        # VWAP analysis
        vwap_behavior = self.vwap_analyzer.analyze_vwap_behavior(df)

        # Determine state
        state = self._determine_state(range_position, vol_regime, vwap_behavior)

        # Determine risk
        risk = self._determine_risk(vol_regime, range_position)

        return {
            'timestamp': datetime.now(IST).isoformat(),
            'current_price': round(current_price, 2),
            # Component analyses
            'pd_levels': pd_levels,
            'or_levels': or_levels,
            'range_position': range_position,
            'volatility': vol_regime,
            'vwap': vwap_behavior,
            # Actionable outputs
            'state': state,
            'risk': risk,
            # Summary
            'summary': self._generate_summary(state, risk, range_position, vwap_behavior)
        }

    def _determine_state(
        self,
        range_pos: Dict,
        vol: Dict,
        vwap: Dict
    ) -> Dict[str, Any]:
        """
        Determine market state: Balance / Expansion / Exhaustion
        """
        range_state = range_pos.get('range_state', 'UNKNOWN')
        vol_regime = vol.get('regime', 'NORMAL') if vol.get('valid') else 'NORMAL'

        # Balance state
        if range_state == 'BALANCE' and vol_regime in ['COMPRESSED', 'LOW']:
            state = 'BALANCE'
            description = 'Price consolidating. Expect range-bound action.'

        # Expansion state
        elif range_state == 'EXPANSION' or vol_regime == 'ELEVATED':
            if vol.get('expansion_data', {}).get('expansion_detected'):
                state = 'EXPANSION'
                description = 'Trend move in progress. Follow momentum.'
            else:
                state = 'BREAKOUT_ATTEMPT'
                description = 'Testing range boundaries. Watch for confirmation.'

        # Exhaustion state (extreme vol after expansion)
        elif vol_regime == 'EXTREME':
            state = 'EXHAUSTION'
            description = 'Extreme volatility. Move may be overextended.'

        # Compression before breakout
        elif vol_regime == 'COMPRESSED':
            if vol.get('compression_data', {}).get('breakout_imminent'):
                state = 'BREAKOUT_IMMINENT'
                description = 'Volatility squeezed. Breakout expected soon.'
            else:
                state = 'BALANCE'
                description = 'Low volatility consolidation.'

        else:
            state = 'TRANSITIONING'
            description = 'Mixed signals. Market in transition.'

        return {
            'state': state,
            'description': description,
            'range_contribution': range_state,
            'vol_contribution': vol_regime
        }

    def _determine_risk(
        self,
        vol: Dict,
        range_pos: Dict
    ) -> Dict[str, Any]:
        """
        Determine risk level: Low / Medium / High
        """
        risk_score = 0

        # Volatility contribution
        vol_regime = vol.get('regime', 'NORMAL') if vol.get('valid') else 'NORMAL'
        vol_risk = {
            'COMPRESSED': 10,
            'LOW': 15,
            'NORMAL': 30,
            'ELEVATED': 50,
            'EXTREME': 80
        }.get(vol_regime, 30)
        risk_score += vol_risk

        # Position contribution (near boundaries = higher risk)
        if range_pos.get('distance_to_key'):
            distance = range_pos['distance_to_key']
            if distance < 20:  # Very close to key level
                risk_score += 20

        # Classify risk
        if risk_score < 30:
            risk_level = 'LOW'
            recommendation = 'Favorable for trading. Normal position size.'
        elif risk_score < 60:
            risk_level = 'MEDIUM'
            recommendation = 'Moderate risk. Consider smaller positions.'
        else:
            risk_level = 'HIGH'
            recommendation = 'High risk environment. Trade with caution.'

        return {
            'level': risk_level,
            'score': risk_score,
            'recommendation': recommendation
        }

    def _generate_summary(
        self,
        state: Dict,
        risk: Dict,
        range_pos: Dict,
        vwap: Dict
    ) -> str:
        """Generate summary string"""
        state_emoji = {
            'BALANCE': '⚖️',
            'EXPANSION': '🚀',
            'EXHAUSTION': '😤',
            'BREAKOUT_IMMINENT': '⏰',
            'BREAKOUT_ATTEMPT': '🔥',
            'TRANSITIONING': '🔄'
        }.get(state['state'], '❓')

        risk_emoji = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴'
        }.get(risk['level'], '⚪')

        parts = [
            f"{state_emoji} State: {state['state']}",
            f"{risk_emoji} Risk: {risk['level']}"
        ]

        if vwap.get('valid'):
            vwap_pos = vwap.get('vwap_position', 'UNKNOWN')
            parts.append(f"VWAP: {vwap_pos}")

        return ' | '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_chart_2026(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick chart analysis with 2026 methodology"""
    analyzer = ChartAnalysis2026()
    return analyzer.analyze(df)


def get_range_levels(df: pd.DataFrame) -> Dict[str, Any]:
    """Get PDH/PDL and Opening Range levels"""
    analyzer = RangeStructureAnalyzer()
    pd_levels = analyzer.calculate_previous_day_levels(df)
    or_levels = analyzer.calculate_opening_range(df)
    return {
        'previous_day': pd_levels,
        'opening_range': or_levels
    }


def get_volatility_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """Get current volatility regime"""
    return VolatilityPatternAnalyzer.classify_volatility_regime(df)


def get_vwap_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Get VWAP behavior analysis"""
    return VWAPBehaviorAnalyzer.analyze_vwap_behavior(df)
