"""
Multi-Timeframe Trend and Support/Resistance Analysis

Analyzes trend direction and support/resistance levels across multiple timeframes:
- 5 minutes
- 15 minutes
- 1 hour
- 4 hours
- 1 day

Author: Claude AI Assistant
Date: 2025-12-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeframeTrendResult:
    """Trend analysis result for a single timeframe"""
    timeframe: str
    trend_direction: str  # UPTREND, DOWNTREND, SIDEWAYS
    trend_strength: float  # 0-100
    current_price: float

    # Support Levels
    support_1: float
    support_2: float
    support_3: float

    # Resistance Levels
    resistance_1: float
    resistance_2: float
    resistance_3: float

    # Technical Indicators
    sma_20: float
    sma_50: float
    sma_200: float
    rsi: float
    macd: float
    macd_signal: float

    # Price Position
    distance_from_sma20_pct: float
    distance_from_support1_pct: float
    distance_from_resistance1_pct: float

    # Trend Quality
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool

    timestamp: datetime = None


class MultiTimeframeAnalyzer:
    """Analyzes trend and S/R across multiple timeframes"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_all_timeframes(
        self,
        data_1d: pd.DataFrame,
        data_4h: Optional[pd.DataFrame] = None,
        data_1h: Optional[pd.DataFrame] = None,
        data_15m: Optional[pd.DataFrame] = None,
        data_5m: Optional[pd.DataFrame] = None
    ) -> Dict[str, TimeframeTrendResult]:
        """
        Analyze all timeframes and return results

        Args:
            data_1d: Daily OHLC data
            data_4h: 4-hour OHLC data (optional)
            data_1h: 1-hour OHLC data (optional)
            data_15m: 15-minute OHLC data (optional)
            data_5m: 5-minute OHLC data (optional)

        Returns:
            Dict mapping timeframe to analysis result
        """
        results = {}

        try:
            # Analyze each timeframe (ordered from smallest to largest)
            if data_5m is not None and not data_5m.empty:
                results['5m'] = self.analyze_timeframe(data_5m, '5m')

            if data_15m is not None and not data_15m.empty:
                results['15m'] = self.analyze_timeframe(data_15m, '15m')

            if data_1h is not None and not data_1h.empty:
                results['1h'] = self.analyze_timeframe(data_1h, '1h')

            if data_4h is not None and not data_4h.empty:
                results['4h'] = self.analyze_timeframe(data_4h, '4h')

            if data_1d is not None and not data_1d.empty:
                results['1d'] = self.analyze_timeframe(data_1d, '1d')

            return results

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}

    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> TimeframeTrendResult:
        """
        Analyze a single timeframe

        Args:
            df: OHLC DataFrame
            timeframe: Timeframe identifier (15m, 1h, 4h, 1d)

        Returns:
            TimeframeTrendResult
        """
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low']
            if not all(col in df.columns or col.lower() in df.columns for col in required_cols):
                # Try to standardize column names
                df = df.rename(columns=str.lower)

            # Get close, high, low prices
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            high = df['high'].values if 'high' in df.columns else df['High'].values
            low = df['low'].values if 'low' in df.columns else df['Low'].values

            current_price = float(close[-1])

            # Calculate moving averages
            sma_20 = self._calculate_sma(close, 20)
            sma_50 = self._calculate_sma(close, 50)
            sma_200 = self._calculate_sma(close, 200)

            # Calculate RSI
            rsi = self._calculate_rsi(close, 14)

            # Calculate MACD
            macd, macd_signal = self._calculate_macd(close)

            # Identify trend direction
            trend_direction, trend_strength = self._identify_trend(
                close, sma_20, sma_50, sma_200
            )

            # Find support and resistance levels
            supports, resistances = self._find_support_resistance(
                high, low, close, current_price
            )

            # Analyze price structure (HH, HL, LH, LL)
            higher_highs, higher_lows, lower_highs, lower_lows = \
                self._analyze_price_structure(high, low)

            # Calculate distances
            distance_from_sma20 = ((current_price - sma_20) / current_price) * 100
            distance_from_support1 = ((current_price - supports[0]) / current_price) * 100 \
                if supports[0] > 0 else 0
            distance_from_resistance1 = ((resistances[0] - current_price) / current_price) * 100 \
                if resistances[0] > 0 else 0

            result = TimeframeTrendResult(
                timeframe=timeframe,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                current_price=current_price,
                support_1=supports[0],
                support_2=supports[1],
                support_3=supports[2],
                resistance_1=resistances[0],
                resistance_2=resistances[1],
                resistance_3=resistances[2],
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                distance_from_sma20_pct=distance_from_sma20,
                distance_from_support1_pct=distance_from_support1,
                distance_from_resistance1_pct=distance_from_resistance1,
                higher_highs=higher_highs,
                higher_lows=higher_lows,
                lower_highs=lower_highs,
                lower_lows=lower_lows,
                timestamp=datetime.now()
            )

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe}: {e}")
            # Return neutral result
            return self._get_neutral_result(timeframe, df)

    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return float(prices[-1])
            return float(np.mean(prices[-period:]))
        except Exception:
            return 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi)
        except Exception:
            return 50.0

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        try:
            if len(prices) < slow:
                return 0.0, 0.0

            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)

            macd = ema_fast - ema_slow

            # For signal, we need MACD history
            macd_line = []
            for i in range(slow, len(prices)):
                fast_ema = self._calculate_ema(prices[:i+1], fast)
                slow_ema = self._calculate_ema(prices[:i+1], slow)
                macd_line.append(fast_ema - slow_ema)

            if len(macd_line) >= signal:
                macd_signal = self._calculate_ema(np.array(macd_line), signal)
            else:
                macd_signal = macd

            return float(macd), float(macd_signal)
        except Exception:
            return 0.0, 0.0

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return float(prices[-1])

            multiplier = 2 / (period + 1)
            ema = np.mean(prices[:period])

            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return float(ema)
        except Exception:
            return 0.0

    def _identify_trend(
        self,
        close: np.ndarray,
        sma_20: float,
        sma_50: float,
        sma_200: float
    ) -> Tuple[str, float]:
        """Identify trend direction and strength"""
        try:
            current_price = float(close[-1])

            # Count bullish and bearish signals
            bullish_signals = 0
            bearish_signals = 0

            # Price vs SMAs
            if current_price > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if current_price > sma_50:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if sma_200 > 0:
                if current_price > sma_200:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # SMA alignment
            if sma_20 > sma_50:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if sma_50 > 0 and sma_200 > 0:
                if sma_50 > sma_200:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # Recent price action
            if len(close) >= 5:
                recent_trend = (close[-1] - close[-5]) / close[-5]
                if recent_trend > 0.005:
                    bullish_signals += 1
                elif recent_trend < -0.005:
                    bearish_signals += 1

            total_signals = bullish_signals + bearish_signals

            if bullish_signals > bearish_signals + 2:
                trend = "UPTREND"
                strength = (bullish_signals / total_signals) * 100
            elif bearish_signals > bullish_signals + 2:
                trend = "DOWNTREND"
                strength = (bearish_signals / total_signals) * 100
            else:
                trend = "SIDEWAYS"
                strength = 50.0

            return trend, strength

        except Exception:
            return "SIDEWAYS", 50.0

    def _find_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        current_price: float
    ) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            # Use pivot points method
            supports = []
            resistances = []

            # Calculate pivot point (standard)
            if len(high) >= 1 and len(low) >= 1 and len(close) >= 1:
                pivot = (high[-1] + low[-1] + close[-1]) / 3

                # Calculate support and resistance levels
                r1 = 2 * pivot - low[-1]
                r2 = pivot + (high[-1] - low[-1])
                r3 = high[-1] + 2 * (pivot - low[-1])

                s1 = 2 * pivot - high[-1]
                s2 = pivot - (high[-1] - low[-1])
                s3 = low[-1] - 2 * (high[-1] - pivot)

                # Sort and filter
                all_resistances = [r for r in [r1, r2, r3] if r > current_price]
                all_supports = [s for s in [s1, s2, s3] if s < current_price]

                resistances = sorted(all_resistances)[:3]
                supports = sorted(all_supports, reverse=True)[:3]

            # Add swing highs/lows
            swing_period = min(20, len(close) - 1)
            if swing_period >= 5:
                for i in range(swing_period, len(close)):
                    # Swing high
                    if all(high[i] > high[i-j] for j in range(1, min(3, i))) and \
                       all(high[i] > high[i+j] for j in range(1, min(3, len(close)-i))):
                        if high[i] > current_price:
                            resistances.append(float(high[i]))
                        elif high[i] < current_price:
                            supports.append(float(high[i]))

                    # Swing low
                    if all(low[i] < low[i-j] for j in range(1, min(3, i))) and \
                       all(low[i] < low[i+j] for j in range(1, min(3, len(close)-i))):
                        if low[i] < current_price:
                            supports.append(float(low[i]))
                        elif low[i] > current_price:
                            resistances.append(float(low[i]))

            # Ensure we have 3 levels each
            resistances = sorted(set(resistances))[:3]
            supports = sorted(set(supports), reverse=True)[:3]

            # Pad with zeros if needed
            while len(resistances) < 3:
                resistances.append(0.0)
            while len(supports) < 3:
                supports.append(0.0)

            return supports[:3], resistances[:3]

        except Exception as e:
            self.logger.warning(f"Error finding S/R: {e}")
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def _analyze_price_structure(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[bool, bool, bool, bool]:
        """Analyze if price is making HH, HL, LH, LL"""
        try:
            if len(high) < 10 or len(low) < 10:
                return False, False, False, False

            # Look at recent swings
            recent_highs = high[-10:]
            recent_lows = low[-10:]

            # Higher highs: last high > previous high
            higher_highs = recent_highs[-1] > recent_highs[-5] and \
                          recent_highs[-5] > recent_highs[0]

            # Higher lows: last low > previous low
            higher_lows = recent_lows[-1] > recent_lows[-5] and \
                         recent_lows[-5] > recent_lows[0]

            # Lower highs: last high < previous high
            lower_highs = recent_highs[-1] < recent_highs[-5] and \
                         recent_highs[-5] < recent_highs[0]

            # Lower lows: last low < previous low
            lower_lows = recent_lows[-1] < recent_lows[-5] and \
                        recent_lows[-5] < recent_lows[0]

            return higher_highs, higher_lows, lower_highs, lower_lows

        except Exception:
            return False, False, False, False

    def _get_neutral_result(
        self,
        timeframe: str,
        df: pd.DataFrame
    ) -> TimeframeTrendResult:
        """Get neutral result when analysis fails"""
        try:
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            current_price = float(close[-1])
        except Exception:
            current_price = 0.0

        return TimeframeTrendResult(
            timeframe=timeframe,
            trend_direction="SIDEWAYS",
            trend_strength=50.0,
            current_price=current_price,
            support_1=0.0,
            support_2=0.0,
            support_3=0.0,
            resistance_1=0.0,
            resistance_2=0.0,
            resistance_3=0.0,
            sma_20=0.0,
            sma_50=0.0,
            sma_200=0.0,
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            distance_from_sma20_pct=0.0,
            distance_from_support1_pct=0.0,
            distance_from_resistance1_pct=0.0,
            higher_highs=False,
            higher_lows=False,
            lower_highs=False,
            lower_lows=False,
            timestamp=datetime.now()
        )


def create_mtf_summary_table(
    results: Dict[str, TimeframeTrendResult]
) -> pd.DataFrame:
    """
    Create a summary table from multi-timeframe results

    Args:
        results: Dict mapping timeframe to analysis result

    Returns:
        DataFrame with summary table
    """
    rows = []

    for timeframe in ['15m', '1h', '4h', '1d']:
        if timeframe not in results:
            continue

        result = results[timeframe]

        # Determine trend emoji
        trend_emoji = {
            'UPTREND': '🟢 ↗',
            'DOWNTREND': '🔴 ↘',
            'SIDEWAYS': '🟡 →'
        }.get(result.trend_direction, '⚪')

        row = {
            'Timeframe': timeframe.upper(),
            'Trend': f"{trend_emoji} {result.trend_direction}",
            'Strength': f"{result.trend_strength:.0f}%",
            'Current Price': f"₹{result.current_price:,.2f}",
            'RSI': f"{result.rsi:.1f}",
            'Support 1': f"₹{result.support_1:,.2f}" if result.support_1 > 0 else 'N/A',
            'Support 2': f"₹{result.support_2:,.2f}" if result.support_2 > 0 else 'N/A',
            'Support 3': f"₹{result.support_3:,.2f}" if result.support_3 > 0 else 'N/A',
            'Resistance 1': f"₹{result.resistance_1:,.2f}" if result.resistance_1 > 0 else 'N/A',
            'Resistance 2': f"₹{result.resistance_2:,.2f}" if result.resistance_2 > 0 else 'N/A',
            'Resistance 3': f"₹{result.resistance_3:,.2f}" if result.resistance_3 > 0 else 'N/A',
            'SMA 20': f"₹{result.sma_20:,.2f}" if result.sma_20 > 0 else 'N/A',
            'SMA 50': f"₹{result.sma_50:,.2f}" if result.sma_50 > 0 else 'N/A',
            'SMA 200': f"₹{result.sma_200:,.2f}" if result.sma_200 > 0 else 'N/A',
            'Price Structure': _get_price_structure_text(result)
        }

        rows.append(row)

    return pd.DataFrame(rows)


def _get_price_structure_text(result: TimeframeTrendResult) -> str:
    """Get price structure description"""
    structures = []

    if result.higher_highs and result.higher_lows:
        structures.append("HH+HL")
    elif result.lower_highs and result.lower_lows:
        structures.append("LH+LL")
    elif result.higher_highs:
        structures.append("HH")
    elif result.higher_lows:
        structures.append("HL")
    elif result.lower_highs:
        structures.append("LH")
    elif result.lower_lows:
        structures.append("LL")
    else:
        structures.append("Choppy")

    return ", ".join(structures) if structures else "Neutral"
