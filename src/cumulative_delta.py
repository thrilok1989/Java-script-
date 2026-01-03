"""
Cumulative Delta (CVD) - Diamond Level Feature
===============================================
Institutional-level Order Flow Analysis

The CVD measures REAL buying vs selling pressure by analyzing:
1. Delta = Buy Volume - Sell Volume at each price level
2. Cumulative Delta = Running sum of delta over time
3. CVD Divergence = Price makes new high/low but CVD doesn't confirm

KEY INSIGHT: When price rises but CVD falls = HIDDEN SELLING (distribution)
             When price falls but CVD rises = HIDDEN BUYING (accumulation)

This is what institutions use - they can't hide their order flow!
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import streamlit as st


@dataclass
class DeltaBar:
    """Single bar delta analysis"""
    timestamp: datetime
    price: float
    delta: float  # Buy volume - Sell volume
    cumulative_delta: float  # Running sum
    delta_percent: float  # Delta as % of total volume
    aggressive_ratio: float  # Aggressive buyers / Aggressive sellers


@dataclass
class CVDSignal:
    """CVD Divergence Signal"""
    signal_type: str  # 'BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE', 'ABSORPTION', 'EXHAUSTION'
    strength: float  # 0-100
    price_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    cvd_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    description: str
    trade_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0-100
    lookback_bars: int
    price_change_percent: float
    cvd_change_percent: float


@dataclass
class CVDAnalysis:
    """Complete CVD Analysis Result"""
    current_cvd: float
    cvd_trend: str  # 'RISING', 'FALLING', 'FLAT'
    cvd_momentum: float  # Rate of change
    signals: List[CVDSignal]
    delta_bars: List[DeltaBar]
    overall_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    institutional_activity: str  # 'ACCUMULATION', 'DISTRIBUTION', 'NONE'
    smart_money_direction: str  # 'BUYING', 'SELLING', 'NEUTRAL'
    score: float  # 0-100 composite score


class CumulativeDeltaAnalyzer:
    """
    Cumulative Delta / Order Flow Analyzer

    Uses market depth and volume data to calculate:
    - Delta per bar (buy volume - sell volume)
    - Cumulative delta (running sum)
    - CVD divergences (institutional footprints)
    - Absorption patterns (hidden strength/weakness)
    """

    def __init__(self):
        self.cvd_history = []
        self.lookback_short = 5
        self.lookback_medium = 13
        self.lookback_long = 21
        self.divergence_threshold = 0.3  # 30% divergence triggers signal

    def estimate_delta_from_candle(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        Estimate delta from OHLCV candle data

        Logic:
        - If close > open: Buyers dominated, estimate buy volume higher
        - If close < open: Sellers dominated, estimate sell volume higher
        - Use candle body and wicks to estimate aggressive vs passive orders

        Returns: (delta, buy_volume, sell_volume)
        """
        open_price = row.get('Open', row.get('open', 0))
        high = row.get('High', row.get('high', 0))
        low = row.get('Low', row.get('low', 0))
        close = row.get('Close', row.get('close', 0))
        volume = row.get('Volume', row.get('volume', 0))

        if volume == 0 or high == low:
            return 0, 0, 0

        # Calculate candle characteristics
        body = abs(close - open_price)
        full_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low

        # Body ratio (how much of candle is body vs wicks)
        body_ratio = body / full_range if full_range > 0 else 0

        # Estimate buy/sell volume based on candle structure
        if close >= open_price:  # Bullish candle
            # More body = more aggressive buying
            # Lower wick = absorption of selling
            buy_aggression = 0.5 + (body_ratio * 0.3) + (lower_wick / full_range * 0.2 if full_range > 0 else 0)
            sell_aggression = 1 - buy_aggression
        else:  # Bearish candle
            # More body = more aggressive selling
            # Upper wick = absorption of buying
            sell_aggression = 0.5 + (body_ratio * 0.3) + (upper_wick / full_range * 0.2 if full_range > 0 else 0)
            buy_aggression = 1 - sell_aggression

        buy_volume = volume * buy_aggression
        sell_volume = volume * sell_aggression
        delta = buy_volume - sell_volume

        return delta, buy_volume, sell_volume

    def enhance_delta_with_depth(self, delta: float, market_depth: Optional[Dict]) -> float:
        """
        Enhance delta calculation using market depth data

        Market depth shows:
        - Bid/Ask imbalance
        - Order absorption
        - Spoofing patterns
        """
        if not market_depth:
            return delta

        try:
            # Get bid/ask totals
            bids = market_depth.get('bids', [])
            asks = market_depth.get('asks', [])

            total_bid_qty = sum(b.get('quantity', 0) for b in bids) if bids else 0
            total_ask_qty = sum(a.get('quantity', 0) for a in asks) if asks else 0

            if total_bid_qty + total_ask_qty == 0:
                return delta

            # Bid/Ask imbalance ratio
            imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

            # Enhance delta based on depth imbalance
            # Strong bid support + positive delta = more bullish
            # Strong ask resistance + negative delta = more bearish
            enhancement = 1 + (imbalance * 0.2)  # Up to 20% enhancement

            return delta * enhancement

        except Exception:
            return delta

    def calculate_cvd(self, df: pd.DataFrame, market_depth: Optional[Dict] = None) -> List[DeltaBar]:
        """
        Calculate Cumulative Volume Delta for all bars
        """
        delta_bars = []
        cumulative_delta = 0

        for idx, row in df.iterrows():
            delta, buy_vol, sell_vol = self.estimate_delta_from_candle(row)

            # Enhance with market depth if available
            delta = self.enhance_delta_with_depth(delta, market_depth)

            cumulative_delta += delta

            total_vol = buy_vol + sell_vol
            delta_percent = (delta / total_vol * 100) if total_vol > 0 else 0
            aggressive_ratio = (buy_vol / sell_vol) if sell_vol > 0 else float('inf')

            close = row.get('Close', row.get('close', 0))
            timestamp = idx if isinstance(idx, datetime) else datetime.now()

            delta_bars.append(DeltaBar(
                timestamp=timestamp,
                price=close,
                delta=delta,
                cumulative_delta=cumulative_delta,
                delta_percent=delta_percent,
                aggressive_ratio=aggressive_ratio
            ))

        return delta_bars

    def detect_divergence(self, delta_bars: List[DeltaBar], lookback: int = 10) -> Optional[CVDSignal]:
        """
        Detect CVD Divergence - The Holy Grail of Order Flow

        Bullish Divergence: Price makes lower low, CVD makes higher low
        Bearish Divergence: Price makes higher high, CVD makes lower high
        """
        if len(delta_bars) < lookback + 2:
            return None

        recent = delta_bars[-lookback:]

        # Get price and CVD changes
        first_price = recent[0].price
        last_price = recent[-1].price
        first_cvd = recent[0].cumulative_delta
        last_cvd = recent[-1].cumulative_delta

        price_change = (last_price - first_price) / first_price * 100 if first_price != 0 else 0
        cvd_change = (last_cvd - first_cvd) / abs(first_cvd) * 100 if first_cvd != 0 else 0

        # Determine directions
        price_dir = 'UP' if price_change > 0.1 else ('DOWN' if price_change < -0.1 else 'SIDEWAYS')
        cvd_dir = 'UP' if cvd_change > 5 else ('DOWN' if cvd_change < -5 else 'SIDEWAYS')

        # Check for divergence
        signal_type = None
        trade_bias = 'NEUTRAL'
        description = ""
        strength = 0

        # BULLISH DIVERGENCE: Price down but CVD up (hidden buying)
        if price_dir == 'DOWN' and cvd_dir == 'UP':
            signal_type = 'BULLISH_DIVERGENCE'
            trade_bias = 'BULLISH'
            strength = min(100, abs(price_change) + abs(cvd_change))
            description = f"HIDDEN ACCUMULATION: Price fell {abs(price_change):.1f}% but smart money buying (CVD +{cvd_change:.1f}%)"

        # BEARISH DIVERGENCE: Price up but CVD down (hidden selling)
        elif price_dir == 'UP' and cvd_dir == 'DOWN':
            signal_type = 'BEARISH_DIVERGENCE'
            trade_bias = 'BEARISH'
            strength = min(100, abs(price_change) + abs(cvd_change))
            description = f"HIDDEN DISTRIBUTION: Price rose {price_change:.1f}% but smart money selling (CVD {cvd_change:.1f}%)"

        # ABSORPTION: Price sideways but CVD strongly directional
        elif price_dir == 'SIDEWAYS' and cvd_dir != 'SIDEWAYS':
            signal_type = 'ABSORPTION'
            trade_bias = 'BULLISH' if cvd_dir == 'UP' else 'BEARISH'
            strength = min(80, abs(cvd_change))
            description = f"ABSORPTION: Price consolidating but CVD shows {trade_bias.lower()} pressure building"

        # EXHAUSTION: Strong price move with declining CVD momentum
        elif abs(price_change) > 0.5:
            # Check if CVD momentum is declining
            mid_cvd = recent[len(recent)//2].cumulative_delta
            first_half_change = mid_cvd - first_cvd
            second_half_change = last_cvd - mid_cvd

            if abs(second_half_change) < abs(first_half_change) * 0.5:
                signal_type = 'EXHAUSTION'
                trade_bias = 'BEARISH' if price_dir == 'UP' else 'BULLISH'
                strength = min(70, abs(price_change) * 10)
                description = f"EXHAUSTION: Price move losing momentum, CVD velocity declining"

        if signal_type:
            return CVDSignal(
                signal_type=signal_type,
                strength=strength,
                price_direction=price_dir,
                cvd_direction=cvd_dir,
                description=description,
                trade_bias=trade_bias,
                confidence=min(95, strength * 0.9),
                lookback_bars=lookback,
                price_change_percent=price_change,
                cvd_change_percent=cvd_change
            )

        return None

    def detect_institutional_activity(self, delta_bars: List[DeltaBar]) -> Tuple[str, str]:
        """
        Detect institutional accumulation/distribution patterns

        Returns: (activity_type, smart_money_direction)
        """
        if len(delta_bars) < 20:
            return 'NONE', 'NEUTRAL'

        recent = delta_bars[-20:]

        # Calculate CVD trend
        cvd_values = [d.cumulative_delta for d in recent]
        cvd_sma_short = np.mean(cvd_values[-5:])
        cvd_sma_long = np.mean(cvd_values[-15:])

        # Calculate price trend
        prices = [d.price for d in recent]
        price_sma_short = np.mean(prices[-5:])
        price_sma_long = np.mean(prices[-15:])

        cvd_trend = cvd_sma_short - cvd_sma_long
        price_trend = price_sma_short - price_sma_long

        # Accumulation: CVD rising faster than price
        if cvd_trend > 0 and (price_trend <= 0 or cvd_trend > price_trend * 100):
            return 'ACCUMULATION', 'BUYING'

        # Distribution: CVD falling while price rising or flat
        if cvd_trend < 0 and price_trend >= 0:
            return 'DISTRIBUTION', 'SELLING'

        # Strong trend confirmation
        if cvd_trend > 0 and price_trend > 0:
            return 'NONE', 'BUYING'
        if cvd_trend < 0 and price_trend < 0:
            return 'NONE', 'SELLING'

        return 'NONE', 'NEUTRAL'

    def calculate_cvd_score(self, delta_bars: List[DeltaBar], signals: List[CVDSignal],
                           inst_activity: str, smart_money_dir: str) -> float:
        """
        Calculate composite CVD score (0-100)

        Higher score = More bullish signals
        Lower score = More bearish signals
        50 = Neutral
        """
        score = 50  # Start neutral

        # Factor 1: CVD Trend (+/- 20 points)
        if len(delta_bars) >= 10:
            cvd_recent = [d.cumulative_delta for d in delta_bars[-10:]]
            cvd_change = cvd_recent[-1] - cvd_recent[0]
            cvd_momentum = cvd_change / abs(cvd_recent[0]) * 100 if cvd_recent[0] != 0 else 0
            score += min(20, max(-20, cvd_momentum * 2))

        # Factor 2: Divergence Signals (+/- 15 points each)
        for signal in signals:
            if signal.trade_bias == 'BULLISH':
                score += min(15, signal.strength * 0.15)
            elif signal.trade_bias == 'BEARISH':
                score -= min(15, signal.strength * 0.15)

        # Factor 3: Institutional Activity (+/- 10 points)
        if inst_activity == 'ACCUMULATION':
            score += 10
        elif inst_activity == 'DISTRIBUTION':
            score -= 10

        # Factor 4: Smart Money Direction (+/- 5 points)
        if smart_money_dir == 'BUYING':
            score += 5
        elif smart_money_dir == 'SELLING':
            score -= 5

        return max(0, min(100, score))

    def analyze(self, df: pd.DataFrame = None, market_depth: Optional[Dict] = None) -> Optional[CVDAnalysis]:
        """
        Main analysis function - calculates CVD and detects signals

        Args:
            df: OHLCV DataFrame (uses session_state.chart_data if None)
            market_depth: Market depth data (uses session_state.market_depth_data if None)

        Returns:
            CVDAnalysis with all findings
        """
        try:
            # Get data from session state if not provided
            if df is None:
                df = st.session_state.get('chart_data')
                if df is None:
                    df = st.session_state.get('nifty_df')

            if df is None or len(df) < 10:
                return None

            if market_depth is None:
                market_depth = st.session_state.get('market_depth_data')

            # Calculate CVD for all bars
            delta_bars = self.calculate_cvd(df, market_depth)

            if not delta_bars:
                return None

            # Detect signals at multiple timeframes
            signals = []
            for lookback in [self.lookback_short, self.lookback_medium, self.lookback_long]:
                signal = self.detect_divergence(delta_bars, lookback)
                if signal:
                    signals.append(signal)

            # Detect institutional activity
            inst_activity, smart_money_dir = self.detect_institutional_activity(delta_bars)

            # Calculate CVD trend
            recent_cvd = [d.cumulative_delta for d in delta_bars[-5:]]
            cvd_momentum = (recent_cvd[-1] - recent_cvd[0]) / abs(recent_cvd[0]) * 100 if recent_cvd[0] != 0 else 0

            if cvd_momentum > 5:
                cvd_trend = 'RISING'
            elif cvd_momentum < -5:
                cvd_trend = 'FALLING'
            else:
                cvd_trend = 'FLAT'

            # Calculate overall bias
            bullish_signals = sum(1 for s in signals if s.trade_bias == 'BULLISH')
            bearish_signals = sum(1 for s in signals if s.trade_bias == 'BEARISH')

            if bullish_signals > bearish_signals:
                overall_bias = 'BULLISH'
            elif bearish_signals > bullish_signals:
                overall_bias = 'BEARISH'
            else:
                overall_bias = smart_money_dir if smart_money_dir != 'NEUTRAL' else 'NEUTRAL'
                if overall_bias == 'BUYING':
                    overall_bias = 'BULLISH'
                elif overall_bias == 'SELLING':
                    overall_bias = 'BEARISH'

            # Calculate composite score
            score = self.calculate_cvd_score(delta_bars, signals, inst_activity, smart_money_dir)

            return CVDAnalysis(
                current_cvd=delta_bars[-1].cumulative_delta,
                cvd_trend=cvd_trend,
                cvd_momentum=cvd_momentum,
                signals=signals,
                delta_bars=delta_bars[-50:],  # Keep last 50 bars
                overall_bias=overall_bias,
                institutional_activity=inst_activity,
                smart_money_direction=smart_money_dir,
                score=score
            )

        except Exception as e:
            st.error(f"CVD Analysis error: {e}")
            return None


# Singleton instance
_cvd_analyzer = None

def get_cvd_analyzer() -> CumulativeDeltaAnalyzer:
    """Get singleton CVD Analyzer instance"""
    global _cvd_analyzer
    if _cvd_analyzer is None:
        _cvd_analyzer = CumulativeDeltaAnalyzer()
    return _cvd_analyzer


def analyze_cumulative_delta(df: pd.DataFrame = None, market_depth: Dict = None) -> Optional[CVDAnalysis]:
    """
    Convenience function to analyze cumulative delta

    Usage:
        from src.cumulative_delta import analyze_cumulative_delta

        result = analyze_cumulative_delta()
        if result:
            print(f"CVD Score: {result.score}")
            print(f"Bias: {result.overall_bias}")
            print(f"Institutional: {result.institutional_activity}")
    """
    analyzer = get_cvd_analyzer()
    return analyzer.analyze(df, market_depth)
