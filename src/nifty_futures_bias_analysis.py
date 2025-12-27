"""
NIFTY Futures Bias Analysis Module

Provides comprehensive bias analysis specifically for NIFTY futures:
- OI Bias (Futures Open Interest patterns)
- Volume Bias (Futures trading volume)
- Basis Bias (Premium/Discount patterns)
- Rollover Bias (Near to next month rollover)
- Participant Bias (FII/DII/Pro positioning)
- Cost of Carry Bias (Arbitrage opportunities)
- Calendar Spread Bias (Inter-month spreads)
- Combined Futures Bias Score

Author: Claude AI Assistant
Date: 2025-12-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FuturesBiasResult:
    """Complete futures bias analysis result"""

    # Individual Bias Scores (-100 to +100, negative = bearish, positive = bullish)
    oi_bias: float
    volume_bias: float
    basis_bias: float
    rollover_bias: float
    participant_bias: float
    cost_of_carry_bias: float
    calendar_spread_bias: float
    price_momentum_bias: float
    buildup_unwinding_bias: float

    # Combined Score
    combined_bias: float

    # Confidence Levels (0-100)
    oi_confidence: float
    volume_confidence: float
    basis_confidence: float
    rollover_confidence: float
    participant_confidence: float

    # Overall
    overall_direction: str  # BULLISH, BEARISH, NEUTRAL
    overall_confidence: float

    # Detailed Metrics
    futures_oi_change_pct: float
    futures_volume_change_pct: float
    basis_points: float
    rollover_percentage: float
    fii_net_position: float
    arbitrage_opportunity: str

    # Signal Strength
    signal_strength: str  # STRONG, MODERATE, WEAK
    confluence_count: int  # How many indicators agree

    timestamp: datetime = None


class NiftyFuturesBiasAnalyzer:
    """Analyzes bias indicators specifically for NIFTY futures"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_comprehensive_bias(
        self,
        futures_data: Dict,
        spot_price: float,
        participant_data: Optional[Dict] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> FuturesBiasResult:
        """
        Comprehensive bias analysis for NIFTY futures

        Args:
            futures_data: Current futures market data
            spot_price: Current NIFTY spot price
            participant_data: FII/DII/Pro positioning data
            historical_data: Historical futures data for comparison

        Returns:
            FuturesBiasResult with all bias indicators
        """
        try:
            self.logger.info("🎯 Analyzing NIFTY Futures Bias...")

            # Extract individual biases
            oi_bias, oi_conf = self._analyze_oi_bias(futures_data, historical_data)
            volume_bias, vol_conf = self._analyze_volume_bias(futures_data, historical_data)
            basis_bias, basis_conf = self._analyze_basis_bias(futures_data, spot_price)
            rollover_bias, roll_conf = self._analyze_rollover_bias(futures_data)
            participant_bias, part_conf = self._analyze_participant_bias(participant_data)
            coc_bias = self._analyze_cost_of_carry_bias(futures_data, spot_price)
            spread_bias = self._analyze_calendar_spread_bias(futures_data)
            momentum_bias = self._analyze_price_momentum_bias(futures_data, historical_data)
            buildup_bias = self._analyze_buildup_unwinding_bias(futures_data)

            # Combine biases with weighted average
            weights = {
                'oi': 0.15,
                'volume': 0.10,
                'basis': 0.15,
                'rollover': 0.12,
                'participant': 0.20,
                'coc': 0.08,
                'spread': 0.05,
                'momentum': 0.10,
                'buildup': 0.05
            }

            combined_bias = (
                oi_bias * weights['oi'] +
                volume_bias * weights['volume'] +
                basis_bias * weights['basis'] +
                rollover_bias * weights['rollover'] +
                participant_bias * weights['participant'] +
                coc_bias * weights['coc'] +
                spread_bias * weights['spread'] +
                momentum_bias * weights['momentum'] +
                buildup_bias * weights['buildup']
            )

            # Overall direction
            if combined_bias > 20:
                direction = "BULLISH"
            elif combined_bias < -20:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            # Calculate overall confidence (average of individual confidences)
            overall_conf = np.mean([oi_conf, vol_conf, basis_conf, roll_conf, part_conf])

            # Count confluence (how many indicators agree with direction)
            biases = [oi_bias, volume_bias, basis_bias, rollover_bias,
                     participant_bias, coc_bias, spread_bias, momentum_bias, buildup_bias]

            if combined_bias > 0:
                confluence = sum(1 for b in biases if b > 10)
            elif combined_bias < 0:
                confluence = sum(1 for b in biases if b < -10)
            else:
                confluence = sum(1 for b in biases if -10 <= b <= 10)

            # Signal strength
            if overall_conf > 70 and confluence >= 6:
                strength = "STRONG"
            elif overall_conf > 50 and confluence >= 4:
                strength = "MODERATE"
            else:
                strength = "WEAK"

            # Extract metrics
            futures_oi_change = futures_data.get('oi_change_pct', 0.0)
            futures_vol_change = futures_data.get('volume_change_pct', 0.0)
            basis_pts = futures_data.get('basis', 0.0)
            rollover_pct = futures_data.get('rollover_percentage', 0.0)
            fii_net = participant_data.get('fii_net_futures', 0.0) if participant_data else 0.0
            arb_opp = "YES" if abs(coc_bias) > 50 else "NO"

            result = FuturesBiasResult(
                oi_bias=oi_bias,
                volume_bias=volume_bias,
                basis_bias=basis_bias,
                rollover_bias=rollover_bias,
                participant_bias=participant_bias,
                cost_of_carry_bias=coc_bias,
                calendar_spread_bias=spread_bias,
                price_momentum_bias=momentum_bias,
                buildup_unwinding_bias=buildup_bias,
                combined_bias=combined_bias,
                oi_confidence=oi_conf,
                volume_confidence=vol_conf,
                basis_confidence=basis_conf,
                rollover_confidence=roll_conf,
                participant_confidence=part_conf,
                overall_direction=direction,
                overall_confidence=overall_conf,
                futures_oi_change_pct=futures_oi_change,
                futures_volume_change_pct=futures_vol_change,
                basis_points=basis_pts,
                rollover_percentage=rollover_pct,
                fii_net_position=fii_net,
                arbitrage_opportunity=arb_opp,
                signal_strength=strength,
                confluence_count=confluence,
                timestamp=datetime.now()
            )

            self.logger.info(f"✅ Futures Bias: {direction} ({combined_bias:.1f}) - {strength}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error in futures bias analysis: {e}")
            # Return neutral bias on error
            return self._get_neutral_bias()

    def _analyze_oi_bias(
        self,
        futures_data: Dict,
        historical_data: Optional[pd.DataFrame]
    ) -> Tuple[float, float]:
        """
        Analyze OI bias for futures

        Returns: (bias_score, confidence)
        """
        try:
            oi_change = futures_data.get('oi_change_pct', 0.0)
            price_change = futures_data.get('price_change_pct', 0.0)

            # Long buildup: OI↑ + Price↑ = Bullish
            # Short buildup: OI↑ + Price↓ = Bearish
            # Long unwinding: OI↓ + Price↓ = Bearish
            # Short covering: OI↓ + Price↑ = Bullish

            if oi_change > 5:  # OI increasing
                if price_change > 1:
                    bias = 60  # Long buildup
                    conf = 70
                elif price_change < -1:
                    bias = -60  # Short buildup
                    conf = 70
                else:
                    bias = 0
                    conf = 40
            elif oi_change < -5:  # OI decreasing
                if price_change > 1:
                    bias = 40  # Short covering
                    conf = 60
                elif price_change < -1:
                    bias = -40  # Long unwinding
                    conf = 60
                else:
                    bias = 0
                    conf = 40
            else:
                bias = 0
                conf = 30

            return bias, conf

        except Exception as e:
            self.logger.warning(f"OI bias calculation error: {e}")
            return 0.0, 30.0

    def _analyze_volume_bias(
        self,
        futures_data: Dict,
        historical_data: Optional[pd.DataFrame]
    ) -> Tuple[float, float]:
        """Analyze volume bias"""
        try:
            volume_change = futures_data.get('volume_change_pct', 0.0)
            price_change = futures_data.get('price_change_pct', 0.0)

            # High volume with price rise = Bullish
            # High volume with price fall = Bearish
            # Low volume = Weak signal

            if volume_change > 20:  # High volume
                if price_change > 1:
                    bias = 50
                    conf = 65
                elif price_change < -1:
                    bias = -50
                    conf = 65
                else:
                    bias = 0
                    conf = 40
            elif volume_change < -20:  # Low volume
                bias = 0
                conf = 25
            else:
                bias = price_change * 10  # Proportional
                conf = 40

            return bias, conf

        except Exception as e:
            self.logger.warning(f"Volume bias calculation error: {e}")
            return 0.0, 30.0

    def _analyze_basis_bias(
        self,
        futures_data: Dict,
        spot_price: float
    ) -> Tuple[float, float]:
        """Analyze basis (futures - spot) bias"""
        try:
            futures_price = futures_data.get('futures_price', spot_price)
            basis = futures_price - spot_price
            basis_pct = (basis / spot_price) * 100

            # Large premium = Bullish (people paying more for futures)
            # Large discount = Bearish (futures cheaper than spot)
            # Normal range: -0.5% to +1.5%

            if basis_pct > 1.5:
                bias = 70  # Strong premium = Bullish
                conf = 75
            elif basis_pct > 0.5:
                bias = 40  # Moderate premium
                conf = 60
            elif basis_pct < -0.5:
                bias = -70  # Discount = Bearish
                conf = 75
            elif basis_pct < 0:
                bias = -30  # Slight discount
                conf = 50
            else:
                bias = 20  # Normal premium
                conf = 50

            return bias, conf

        except Exception as e:
            self.logger.warning(f"Basis bias calculation error: {e}")
            return 0.0, 30.0

    def _analyze_rollover_bias(self, futures_data: Dict) -> Tuple[float, float]:
        """Analyze rollover bias"""
        try:
            rollover_pct = futures_data.get('rollover_percentage', 0.0)
            rollover_trend = futures_data.get('rollover_trend', 'NEUTRAL')

            # High rollover = Bullish (traders rolling positions forward)
            # Low rollover = Bearish (traders exiting)
            # Early rollover = Very bullish

            if rollover_pct > 80:
                bias = 70 if rollover_trend == 'EARLY' else 50
                conf = 70
            elif rollover_pct > 60:
                bias = 40
                conf = 60
            elif rollover_pct < 30:
                bias = -60
                conf = 70
            elif rollover_pct < 50:
                bias = -30
                conf = 50
            else:
                bias = 0
                conf = 40

            return bias, conf

        except Exception as e:
            self.logger.warning(f"Rollover bias calculation error: {e}")
            return 0.0, 30.0

    def _analyze_participant_bias(
        self,
        participant_data: Optional[Dict]
    ) -> Tuple[float, float]:
        """Analyze FII/DII/Pro participant bias"""
        try:
            if not participant_data:
                return 0.0, 20.0

            fii_net = participant_data.get('fii_net_futures', 0.0)
            dii_net = participant_data.get('dii_net_futures', 0.0)
            pro_net = participant_data.get('pro_net_futures', 0.0)

            # FII positioning is most important (weight: 60%)
            # DII positioning (weight: 25%)
            # Pro positioning (weight: 15%)

            # Normalize to -100 to +100 range
            fii_bias = np.clip(fii_net / 1000, -100, 100) * 0.60
            dii_bias = np.clip(dii_net / 500, -100, 100) * 0.25
            pro_bias = np.clip(pro_net / 300, -100, 100) * 0.15

            total_bias = fii_bias + dii_bias + pro_bias

            # Confidence based on agreement
            if (fii_net > 0 and dii_net > 0) or (fii_net < 0 and dii_net < 0):
                conf = 80  # FII and DII agree
            elif fii_net != 0:
                conf = 60  # At least FII has position
            else:
                conf = 30

            return total_bias, conf

        except Exception as e:
            self.logger.warning(f"Participant bias calculation error: {e}")
            return 0.0, 30.0

    def _analyze_cost_of_carry_bias(
        self,
        futures_data: Dict,
        spot_price: float
    ) -> float:
        """Analyze cost of carry bias"""
        try:
            theoretical_futures = futures_data.get('theoretical_futures_price', 0.0)
            actual_futures = futures_data.get('futures_price', spot_price)

            if theoretical_futures == 0:
                return 0.0

            mispricing = actual_futures - theoretical_futures
            mispricing_pct = (mispricing / theoretical_futures) * 100

            # Overpriced futures = Bullish sentiment
            # Underpriced futures = Bearish sentiment

            if mispricing_pct > 0.3:
                return 60  # Overpriced = Bullish
            elif mispricing_pct > 0.1:
                return 30
            elif mispricing_pct < -0.3:
                return -60  # Underpriced = Bearish
            elif mispricing_pct < -0.1:
                return -30
            else:
                return 0

        except Exception as e:
            self.logger.warning(f"Cost of carry bias error: {e}")
            return 0.0

    def _analyze_calendar_spread_bias(self, futures_data: Dict) -> float:
        """Analyze calendar spread bias"""
        try:
            near_month = futures_data.get('near_month_price', 0.0)
            next_month = futures_data.get('next_month_price', 0.0)

            if near_month == 0 or next_month == 0:
                return 0.0

            spread = next_month - near_month
            spread_pct = (spread / near_month) * 100

            # Widening spread = Bullish (contango increasing)
            # Narrowing spread = Bearish
            # Backwardation = Very bearish

            if spread_pct > 1.0:
                return 50  # Wide contango
            elif spread_pct > 0.5:
                return 25
            elif spread_pct < -0.5:
                return -60  # Backwardation
            elif spread_pct < 0:
                return -30
            else:
                return 0

        except Exception as e:
            self.logger.warning(f"Calendar spread bias error: {e}")
            return 0.0

    def _analyze_price_momentum_bias(
        self,
        futures_data: Dict,
        historical_data: Optional[pd.DataFrame]
    ) -> float:
        """Analyze price momentum bias"""
        try:
            price_change_1d = futures_data.get('price_change_pct', 0.0)
            price_change_5d = futures_data.get('price_change_5d_pct', 0.0)

            # Strong uptrend = Bullish
            # Strong downtrend = Bearish

            if price_change_5d > 3 and price_change_1d > 0.5:
                return 70  # Strong uptrend
            elif price_change_5d > 1.5:
                return 40
            elif price_change_5d < -3 and price_change_1d < -0.5:
                return -70  # Strong downtrend
            elif price_change_5d < -1.5:
                return -40
            else:
                return price_change_1d * 15  # Proportional

        except Exception as e:
            self.logger.warning(f"Momentum bias error: {e}")
            return 0.0

    def _analyze_buildup_unwinding_bias(self, futures_data: Dict) -> float:
        """Analyze buildup/unwinding patterns"""
        try:
            pattern = futures_data.get('buildup_pattern', 'NEUTRAL')

            pattern_scores = {
                'LONG_BUILDUP': 70,
                'SHORT_BUILDUP': -70,
                'LONG_UNWINDING': -50,
                'SHORT_COVERING': 50,
                'NEUTRAL': 0
            }

            return pattern_scores.get(pattern, 0.0)

        except Exception as e:
            self.logger.warning(f"Buildup/unwinding bias error: {e}")
            return 0.0

    def _get_neutral_bias(self) -> FuturesBiasResult:
        """Return neutral bias result on errors"""
        return FuturesBiasResult(
            oi_bias=0.0,
            volume_bias=0.0,
            basis_bias=0.0,
            rollover_bias=0.0,
            participant_bias=0.0,
            cost_of_carry_bias=0.0,
            calendar_spread_bias=0.0,
            price_momentum_bias=0.0,
            buildup_unwinding_bias=0.0,
            combined_bias=0.0,
            oi_confidence=30.0,
            volume_confidence=30.0,
            basis_confidence=30.0,
            rollover_confidence=30.0,
            participant_confidence=30.0,
            overall_direction="NEUTRAL",
            overall_confidence=30.0,
            futures_oi_change_pct=0.0,
            futures_volume_change_pct=0.0,
            basis_points=0.0,
            rollover_percentage=0.0,
            fii_net_position=0.0,
            arbitrage_opportunity="NO",
            signal_strength="WEAK",
            confluence_count=0,
            timestamp=datetime.now()
        )

    def get_bias_summary(self, result: FuturesBiasResult) -> Dict:
        """Get human-readable bias summary"""
        return {
            'overall_direction': result.overall_direction,
            'combined_bias_score': round(result.combined_bias, 1),
            'signal_strength': result.signal_strength,
            'confidence': round(result.overall_confidence, 1),
            'confluence': f"{result.confluence_count}/9 indicators agree",
            'top_bullish_factors': self._get_top_factors(result, bullish=True),
            'top_bearish_factors': self._get_top_factors(result, bullish=False),
            'key_metrics': {
                'futures_oi_change': f"{result.futures_oi_change_pct:+.1f}%",
                'futures_volume_change': f"{result.futures_volume_change_pct:+.1f}%",
                'basis': f"{result.basis_points:+.1f} pts",
                'rollover': f"{result.rollover_percentage:.1f}%",
                'fii_position': f"{result.fii_net_position:+.0f} cr",
                'arbitrage': result.arbitrage_opportunity
            }
        }

    def _get_top_factors(
        self,
        result: FuturesBiasResult,
        bullish: bool = True
    ) -> List[str]:
        """Get top 3 bullish or bearish factors"""
        factors = {
            'OI': result.oi_bias,
            'Volume': result.volume_bias,
            'Basis': result.basis_bias,
            'Rollover': result.rollover_bias,
            'Participants': result.participant_bias,
            'Cost of Carry': result.cost_of_carry_bias,
            'Calendar Spread': result.calendar_spread_bias,
            'Momentum': result.price_momentum_bias,
            'Buildup/Unwinding': result.buildup_unwinding_bias
        }

        if bullish:
            sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
            return [f"{name} ({score:+.0f})" for name, score in sorted_factors[:3] if score > 0]
        else:
            sorted_factors = sorted(factors.items(), key=lambda x: x[1])
            return [f"{name} ({score:+.0f})" for name, score in sorted_factors[:3] if score < 0]
