"""
Option Chain 2026 Module
========================
Modern option chain analysis replacing outdated 2025 methods.

STOP (2025 logic):
- Highest OI only
- Static PCR
- CE vs PE OI comparison only
- "Support = max PE OI" blindly

ADD (2026 logic):
- OI Change Velocity (speed of change)
- Premium Decay Speed
- Writer Stress Zones
- ATM Gamma Zone
- Time-Weighted PCR (intraday evolving)

Output: Market Mood, Writer Control, Action recommendation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pytz

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# OI CHANGE VELOCITY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class OIVelocityTracker:
    """
    Track OI change velocity (speed of change) - not just OI value.

    Key Insight:
    - OI velocity tells you: Writers entering (range) or exiting (move coming)
    - Acceleration (velocity increasing) = intensity building
    """

    def __init__(self, history_size: int = 60):
        """
        Initialize tracker.

        Args:
            history_size: Number of snapshots to keep (default 60 for 1 hour of 1-min data)
        """
        self.ce_oi_history = deque(maxlen=history_size)
        self.pe_oi_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

    def update(self, ce_oi: float, pe_oi: float, timestamp: datetime = None):
        """Add new OI snapshot"""
        ts = timestamp or datetime.now(IST)
        self.ce_oi_history.append(ce_oi)
        self.pe_oi_history.append(pe_oi)
        self.timestamps.append(ts)

    def get_velocity(self, period_minutes: int = 5) -> Dict[str, float]:
        """
        Calculate OI change velocity over specified period.

        Returns:
            Dict with CE velocity, PE velocity, total velocity, and interpretation
        """
        if len(self.ce_oi_history) < 2:
            return {
                'ce_velocity': 0, 'pe_velocity': 0, 'total_velocity': 0,
                'ce_velocity_1min': 0, 'pe_velocity_1min': 0,
                'interpretation': 'INSUFFICIENT_DATA'
            }

        # Use available data or specified period
        lookback = min(period_minutes, len(self.ce_oi_history) - 1)

        ce_change = self.ce_oi_history[-1] - self.ce_oi_history[-lookback-1]
        pe_change = self.pe_oi_history[-1] - self.pe_oi_history[-lookback-1]
        total_change = ce_change + pe_change

        # Per minute velocity
        ce_velocity_1min = ce_change / lookback if lookback > 0 else 0
        pe_velocity_1min = pe_change / lookback if lookback > 0 else 0

        # Interpretation
        interpretation = self._interpret_velocity(ce_velocity_1min, pe_velocity_1min)

        return {
            'ce_velocity': ce_change,
            'pe_velocity': pe_change,
            'total_velocity': total_change,
            'ce_velocity_1min': round(ce_velocity_1min, 0),
            'pe_velocity_1min': round(pe_velocity_1min, 0),
            'period_minutes': lookback,
            'interpretation': interpretation
        }

    def get_acceleration(self, period_minutes: int = 5) -> Dict[str, float]:
        """
        Calculate OI acceleration (velocity of velocity).

        Acceleration tells you if writers are:
        - Aggressively entering (positive acceleration + positive velocity)
        - Slowing down entry (negative acceleration + positive velocity)
        - Aggressively exiting (positive acceleration + negative velocity)
        - Slowing down exit (negative acceleration + negative velocity)
        """
        if len(self.ce_oi_history) < period_minutes * 2 + 1:
            return {
                'ce_acceleration': 0, 'pe_acceleration': 0,
                'interpretation': 'INSUFFICIENT_DATA'
            }

        # Current velocity (last period)
        current_ce_vel = (self.ce_oi_history[-1] - self.ce_oi_history[-period_minutes-1]) / period_minutes
        current_pe_vel = (self.pe_oi_history[-1] - self.pe_oi_history[-period_minutes-1]) / period_minutes

        # Previous velocity (period before that)
        prev_ce_vel = (self.ce_oi_history[-period_minutes-1] - self.ce_oi_history[-2*period_minutes-1]) / period_minutes
        prev_pe_vel = (self.pe_oi_history[-period_minutes-1] - self.pe_oi_history[-2*period_minutes-1]) / period_minutes

        ce_accel = current_ce_vel - prev_ce_vel
        pe_accel = current_pe_vel - prev_pe_vel

        interpretation = self._interpret_acceleration(ce_accel, pe_accel, current_ce_vel, current_pe_vel)

        return {
            'ce_acceleration': round(ce_accel, 2),
            'pe_acceleration': round(pe_accel, 2),
            'ce_current_velocity': round(current_ce_vel, 0),
            'pe_current_velocity': round(current_pe_vel, 0),
            'interpretation': interpretation
        }

    def _interpret_velocity(self, ce_vel: float, pe_vel: float) -> str:
        """Interpret velocity signals"""
        threshold = 5000  # Significant velocity threshold

        if ce_vel > threshold and pe_vel > threshold:
            return 'WRITERS_ENTERING_BOTH'  # Range likely
        elif ce_vel > threshold and pe_vel < -threshold:
            return 'CALL_WRITING_PUT_EXIT'  # Bearish bias
        elif ce_vel < -threshold and pe_vel > threshold:
            return 'PUT_WRITING_CALL_EXIT'  # Bullish bias
        elif ce_vel < -threshold and pe_vel < -threshold:
            return 'WRITERS_EXITING_BOTH'  # Move coming
        else:
            return 'NEUTRAL'

    def _interpret_acceleration(self, ce_accel: float, pe_accel: float,
                                ce_vel: float, pe_vel: float) -> str:
        """Interpret acceleration signals"""
        accel_threshold = 1000

        # Total velocity and acceleration
        total_vel = ce_vel + pe_vel
        total_accel = ce_accel + pe_accel

        if total_accel > accel_threshold:
            if total_vel > 0:
                return 'AGGRESSIVE_WRITING'  # Range intensifying
            else:
                return 'AGGRESSIVE_EXIT'  # Big move coming
        elif total_accel < -accel_threshold:
            if total_vel > 0:
                return 'WRITING_SLOWING'  # Range weakening
            else:
                return 'EXIT_SLOWING'  # Move may be ending
        else:
            return 'STABLE'


# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM DECAY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class PremiumDecayTracker:
    """
    Track premium decay speed - Gold in 2026.

    Key Signals:
    - Both decaying fast → Balance (range)
    - One decays, other holds → Directional bias
    - Price moves but premium doesn't → TRAP
    """

    def __init__(self, history_size: int = 60):
        self.ce_premium_history = deque(maxlen=history_size)
        self.pe_premium_history = deque(maxlen=history_size)
        self.spot_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

    def update(self, ce_premium: float, pe_premium: float, spot: float, timestamp: datetime = None):
        """Add new premium snapshot"""
        ts = timestamp or datetime.now(IST)
        self.ce_premium_history.append(ce_premium)
        self.pe_premium_history.append(pe_premium)
        self.spot_history.append(spot)
        self.timestamps.append(ts)

    def get_decay_rate(self, period_minutes: int = 5) -> Dict[str, Any]:
        """
        Calculate premium decay rate.

        Returns:
            Decay rate per minute, decay % over period, interpretation
        """
        if len(self.ce_premium_history) < 2:
            return {
                'ce_decay_pct': 0, 'pe_decay_pct': 0,
                'interpretation': 'INSUFFICIENT_DATA', 'signal': 'WAIT'
            }

        lookback = min(period_minutes, len(self.ce_premium_history) - 1)

        # Calculate decay
        ce_start = self.ce_premium_history[-lookback-1]
        ce_end = self.ce_premium_history[-1]
        pe_start = self.pe_premium_history[-lookback-1]
        pe_end = self.pe_premium_history[-1]

        ce_decay_pct = ((ce_end - ce_start) / ce_start * 100) if ce_start > 0 else 0
        pe_decay_pct = ((pe_end - pe_start) / pe_start * 100) if pe_start > 0 else 0

        # Per minute decay
        ce_decay_per_min = ce_decay_pct / lookback if lookback > 0 else 0
        pe_decay_per_min = pe_decay_pct / lookback if lookback > 0 else 0

        # Get interpretation and signal
        interpretation, signal = self._interpret_decay(ce_decay_pct, pe_decay_pct)

        return {
            'ce_decay_pct': round(ce_decay_pct, 2),
            'pe_decay_pct': round(pe_decay_pct, 2),
            'ce_decay_per_min': round(ce_decay_per_min, 3),
            'pe_decay_per_min': round(pe_decay_per_min, 3),
            'period_minutes': lookback,
            'interpretation': interpretation,
            'signal': signal
        }

    def detect_premium_trap(self, period_minutes: int = 5) -> Dict[str, Any]:
        """
        Detect price vs premium divergence (TRAP).

        TRAP: Price moves but premium doesn't follow
        - Price up but CE premium not up → Bullish trap
        - Price down but PE premium not up → Bearish trap
        """
        if len(self.spot_history) < period_minutes + 1:
            return {'trap_detected': False, 'trap_type': None}

        lookback = min(period_minutes, len(self.spot_history) - 1)

        spot_change = self.spot_history[-1] - self.spot_history[-lookback-1]
        spot_change_pct = (spot_change / self.spot_history[-lookback-1] * 100)

        ce_change_pct = ((self.ce_premium_history[-1] - self.ce_premium_history[-lookback-1]) /
                         self.ce_premium_history[-lookback-1] * 100) if self.ce_premium_history[-lookback-1] > 0 else 0
        pe_change_pct = ((self.pe_premium_history[-1] - self.pe_premium_history[-lookback-1]) /
                         self.pe_premium_history[-lookback-1] * 100) if self.pe_premium_history[-lookback-1] > 0 else 0

        trap_detected = False
        trap_type = None

        # Significant price move threshold
        price_threshold = 0.3  # 0.3% move
        premium_threshold = 5  # 5% premium change expected

        if spot_change_pct > price_threshold:  # Price went up
            if ce_change_pct < premium_threshold:  # But CE premium didn't rise much
                trap_detected = True
                trap_type = 'BULLISH_TRAP'
        elif spot_change_pct < -price_threshold:  # Price went down
            if pe_change_pct < premium_threshold:  # But PE premium didn't rise much
                trap_detected = True
                trap_type = 'BEARISH_TRAP'

        return {
            'trap_detected': trap_detected,
            'trap_type': trap_type,
            'spot_change_pct': round(spot_change_pct, 2),
            'ce_premium_change_pct': round(ce_change_pct, 2),
            'pe_premium_change_pct': round(pe_change_pct, 2),
            'warning': f'Price moved {spot_change_pct:.2f}% but premium lagged' if trap_detected else None
        }

    def _interpret_decay(self, ce_decay: float, pe_decay: float) -> Tuple[str, str]:
        """Interpret decay pattern"""
        fast_decay_threshold = -2  # More than 2% decay

        if ce_decay < fast_decay_threshold and pe_decay < fast_decay_threshold:
            return 'BOTH_DECAYING_FAST', 'RANGE_LIKELY'
        elif ce_decay < fast_decay_threshold and pe_decay > 0:
            return 'CE_DECAY_PE_HOLD', 'BEARISH_BIAS'
        elif pe_decay < fast_decay_threshold and ce_decay > 0:
            return 'PE_DECAY_CE_HOLD', 'BULLISH_BIAS'
        elif ce_decay > 2 and pe_decay > 2:
            return 'BOTH_EXPANDING', 'VOLATILITY_SPIKE'
        else:
            return 'NEUTRAL', 'NO_CLEAR_SIGNAL'


# ═══════════════════════════════════════════════════════════════════════════════
# WRITER STRESS ZONE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class WriterStressZoneDetector:
    """
    Detect writer stress zones.

    Stress = Price distance from strike + Premium expansion + OI unwind

    High stress zones = possible move magnets
    Helps detect: Breakouts that will actually run vs Fake breakouts
    """

    @staticmethod
    def calculate_stress_score(
        strike: float,
        spot: float,
        premium: float,
        premium_change_pct: float,
        oi: float,
        oi_change: float,
        option_type: str = 'CE'
    ) -> Dict[str, Any]:
        """
        Calculate stress score for a strike.

        Components:
        1. Price distance stress (closer = more stress)
        2. Premium expansion (expanding = more stress)
        3. OI unwind (negative change = stress, writers exiting)
        """
        stress_score = 0
        stress_components = {}

        # 1. Price Distance Stress
        distance = abs(spot - strike)
        distance_pct = (distance / spot) * 100

        if option_type == 'CE':
            # For calls, stress increases as price approaches from below
            if spot < strike:  # OTM
                distance_stress = max(0, 2 - distance_pct) * 20  # Max 40 for ATM
            else:  # ITM - already breached
                distance_stress = 40 + min(distance_pct, 1) * 20  # 40-60 for ITM
        else:  # PE
            # For puts, stress increases as price approaches from above
            if spot > strike:  # OTM
                distance_stress = max(0, 2 - distance_pct) * 20
            else:  # ITM
                distance_stress = 40 + min(distance_pct, 1) * 20

        stress_score += distance_stress
        stress_components['distance'] = round(distance_stress, 1)

        # 2. Premium Expansion Stress
        if premium_change_pct > 10:
            premium_stress = 30
        elif premium_change_pct > 5:
            premium_stress = 20
        elif premium_change_pct > 0:
            premium_stress = 10
        else:
            premium_stress = 0  # Decaying = no stress

        stress_score += premium_stress
        stress_components['premium'] = premium_stress

        # 3. OI Unwind Stress
        oi_change_pct = (oi_change / oi * 100) if oi > 0 else 0

        if oi_change_pct < -5:  # More than 5% unwound
            oi_stress = 30
        elif oi_change_pct < -2:
            oi_stress = 20
        elif oi_change_pct < 0:
            oi_stress = 10
        else:
            oi_stress = 0  # Adding = no stress

        stress_score += oi_stress
        stress_components['oi_unwind'] = oi_stress

        # Classify stress level
        if stress_score >= 70:
            level = 'CRITICAL'
            implication = 'Writers under extreme pressure. Move magnet.'
        elif stress_score >= 50:
            level = 'HIGH'
            implication = 'Significant writer stress. Watch for break.'
        elif stress_score >= 30:
            level = 'MODERATE'
            implication = 'Some stress. Monitor closely.'
        else:
            level = 'LOW'
            implication = 'Writers comfortable.'

        return {
            'strike': strike,
            'stress_score': round(stress_score, 1),
            'stress_level': level,
            'components': stress_components,
            'implication': implication,
            'is_stress_zone': stress_score >= 50
        }

    @staticmethod
    def find_stress_zones(
        option_chain_df: pd.DataFrame,
        spot: float,
        num_strikes: int = 5
    ) -> Dict[str, Any]:
        """
        Find high stress zones in option chain.

        Returns top stress zones for both CE and PE.
        """
        ce_stress_zones = []
        pe_stress_zones = []

        for _, row in option_chain_df.iterrows():
            strike = row.get('strikePrice', row.get('strike', 0))

            # CE stress
            ce_premium = row.get('LTP_CE', row.get('ce_ltp', 0)) or 0
            ce_premium_change = row.get('ce_premium_change_pct', 0)
            ce_oi = row.get('OI_CE', row.get('ce_oi', 0)) or 0
            ce_oi_change = row.get('Chg_OI_CE', row.get('ce_oi_change', 0)) or 0

            if ce_oi > 0:
                ce_stress = WriterStressZoneDetector.calculate_stress_score(
                    strike, spot, ce_premium, ce_premium_change, ce_oi, ce_oi_change, 'CE'
                )
                ce_stress_zones.append(ce_stress)

            # PE stress
            pe_premium = row.get('LTP_PE', row.get('pe_ltp', 0)) or 0
            pe_premium_change = row.get('pe_premium_change_pct', 0)
            pe_oi = row.get('OI_PE', row.get('pe_oi', 0)) or 0
            pe_oi_change = row.get('Chg_OI_PE', row.get('pe_oi_change', 0)) or 0

            if pe_oi > 0:
                pe_stress = WriterStressZoneDetector.calculate_stress_score(
                    strike, spot, pe_premium, pe_premium_change, pe_oi, pe_oi_change, 'PE'
                )
                pe_stress_zones.append(pe_stress)

        # Sort by stress score
        ce_stress_zones = sorted(ce_stress_zones, key=lambda x: x['stress_score'], reverse=True)[:num_strikes]
        pe_stress_zones = sorted(pe_stress_zones, key=lambda x: x['stress_score'], reverse=True)[:num_strikes]

        # Find move magnets (strikes with both CE and PE stress)
        move_magnets = []
        ce_strikes = {z['strike'] for z in ce_stress_zones if z['is_stress_zone']}
        pe_strikes = {z['strike'] for z in pe_stress_zones if z['is_stress_zone']}
        magnet_strikes = ce_strikes.intersection(pe_strikes)

        for strike in magnet_strikes:
            move_magnets.append({
                'strike': strike,
                'reason': 'Both CE and PE writers stressed at this level'
            })

        return {
            'ce_stress_zones': ce_stress_zones,
            'pe_stress_zones': pe_stress_zones,
            'move_magnets': move_magnets,
            'highest_ce_stress': ce_stress_zones[0] if ce_stress_zones else None,
            'highest_pe_stress': pe_stress_zones[0] if pe_stress_zones else None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ATM GAMMA ZONE
# ═══════════════════════════════════════════════════════════════════════════════

class ATMGammaZone:
    """
    ATM Gamma Zone - where price gets "stuck" due to dealer hedging.

    ATM ± 1 strike = Gamma Zone
    Rules:
    - Chop inside gamma zone
    - Expansion only when price escapes gamma zone
    Prevents overtrading.
    """

    @staticmethod
    def identify_gamma_zone(
        atm_strike: float,
        strike_gap: float,
        spot: float,
        ce_gamma: float = None,
        pe_gamma: float = None
    ) -> Dict[str, Any]:
        """
        Identify ATM gamma zone.

        Returns zone boundaries and current position.
        """
        lower_bound = atm_strike - strike_gap
        upper_bound = atm_strike + strike_gap

        in_gamma_zone = lower_bound <= spot <= upper_bound

        # Distance to boundaries
        distance_to_upper = upper_bound - spot
        distance_to_lower = spot - lower_bound
        distance_to_nearest = min(distance_to_upper, distance_to_lower)

        # Zone position
        if in_gamma_zone:
            zone_pct = ((spot - lower_bound) / (upper_bound - lower_bound)) * 100
            if zone_pct < 33:
                position = 'LOWER_THIRD'
            elif zone_pct > 67:
                position = 'UPPER_THIRD'
            else:
                position = 'MIDDLE'
        else:
            if spot > upper_bound:
                position = 'ABOVE_ZONE'
            else:
                position = 'BELOW_ZONE'

        return {
            'atm_strike': atm_strike,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'spot': spot,
            'in_gamma_zone': in_gamma_zone,
            'position': position,
            'distance_to_upper': round(distance_to_upper, 2),
            'distance_to_lower': round(distance_to_lower, 2),
            'distance_to_escape': round(distance_to_nearest, 2),
            'trading_implication': ATMGammaZone._get_implication(in_gamma_zone, position)
        }

    @staticmethod
    def _get_implication(in_zone: bool, position: str) -> str:
        """Get trading implication"""
        if not in_zone:
            if position == 'ABOVE_ZONE':
                return 'Escaped gamma zone upward. Bullish expansion possible.'
            else:
                return 'Escaped gamma zone downward. Bearish expansion possible.'
        else:
            if position == 'MIDDLE':
                return 'Deep in gamma zone. Expect chop. Avoid directional trades.'
            elif position == 'LOWER_THIRD':
                return 'Lower gamma zone. Watch for breakdown or bounce.'
            else:
                return 'Upper gamma zone. Watch for breakout or rejection.'


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-WEIGHTED PCR
# ═══════════════════════════════════════════════════════════════════════════════

class TimeWeightedPCR:
    """
    Time-Weighted PCR - Intraday evolving, reset daily.

    NOT weekly PCR. PCR should evolve through the day.
    Recent data weighted more heavily.
    """

    def __init__(self, history_size: int = 390):  # Full day of 1-min data
        self.pcr_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.ce_oi_history = deque(maxlen=history_size)
        self.pe_oi_history = deque(maxlen=history_size)

    def update(self, ce_oi: float, pe_oi: float, timestamp: datetime = None):
        """Add new OI snapshot"""
        ts = timestamp or datetime.now(IST)

        # Check if new day - reset if so
        if self.timestamps and ts.date() != self.timestamps[-1].date():
            self.pcr_history.clear()
            self.timestamps.clear()
            self.ce_oi_history.clear()
            self.pe_oi_history.clear()

        pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0
        self.pcr_history.append(pcr)
        self.timestamps.append(ts)
        self.ce_oi_history.append(ce_oi)
        self.pe_oi_history.append(pe_oi)

    def get_time_weighted_pcr(self, decay_factor: float = 0.95) -> Dict[str, Any]:
        """
        Calculate time-weighted PCR.

        Recent data has higher weight using exponential decay.
        """
        if len(self.pcr_history) == 0:
            return {
                'tw_pcr': 1.0, 'simple_pcr': 1.0,
                'interpretation': 'NO_DATA', 'bias': 'NEUTRAL'
            }

        # Simple current PCR
        simple_pcr = self.pcr_history[-1]

        # Time-weighted PCR (exponential weighting)
        weights = [decay_factor ** i for i in range(len(self.pcr_history) - 1, -1, -1)]
        weight_sum = sum(weights)
        tw_pcr = sum(p * w for p, w in zip(self.pcr_history, weights)) / weight_sum

        # PCR trend (is it increasing or decreasing?)
        if len(self.pcr_history) >= 5:
            recent_avg = sum(list(self.pcr_history)[-5:]) / 5
            older_avg = sum(list(self.pcr_history)[-10:-5]) / 5 if len(self.pcr_history) >= 10 else recent_avg
            trend = 'RISING' if recent_avg > older_avg * 1.02 else 'FALLING' if recent_avg < older_avg * 0.98 else 'STABLE'
        else:
            trend = 'INSUFFICIENT_DATA'

        # Interpretation
        interpretation, bias = self._interpret_pcr(tw_pcr, trend)

        return {
            'tw_pcr': round(tw_pcr, 3),
            'simple_pcr': round(simple_pcr, 3),
            'trend': trend,
            'interpretation': interpretation,
            'bias': bias,
            'data_points': len(self.pcr_history)
        }

    def get_pcr_momentum(self, period: int = 15) -> Dict[str, Any]:
        """
        Calculate PCR momentum (rate of change).
        """
        if len(self.pcr_history) < period + 1:
            return {'pcr_momentum': 0, 'momentum_bias': 'NEUTRAL'}

        current = sum(list(self.pcr_history)[-5:]) / 5
        past = sum(list(self.pcr_history)[-period-5:-period]) / 5 if len(self.pcr_history) > period + 5 else current

        momentum = ((current - past) / past * 100) if past > 0 else 0

        if momentum > 5:
            momentum_bias = 'BULLISH'  # PCR rising = more puts = bullish
        elif momentum < -5:
            momentum_bias = 'BEARISH'  # PCR falling = less puts = bearish
        else:
            momentum_bias = 'NEUTRAL'

        return {
            'pcr_momentum': round(momentum, 2),
            'current_pcr': round(current, 3),
            'past_pcr': round(past, 3),
            'momentum_bias': momentum_bias
        }

    def _interpret_pcr(self, pcr: float, trend: str) -> Tuple[str, str]:
        """Interpret PCR value and trend"""
        if pcr > 1.3:
            if trend == 'RISING':
                return 'EXTREME_PUT_BUYING', 'STRONGLY_BULLISH'
            else:
                return 'HIGH_PUT_OI', 'BULLISH'
        elif pcr > 1.0:
            if trend == 'RISING':
                return 'PUT_ACCUMULATION', 'BULLISH'
            else:
                return 'BALANCED_BULLISH', 'MILDLY_BULLISH'
        elif pcr > 0.7:
            return 'BALANCED', 'NEUTRAL'
        elif pcr > 0.5:
            if trend == 'FALLING':
                return 'CALL_ACCUMULATION', 'BEARISH'
            else:
                return 'BALANCED_BEARISH', 'MILDLY_BEARISH'
        else:
            if trend == 'FALLING':
                return 'EXTREME_CALL_BUYING', 'STRONGLY_BEARISH'
            else:
                return 'HIGH_CALL_OI', 'BEARISH'


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED OPTION CHAIN 2026 ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class OptionChain2026Analyzer:
    """
    Unified option chain analyzer for 2026.

    Combines all new metrics and produces actionable output.

    Output Format:
    - Market Mood: Calm / Stressed / Aggressive
    - Writer Control: Strong / Weak
    - Action: Range / Expansion / Wait
    """

    def __init__(self):
        self.oi_tracker = OIVelocityTracker()
        self.premium_tracker = PremiumDecayTracker()
        self.tw_pcr = TimeWeightedPCR()

    def update(
        self,
        option_chain_df: pd.DataFrame,
        spot: float,
        atm_strike: float,
        strike_gap: float
    ):
        """
        Update all trackers with new data.
        """
        # Calculate totals
        total_ce_oi = option_chain_df['OI_CE'].sum() if 'OI_CE' in option_chain_df.columns else 0
        total_pe_oi = option_chain_df['OI_PE'].sum() if 'OI_PE' in option_chain_df.columns else 0

        # ATM premiums
        atm_row = option_chain_df[option_chain_df['strikePrice'] == atm_strike]
        if not atm_row.empty:
            ce_premium = atm_row['LTP_CE'].iloc[0] if 'LTP_CE' in atm_row.columns else 0
            pe_premium = atm_row['LTP_PE'].iloc[0] if 'LTP_PE' in atm_row.columns else 0
        else:
            ce_premium = pe_premium = 0

        # Update trackers
        self.oi_tracker.update(total_ce_oi, total_pe_oi)
        self.premium_tracker.update(ce_premium or 0, pe_premium or 0, spot)
        self.tw_pcr.update(total_ce_oi, total_pe_oi)

    def analyze(
        self,
        option_chain_df: pd.DataFrame,
        spot: float,
        atm_strike: float,
        strike_gap: float
    ) -> Dict[str, Any]:
        """
        Perform comprehensive 2026 option chain analysis.
        """
        # Update trackers first
        self.update(option_chain_df, spot, atm_strike, strike_gap)

        # Get all metrics
        oi_velocity = self.oi_tracker.get_velocity(5)
        oi_acceleration = self.oi_tracker.get_acceleration(5)
        premium_decay = self.premium_tracker.get_decay_rate(5)
        trap_detection = self.premium_tracker.detect_premium_trap(5)
        tw_pcr = self.tw_pcr.get_time_weighted_pcr()
        pcr_momentum = self.tw_pcr.get_pcr_momentum()

        # Gamma zone
        gamma_zone = ATMGammaZone.identify_gamma_zone(atm_strike, strike_gap, spot)

        # Stress zones
        stress_zones = WriterStressZoneDetector.find_stress_zones(option_chain_df, spot)

        # Generate market mood
        market_mood = self._determine_market_mood(
            oi_velocity, oi_acceleration, premium_decay, stress_zones
        )

        # Generate writer control
        writer_control = self._determine_writer_control(
            oi_velocity, premium_decay, trap_detection
        )

        # Generate action recommendation
        action = self._determine_action(
            market_mood, writer_control, gamma_zone, tw_pcr
        )

        return {
            'timestamp': datetime.now(IST).isoformat(),
            'spot': spot,
            'atm_strike': atm_strike,
            # Core 2026 metrics
            'oi_velocity': oi_velocity,
            'oi_acceleration': oi_acceleration,
            'premium_decay': premium_decay,
            'trap_detection': trap_detection,
            'tw_pcr': tw_pcr,
            'pcr_momentum': pcr_momentum,
            'gamma_zone': gamma_zone,
            'stress_zones': stress_zones,
            # Actionable outputs
            'market_mood': market_mood,
            'writer_control': writer_control,
            'action': action,
            # Summary for dashboard
            'summary': self._generate_summary(market_mood, writer_control, action, gamma_zone)
        }

    def _determine_market_mood(
        self,
        oi_velocity: Dict,
        oi_acceleration: Dict,
        premium_decay: Dict,
        stress_zones: Dict
    ) -> Dict[str, Any]:
        """Determine market mood: Calm / Stressed / Aggressive"""
        mood_score = 0

        # OI velocity contribution
        if 'EXIT' in oi_velocity.get('interpretation', ''):
            mood_score += 30
        elif 'ENTERING' in oi_velocity.get('interpretation', ''):
            mood_score -= 10  # Writers entering = calmer

        # Acceleration contribution
        if 'AGGRESSIVE' in oi_acceleration.get('interpretation', ''):
            mood_score += 25

        # Premium behavior
        if premium_decay.get('interpretation') == 'BOTH_EXPANDING':
            mood_score += 20

        # Stress zones
        high_stress_count = sum(
            1 for z in stress_zones.get('ce_stress_zones', [])
            if z.get('stress_level') in ['HIGH', 'CRITICAL']
        )
        high_stress_count += sum(
            1 for z in stress_zones.get('pe_stress_zones', [])
            if z.get('stress_level') in ['HIGH', 'CRITICAL']
        )
        mood_score += high_stress_count * 10

        # Classify
        if mood_score >= 50:
            mood = 'AGGRESSIVE'
            emoji = '🔥'
        elif mood_score >= 25:
            mood = 'STRESSED'
            emoji = '😰'
        else:
            mood = 'CALM'
            emoji = '😌'

        return {
            'mood': mood,
            'emoji': emoji,
            'score': mood_score
        }

    def _determine_writer_control(
        self,
        oi_velocity: Dict,
        premium_decay: Dict,
        trap_detection: Dict
    ) -> Dict[str, Any]:
        """Determine writer control: Strong / Weak"""
        control_score = 50  # Start neutral

        # OI velocity - writers entering = strong control
        if 'ENTERING' in oi_velocity.get('interpretation', ''):
            control_score += 20
        elif 'EXIT' in oi_velocity.get('interpretation', ''):
            control_score -= 25

        # Premium decay - decaying = strong control
        decay_signal = premium_decay.get('signal', '')
        if decay_signal == 'RANGE_LIKELY':
            control_score += 15
        elif 'BIAS' in decay_signal:
            control_score -= 10

        # Trap detection - traps mean writers winning
        if trap_detection.get('trap_detected'):
            control_score += 15

        # Classify
        if control_score >= 60:
            control = 'STRONG'
        elif control_score <= 40:
            control = 'WEAK'
        else:
            control = 'MIXED'

        return {
            'control': control,
            'score': control_score
        }

    def _determine_action(
        self,
        mood: Dict,
        control: Dict,
        gamma_zone: Dict,
        pcr: Dict
    ) -> Dict[str, Any]:
        """Determine action: Range / Expansion / Wait"""
        mood_val = mood.get('mood', 'CALM')
        control_val = control.get('control', 'MIXED')
        in_gamma = gamma_zone.get('in_gamma_zone', True)

        # Decision matrix
        if mood_val == 'CALM' and control_val == 'STRONG' and in_gamma:
            action = 'RANGE'
            reason = 'Calm mood, strong writers, in gamma zone. Play range.'
        elif mood_val == 'AGGRESSIVE' and control_val == 'WEAK' and not in_gamma:
            action = 'EXPANSION'
            reason = 'Aggressive mood, weak writers, escaped gamma. Trade expansion.'
        elif mood_val == 'STRESSED':
            action = 'WAIT'
            reason = 'Market stressed. Wait for clarity.'
        elif control_val == 'STRONG':
            action = 'RANGE'
            reason = 'Writers in control. Expect range.'
        elif control_val == 'WEAK':
            action = 'EXPANSION'
            reason = 'Writers losing control. Expansion likely.'
        else:
            action = 'WAIT'
            reason = 'Mixed signals. Wait for setup.'

        return {
            'action': action,
            'reason': reason,
            'confidence': 'HIGH' if (mood_val != 'STRESSED' and control_val != 'MIXED') else 'LOW'
        }

    def _generate_summary(
        self,
        mood: Dict,
        control: Dict,
        action: Dict,
        gamma_zone: Dict
    ) -> str:
        """Generate human-readable summary"""
        lines = [
            f"{mood['emoji']} Mood: {mood['mood']}",
            f"Writers: {control['control']}",
            f"Gamma: {'IN' if gamma_zone['in_gamma_zone'] else 'OUT'} ({gamma_zone['position']})",
            f"Action: {action['action']}"
        ]
        return ' | '.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_option_chain_2026(
    option_chain_df: pd.DataFrame,
    spot: float,
    atm_strike: float,
    strike_gap: float = 50
) -> Dict[str, Any]:
    """
    Quick function to analyze option chain with 2026 logic.
    """
    analyzer = OptionChain2026Analyzer()
    return analyzer.analyze(option_chain_df, spot, atm_strike, strike_gap)


def get_gamma_zone(atm_strike: float, strike_gap: float, spot: float) -> Dict[str, Any]:
    """Quick gamma zone check"""
    return ATMGammaZone.identify_gamma_zone(atm_strike, strike_gap, spot)


def find_writer_stress_zones(option_chain_df: pd.DataFrame, spot: float) -> Dict[str, Any]:
    """Quick stress zone finder"""
    return WriterStressZoneDetector.find_stress_zones(option_chain_df, spot)
