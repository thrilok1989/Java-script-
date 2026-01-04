"""
Expiry Structure Detector
=========================
Special patterns and structures for expiry day/week trading.

Expiry days have UNIQUE dynamics:
- Gamma effects amplified
- Max pain magnet effect
- Time decay acceleration
- Manipulation peaks
- False breakouts common
- Last hour chaos

This module detects these BEFORE they happen.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, time, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EXPIRY ENUMS
# =============================================================================

class ExpiryPhase(Enum):
    """Phase of expiry day/week"""
    OPENING_HUNT = "OPENING_HUNT"           # 9:15-10:00 - SL hunts common
    MORNING_TREND = "MORNING_TREND"         # 10:00-12:00 - Trend attempts
    MIDDAY_CHOP = "MIDDAY_CHOP"             # 12:00-14:00 - Range bound
    AFTERNOON_SETUP = "AFTERNOON_SETUP"     # 14:00-14:30 - Setup forms
    GAMMA_HOUR = "GAMMA_HOUR"               # 14:30-15:00 - Gamma effects peak
    LAST_MINUTE_CHAOS = "LAST_MINUTE_CHAOS" # 15:00-15:30 - Unpredictable
    NON_EXPIRY = "NON_EXPIRY"               # Not expiry day


class ExpiryPattern(Enum):
    """Expiry-specific patterns"""
    GAMMA_PIN = "GAMMA_PIN"                 # Price pinned to strike
    MAX_PAIN_MAGNET = "MAX_PAIN_MAGNET"     # Price pulled to max pain
    OPENING_SL_HUNT = "OPENING_SL_HUNT"     # Morning stop hunt
    AFTERNOON_REVERSAL = "AFTERNOON_REVERSAL"  # 2:30 reversal
    EXPIRY_COIL = "EXPIRY_COIL"             # Tight range → explosion
    FAKE_BREAK_REVERSAL = "FAKE_BREAK_REVERSAL"  # False breakout
    GAMMA_SNAP = "GAMMA_SNAP"               # Sudden violent move
    OI_UNWINDING_SQUEEZE = "OI_UNWINDING"   # Short covering / long liquidation
    IV_CRUSH_DROP = "IV_CRUSH_DROP"         # IV crush causes option drop
    TIME_DECAY_KILL = "TIME_DECAY_KILL"     # OTM options go to zero


class ExpiryRisk(Enum):
    """Risk levels for expiry trading"""
    EXTREME = "EXTREME"     # Don't trade
    HIGH = "HIGH"           # Very small size only
    MEDIUM = "MEDIUM"       # Normal caution
    LOW = "LOW"             # Relatively safe


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExpiryPatternMatch:
    """Detected expiry pattern"""
    pattern: ExpiryPattern
    confidence: float               # 0-100
    expected_outcome: str
    expected_direction: str         # UP/DOWN/NEUTRAL
    expected_timing: str            # IMMEDIATE/WITHIN_30_MIN/END_OF_DAY
    risk_level: ExpiryRisk
    key_level: float                # Key price level
    action_recommendation: str
    warnings: List[str]


@dataclass
class GammaPinAnalysis:
    """Gamma pinning analysis"""
    is_pinned: bool
    pin_strike: float
    pin_strength: float             # 0-100
    snap_probability: float         # Probability of snap move
    snap_direction: str             # UP/DOWN/UNKNOWN
    distance_to_snap: float         # Points to next strike
    time_to_expiry_mins: int


@dataclass
class MaxPainMagnet:
    """Max pain magnet analysis"""
    max_pain: float
    current_price: float
    distance_pct: float
    magnet_strength: float          # 0-100
    expected_drift: str             # UP/DOWN/AT_LEVEL
    time_factor: float              # Higher near close


@dataclass
class ExpiryStructureAnalysis:
    """Complete expiry structure analysis"""
    timestamp: datetime
    spot_price: float

    # Expiry context
    is_expiry_day: bool
    is_expiry_week: bool
    days_to_expiry: float
    hours_to_close: float
    current_phase: ExpiryPhase

    # Detected patterns
    patterns_detected: List[ExpiryPatternMatch]
    primary_pattern: Optional[ExpiryPatternMatch]

    # Gamma analysis
    gamma_analysis: GammaPinAnalysis

    # Max pain analysis
    max_pain_analysis: MaxPainMagnet

    # OI dynamics
    oi_unwinding: bool
    unwinding_direction: str        # CALLS/PUTS/BOTH

    # IV analysis
    iv_percentile: float
    iv_crush_expected: bool

    # Risk assessment
    overall_risk: ExpiryRisk
    tradeable: bool
    position_size_factor: float     # 0.25 to 1.0

    # Probabilities
    outcome_probabilities: Dict[str, float]

    # Recommendations
    action_recommendation: str
    key_levels: Dict[str, float]
    warnings: List[str]
    opportunities: List[str]


# =============================================================================
# EXPIRY STRUCTURE DETECTOR
# =============================================================================

class ExpiryStructureDetector:
    """
    Detects expiry-specific structures and patterns

    Expiry days are fundamentally different:
    - Option Greeks dominate price action
    - Time decay accelerates
    - Manipulation increases
    - Fake moves are common

    This detector identifies these dynamics BEFORE they unfold.
    """

    def __init__(self):
        """Initialize detector"""
        self.strike_interval = 50  # NIFTY strike interval
        self.gamma_hours = ['14:30', '15:00', '15:15']
        self.sl_hunt_hours = ['09:15', '09:30', '09:45']

    def analyze(
        self,
        df: pd.DataFrame,
        spot_price: float,
        option_chain: Optional[Dict] = None,
        merged_df: Optional[pd.DataFrame] = None,
        days_to_expiry: float = 7.0,
        current_time: Optional[datetime] = None
    ) -> ExpiryStructureAnalysis:
        """
        Analyze expiry-specific structures

        Args:
            df: OHLCV DataFrame
            spot_price: Current spot price
            option_chain: Option chain data
            merged_df: Merged option chain DataFrame
            days_to_expiry: Days until expiry
            current_time: Current time (for phase detection)

        Returns:
            ExpiryStructureAnalysis with complete analysis
        """
        if current_time is None:
            current_time = datetime.now()

        # Determine expiry context
        is_expiry_day = days_to_expiry < 1
        is_expiry_week = days_to_expiry <= 5

        # Calculate time to close
        close_time = current_time.replace(hour=15, minute=30, second=0)
        hours_to_close = max(0, (close_time - current_time).total_seconds() / 3600)

        # Determine current phase
        current_phase = self._determine_phase(current_time, is_expiry_day)

        # Gamma analysis
        gamma_analysis = self._analyze_gamma_pinning(
            spot_price, option_chain, merged_df, hours_to_close
        )

        # Max pain analysis
        max_pain_analysis = self._analyze_max_pain_magnet(
            spot_price, option_chain, days_to_expiry, hours_to_close
        )

        # Detect patterns
        patterns = self._detect_expiry_patterns(
            df, spot_price, option_chain, merged_df,
            is_expiry_day, is_expiry_week, current_phase,
            gamma_analysis, max_pain_analysis
        )

        # OI unwinding analysis
        oi_unwinding, unwinding_direction = self._analyze_oi_unwinding(
            option_chain, merged_df
        )

        # IV analysis
        iv_percentile = self._get_iv_percentile(option_chain)
        iv_crush_expected = is_expiry_day and iv_percentile > 50

        # Risk assessment
        overall_risk = self._assess_expiry_risk(
            is_expiry_day, current_phase, patterns, gamma_analysis
        )

        # Tradeable assessment
        tradeable = overall_risk not in [ExpiryRisk.EXTREME]
        position_size_factor = self._get_position_size_factor(overall_risk, current_phase)

        # Calculate probabilities
        outcome_probs = self._calculate_expiry_probabilities(
            patterns, gamma_analysis, max_pain_analysis,
            is_expiry_day, current_phase
        )

        # Generate recommendations
        action, key_levels, warnings, opportunities = self._generate_recommendations(
            patterns, gamma_analysis, max_pain_analysis,
            is_expiry_day, current_phase, spot_price
        )

        return ExpiryStructureAnalysis(
            timestamp=current_time,
            spot_price=spot_price,
            is_expiry_day=is_expiry_day,
            is_expiry_week=is_expiry_week,
            days_to_expiry=days_to_expiry,
            hours_to_close=hours_to_close,
            current_phase=current_phase,
            patterns_detected=patterns,
            primary_pattern=patterns[0] if patterns else None,
            gamma_analysis=gamma_analysis,
            max_pain_analysis=max_pain_analysis,
            oi_unwinding=oi_unwinding,
            unwinding_direction=unwinding_direction,
            iv_percentile=iv_percentile,
            iv_crush_expected=iv_crush_expected,
            overall_risk=overall_risk,
            tradeable=tradeable,
            position_size_factor=position_size_factor,
            outcome_probabilities=outcome_probs,
            action_recommendation=action,
            key_levels=key_levels,
            warnings=warnings,
            opportunities=opportunities
        )

    def _determine_phase(
        self,
        current_time: datetime,
        is_expiry_day: bool
    ) -> ExpiryPhase:
        """Determine current expiry phase"""
        if not is_expiry_day:
            return ExpiryPhase.NON_EXPIRY

        hour = current_time.hour
        minute = current_time.minute
        time_val = hour * 60 + minute

        if time_val < 600:  # Before 10:00
            return ExpiryPhase.OPENING_HUNT
        elif time_val < 720:  # 10:00 - 12:00
            return ExpiryPhase.MORNING_TREND
        elif time_val < 840:  # 12:00 - 14:00
            return ExpiryPhase.MIDDAY_CHOP
        elif time_val < 870:  # 14:00 - 14:30
            return ExpiryPhase.AFTERNOON_SETUP
        elif time_val < 900:  # 14:30 - 15:00
            return ExpiryPhase.GAMMA_HOUR
        else:  # After 15:00
            return ExpiryPhase.LAST_MINUTE_CHAOS

    def _analyze_gamma_pinning(
        self,
        spot_price: float,
        option_chain: Optional[Dict],
        merged_df: Optional[pd.DataFrame],
        hours_to_close: float
    ) -> GammaPinAnalysis:
        """Analyze gamma pinning dynamics"""
        # Find nearest strike
        nearest_strike = round(spot_price / self.strike_interval) * self.strike_interval
        distance_to_strike = abs(spot_price - nearest_strike)

        # Default values
        pin_strength = 0
        snap_probability = 0
        snap_direction = "UNKNOWN"
        next_strike_distance = self.strike_interval - distance_to_strike

        # Check if pinned (within 0.3% of strike)
        is_pinned = (distance_to_strike / spot_price) < 0.003

        if is_pinned:
            # Stronger pin = closer to strike + more time passed
            distance_factor = 1 - (distance_to_strike / self.strike_interval)
            time_factor = max(0, 1 - hours_to_close / 6)  # Stronger as close approaches
            pin_strength = (distance_factor * 0.6 + time_factor * 0.4) * 100

            # Snap probability increases as pin weakens
            if distance_to_strike > self.strike_interval * 0.4:
                snap_probability = 60 + (distance_to_strike / self.strike_interval) * 40
        else:
            # Calculate snap probability based on gamma exposure
            if hours_to_close < 2:
                snap_probability = 40 + (2 - hours_to_close) * 20

        # Determine snap direction from OI if available
        if merged_df is not None and len(merged_df) > 0:
            try:
                atm_row = merged_df[merged_df['strikePrice'] == nearest_strike]
                if len(atm_row) > 0:
                    call_oi = atm_row['OI_CE'].iloc[0]
                    put_oi = atm_row['OI_PE'].iloc[0]
                    if put_oi > call_oi * 1.3:
                        snap_direction = "UP"
                    elif call_oi > put_oi * 1.3:
                        snap_direction = "DOWN"
            except:
                pass

        return GammaPinAnalysis(
            is_pinned=is_pinned,
            pin_strike=nearest_strike,
            pin_strength=pin_strength,
            snap_probability=snap_probability,
            snap_direction=snap_direction,
            distance_to_snap=next_strike_distance,
            time_to_expiry_mins=int(hours_to_close * 60)
        )

    def _analyze_max_pain_magnet(
        self,
        spot_price: float,
        option_chain: Optional[Dict],
        days_to_expiry: float,
        hours_to_close: float
    ) -> MaxPainMagnet:
        """Analyze max pain magnet effect"""
        max_pain = 0
        if option_chain:
            max_pain = option_chain.get('max_pain', 0)

        if max_pain <= 0:
            max_pain = spot_price  # Default to current price

        distance_pct = (spot_price - max_pain) / spot_price * 100

        # Magnet strength increases as expiry approaches
        if days_to_expiry < 1:
            time_factor = min(1, (6 - hours_to_close) / 6) if hours_to_close < 6 else 0
            magnet_strength = 50 + time_factor * 50
        elif days_to_expiry < 2:
            magnet_strength = 40
        else:
            magnet_strength = max(0, 30 - days_to_expiry * 3)

        # Expected drift
        if abs(distance_pct) < 0.3:
            expected_drift = "AT_LEVEL"
        elif distance_pct > 0:
            expected_drift = "DOWN"  # Price above max pain → drift down
        else:
            expected_drift = "UP"    # Price below max pain → drift up

        return MaxPainMagnet(
            max_pain=max_pain,
            current_price=spot_price,
            distance_pct=distance_pct,
            magnet_strength=magnet_strength,
            expected_drift=expected_drift,
            time_factor=time_factor if days_to_expiry < 1 else 0
        )

    def _detect_expiry_patterns(
        self,
        df: pd.DataFrame,
        spot_price: float,
        option_chain: Optional[Dict],
        merged_df: Optional[pd.DataFrame],
        is_expiry_day: bool,
        is_expiry_week: bool,
        current_phase: ExpiryPhase,
        gamma_analysis: GammaPinAnalysis,
        max_pain_analysis: MaxPainMagnet
    ) -> List[ExpiryPatternMatch]:
        """Detect expiry-specific patterns"""
        patterns = []

        # 1. GAMMA PIN PATTERN
        if gamma_analysis.is_pinned and gamma_analysis.pin_strength > 50:
            patterns.append(ExpiryPatternMatch(
                pattern=ExpiryPattern.GAMMA_PIN,
                confidence=gamma_analysis.pin_strength,
                expected_outcome="Price stays near strike until snap or expiry",
                expected_direction="NEUTRAL",
                expected_timing="UNTIL_SNAP",
                risk_level=ExpiryRisk.HIGH,
                key_level=gamma_analysis.pin_strike,
                action_recommendation="Wait for snap or sell premium near strike",
                warnings=[
                    "Snap moves are violent when they occur",
                    "Don't fight the pin - wait for break"
                ]
            ))

        # 2. MAX PAIN MAGNET
        if max_pain_analysis.magnet_strength > 60:
            patterns.append(ExpiryPatternMatch(
                pattern=ExpiryPattern.MAX_PAIN_MAGNET,
                confidence=max_pain_analysis.magnet_strength,
                expected_outcome=f"Price drifts toward max pain ({max_pain_analysis.expected_drift})",
                expected_direction=max_pain_analysis.expected_drift,
                expected_timing="END_OF_DAY",
                risk_level=ExpiryRisk.MEDIUM,
                key_level=max_pain_analysis.max_pain,
                action_recommendation=f"Bias toward max pain direction: {max_pain_analysis.expected_drift}",
                warnings=[
                    "Max pain is not guaranteed - just probability",
                    "Can overshoot before reverting"
                ]
            ))

        # 3. OPENING SL HUNT (expiry day morning)
        if is_expiry_day and current_phase == ExpiryPhase.OPENING_HUNT:
            patterns.append(ExpiryPatternMatch(
                pattern=ExpiryPattern.OPENING_SL_HUNT,
                confidence=70,
                expected_outcome="Sharp move to hunt stops, then reverse",
                expected_direction="UNKNOWN",
                expected_timing="WITHIN_30_MIN",
                risk_level=ExpiryRisk.HIGH,
                key_level=spot_price,
                action_recommendation="Wait for hunt to complete before entering",
                warnings=[
                    "First move on expiry is often FALSE",
                    "Equal highs/lows from yesterday are targets"
                ]
            ))

        # 4. GAMMA SNAP (gamma hour with high snap probability)
        if current_phase == ExpiryPhase.GAMMA_HOUR and gamma_analysis.snap_probability > 60:
            patterns.append(ExpiryPatternMatch(
                pattern=ExpiryPattern.GAMMA_SNAP,
                confidence=gamma_analysis.snap_probability,
                expected_outcome="Sudden violent move breaking gamma pin",
                expected_direction=gamma_analysis.snap_direction,
                expected_timing="IMMEDIATE",
                risk_level=ExpiryRisk.EXTREME,
                key_level=gamma_analysis.pin_strike + self.strike_interval * (1 if gamma_analysis.snap_direction == "UP" else -1),
                action_recommendation="Reduce position or hedge. Move can be violent.",
                warnings=[
                    "Gamma snaps can move 100+ points in minutes",
                    "Options can 5x or go to zero",
                    "Most dangerous time to hold naked positions"
                ]
            ))

        # 5. EXPIRY COIL (tight range building pressure)
        if is_expiry_day and df is not None and len(df) >= 10:
            try:
                high_col = 'High' if 'High' in df.columns else 'high'
                low_col = 'Low' if 'Low' in df.columns else 'low'

                recent_range = df[high_col].tail(10).max() - df[low_col].tail(10).min()
                day_range = df[high_col].iloc[-1] - df[low_col].iloc[-1]

                # Coil = range compressing
                if day_range < recent_range * 0.5:
                    patterns.append(ExpiryPatternMatch(
                        pattern=ExpiryPattern.EXPIRY_COIL,
                        confidence=70,
                        expected_outcome="Tight range will break with force",
                        expected_direction="UNKNOWN",
                        expected_timing="WITHIN_30_MIN",
                        risk_level=ExpiryRisk.MEDIUM,
                        key_level=spot_price,
                        action_recommendation="Prepare for breakout. Direction from delta/OI.",
                        warnings=[
                            "Breakout direction unknown until it happens",
                            "Size smaller, trail stops tighter"
                        ]
                    ))
            except:
                pass

        # 6. OI UNWINDING SQUEEZE
        if option_chain:
            try:
                # Check for significant OI change (would need historical comparison)
                # For now, use PCR extreme as proxy
                call_oi = sum(option_chain.get('CE', {}).get('openInterest', [0]))
                put_oi = sum(option_chain.get('PE', {}).get('openInterest', [0]))
                pcr = put_oi / call_oi if call_oi > 0 else 1

                if pcr > 1.5:  # Heavy puts → squeeze up
                    patterns.append(ExpiryPatternMatch(
                        pattern=ExpiryPattern.OI_UNWINDING_SQUEEZE,
                        confidence=65,
                        expected_outcome="Put writers cover → short squeeze up",
                        expected_direction="UP",
                        expected_timing="GRADUAL",
                        risk_level=ExpiryRisk.MEDIUM,
                        key_level=spot_price + self.strike_interval,
                        action_recommendation="Bias bullish as puts unwind",
                        warnings=["Can reverse if new puts added"]
                    ))
                elif pcr < 0.7:  # Heavy calls → squeeze down
                    patterns.append(ExpiryPatternMatch(
                        pattern=ExpiryPattern.OI_UNWINDING_SQUEEZE,
                        confidence=65,
                        expected_outcome="Call writers cover → squeeze down",
                        expected_direction="DOWN",
                        expected_timing="GRADUAL",
                        risk_level=ExpiryRisk.MEDIUM,
                        key_level=spot_price - self.strike_interval,
                        action_recommendation="Bias bearish as calls unwind",
                        warnings=["Can reverse if new calls added"]
                    ))
            except:
                pass

        # 7. AFTERNOON REVERSAL (2:30 phenomenon)
        if current_phase == ExpiryPhase.GAMMA_HOUR:
            # Check if morning had a trend
            if df is not None and len(df) >= 20:
                try:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    morning_change = df[close_col].iloc[-10] - df[close_col].iloc[0]

                    if abs(morning_change) > 50:  # Had a trend
                        patterns.append(ExpiryPatternMatch(
                            pattern=ExpiryPattern.AFTERNOON_REVERSAL,
                            confidence=55,
                            expected_outcome="Morning trend reverses in afternoon",
                            expected_direction="UP" if morning_change < 0 else "DOWN",
                            expected_timing="WITHIN_30_MIN",
                            risk_level=ExpiryRisk.MEDIUM,
                            key_level=spot_price,
                            action_recommendation="Watch for reversal trigger, fade morning move",
                            warnings=[
                                "Not always happens - need confirmation",
                                "Stronger morning trend = higher reversal chance"
                            ]
                        ))
                except:
                    pass

        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)

        return patterns

    def _analyze_oi_unwinding(
        self,
        option_chain: Optional[Dict],
        merged_df: Optional[pd.DataFrame]
    ) -> Tuple[bool, str]:
        """Analyze OI unwinding dynamics"""
        # Would need historical OI to detect actual unwinding
        # For now, return defaults
        return False, "NEUTRAL"

    def _get_iv_percentile(self, option_chain: Optional[Dict]) -> float:
        """Get IV percentile"""
        if not option_chain:
            return 50.0
        return option_chain.get('iv_percentile', 50.0)

    def _assess_expiry_risk(
        self,
        is_expiry_day: bool,
        current_phase: ExpiryPhase,
        patterns: List[ExpiryPatternMatch],
        gamma_analysis: GammaPinAnalysis
    ) -> ExpiryRisk:
        """Assess overall expiry risk"""
        if not is_expiry_day:
            return ExpiryRisk.LOW

        # Last minute chaos = extreme
        if current_phase == ExpiryPhase.LAST_MINUTE_CHAOS:
            return ExpiryRisk.EXTREME

        # Gamma hour with snap probability = extreme
        if current_phase == ExpiryPhase.GAMMA_HOUR:
            if gamma_analysis.snap_probability > 70:
                return ExpiryRisk.EXTREME
            return ExpiryRisk.HIGH

        # Opening hunt = high
        if current_phase == ExpiryPhase.OPENING_HUNT:
            return ExpiryRisk.HIGH

        # Multiple high-confidence patterns = high
        high_conf_patterns = [p for p in patterns if p.confidence > 70]
        if len(high_conf_patterns) >= 2:
            return ExpiryRisk.HIGH

        return ExpiryRisk.MEDIUM

    def _get_position_size_factor(
        self,
        risk: ExpiryRisk,
        phase: ExpiryPhase
    ) -> float:
        """Get position size factor based on risk"""
        factors = {
            ExpiryRisk.LOW: 1.0,
            ExpiryRisk.MEDIUM: 0.7,
            ExpiryRisk.HIGH: 0.4,
            ExpiryRisk.EXTREME: 0.25
        }

        base = factors.get(risk, 0.5)

        # Further reduce for dangerous phases
        if phase == ExpiryPhase.LAST_MINUTE_CHAOS:
            base *= 0.5
        elif phase == ExpiryPhase.GAMMA_HOUR:
            base *= 0.7

        return max(0.25, base)

    def _calculate_expiry_probabilities(
        self,
        patterns: List[ExpiryPatternMatch],
        gamma_analysis: GammaPinAnalysis,
        max_pain_analysis: MaxPainMagnet,
        is_expiry_day: bool,
        current_phase: ExpiryPhase
    ) -> Dict[str, float]:
        """Calculate expiry-specific outcome probabilities"""
        probs = {
            "PIN_TO_STRIKE": 0.0,
            "DRIFT_TO_MAX_PAIN": 0.0,
            "SNAP_UP": 0.0,
            "SNAP_DOWN": 0.0,
            "CONTINUE_TREND": 0.0,
            "REVERSAL": 0.0,
            "CHOP": 0.0
        }

        if not is_expiry_day:
            probs["CHOP"] = 0.3
            probs["CONTINUE_TREND"] = 0.4
            probs["REVERSAL"] = 0.3
            return probs

        # Gamma pin probability
        if gamma_analysis.is_pinned:
            probs["PIN_TO_STRIKE"] = gamma_analysis.pin_strength / 100 * 0.5

        # Max pain drift
        if max_pain_analysis.magnet_strength > 50:
            drift_prob = max_pain_analysis.magnet_strength / 100 * 0.4
            probs["DRIFT_TO_MAX_PAIN"] = drift_prob

        # Snap probabilities
        snap_base = gamma_analysis.snap_probability / 100
        if gamma_analysis.snap_direction == "UP":
            probs["SNAP_UP"] = snap_base * 0.7
            probs["SNAP_DOWN"] = snap_base * 0.3
        elif gamma_analysis.snap_direction == "DOWN":
            probs["SNAP_DOWN"] = snap_base * 0.7
            probs["SNAP_UP"] = snap_base * 0.3
        else:
            probs["SNAP_UP"] = snap_base * 0.5
            probs["SNAP_DOWN"] = snap_base * 0.5

        # Phase-specific adjustments
        if current_phase == ExpiryPhase.OPENING_HUNT:
            probs["REVERSAL"] += 0.2
            probs["CONTINUE_TREND"] -= 0.1
        elif current_phase == ExpiryPhase.GAMMA_HOUR:
            probs["SNAP_UP"] *= 1.5
            probs["SNAP_DOWN"] *= 1.5
            probs["PIN_TO_STRIKE"] *= 0.5

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def _generate_recommendations(
        self,
        patterns: List[ExpiryPatternMatch],
        gamma_analysis: GammaPinAnalysis,
        max_pain_analysis: MaxPainMagnet,
        is_expiry_day: bool,
        current_phase: ExpiryPhase,
        spot_price: float
    ) -> Tuple[str, Dict[str, float], List[str], List[str]]:
        """Generate actionable recommendations"""
        warnings = []
        opportunities = []
        key_levels = {}

        # Add key levels
        key_levels['current_price'] = spot_price
        key_levels['gamma_pin_strike'] = gamma_analysis.pin_strike
        key_levels['max_pain'] = max_pain_analysis.max_pain
        key_levels['snap_target_up'] = gamma_analysis.pin_strike + self.strike_interval
        key_levels['snap_target_down'] = gamma_analysis.pin_strike - self.strike_interval

        # Generate action based on phase and patterns
        if not is_expiry_day:
            action = "Normal trading conditions. Follow unified ML signal."
        elif current_phase == ExpiryPhase.LAST_MINUTE_CHAOS:
            action = "⚠️ AVOID TRADING - Last minute chaos. Close positions or hedge."
            warnings.append("This is the most dangerous time to trade")
        elif current_phase == ExpiryPhase.GAMMA_HOUR:
            if gamma_analysis.snap_probability > 70:
                action = "⚠️ HIGH GAMMA RISK - Reduce exposure. Snap move imminent."
                warnings.append("Options can move 5x in minutes")
            else:
                action = f"Gamma hour active. Bias: {gamma_analysis.snap_direction}"
                opportunities.append("Premium selling works if pin holds")
        elif current_phase == ExpiryPhase.OPENING_HUNT:
            action = "🎯 Wait for opening hunt to complete. First move often false."
            warnings.append("Don't chase opening moves")
            opportunities.append("Fade the hunt after confirmation")
        elif current_phase == ExpiryPhase.AFTERNOON_SETUP:
            action = "Setup phase - Watch for reversal or continuation signals."
            opportunities.append("Afternoon often reverses morning trend")
        else:
            if patterns:
                action = patterns[0].action_recommendation
            else:
                action = "No clear expiry pattern. Follow unified ML signal."

        # Add pattern-specific warnings
        for p in patterns[:3]:  # Top 3 patterns
            warnings.extend(p.warnings)

        # Add opportunities
        if gamma_analysis.is_pinned:
            opportunities.append(f"Gamma pin at {gamma_analysis.pin_strike:.0f} - Premium decay accelerated")
        if abs(max_pain_analysis.distance_pct) > 0.5:
            opportunities.append(f"Max pain at {max_pain_analysis.max_pain:.0f} - Drift expected {max_pain_analysis.expected_drift}")

        return action, key_levels, warnings[:5], opportunities[:3]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def expiry_analysis_to_dict(analysis: ExpiryStructureAnalysis) -> Dict:
    """Convert analysis to dict for storage/display"""
    return {
        'timestamp': analysis.timestamp.isoformat(),
        'spot_price': analysis.spot_price,
        'is_expiry_day': analysis.is_expiry_day,
        'is_expiry_week': analysis.is_expiry_week,
        'days_to_expiry': analysis.days_to_expiry,
        'hours_to_close': analysis.hours_to_close,
        'current_phase': analysis.current_phase.value,
        'patterns': [
            {
                'pattern': p.pattern.value,
                'confidence': p.confidence,
                'direction': p.expected_direction,
                'timing': p.expected_timing,
                'risk': p.risk_level.value,
                'action': p.action_recommendation
            }
            for p in analysis.patterns_detected
        ],
        'gamma': {
            'is_pinned': analysis.gamma_analysis.is_pinned,
            'pin_strike': analysis.gamma_analysis.pin_strike,
            'pin_strength': analysis.gamma_analysis.pin_strength,
            'snap_probability': analysis.gamma_analysis.snap_probability,
            'snap_direction': analysis.gamma_analysis.snap_direction,
        },
        'max_pain': {
            'level': analysis.max_pain_analysis.max_pain,
            'distance_pct': analysis.max_pain_analysis.distance_pct,
            'magnet_strength': analysis.max_pain_analysis.magnet_strength,
            'expected_drift': analysis.max_pain_analysis.expected_drift,
        },
        'overall_risk': analysis.overall_risk.value,
        'tradeable': analysis.tradeable,
        'position_size_factor': analysis.position_size_factor,
        'outcome_probabilities': analysis.outcome_probabilities,
        'action_recommendation': analysis.action_recommendation,
        'key_levels': analysis.key_levels,
        'warnings': analysis.warnings,
        'opportunities': analysis.opportunities,
    }
