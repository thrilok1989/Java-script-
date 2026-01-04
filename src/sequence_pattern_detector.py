"""
Sequence Pattern Detector
=========================
Detects time-ordered market structure patterns BEFORE moves happen.

Patterns are sequences of market states that historically precede specific outcomes.
Each pattern has:
- Sequence of conditions (ordered in time)
- Historical probability of each outcome
- Confidence based on similarity and occurrences

NO DEEP LEARNING - Pure statistical pattern matching.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import os

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

class PatternType(Enum):
    """Types of sequence patterns"""
    SILENT_BUILDUP = "SILENT_BUILDUP"           # OI↑ Volume↓ Price flat → Fast move
    FAKE_BREAK_REVERSAL = "FAKE_BREAK_REVERSAL" # Break S/R weak → Reverse
    GAMMA_PIN_SNAP = "GAMMA_PIN_SNAP"           # Near gamma wall → Snap move
    ABSORPTION_BREAK = "ABSORPTION_BREAK"       # Repeated tests → Break direction
    SL_HUNT_SETUP = "SL_HUNT_SETUP"             # Equal H/L + thin liquidity → Wick
    COMPRESSION_EXPLOSION = "COMPRESSION_EXPLOSION"  # ATR dies → Explosion
    DELTA_DIVERGENCE = "DELTA_DIVERGENCE"       # CVD vs Price diverge → Reversal
    OI_UNWINDING = "OI_UNWINDING"               # OI decreasing → Trend continuation


@dataclass
class PatternCondition:
    """A single condition in a pattern sequence"""
    name: str                           # Condition name
    feature: str                        # Feature to check
    operator: str                       # >, <, ==, !=, between
    value: Any                          # Threshold value
    lookback: int = 1                   # How many bars ago to check
    weight: float = 1.0                 # Importance weight


@dataclass
class SequencePattern:
    """Definition of a sequence pattern"""
    pattern_type: PatternType
    name: str
    description: str
    conditions: List[PatternCondition]  # Ordered conditions
    min_duration: int = 3               # Minimum bars for pattern
    max_duration: int = 20              # Maximum bars for pattern
    expected_outcomes: Dict[str, float] # Outcome → base probability
    risk_warnings: List[str]            # Risk warnings for this pattern
    recommended_action: str             # What to do when detected


@dataclass
class PatternMatch:
    """Result of pattern matching"""
    pattern_type: PatternType
    pattern_name: str
    similarity_score: float             # 0-1 how well it matches
    conditions_met: int                 # How many conditions met
    total_conditions: int               # Total conditions in pattern
    outcome_probabilities: Dict[str, float]  # Adjusted probabilities
    confidence: float                   # Overall confidence
    bars_since_start: int               # How long pattern has been forming
    expected_move_within: int           # Expected bars until move
    risk_warnings: List[str]
    recommended_action: str
    reasoning: List[str]                # Why pattern was detected


@dataclass
class SequenceAnalysisResult:
    """Complete sequence analysis result"""
    timestamp: datetime
    patterns_detected: List[PatternMatch]
    primary_pattern: Optional[PatternMatch]
    structure_state: str                # Current structure state
    phase: str                          # early/developing/mature
    probability_summary: Dict[str, float]  # Aggregated probabilities
    key_observations: List[str]
    risk_level: str                     # LOW/MEDIUM/HIGH/EXTREME
    action_recommendation: str


# =============================================================================
# PATTERN LIBRARY
# =============================================================================

class PatternLibrary:
    """
    Library of predefined sequence patterns

    These patterns are based on institutional trading knowledge:
    - How smart money accumulates/distributes
    - How markets set up before moves
    - How manipulation creates opportunities
    """

    @staticmethod
    def get_all_patterns() -> List[SequencePattern]:
        """Get all defined patterns"""
        return [
            PatternLibrary.silent_buildup_pattern(),
            PatternLibrary.fake_break_reversal_pattern(),
            PatternLibrary.gamma_pin_snap_pattern(),
            PatternLibrary.absorption_break_pattern(),
            PatternLibrary.sl_hunt_setup_pattern(),
            PatternLibrary.compression_explosion_pattern(),
            PatternLibrary.delta_divergence_pattern(),
            PatternLibrary.oi_unwinding_pattern(),
        ]

    @staticmethod
    def silent_buildup_pattern() -> SequencePattern:
        """
        SILENT BUILDUP → FAST MOVE

        Sequence:
        1. OI increasing
        2. Volume decreasing
        3. Price flat (range-bound)
        4. Delta builds up
        5. Sudden delta burst

        Outcome: Directional move within 3-5 candles
        """
        return SequencePattern(
            pattern_type=PatternType.SILENT_BUILDUP,
            name="Silent Buildup",
            description="OI building while volume dries up - explosion imminent",
            conditions=[
                PatternCondition("OI Increasing", "oi_slope", ">", 0, lookback=5, weight=1.0),
                PatternCondition("Volume Declining", "volume_trend", "<", -0.1, lookback=5, weight=1.0),
                PatternCondition("Price Flat", "price_range_atr_ratio", "<", 1.2, lookback=3, weight=0.8),
                PatternCondition("Low Volatility", "atr_percentile", "<", 30, lookback=1, weight=0.7),
                PatternCondition("OI Building", "oi_buildup_score", ">", 50, lookback=1, weight=0.9),
            ],
            min_duration=5,
            max_duration=20,
            expected_outcomes={
                "EXPANSION_UP": 0.40,
                "EXPANSION_DOWN": 0.35,
                "FAKE_BREAK_FIRST": 0.15,
                "NO_MOVE": 0.10,
            },
            risk_warnings=[
                "Direction unknown until delta burst",
                "Wait for confirmation before entry",
                "SL hunt may precede real move"
            ],
            recommended_action="WAIT for delta burst direction"
        )

    @staticmethod
    def fake_break_reversal_pattern() -> SequencePattern:
        """
        FAKE BREAK → REAL MOVE OPPOSITE

        Sequence:
        1. Break S/R with low volume
        2. MAE > MFE (move against exceeds move for)
        3. OI spikes (trapped traders)
        4. Reverse candle forms

        Outcome: Reversal in opposite direction
        """
        return SequencePattern(
            pattern_type=PatternType.FAKE_BREAK_REVERSAL,
            name="Fake Break Reversal",
            description="Break on low volume with quick reversal - trap detected",
            conditions=[
                PatternCondition("Low Volume Break", "volume_ratio", "<", 1.0, lookback=1, weight=1.0),
                PatternCondition("High Rejection", "rejection_score", ">", 50, lookback=1, weight=1.0),
                PatternCondition("Wick Dominates", "wick_imbalance", "!=", 0, lookback=1, weight=0.8),
                PatternCondition("OI Spike", "oi_change_pct", ">", 5, lookback=1, weight=0.9),
                PatternCondition("Delta Reversal", "delta_imbalance", "changed", 0, lookback=2, weight=0.7),
            ],
            min_duration=2,
            max_duration=5,
            expected_outcomes={
                "REVERSAL_OPPOSITE": 0.65,
                "CONTINUATION": 0.20,
                "CHOP": 0.15,
            },
            risk_warnings=[
                "Wait for confirmation candle",
                "Stop above/below fake break high/low",
                "Size smaller until confirmed"
            ],
            recommended_action="WAIT for reversal confirmation, then trade opposite"
        )

    @staticmethod
    def gamma_pin_snap_pattern() -> SequencePattern:
        """
        GAMMA PIN → SNAP MOVE

        Sequence:
        1. Price near gamma wall (max OI strike)
        2. High dealer GEX
        3. Compression near wall
        4. Wall flips or breaks

        Outcome: Sudden violent move when wall breaks
        """
        return SequencePattern(
            pattern_type=PatternType.GAMMA_PIN_SNAP,
            name="Gamma Pin Snap",
            description="Price pinned near gamma wall - snap move when wall breaks",
            conditions=[
                PatternCondition("Near Gamma Wall", "gex_flip_distance", "<", 50, lookback=1, weight=1.0),
                PatternCondition("High Pin Probability", "pin_probability", ">", 50, lookback=1, weight=0.9),
                PatternCondition("Compression", "compression_score", ">", 40, lookback=1, weight=0.8),
                PatternCondition("Expiry Proximity", "days_to_expiry", "<", 3, lookback=1, weight=0.7),
            ],
            min_duration=1,
            max_duration=10,
            expected_outcomes={
                "SNAP_UP": 0.35,
                "SNAP_DOWN": 0.35,
                "CONTINUE_PIN": 0.20,
                "GRADUAL_MOVE": 0.10,
            },
            risk_warnings=[
                "Snap moves are violent - use tight stops",
                "Don't fight gamma pinning",
                "Best to trade AFTER snap starts"
            ],
            recommended_action="WAIT for wall break, then follow momentum"
        )

    @staticmethod
    def absorption_break_pattern() -> SequencePattern:
        """
        ABSORPTION AT LEVEL → BREAK

        Sequence:
        1. Repeated tests of same level
        2. High volume at level
        3. Price doesn't move
        4. Delta accumulates

        Outcome: Break in absorption direction
        """
        return SequencePattern(
            pattern_type=PatternType.ABSORPTION_BREAK,
            name="Absorption Break",
            description="Large orders absorbing at level - break in absorption direction",
            conditions=[
                PatternCondition("Delta Absorption", "delta_absorption", "==", True, lookback=1, weight=1.0),
                PatternCondition("High Volume", "volume_ratio", ">", 1.3, lookback=3, weight=0.9),
                PatternCondition("Price Flat", "price_range_atr_ratio", "<", 1.0, lookback=3, weight=0.8),
                PatternCondition("CVD Building", "cvd_slope", "!=", 0, lookback=3, weight=0.9),
            ],
            min_duration=3,
            max_duration=15,
            expected_outcomes={
                "BREAK_CVD_DIRECTION": 0.60,
                "BREAK_OPPOSITE": 0.20,
                "CONTINUE_ABSORPTION": 0.20,
            },
            risk_warnings=[
                "Break direction = CVD direction",
                "Wait for volume confirmation on break",
                "False breaks possible before real break"
            ],
            recommended_action="Trade in CVD direction when absorption completes"
        )

    @staticmethod
    def sl_hunt_setup_pattern() -> SequencePattern:
        """
        SL HUNT PREPARATION

        Sequence:
        1. Equal highs or equal lows (liquidity pool)
        2. Thin liquidity in order book
        3. Sudden depth imbalance
        4. Probe toward liquidity

        Outcome: Wick + reversal (hunt then reverse)
        """
        return SequencePattern(
            pattern_type=PatternType.SL_HUNT_SETUP,
            name="SL Hunt Setup",
            description="Equal highs/lows creating liquidity pool - hunt imminent",
            conditions=[
                PatternCondition("Equal Highs", "equal_highs_count", ">=", 2, lookback=10, weight=1.0),
                PatternCondition("Equal Lows", "equal_lows_count", ">=", 2, lookback=10, weight=1.0),
                PatternCondition("Depth Imbalance", "bid_ask_imbalance", "!=", 0, lookback=1, weight=0.8),
                PatternCondition("Volume Dry", "volume_ratio", "<", 0.8, lookback=3, weight=0.6),
            ],
            min_duration=5,
            max_duration=30,
            expected_outcomes={
                "HUNT_ABOVE_REVERSE_DOWN": 0.35,
                "HUNT_BELOW_REVERSE_UP": 0.35,
                "NO_HUNT": 0.20,
                "BREAK_THROUGH": 0.10,
            },
            risk_warnings=[
                "DO NOT place stops at equal highs/lows",
                "Wait for hunt to complete before entry",
                "Hunt candle = entry signal in opposite direction"
            ],
            recommended_action="WAIT for hunt, then fade the wick"
        )

    @staticmethod
    def compression_explosion_pattern() -> SequencePattern:
        """
        COMPRESSION / SPRING LOADING

        Volatility dies before explosion:
        1. ATR at historical lows
        2. Volume declining
        3. OI increasing
        4. Direction unknown

        Outcome: Explosive move in either direction
        """
        return SequencePattern(
            pattern_type=PatternType.COMPRESSION_EXPLOSION,
            name="Compression Explosion",
            description="Volatility at extremes - explosion imminent, direction unknown",
            conditions=[
                PatternCondition("ATR Compressed", "atr_percentile", "<", 25, lookback=1, weight=1.0),
                PatternCondition("Bollinger Squeeze", "bollinger_squeeze", "==", True, lookback=1, weight=0.9),
                PatternCondition("Volume Declining", "volume_dry_up", "==", True, lookback=1, weight=0.8),
                PatternCondition("OI Building", "oi_slope", ">", 0, lookback=5, weight=0.7),
                PatternCondition("Long Compression", "compression_duration", ">", 5, lookback=1, weight=0.8),
            ],
            min_duration=5,
            max_duration=30,
            expected_outcomes={
                "EXPLOSION_UP": 0.40,
                "EXPLOSION_DOWN": 0.40,
                "CONTINUED_COMPRESSION": 0.15,
                "GRADUAL_BREAKOUT": 0.05,
            },
            risk_warnings=[
                "Direction UNKNOWN until explosion starts",
                "DO NOT predict direction",
                "Wait for first momentum bar"
            ],
            recommended_action="WAIT for direction, then follow momentum"
        )

    @staticmethod
    def delta_divergence_pattern() -> SequencePattern:
        """
        DELTA DIVERGENCE → REVERSAL

        CVD and price moving opposite:
        1. Price making new highs/lows
        2. CVD making opposite highs/lows
        3. Volume declining on price moves
        4. Momentum weakening

        Outcome: Reversal toward CVD direction
        """
        return SequencePattern(
            pattern_type=PatternType.DELTA_DIVERGENCE,
            name="Delta Divergence",
            description="CVD diverging from price - reversal likely",
            conditions=[
                PatternCondition("CVD Divergence", "cvd_price_divergence", "==", True, lookback=1, weight=1.0),
                PatternCondition("Momentum Weak", "momentum_divergence", "==", True, lookback=1, weight=0.9),
                PatternCondition("Volume Declining", "volume_trend", "<", 0, lookback=5, weight=0.8),
                PatternCondition("Price Extended", "price_momentum_5", "!=", 0, lookback=1, weight=0.7),
            ],
            min_duration=3,
            max_duration=15,
            expected_outcomes={
                "REVERSAL_TO_CVD": 0.55,
                "CONTINUATION_ANYWAY": 0.25,
                "CHOP": 0.20,
            },
            risk_warnings=[
                "Divergence can persist - wait for trigger",
                "Need confirmation candle",
                "Best in exhausted trends"
            ],
            recommended_action="WAIT for reversal trigger, trade toward CVD"
        )

    @staticmethod
    def oi_unwinding_pattern() -> SequencePattern:
        """
        OI UNWINDING → TREND CONTINUATION

        Positions closing:
        1. OI decreasing
        2. Price moving directionally
        3. Volume moderate to high
        4. Delta aligned with price

        Outcome: Trend continues as shorts/longs cover
        """
        return SequencePattern(
            pattern_type=PatternType.OI_UNWINDING,
            name="OI Unwinding",
            description="Positions unwinding - trend likely to continue",
            conditions=[
                PatternCondition("OI Decreasing", "oi_unwinding", "==", True, lookback=1, weight=1.0),
                PatternCondition("Price Trending", "price_momentum_5", "!=", 0, lookback=1, weight=0.9),
                PatternCondition("Volume Present", "volume_ratio", ">", 0.8, lookback=1, weight=0.7),
                PatternCondition("Delta Aligned", "delta_imbalance", "!=", 0, lookback=1, weight=0.8),
            ],
            min_duration=2,
            max_duration=10,
            expected_outcomes={
                "TREND_CONTINUES": 0.60,
                "REVERSAL": 0.20,
                "CONSOLIDATION": 0.20,
            },
            risk_warnings=[
                "Can reverse when unwinding completes",
                "Watch for volume exhaustion",
                "Trail stops in direction of trend"
            ],
            recommended_action="Follow trend with trailing stop"
        )


# =============================================================================
# PATTERN DETECTOR
# =============================================================================

class SequencePatternDetector:
    """
    Detects sequence patterns in market data

    Uses the pattern library to match current market conditions
    against known pre-move patterns.
    """

    def __init__(self, patterns: Optional[List[SequencePattern]] = None):
        """Initialize detector with pattern library"""
        self.patterns = patterns or PatternLibrary.get_all_patterns()
        self.min_similarity_threshold = 0.5  # Minimum similarity to report

    def analyze(
        self,
        features: Dict[str, Any],
        historical_features: Optional[List[Dict]] = None
    ) -> SequenceAnalysisResult:
        """
        Analyze current market state for patterns

        Args:
            features: Current market structure features (flat dict)
            historical_features: List of past feature snapshots

        Returns:
            SequenceAnalysisResult with detected patterns
        """
        patterns_detected = []
        key_observations = []

        for pattern in self.patterns:
            match = self._match_pattern(pattern, features, historical_features)
            if match and match.similarity_score >= self.min_similarity_threshold:
                patterns_detected.append(match)

        # Sort by confidence
        patterns_detected.sort(key=lambda x: x.confidence, reverse=True)

        # Primary pattern
        primary = patterns_detected[0] if patterns_detected else None

        # Aggregate probabilities
        prob_summary = self._aggregate_probabilities(patterns_detected)

        # Determine structure state
        structure_state = self._determine_structure_state(features, patterns_detected)

        # Determine phase
        phase = self._determine_phase(features, patterns_detected)

        # Generate observations
        key_observations = self._generate_observations(features, patterns_detected)

        # Risk level
        risk_level = self._assess_risk_level(features, patterns_detected)

        # Action recommendation
        action = self._generate_action_recommendation(primary, risk_level)

        return SequenceAnalysisResult(
            timestamp=datetime.now(),
            patterns_detected=patterns_detected,
            primary_pattern=primary,
            structure_state=structure_state,
            phase=phase,
            probability_summary=prob_summary,
            key_observations=key_observations,
            risk_level=risk_level,
            action_recommendation=action
        )

    def _match_pattern(
        self,
        pattern: SequencePattern,
        features: Dict[str, Any],
        historical: Optional[List[Dict]] = None
    ) -> Optional[PatternMatch]:
        """Match a single pattern against current features"""
        conditions_met = 0
        total_weight = 0
        met_weight = 0
        reasoning = []

        for condition in pattern.conditions:
            total_weight += condition.weight
            met = self._check_condition(condition, features, historical)

            if met:
                conditions_met += 1
                met_weight += condition.weight
                reasoning.append(f"✓ {condition.name}")
            else:
                reasoning.append(f"✗ {condition.name}")

        # Calculate similarity score (weighted)
        similarity = met_weight / total_weight if total_weight > 0 else 0

        if similarity < self.min_similarity_threshold:
            return None

        # Adjust outcome probabilities based on similarity
        adjusted_outcomes = {}
        for outcome, base_prob in pattern.expected_outcomes.items():
            adjusted_outcomes[outcome] = base_prob * similarity

        # Confidence = similarity * pattern specificity
        specificity = len(pattern.conditions) / 10  # More conditions = more specific
        confidence = similarity * min(specificity + 0.5, 1.0) * 100

        return PatternMatch(
            pattern_type=pattern.pattern_type,
            pattern_name=pattern.name,
            similarity_score=similarity,
            conditions_met=conditions_met,
            total_conditions=len(pattern.conditions),
            outcome_probabilities=adjusted_outcomes,
            confidence=confidence,
            bars_since_start=features.get('compression_duration', 0),
            expected_move_within=pattern.max_duration - features.get('compression_duration', 0),
            risk_warnings=pattern.risk_warnings,
            recommended_action=pattern.recommended_action,
            reasoning=reasoning
        )

    def _check_condition(
        self,
        condition: PatternCondition,
        features: Dict[str, Any],
        historical: Optional[List[Dict]] = None
    ) -> bool:
        """Check if a single condition is met"""
        value = features.get(condition.feature)

        if value is None:
            return False

        try:
            if condition.operator == ">":
                return value > condition.value
            elif condition.operator == "<":
                return value < condition.value
            elif condition.operator == ">=":
                return value >= condition.value
            elif condition.operator == "<=":
                return value <= condition.value
            elif condition.operator == "==":
                return value == condition.value
            elif condition.operator == "!=":
                if isinstance(condition.value, (int, float)):
                    return abs(value - condition.value) > 0.001 if isinstance(value, (int, float)) else value != condition.value
                return value != condition.value
            elif condition.operator == "between":
                low, high = condition.value
                return low <= value <= high
            elif condition.operator == "changed":
                # Need historical data
                if historical and len(historical) >= condition.lookback:
                    old_value = historical[-condition.lookback].get(condition.feature)
                    if old_value is not None:
                        return (value > 0) != (old_value > 0)  # Sign change
                return False
        except Exception as e:
            logger.warning(f"Error checking condition {condition.name}: {e}")

        return False

    def _aggregate_probabilities(
        self,
        patterns: List[PatternMatch]
    ) -> Dict[str, float]:
        """Aggregate probabilities from all detected patterns"""
        aggregated = {}

        if not patterns:
            return {"NO_PATTERN": 1.0}

        # Weight by confidence
        total_confidence = sum(p.confidence for p in patterns)

        for pattern in patterns:
            weight = pattern.confidence / total_confidence if total_confidence > 0 else 1

            for outcome, prob in pattern.outcome_probabilities.items():
                if outcome in aggregated:
                    aggregated[outcome] += prob * weight
                else:
                    aggregated[outcome] = prob * weight

        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return aggregated

    def _determine_structure_state(
        self,
        features: Dict[str, Any],
        patterns: List[PatternMatch]
    ) -> str:
        """Determine overall structure state"""
        if not patterns:
            return "NO_CLEAR_STRUCTURE"

        primary_type = patterns[0].pattern_type

        if primary_type == PatternType.SILENT_BUILDUP:
            return "ACCUMULATION_PHASE"
        elif primary_type == PatternType.COMPRESSION_EXPLOSION:
            return "COMPRESSION_PHASE"
        elif primary_type == PatternType.SL_HUNT_SETUP:
            return "MANIPULATION_PHASE"
        elif primary_type == PatternType.ABSORPTION_BREAK:
            return "ABSORPTION_PHASE"
        elif primary_type in [PatternType.FAKE_BREAK_REVERSAL, PatternType.DELTA_DIVERGENCE]:
            return "REVERSAL_SETUP"
        elif primary_type == PatternType.OI_UNWINDING:
            return "TREND_CONTINUATION"
        elif primary_type == PatternType.GAMMA_PIN_SNAP:
            return "GAMMA_PINNING"

        return "TRANSITION"

    def _determine_phase(
        self,
        features: Dict[str, Any],
        patterns: List[PatternMatch]
    ) -> str:
        """Determine phase within structure"""
        if not patterns:
            return "unclear"

        primary = patterns[0]
        maturity = features.get('structure_maturity', 0)
        breakout_imminence = features.get('breakout_imminence', 0)

        if breakout_imminence > 70:
            return "breaking"
        elif maturity > 80:
            return "mature"
        elif maturity > 50:
            return "developing"
        else:
            return "early"

    def _generate_observations(
        self,
        features: Dict[str, Any],
        patterns: List[PatternMatch]
    ) -> List[str]:
        """Generate key observations"""
        observations = []

        # Volatility state
        atr_pct = features.get('atr_percentile', 50)
        if atr_pct < 25:
            observations.append(f"🔸 Volatility COMPRESSED (ATR {atr_pct:.0f}th percentile)")
        elif atr_pct > 75:
            observations.append(f"🔹 Volatility EXPANDED (ATR {atr_pct:.0f}th percentile)")

        # Volume state
        vol_ratio = features.get('volume_ratio', 1)
        if vol_ratio < 0.7:
            observations.append(f"🔸 Volume DRY ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 1.5:
            observations.append(f"🔹 Volume HIGH ({vol_ratio:.1f}x avg)")

        # Liquidity traps
        equal_highs = features.get('equal_highs_count', 0)
        equal_lows = features.get('equal_lows_count', 0)
        if equal_highs >= 2:
            observations.append(f"⚠️ {equal_highs} EQUAL HIGHS (liquidity above)")
        if equal_lows >= 2:
            observations.append(f"⚠️ {equal_lows} EQUAL LOWS (liquidity below)")

        # Absorption
        if features.get('delta_absorption'):
            observations.append("🔸 ABSORPTION detected (high delta, flat price)")

        # Gamma
        if features.get('days_to_expiry', 7) < 2:
            observations.append("⚠️ EXPIRY PROXIMITY - gamma effects amplified")

        # Primary pattern
        if patterns:
            primary = patterns[0]
            observations.append(f"📊 Primary Pattern: {primary.pattern_name} ({primary.confidence:.0f}%)")

        return observations

    def _assess_risk_level(
        self,
        features: Dict[str, Any],
        patterns: List[PatternMatch]
    ) -> str:
        """Assess overall risk level"""
        risk_score = 0

        # High volatility = risk
        if features.get('atr_percentile', 50) > 80:
            risk_score += 30

        # Manipulation patterns = risk
        for p in patterns:
            if p.pattern_type == PatternType.SL_HUNT_SETUP:
                risk_score += 20
            if p.pattern_type == PatternType.FAKE_BREAK_REVERSAL:
                risk_score += 15

        # Expiry = risk
        if features.get('days_to_expiry', 7) < 1:
            risk_score += 25

        # Low clarity = risk
        if features.get('structure_clarity', 0) < 30:
            risk_score += 20

        if risk_score >= 60:
            return "EXTREME"
        elif risk_score >= 40:
            return "HIGH"
        elif risk_score >= 20:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_action_recommendation(
        self,
        primary: Optional[PatternMatch],
        risk_level: str
    ) -> str:
        """Generate action recommendation"""
        if not primary:
            return "NO CLEAR PATTERN - Wait for structure to develop"

        if risk_level == "EXTREME":
            return f"⚠️ HIGH RISK - Reduce size. {primary.recommended_action}"
        elif risk_level == "HIGH":
            return f"⚡ Caution advised. {primary.recommended_action}"
        else:
            return primary.recommended_action


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def flatten_features(snapshot) -> Dict[str, Any]:
    """Flatten MarketStructureSnapshot to dict for pattern matching"""
    features = {}

    if hasattr(snapshot, 'price_features'):
        pf = snapshot.price_features
        features['price_range_atr_ratio'] = pf.price_range_atr_ratio
        features['clv'] = pf.clv
        features['wick_imbalance'] = pf.wick_imbalance
        features['rejection_score'] = pf.rejection_score
        features['equal_highs_count'] = pf.equal_highs_count
        features['equal_lows_count'] = pf.equal_lows_count
        features['price_momentum_5'] = pf.price_momentum_5
        features['price_momentum_20'] = pf.price_momentum_20
        features['momentum_divergence'] = pf.momentum_divergence

    if hasattr(snapshot, 'volume_oi_features'):
        vf = snapshot.volume_oi_features
        features['volume_ratio'] = vf.volume_ratio
        features['volume_trend'] = vf.volume_trend
        features['volume_price_divergence'] = vf.volume_price_divergence
        features['volume_climax'] = vf.volume_climax
        features['volume_dry_up'] = vf.volume_dry_up
        features['oi_change_pct'] = vf.oi_change_pct
        features['oi_price_divergence'] = vf.oi_price_divergence
        features['oi_slope'] = vf.oi_slope
        features['oi_buildup_score'] = vf.oi_buildup_score
        features['oi_pcr'] = vf.oi_pcr
        features['oi_unwinding'] = vf.oi_unwinding

    if hasattr(snapshot, 'delta_flow_features'):
        df = snapshot.delta_flow_features
        features['cvd_value'] = df.cvd_value
        features['cvd_slope'] = df.cvd_slope
        features['cvd_price_divergence'] = df.cvd_price_divergence
        features['delta_imbalance'] = df.delta_imbalance
        features['delta_absorption'] = df.delta_absorption
        features['delta_exhaustion'] = df.delta_exhaustion

    if hasattr(snapshot, 'volatility_features'):
        volf = snapshot.volatility_features
        features['atr_current'] = volf.atr_current
        features['atr_ratio'] = volf.atr_ratio
        features['atr_percentile'] = volf.atr_percentile
        features['compression_score'] = volf.compression_score
        features['compression_duration'] = volf.compression_duration
        features['bollinger_squeeze'] = volf.bollinger_squeeze
        features['vix_level'] = volf.vix_level
        features['vix_percentile'] = volf.vix_percentile

    if hasattr(snapshot, 'market_depth_features'):
        mdf = snapshot.market_depth_features
        features['bid_ask_imbalance'] = mdf.bid_ask_imbalance
        features['spoof_score'] = mdf.spoof_score
        features['phantom_liquidity'] = mdf.phantom_liquidity

    if hasattr(snapshot, 'gamma_features'):
        gf = snapshot.gamma_features
        features['gex_flip_distance'] = gf.gex_flip_distance
        features['pin_probability'] = gf.pin_probability
        features['days_to_expiry'] = gf.days_to_expiry
        features['is_expiry_day'] = gf.is_expiry_day

    if hasattr(snapshot, 'derived_indicators'):
        di = snapshot.derived_indicators
        features['accumulation_score'] = di.accumulation_score
        features['distribution_score'] = di.distribution_score
        features['manipulation_score'] = di.manipulation_score
        features['structure_maturity'] = di.structure_maturity
        features['breakout_imminence'] = di.breakout_imminence
        features['sl_hunt_probability_above'] = di.sl_hunt_probability_above
        features['sl_hunt_probability_below'] = di.sl_hunt_probability_below
        features['structure_clarity'] = di.structure_clarity

    return features


def result_to_dict(result: SequenceAnalysisResult) -> Dict:
    """Convert SequenceAnalysisResult to dict for storage/display"""
    return {
        'timestamp': result.timestamp.isoformat(),
        'structure_state': result.structure_state,
        'phase': result.phase,
        'risk_level': result.risk_level,
        'action_recommendation': result.action_recommendation,
        'probability_summary': result.probability_summary,
        'key_observations': result.key_observations,
        'patterns': [
            {
                'name': p.pattern_name,
                'type': p.pattern_type.value,
                'similarity': p.similarity_score,
                'confidence': p.confidence,
                'conditions_met': f"{p.conditions_met}/{p.total_conditions}",
                'outcome_probabilities': p.outcome_probabilities,
                'risk_warnings': p.risk_warnings,
                'recommended_action': p.recommended_action,
                'reasoning': p.reasoning,
            }
            for p in result.patterns_detected
        ],
        'primary_pattern': result.primary_pattern.pattern_name if result.primary_pattern else None,
    }
