"""
Probability Engine
==================
Historical pattern matching and probability calculation.

Philosophy:
- We NEVER say "This WILL happen"
- We say "Given history, probability is X%"

Probability = (Similar patterns leading to outcome) / (Total similar patterns)
Confidence = Similarity_Score * log(Occurrences)

Weighted by:
- Recency (recent patterns matter more)
- Regime match (same market regime)
- Expiry proximity (expiry vs non-expiry patterns)
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
import math

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HistoricalPattern:
    """A historical pattern with known outcome"""
    timestamp: datetime
    features: Dict[str, float]          # Feature values at pattern time
    structure_type: str                 # Accumulation, Distribution, etc.
    outcome: str                        # What happened after
    outcome_magnitude: float            # How big was the move (%)
    outcome_bars: int                   # How many bars until outcome
    regime: str                         # Market regime at time
    is_expiry: bool                     # Was this on expiry day/week
    is_expiry_week: bool
    vix_level: float                    # VIX at time
    similarity_hash: str = ""           # For quick lookup


@dataclass
class PatternMatchResult:
    """Result of matching current state to historical patterns"""
    historical_pattern: HistoricalPattern
    similarity_score: float             # 0-1 similarity
    recency_weight: float               # Weight for recency
    regime_weight: float                # Weight for regime match
    final_score: float                  # Combined weighted score


@dataclass
class ProbabilityResult:
    """Probability calculation result"""
    outcome: str
    probability: float                  # 0-1 probability
    confidence: float                   # 0-100 confidence
    occurrences: int                    # How many similar patterns
    avg_magnitude: float                # Average move size
    avg_bars_to_outcome: int            # Average time to outcome
    recent_success_rate: float          # Success rate in recent history
    regime_specific_prob: float         # Probability in current regime


@dataclass
class ProbabilityAnalysis:
    """Complete probability analysis"""
    timestamp: datetime
    current_features: Dict[str, float]
    similar_patterns_found: int
    outcome_probabilities: Dict[str, ProbabilityResult]
    primary_outcome: str
    primary_probability: float
    primary_confidence: float
    expected_move_magnitude: float
    expected_bars_to_move: int
    reliability_score: float            # How reliable is this analysis
    warnings: List[str]
    reasoning: List[str]


# =============================================================================
# PATTERN DATABASE
# =============================================================================

class PatternDatabase:
    """
    In-memory pattern database for fast lookup

    In production, this could be backed by a database.
    For now, we use in-memory storage with optional file persistence.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize pattern database"""
        self.patterns: List[HistoricalPattern] = []
        self.db_path = db_path or "pattern_database.json"
        self.feature_stats: Dict[str, Dict] = {}  # For normalization

    def add_pattern(self, pattern: HistoricalPattern) -> None:
        """Add a pattern to the database"""
        self.patterns.append(pattern)
        self._update_feature_stats(pattern.features)

    def add_patterns(self, patterns: List[HistoricalPattern]) -> None:
        """Add multiple patterns"""
        for p in patterns:
            self.add_pattern(p)

    def _update_feature_stats(self, features: Dict[str, float]) -> None:
        """Update feature statistics for normalization"""
        for key, value in features.items():
            if key not in self.feature_stats:
                self.feature_stats[key] = {'values': [], 'mean': 0, 'std': 1}

            self.feature_stats[key]['values'].append(value)

            # Update mean and std
            vals = self.feature_stats[key]['values'][-1000:]  # Keep last 1000
            self.feature_stats[key]['mean'] = np.mean(vals)
            self.feature_stats[key]['std'] = max(np.std(vals), 0.001)

    def get_similar_patterns(
        self,
        current_features: Dict[str, float],
        min_similarity: float = 0.6,
        max_patterns: int = 100,
        regime_filter: Optional[str] = None,
        expiry_filter: Optional[bool] = None
    ) -> List[PatternMatchResult]:
        """
        Find patterns similar to current state

        Args:
            current_features: Current market features
            min_similarity: Minimum similarity threshold
            max_patterns: Maximum patterns to return
            regime_filter: Only match patterns from this regime
            expiry_filter: True = only expiry, False = no expiry, None = all

        Returns:
            List of PatternMatchResult sorted by final score
        """
        results = []

        for pattern in self.patterns:
            # Apply filters
            if regime_filter and pattern.regime != regime_filter:
                continue
            if expiry_filter is not None:
                if expiry_filter and not (pattern.is_expiry or pattern.is_expiry_week):
                    continue
                if not expiry_filter and (pattern.is_expiry or pattern.is_expiry_week):
                    continue

            # Calculate similarity
            similarity = self._calculate_similarity(current_features, pattern.features)

            if similarity >= min_similarity:
                # Calculate weights
                recency_weight = self._calculate_recency_weight(pattern.timestamp)
                regime_weight = 1.0 if regime_filter else 0.8

                final_score = similarity * recency_weight * regime_weight

                results.append(PatternMatchResult(
                    historical_pattern=pattern,
                    similarity_score=similarity,
                    recency_weight=recency_weight,
                    regime_weight=regime_weight,
                    final_score=final_score
                ))

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results[:max_patterns]

    def _calculate_similarity(
        self,
        current: Dict[str, float],
        historical: Dict[str, float]
    ) -> float:
        """
        Calculate similarity between current and historical features

        Uses normalized Euclidean distance converted to similarity score
        """
        common_keys = set(current.keys()) & set(historical.keys())

        if not common_keys:
            return 0.0

        distances = []
        weights = []

        # Feature importance weights
        importance = {
            'atr_percentile': 2.0,
            'compression_score': 2.0,
            'volume_ratio': 1.5,
            'oi_pcr': 1.5,
            'delta_imbalance': 1.5,
            'equal_highs_count': 2.0,
            'equal_lows_count': 2.0,
            'structure_maturity': 1.8,
        }

        for key in common_keys:
            curr_val = current[key]
            hist_val = historical[key]

            # Normalize using stored statistics
            if key in self.feature_stats:
                mean = self.feature_stats[key]['mean']
                std = self.feature_stats[key]['std']
                curr_norm = (curr_val - mean) / std
                hist_norm = (hist_val - mean) / std
            else:
                curr_norm = curr_val
                hist_norm = hist_val

            distance = abs(curr_norm - hist_norm)
            weight = importance.get(key, 1.0)

            distances.append(distance * weight)
            weights.append(weight)

        # Weighted average distance
        avg_distance = sum(distances) / sum(weights) if weights else 0

        # Convert to similarity (0-1)
        similarity = 1 / (1 + avg_distance)

        return similarity

    def _calculate_recency_weight(self, pattern_time: datetime) -> float:
        """
        Calculate recency weight - recent patterns matter more

        Uses exponential decay with half-life of 30 days
        """
        now = datetime.now()
        days_ago = (now - pattern_time).days

        half_life = 30  # days
        decay = 0.5 ** (days_ago / half_life)

        # Minimum weight of 0.3 even for old patterns
        return max(decay, 0.3)

    def save(self) -> bool:
        """Save database to file"""
        try:
            data = []
            for p in self.patterns:
                data.append({
                    'timestamp': p.timestamp.isoformat(),
                    'features': p.features,
                    'structure_type': p.structure_type,
                    'outcome': p.outcome,
                    'outcome_magnitude': p.outcome_magnitude,
                    'outcome_bars': p.outcome_bars,
                    'regime': p.regime,
                    'is_expiry': p.is_expiry,
                    'is_expiry_week': p.is_expiry_week,
                    'vix_level': p.vix_level,
                })

            with open(self.db_path, 'w') as f:
                json.dump({
                    'patterns': data,
                    'feature_stats': {
                        k: {'mean': v['mean'], 'std': v['std']}
                        for k, v in self.feature_stats.items()
                    }
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving pattern database: {e}")
            return False

    def load(self) -> bool:
        """Load database from file"""
        if not os.path.exists(self.db_path):
            return False

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            self.patterns = []
            for p in data.get('patterns', []):
                self.patterns.append(HistoricalPattern(
                    timestamp=datetime.fromisoformat(p['timestamp']),
                    features=p['features'],
                    structure_type=p['structure_type'],
                    outcome=p['outcome'],
                    outcome_magnitude=p['outcome_magnitude'],
                    outcome_bars=p['outcome_bars'],
                    regime=p['regime'],
                    is_expiry=p['is_expiry'],
                    is_expiry_week=p['is_expiry_week'],
                    vix_level=p['vix_level'],
                ))

            stats = data.get('feature_stats', {})
            for k, v in stats.items():
                self.feature_stats[k] = {
                    'values': [],
                    'mean': v['mean'],
                    'std': v['std']
                }

            return True
        except Exception as e:
            logger.error(f"Error loading pattern database: {e}")
            return False

    @property
    def size(self) -> int:
        """Number of patterns in database"""
        return len(self.patterns)


# =============================================================================
# PROBABILITY ENGINE
# =============================================================================

class ProbabilityEngine:
    """
    Main probability calculation engine

    Uses historical pattern matching to calculate outcome probabilities.
    """

    def __init__(self, database: Optional[PatternDatabase] = None):
        """Initialize probability engine"""
        self.database = database or PatternDatabase()

        # Outcome definitions
        self.outcomes = [
            "EXPANSION_UP",
            "EXPANSION_DOWN",
            "FAKE_BREAK_UP",
            "FAKE_BREAK_DOWN",
            "SL_HUNT_ABOVE",
            "SL_HUNT_BELOW",
            "CONTINUED_RANGE",
            "GAMMA_SNAP",
            "NO_MOVE"
        ]

    def analyze(
        self,
        current_features: Dict[str, float],
        current_regime: Optional[str] = None,
        is_expiry: bool = False,
        is_expiry_week: bool = False
    ) -> ProbabilityAnalysis:
        """
        Calculate probabilities for all outcomes

        Args:
            current_features: Current market structure features
            current_regime: Current market regime
            is_expiry: Is today expiry day
            is_expiry_week: Is this expiry week

        Returns:
            ProbabilityAnalysis with all outcome probabilities
        """
        warnings = []
        reasoning = []

        # Find similar historical patterns
        expiry_filter = True if is_expiry or is_expiry_week else None
        similar_patterns = self.database.get_similar_patterns(
            current_features,
            min_similarity=0.5,
            max_patterns=200,
            regime_filter=current_regime,
            expiry_filter=expiry_filter
        )

        if len(similar_patterns) < 10:
            warnings.append(f"Low sample size ({len(similar_patterns)} patterns)")
            reasoning.append("Confidence reduced due to limited historical data")

        # Calculate outcome probabilities
        outcome_probs = self._calculate_outcome_probabilities(
            similar_patterns,
            current_regime,
            is_expiry or is_expiry_week
        )

        # Find primary outcome
        primary = max(outcome_probs.values(), key=lambda x: x.probability)
        primary_outcome = primary.outcome
        primary_probability = primary.probability
        primary_confidence = primary.confidence

        # Expected magnitude and timing
        expected_magnitude = primary.avg_magnitude
        expected_bars = primary.avg_bars_to_outcome

        # Reliability score
        reliability = self._calculate_reliability(
            len(similar_patterns),
            primary_confidence,
            current_features
        )

        # Generate reasoning
        reasoning.extend(self._generate_reasoning(
            similar_patterns,
            outcome_probs,
            current_features
        ))

        return ProbabilityAnalysis(
            timestamp=datetime.now(),
            current_features=current_features,
            similar_patterns_found=len(similar_patterns),
            outcome_probabilities=outcome_probs,
            primary_outcome=primary_outcome,
            primary_probability=primary_probability,
            primary_confidence=primary_confidence,
            expected_move_magnitude=expected_magnitude,
            expected_bars_to_move=expected_bars,
            reliability_score=reliability,
            warnings=warnings,
            reasoning=reasoning
        )

    def _calculate_outcome_probabilities(
        self,
        similar_patterns: List[PatternMatchResult],
        current_regime: Optional[str],
        is_expiry: bool
    ) -> Dict[str, ProbabilityResult]:
        """Calculate probability for each outcome"""
        results = {}

        if not similar_patterns:
            # Return uniform probabilities if no data
            for outcome in self.outcomes:
                results[outcome] = ProbabilityResult(
                    outcome=outcome,
                    probability=1.0 / len(self.outcomes),
                    confidence=0,
                    occurrences=0,
                    avg_magnitude=0,
                    avg_bars_to_outcome=0,
                    recent_success_rate=0,
                    regime_specific_prob=0
                )
            return results

        total_weight = sum(p.final_score for p in similar_patterns)

        for outcome in self.outcomes:
            # Filter patterns with this outcome
            outcome_patterns = [
                p for p in similar_patterns
                if p.historical_pattern.outcome == outcome
            ]

            if not outcome_patterns:
                results[outcome] = ProbabilityResult(
                    outcome=outcome,
                    probability=0,
                    confidence=0,
                    occurrences=0,
                    avg_magnitude=0,
                    avg_bars_to_outcome=0,
                    recent_success_rate=0,
                    regime_specific_prob=0
                )
                continue

            # Weighted probability
            outcome_weight = sum(p.final_score for p in outcome_patterns)
            probability = outcome_weight / total_weight if total_weight > 0 else 0

            # Occurrences
            occurrences = len(outcome_patterns)

            # Confidence = similarity * log(occurrences)
            avg_similarity = np.mean([p.similarity_score for p in outcome_patterns])
            confidence = avg_similarity * math.log(max(occurrences, 1) + 1) * 20
            confidence = min(confidence, 100)

            # Average magnitude and bars
            avg_magnitude = np.mean([
                p.historical_pattern.outcome_magnitude
                for p in outcome_patterns
            ])
            avg_bars = int(np.mean([
                p.historical_pattern.outcome_bars
                for p in outcome_patterns
            ]))

            # Recent success rate (last 30 days)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_patterns = [
                p for p in outcome_patterns
                if p.historical_pattern.timestamp > recent_cutoff
            ]
            total_recent = [
                p for p in similar_patterns
                if p.historical_pattern.timestamp > recent_cutoff
            ]
            recent_success = len(recent_patterns) / len(total_recent) if total_recent else 0

            # Regime-specific probability
            if current_regime:
                regime_patterns = [
                    p for p in outcome_patterns
                    if p.historical_pattern.regime == current_regime
                ]
                regime_total = [
                    p for p in similar_patterns
                    if p.historical_pattern.regime == current_regime
                ]
                regime_prob = len(regime_patterns) / len(regime_total) if regime_total else probability
            else:
                regime_prob = probability

            results[outcome] = ProbabilityResult(
                outcome=outcome,
                probability=probability,
                confidence=confidence,
                occurrences=occurrences,
                avg_magnitude=avg_magnitude,
                avg_bars_to_outcome=avg_bars,
                recent_success_rate=recent_success,
                regime_specific_prob=regime_prob
            )

        return results

    def _calculate_reliability(
        self,
        num_patterns: int,
        confidence: float,
        features: Dict[str, float]
    ) -> float:
        """
        Calculate overall reliability score

        Based on:
        - Sample size
        - Confidence
        - Structure clarity
        """
        # Sample size factor
        sample_factor = min(num_patterns / 50, 1.0)  # Max at 50 patterns

        # Confidence factor
        conf_factor = confidence / 100

        # Structure clarity
        clarity = features.get('structure_clarity', 50)
        clarity_factor = clarity / 100

        # Combined reliability
        reliability = (sample_factor * 0.4 + conf_factor * 0.4 + clarity_factor * 0.2) * 100

        return min(reliability, 100)

    def _generate_reasoning(
        self,
        similar_patterns: List[PatternMatchResult],
        outcome_probs: Dict[str, ProbabilityResult],
        features: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []

        if similar_patterns:
            best_match = similar_patterns[0]
            reasoning.append(
                f"Best match: {best_match.historical_pattern.structure_type} "
                f"({best_match.similarity_score:.0%} similar)"
            )
            reasoning.append(
                f"Historical outcome: {best_match.historical_pattern.outcome} "
                f"({best_match.historical_pattern.outcome_magnitude:.1f}% move)"
            )

        # Top 3 outcomes
        top_outcomes = sorted(
            outcome_probs.values(),
            key=lambda x: x.probability,
            reverse=True
        )[:3]

        reasoning.append("Top 3 probable outcomes:")
        for i, outcome in enumerate(top_outcomes, 1):
            reasoning.append(
                f"  {i}. {outcome.outcome}: {outcome.probability:.0%} "
                f"({outcome.occurrences} occurrences)"
            )

        # Feature-based insights
        if features.get('atr_percentile', 50) < 25:
            reasoning.append("Volatility compressed - explosion likely")
        if features.get('equal_highs_count', 0) >= 2:
            reasoning.append("Equal highs detected - SL hunt above probable")
        if features.get('equal_lows_count', 0) >= 2:
            reasoning.append("Equal lows detected - SL hunt below probable")

        return reasoning

    def record_outcome(
        self,
        features: Dict[str, float],
        structure_type: str,
        outcome: str,
        outcome_magnitude: float,
        outcome_bars: int,
        regime: str,
        is_expiry: bool,
        is_expiry_week: bool,
        vix_level: float
    ) -> None:
        """
        Record an outcome for future pattern matching

        Call this when you know what happened after a structure.
        """
        pattern = HistoricalPattern(
            timestamp=datetime.now(),
            features=features,
            structure_type=structure_type,
            outcome=outcome,
            outcome_magnitude=outcome_magnitude,
            outcome_bars=outcome_bars,
            regime=regime,
            is_expiry=is_expiry,
            is_expiry_week=is_expiry_week,
            vix_level=vix_level
        )
        self.database.add_pattern(pattern)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_base_probabilities(
    features: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate base probabilities from features alone

    Used when historical data is limited.
    These are rule-based defaults.
    """
    probs = {
        "EXPANSION_UP": 0.2,
        "EXPANSION_DOWN": 0.2,
        "FAKE_BREAK_UP": 0.1,
        "FAKE_BREAK_DOWN": 0.1,
        "SL_HUNT_ABOVE": 0.1,
        "SL_HUNT_BELOW": 0.1,
        "CONTINUED_RANGE": 0.15,
        "GAMMA_SNAP": 0.05,
        "NO_MOVE": 0.0,
    }

    # Adjust based on features
    compression = features.get('compression_score', 0)
    if compression > 70:
        # High compression = explosion likely
        probs["EXPANSION_UP"] += 0.1
        probs["EXPANSION_DOWN"] += 0.1
        probs["CONTINUED_RANGE"] -= 0.15
        probs["NO_MOVE"] -= 0.05

    equal_highs = features.get('equal_highs_count', 0)
    if equal_highs >= 2:
        probs["SL_HUNT_ABOVE"] += 0.15
        probs["EXPANSION_UP"] -= 0.1

    equal_lows = features.get('equal_lows_count', 0)
    if equal_lows >= 2:
        probs["SL_HUNT_BELOW"] += 0.15
        probs["EXPANSION_DOWN"] -= 0.1

    delta_imbalance = features.get('delta_imbalance', 0)
    if delta_imbalance > 0.3:
        probs["EXPANSION_UP"] += 0.1
        probs["EXPANSION_DOWN"] -= 0.05
    elif delta_imbalance < -0.3:
        probs["EXPANSION_DOWN"] += 0.1
        probs["EXPANSION_UP"] -= 0.05

    is_expiry = features.get('is_expiry_day', False)
    if is_expiry:
        probs["GAMMA_SNAP"] += 0.15
        probs["FAKE_BREAK_UP"] += 0.05
        probs["FAKE_BREAK_DOWN"] += 0.05

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: max(0, v / total) for k, v in probs.items()}

    return probs


def format_probability_output(analysis: ProbabilityAnalysis) -> str:
    """Format probability analysis for display"""
    lines = []

    lines.append("=" * 50)
    lines.append("🧠 MARKET STRUCTURE PROBABILITY ANALYSIS")
    lines.append("=" * 50)
    lines.append("")

    lines.append(f"📊 Similar Patterns Found: {analysis.similar_patterns_found}")
    lines.append(f"📈 Reliability Score: {analysis.reliability_score:.0f}/100")
    lines.append("")

    lines.append("📉 OUTCOME PROBABILITIES:")
    lines.append("-" * 40)

    # Sort by probability
    sorted_outcomes = sorted(
        analysis.outcome_probabilities.values(),
        key=lambda x: x.probability,
        reverse=True
    )

    for outcome in sorted_outcomes:
        if outcome.probability > 0.01:  # Only show >1%
            bar = "█" * int(outcome.probability * 20)
            lines.append(
                f"  {outcome.outcome:20} {outcome.probability:5.1%} {bar}"
            )
            lines.append(
                f"    Confidence: {outcome.confidence:.0f}% | "
                f"Occurrences: {outcome.occurrences} | "
                f"Avg Move: {outcome.avg_magnitude:.1f}%"
            )

    lines.append("")
    lines.append("🎯 PRIMARY EXPECTATION:")
    lines.append(f"  Outcome: {analysis.primary_outcome}")
    lines.append(f"  Probability: {analysis.primary_probability:.1%}")
    lines.append(f"  Confidence: {analysis.primary_confidence:.0f}%")
    lines.append(f"  Expected Move: {analysis.expected_move_magnitude:.1f}%")
    lines.append(f"  Expected Within: {analysis.expected_bars_to_move} bars")

    if analysis.warnings:
        lines.append("")
        lines.append("⚠️ WARNINGS:")
        for w in analysis.warnings:
            lines.append(f"  • {w}")

    lines.append("")
    lines.append("💡 REASONING:")
    for r in analysis.reasoning:
        lines.append(f"  {r}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def analysis_to_dict(analysis: ProbabilityAnalysis) -> Dict:
    """Convert ProbabilityAnalysis to dict for storage/display"""
    return {
        'timestamp': analysis.timestamp.isoformat(),
        'similar_patterns_found': analysis.similar_patterns_found,
        'primary_outcome': analysis.primary_outcome,
        'primary_probability': analysis.primary_probability,
        'primary_confidence': analysis.primary_confidence,
        'expected_move_magnitude': analysis.expected_move_magnitude,
        'expected_bars_to_move': analysis.expected_bars_to_move,
        'reliability_score': analysis.reliability_score,
        'outcome_probabilities': {
            k: {
                'probability': v.probability,
                'confidence': v.confidence,
                'occurrences': v.occurrences,
                'avg_magnitude': v.avg_magnitude,
                'recent_success_rate': v.recent_success_rate,
            }
            for k, v in analysis.outcome_probabilities.items()
        },
        'warnings': analysis.warnings,
        'reasoning': analysis.reasoning,
    }
