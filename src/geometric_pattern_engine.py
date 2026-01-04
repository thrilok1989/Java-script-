"""
Geometric Pattern Engine - Institution-Grade Pattern Detection
Converts geometric shapes into numeric features ML can learn

Philosophy: If a pattern cannot be expressed in math, ignore it.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════

class GeometricPatternType(Enum):
    """Patterns that can be expressed mathematically"""
    # Triangles
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRIC_TRIANGLE = "SYMMETRIC_TRIANGLE"

    # Channels & Ranges
    PARALLEL_CHANNEL = "PARALLEL_CHANNEL"
    RISING_CHANNEL = "RISING_CHANNEL"
    FALLING_CHANNEL = "FALLING_CHANNEL"
    HORIZONTAL_RANGE = "HORIZONTAL_RANGE"

    # Wedges
    RISING_WEDGE = "RISING_WEDGE"      # Bearish
    FALLING_WEDGE = "FALLING_WEDGE"    # Bullish

    # Flags & Pennants
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    PENNANT = "PENNANT"

    # Tops & Bottoms
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"

    # Head & Shoulders
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"

    # Compression
    COIL_BOX = "COIL_BOX"  # Compression < 1.2 ATR


class PatternDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    """Detected swing high or low"""
    index: int
    price: float
    timestamp: datetime
    is_high: bool
    strength: int  # Number of bars confirming the swing


@dataclass
class TrendLine:
    """Fitted trendline through swing points"""
    slope: float
    intercept: float
    start_idx: int
    end_idx: int
    points: List[SwingPoint]
    r_squared: float  # Fit quality


@dataclass
class GeometricFeatures:
    """Numeric features extracted from geometry"""
    # Slope features
    slope_highs: float = 0.0
    slope_lows: float = 0.0
    norm_slope_highs: float = 0.0  # Normalized by ATR
    norm_slope_lows: float = 0.0

    # Range features
    range_width: float = 0.0
    compression_ratio: float = 0.0  # Range / ATR

    # Convergence (triangles)
    convergence_rate: float = 0.0  # |slope_highs - slope_lows|
    is_converging: bool = False

    # Channel features
    channel_parallelism: float = 0.0  # How parallel are the lines
    channel_width_stability: float = 0.0  # std of width

    # Retracement
    retracement_pct: float = 0.0

    # Structure counts
    swing_high_count: int = 0
    swing_low_count: int = 0


@dataclass
class DetectedPattern:
    """A detected geometric pattern with all metadata"""
    pattern_type: GeometricPatternType
    direction: PatternDirection

    # Geometry score (0-1)
    geometry_score: float

    # Validation flags
    volume_confirmed: bool = False
    oi_confirmed: bool = False
    delta_confirmed: bool = False

    # Risk assessment
    fake_move_risk: float = 0.0

    # Expected outcomes (probabilities)
    expected_outcomes: Dict[str, float] = field(default_factory=dict)

    # Pattern boundaries
    start_idx: int = 0
    end_idx: int = 0

    # Key price levels
    resistance: float = 0.0
    support: float = 0.0
    target: float = 0.0

    # Geometric features used
    features: Optional[GeometricFeatures] = None

    # Timestamp
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatternAnalysisResult:
    """Complete pattern analysis output"""
    patterns: List[DetectedPattern]
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    geometric_features: GeometricFeatures
    dominant_pattern: Optional[DetectedPattern] = None
    market_structure: str = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# SWING POINT DETECTOR (MANDATORY FIRST STEP)
# ═══════════════════════════════════════════════════════════════════════

class SwingPointDetector:
    """
    Detects swing highs and lows using fractal logic
    All geometry starts here
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def detect_swings(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low'
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows

        Swing_High: High[i] > High[i-1..i-n] AND High[i] > High[i+1..i+n]
        Swing_Low:  Low[i] < Low[i-1..i-n] AND Low[i] < Low[i+1..i+n]
        """
        swing_highs = []
        swing_lows = []

        highs = df[high_col].values
        lows = df[low_col].values
        n = self.lookback

        # Get timestamps if available
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.tolist()
        else:
            timestamps = [datetime.now() for _ in range(len(df))]

        for i in range(n, len(df) - n):
            # Check swing high
            is_swing_high = True
            strength = 0

            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
                strength += 1

            if is_swing_high:
                swing_highs.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    is_high=True,
                    strength=strength
                ))

            # Check swing low
            is_swing_low = True
            strength = 0

            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
                strength += 1

            if is_swing_low:
                swing_lows.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    is_high=False,
                    strength=strength
                ))

        return swing_highs, swing_lows

    def get_recent_swings(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        count: int = 5
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Get most recent swing points"""
        return swing_highs[-count:], swing_lows[-count:]


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class GeometryCalculator:
    """
    Converts geometric shapes into numeric features
    These formulas are the core of pattern detection
    """

    @staticmethod
    def calculate_slope(points: List[SwingPoint]) -> Tuple[float, float, float]:
        """
        Calculate slope of line through swing points
        Returns: (slope, intercept, r_squared)

        Slope = (Price_end - Price_start) / Time
        """
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        x = np.array([p.index for p in points])
        y = np.array([p.price for p in points])

        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0, y.mean(), 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return slope, intercept, r_squared

    @staticmethod
    def normalize_slope(slope: float, atr: float) -> float:
        """
        Normalize slope by ATR for comparability
        Norm_Slope = Slope / ATR
        """
        if atr <= 0:
            return 0.0
        return slope / atr

    @staticmethod
    def calculate_range_width(highs: List[float], lows: List[float]) -> float:
        """
        Range_Width = High_N - Low_N
        """
        if not highs or not lows:
            return 0.0
        return max(highs) - min(lows)

    @staticmethod
    def calculate_compression_ratio(range_width: float, atr: float) -> float:
        """
        Compression = Range_Width / ATR
        Compression < 1.2 → Coil / Box
        """
        if atr <= 0:
            return 0.0
        return range_width / atr

    @staticmethod
    def calculate_convergence(slope_highs: float, slope_lows: float) -> float:
        """
        Convergence Rate = |Slope_highs - Slope_lows|
        Triangles: Convergence → 0
        Expansion: Convergence increases
        """
        return abs(slope_highs - slope_lows)

    @staticmethod
    def calculate_retracement(
        impulse_start: float,
        impulse_end: float,
        pullback_end: float
    ) -> float:
        """
        Retracement % = Pullback / Impulse
        Flag: 30–50%
        Deep pullback: >61.8%
        """
        impulse = abs(impulse_end - impulse_start)
        if impulse <= 0:
            return 0.0
        pullback = abs(pullback_end - impulse_end)
        return pullback / impulse

    @staticmethod
    def check_channel_parallelism(
        slope_highs: float,
        slope_lows: float,
        epsilon: float = 0.001
    ) -> float:
        """
        Channel Parallelism = |Slope_highs - Slope_lows| < ε
        Returns similarity score (1 = perfectly parallel)
        """
        diff = abs(slope_highs - slope_lows)
        # Convert to similarity score
        return max(0, 1 - diff / (epsilon * 10))

    @staticmethod
    def check_equal_levels(
        level1: float,
        level2: float,
        atr: float,
        threshold: float = 0.15
    ) -> bool:
        """
        Equal Level Condition: |Top1 - Top2| < 0.15 * ATR
        Used for double/triple tops/bottoms
        """
        if atr <= 0:
            return False
        return abs(level1 - level2) < threshold * atr

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        return np.mean(tr[-period:])


# ═══════════════════════════════════════════════════════════════════════
# PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════════════

class GeometricPatternDetector:
    """
    Detects geometric patterns from swing points
    Step 2: BUILD STRUCTURES FROM SWINGS
    """

    def __init__(self):
        self.swing_detector = SwingPointDetector(lookback=5)
        self.geometry = GeometryCalculator()

    def detect_triangle(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect triangle patterns
        - Ascending: Flat highs + Higher lows
        - Descending: Lower highs + Flat lows
        - Symmetric: Lower highs + Higher lows (converging)
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        slope_highs, _, r2_h = self.geometry.calculate_slope(swing_highs[-3:])
        slope_lows, _, r2_l = self.geometry.calculate_slope(swing_lows[-3:])

        norm_slope_h = self.geometry.normalize_slope(slope_highs, atr)
        norm_slope_l = self.geometry.normalize_slope(slope_lows, atr)

        convergence = self.geometry.calculate_convergence(slope_highs, slope_lows)

        # Check for convergence
        if not (slope_highs <= 0 and slope_lows >= 0):
            # Not converging (one could be flat)
            pass

        pattern_type = None
        direction = PatternDirection.NEUTRAL
        geometry_score = 0.0

        # Ascending Triangle: Flat highs + Higher lows
        if abs(norm_slope_h) < 0.0005 and norm_slope_l > 0.0002:
            pattern_type = GeometricPatternType.ASCENDING_TRIANGLE
            direction = PatternDirection.BULLISH
            geometry_score = min(1.0, r2_l * (1 - abs(norm_slope_h) * 100))

        # Descending Triangle: Lower highs + Flat lows
        elif norm_slope_h < -0.0002 and abs(norm_slope_l) < 0.0005:
            pattern_type = GeometricPatternType.DESCENDING_TRIANGLE
            direction = PatternDirection.BEARISH
            geometry_score = min(1.0, r2_h * (1 - abs(norm_slope_l) * 100))

        # Symmetric Triangle: Lower highs + Higher lows
        elif norm_slope_h < -0.0001 and norm_slope_l > 0.0001:
            pattern_type = GeometricPatternType.SYMMETRIC_TRIANGLE
            direction = PatternDirection.NEUTRAL
            geometry_score = min(1.0, (r2_h + r2_l) / 2 * convergence * 10)

        if pattern_type is None:
            return None

        return DetectedPattern(
            pattern_type=pattern_type,
            direction=direction,
            geometry_score=geometry_score,
            start_idx=min(swing_highs[0].index, swing_lows[0].index),
            end_idx=max(swing_highs[-1].index, swing_lows[-1].index),
            resistance=max(sh.price for sh in swing_highs[-3:]),
            support=min(sl.price for sl in swing_lows[-3:]),
            features=GeometricFeatures(
                slope_highs=slope_highs,
                slope_lows=slope_lows,
                norm_slope_highs=norm_slope_h,
                norm_slope_lows=norm_slope_l,
                convergence_rate=convergence,
                is_converging=True
            )
        )

    def detect_wedge(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect wedge patterns
        Rising Wedge: Slope_highs > 0 AND Slope_lows > 0 AND Convergence ↑ → Bearish
        Falling Wedge: Slope_highs < 0 AND Slope_lows < 0 AND Convergence ↑ → Bullish
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return None

        slope_highs, _, r2_h = self.geometry.calculate_slope(swing_highs[-4:])
        slope_lows, _, r2_l = self.geometry.calculate_slope(swing_lows[-4:])

        norm_slope_h = self.geometry.normalize_slope(slope_highs, atr)
        norm_slope_l = self.geometry.normalize_slope(slope_lows, atr)

        # Check for convergence (wedges converge)
        convergence = self.geometry.calculate_convergence(slope_highs, slope_lows)

        pattern_type = None
        direction = PatternDirection.NEUTRAL

        # Rising Wedge: Both slopes positive, converging → Bearish
        if norm_slope_h > 0.0001 and norm_slope_l > 0.0001:
            # Check convergence: highs slope less steep than lows
            if slope_highs < slope_lows:
                pattern_type = GeometricPatternType.RISING_WEDGE
                direction = PatternDirection.BEARISH

        # Falling Wedge: Both slopes negative, converging → Bullish
        elif norm_slope_h < -0.0001 and norm_slope_l < -0.0001:
            # Check convergence: lows slope less steep than highs
            if abs(slope_lows) < abs(slope_highs):
                pattern_type = GeometricPatternType.FALLING_WEDGE
                direction = PatternDirection.BULLISH

        if pattern_type is None:
            return None

        geometry_score = min(1.0, (r2_h + r2_l) / 2)

        return DetectedPattern(
            pattern_type=pattern_type,
            direction=direction,
            geometry_score=geometry_score,
            start_idx=min(swing_highs[0].index, swing_lows[0].index),
            end_idx=max(swing_highs[-1].index, swing_lows[-1].index),
            resistance=max(sh.price for sh in swing_highs[-4:]),
            support=min(sl.price for sl in swing_lows[-4:]),
            features=GeometricFeatures(
                slope_highs=slope_highs,
                slope_lows=slope_lows,
                norm_slope_highs=norm_slope_h,
                norm_slope_lows=norm_slope_l,
                convergence_rate=convergence,
                is_converging=True
            )
        )

    def detect_channel(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect channel patterns
        Channel: |Slope_highs - Slope_lows| < ε (parallel)
        + std(Channel_Width) < threshold (width stability)
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return None

        slope_highs, int_h, r2_h = self.geometry.calculate_slope(swing_highs[-4:])
        slope_lows, int_l, r2_l = self.geometry.calculate_slope(swing_lows[-4:])

        # Check parallelism
        parallelism = self.geometry.check_channel_parallelism(slope_highs, slope_lows)

        if parallelism < 0.7:  # Not parallel enough
            return None

        # Calculate channel width stability
        widths = []
        for sh in swing_highs[-4:]:
            # Estimate low at same index
            estimated_low = slope_lows * sh.index + int_l
            widths.append(sh.price - estimated_low)

        width_stability = 1 - (np.std(widths) / np.mean(widths)) if widths else 0

        if width_stability < 0.6:  # Width too variable
            return None

        # Determine channel type
        norm_slope = self.geometry.normalize_slope((slope_highs + slope_lows) / 2, atr)

        if abs(norm_slope) < 0.0002:
            pattern_type = GeometricPatternType.HORIZONTAL_RANGE
            direction = PatternDirection.NEUTRAL
        elif norm_slope > 0:
            pattern_type = GeometricPatternType.RISING_CHANNEL
            direction = PatternDirection.BULLISH
        else:
            pattern_type = GeometricPatternType.FALLING_CHANNEL
            direction = PatternDirection.BEARISH

        geometry_score = (parallelism + width_stability + (r2_h + r2_l) / 2) / 3

        return DetectedPattern(
            pattern_type=pattern_type,
            direction=direction,
            geometry_score=geometry_score,
            start_idx=min(swing_highs[0].index, swing_lows[0].index),
            end_idx=max(swing_highs[-1].index, swing_lows[-1].index),
            resistance=max(sh.price for sh in swing_highs[-4:]),
            support=min(sl.price for sl in swing_lows[-4:]),
            features=GeometricFeatures(
                slope_highs=slope_highs,
                slope_lows=slope_lows,
                channel_parallelism=parallelism,
                channel_width_stability=width_stability
            )
        )

    def detect_flag(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect flag patterns
        Flag: Strong impulse → Parallel pullback → Volume contraction
        Retrace: 30-50%
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Need to identify impulse vs pullback
        # Look for strong move followed by consolidation

        closes = df['close'].values
        if len(closes) < 30:
            return None

        # Find impulse (last 20-30 bars before consolidation)
        impulse_start_idx = max(0, len(closes) - 30)
        impulse_end_idx = len(closes) - 10

        # Calculate impulse
        impulse_start = closes[impulse_start_idx]
        impulse_end = closes[impulse_end_idx]
        impulse = impulse_end - impulse_start

        if abs(impulse) < atr * 2:  # Need significant impulse
            return None

        # Calculate pullback
        pullback_end = closes[-1]
        retracement = self.geometry.calculate_retracement(
            impulse_start, impulse_end, pullback_end
        )

        # Flag retracement: 30-50%
        if not (0.30 <= retracement <= 0.50):
            return None

        # Check for parallel pullback (channel-like)
        recent_highs = [sh for sh in swing_highs if sh.index >= impulse_end_idx]
        recent_lows = [sl for sl in swing_lows if sl.index >= impulse_end_idx]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        slope_h, _, _ = self.geometry.calculate_slope(recent_highs)
        slope_l, _, _ = self.geometry.calculate_slope(recent_lows)

        parallelism = self.geometry.check_channel_parallelism(slope_h, slope_l)

        if parallelism < 0.6:
            return None

        # Determine direction
        if impulse > 0:
            pattern_type = GeometricPatternType.BULL_FLAG
            direction = PatternDirection.BULLISH
        else:
            pattern_type = GeometricPatternType.BEAR_FLAG
            direction = PatternDirection.BEARISH

        geometry_score = parallelism * (1 - abs(retracement - 0.40) * 2)

        return DetectedPattern(
            pattern_type=pattern_type,
            direction=direction,
            geometry_score=geometry_score,
            start_idx=impulse_start_idx,
            end_idx=len(closes) - 1,
            target=impulse_end + impulse,  # Measured move target
            features=GeometricFeatures(
                retracement_pct=retracement,
                channel_parallelism=parallelism
            )
        )

    def detect_double_top_bottom(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect double/triple tops and bottoms
        Equal Level: |Top1 - Top2| < 0.15 * ATR
        """
        patterns = []

        # Check for double/triple top
        if len(swing_highs) >= 2:
            recent_highs = swing_highs[-3:]

            if len(recent_highs) >= 2:
                # Check if last 2 highs are equal
                if self.geometry.check_equal_levels(
                    recent_highs[-1].price,
                    recent_highs[-2].price,
                    atr
                ):
                    # Double top
                    if len(recent_highs) >= 3 and self.geometry.check_equal_levels(
                        recent_highs[-2].price,
                        recent_highs[-3].price,
                        atr
                    ):
                        pattern_type = GeometricPatternType.TRIPLE_TOP
                    else:
                        pattern_type = GeometricPatternType.DOUBLE_TOP

                    resistance = max(sh.price for sh in recent_highs[-2:])

                    patterns.append(DetectedPattern(
                        pattern_type=pattern_type,
                        direction=PatternDirection.BEARISH,
                        geometry_score=0.8,
                        resistance=resistance,
                        start_idx=recent_highs[-2].index,
                        end_idx=recent_highs[-1].index
                    ))

        # Check for double/triple bottom
        if len(swing_lows) >= 2:
            recent_lows = swing_lows[-3:]

            if len(recent_lows) >= 2:
                if self.geometry.check_equal_levels(
                    recent_lows[-1].price,
                    recent_lows[-2].price,
                    atr
                ):
                    if len(recent_lows) >= 3 and self.geometry.check_equal_levels(
                        recent_lows[-2].price,
                        recent_lows[-3].price,
                        atr
                    ):
                        pattern_type = GeometricPatternType.TRIPLE_BOTTOM
                    else:
                        pattern_type = GeometricPatternType.DOUBLE_BOTTOM

                    support = min(sl.price for sl in recent_lows[-2:])

                    patterns.append(DetectedPattern(
                        pattern_type=pattern_type,
                        direction=PatternDirection.BULLISH,
                        geometry_score=0.8,
                        support=support,
                        start_idx=recent_lows[-2].index,
                        end_idx=recent_lows[-1].index
                    ))

        return patterns[0] if patterns else None

    def detect_head_shoulders(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect Head & Shoulders patterns
        Head > Shoulder1 AND Head > Shoulder2
        |Shoulder1 - Shoulder2| < ε
        Neckline_Slope < small_threshold
        """
        # Regular H&S (tops)
        if len(swing_highs) >= 3:
            recent = swing_highs[-3:]
            left_shoulder = recent[0].price
            head = recent[1].price
            right_shoulder = recent[2].price

            # Check H&S conditions
            if (head > left_shoulder and
                head > right_shoulder and
                self.geometry.check_equal_levels(left_shoulder, right_shoulder, atr)):

                # Calculate neckline
                neckline_points = [sl for sl in swing_lows
                                  if recent[0].index <= sl.index <= recent[2].index]

                if len(neckline_points) >= 2:
                    neckline_slope, _, _ = self.geometry.calculate_slope(neckline_points)
                    norm_neckline = self.geometry.normalize_slope(neckline_slope, atr)

                    # Neckline should be relatively flat
                    if abs(norm_neckline) < 0.0005:
                        return DetectedPattern(
                            pattern_type=GeometricPatternType.HEAD_SHOULDERS,
                            direction=PatternDirection.BEARISH,
                            geometry_score=0.85,
                            resistance=head,
                            support=min(np.price for np in neckline_points),
                            start_idx=recent[0].index,
                            end_idx=recent[2].index
                        )

        # Inverse H&S (bottoms)
        if len(swing_lows) >= 3:
            recent = swing_lows[-3:]
            left_shoulder = recent[0].price
            head = recent[1].price
            right_shoulder = recent[2].price

            if (head < left_shoulder and
                head < right_shoulder and
                self.geometry.check_equal_levels(left_shoulder, right_shoulder, atr)):

                neckline_points = [sh for sh in swing_highs
                                  if recent[0].index <= sh.index <= recent[2].index]

                if len(neckline_points) >= 2:
                    neckline_slope, _, _ = self.geometry.calculate_slope(neckline_points)
                    norm_neckline = self.geometry.normalize_slope(neckline_slope, atr)

                    if abs(norm_neckline) < 0.0005:
                        return DetectedPattern(
                            pattern_type=GeometricPatternType.INVERSE_HEAD_SHOULDERS,
                            direction=PatternDirection.BULLISH,
                            geometry_score=0.85,
                            support=head,
                            resistance=max(np.price for np in neckline_points),
                            start_idx=recent[0].index,
                            end_idx=recent[2].index
                        )

        return None

    def detect_compression(
        self,
        df: pd.DataFrame,
        atr: float
    ) -> Optional[DetectedPattern]:
        """
        Detect compression/coil patterns
        Compression < 1.2 → Coil / Box
        """
        if len(df) < 10:
            return None

        recent = df.tail(10)
        range_width = recent['high'].max() - recent['low'].min()
        compression = self.geometry.calculate_compression_ratio(range_width, atr)

        if compression < 1.2:
            return DetectedPattern(
                pattern_type=GeometricPatternType.COIL_BOX,
                direction=PatternDirection.NEUTRAL,  # Could break either way
                geometry_score=1 - compression / 1.2,  # Lower compression = higher score
                resistance=recent['high'].max(),
                support=recent['low'].min(),
                start_idx=len(df) - 10,
                end_idx=len(df) - 1,
                features=GeometricFeatures(
                    range_width=range_width,
                    compression_ratio=compression
                )
            )

        return None


# ═══════════════════════════════════════════════════════════════════════
# PATTERN VALIDATOR (STEP 3)
# ═══════════════════════════════════════════════════════════════════════

class PatternValidator:
    """
    Validate patterns with volume, OI, and delta confirmation
    Reject patterns missing confirmation
    """

    def validate(
        self,
        pattern: DetectedPattern,
        df: pd.DataFrame,
        option_data: Optional[Dict] = None
    ) -> DetectedPattern:
        """
        Validate pattern with additional confirmations

        Reject if:
        - Volume confirmation missing
        - ATR expansion already happened (late pattern)
        """
        # Volume confirmation
        if 'volume' in df.columns:
            recent_vol = df['volume'].tail(5).mean()
            prev_vol = df['volume'].tail(20).head(15).mean()

            if pattern.direction == PatternDirection.BULLISH:
                # Volume should increase on up moves
                up_vol = df[df['close'] > df['open']]['volume'].tail(5).mean()
                down_vol = df[df['close'] <= df['open']]['volume'].tail(5).mean()
                pattern.volume_confirmed = up_vol > down_vol
            elif pattern.direction == PatternDirection.BEARISH:
                up_vol = df[df['close'] > df['open']]['volume'].tail(5).mean()
                down_vol = df[df['close'] <= df['open']]['volume'].tail(5).mean()
                pattern.volume_confirmed = down_vol > up_vol
            else:
                # Neutral patterns: look for volume contraction
                pattern.volume_confirmed = recent_vol < prev_vol * 0.8

        # OI confirmation (if option data available)
        if option_data:
            # This would integrate with your option chain data
            # For now, mark as requiring implementation
            pattern.oi_confirmed = self._check_oi_confirmation(pattern, option_data)

        # Delta confirmation
        if option_data:
            pattern.delta_confirmed = self._check_delta_confirmation(pattern, option_data)

        # Calculate fake move risk
        pattern.fake_move_risk = self._calculate_fake_move_risk(pattern)

        # Update expected outcomes based on validation
        pattern.expected_outcomes = self._calculate_expected_outcomes(pattern)

        return pattern

    def _check_oi_confirmation(
        self,
        pattern: DetectedPattern,
        option_data: Dict
    ) -> bool:
        """Check if OI confirms the pattern"""
        # OI logic from user's bible:
        # Accumulation: OI ↑ + Price flat
        # Distribution: OI ↑ + Price at top
        # Breakout: OI spike = fake, OI stable = real

        if 'total_ce_oi' in option_data and 'total_pe_oi' in option_data:
            ce_oi = option_data.get('total_ce_oi', 0)
            pe_oi = option_data.get('total_pe_oi', 0)

            if pattern.direction == PatternDirection.BULLISH:
                # PE OI should be higher (puts being written = bullish)
                return pe_oi > ce_oi
            elif pattern.direction == PatternDirection.BEARISH:
                return ce_oi > pe_oi

        return False

    def _check_delta_confirmation(
        self,
        pattern: DetectedPattern,
        option_data: Dict
    ) -> bool:
        """Check if delta flow confirms pattern"""
        # Delta alignment from user's bible:
        # Breakout + Delta aligns = valid
        # Breakout + Delta diverges = fake

        if 'delta' in option_data:
            delta = option_data['delta']
            if pattern.direction == PatternDirection.BULLISH:
                return delta > 0
            elif pattern.direction == PatternDirection.BEARISH:
                return delta < 0

        return False

    def _calculate_fake_move_risk(self, pattern: DetectedPattern) -> float:
        """
        Calculate risk of fake breakout

        Fake Break Detection (from bible):
        Geometry Break + Volume weak + OI spikes + Delta diverges = SL Hunt likely
        """
        risk = 0.3  # Base risk

        if not pattern.volume_confirmed:
            risk += 0.2
        if not pattern.oi_confirmed:
            risk += 0.15
        if not pattern.delta_confirmed:
            risk += 0.15

        # Lower geometry score = higher risk
        risk += (1 - pattern.geometry_score) * 0.2

        return min(1.0, risk)

    def _calculate_expected_outcomes(
        self,
        pattern: DetectedPattern
    ) -> Dict[str, float]:
        """Calculate expected outcomes based on pattern and validation"""

        base_outcomes = {
            "expansion": 0.0,
            "fake": 0.0,
            "chop": 0.0
        }

        # Start with geometry score
        geometry_factor = pattern.geometry_score

        # Adjust based on confirmation
        confirmation_count = sum([
            pattern.volume_confirmed,
            pattern.oi_confirmed,
            pattern.delta_confirmed
        ])

        if confirmation_count == 3:
            base_outcomes["expansion"] = geometry_factor * 0.9
            base_outcomes["fake"] = 0.05
            base_outcomes["chop"] = 1 - base_outcomes["expansion"] - base_outcomes["fake"]
        elif confirmation_count == 2:
            base_outcomes["expansion"] = geometry_factor * 0.7
            base_outcomes["fake"] = 0.15
            base_outcomes["chop"] = 1 - base_outcomes["expansion"] - base_outcomes["fake"]
        elif confirmation_count == 1:
            base_outcomes["expansion"] = geometry_factor * 0.5
            base_outcomes["fake"] = 0.25
            base_outcomes["chop"] = 1 - base_outcomes["expansion"] - base_outcomes["fake"]
        else:
            base_outcomes["expansion"] = geometry_factor * 0.3
            base_outcomes["fake"] = pattern.fake_move_risk
            base_outcomes["chop"] = 1 - base_outcomes["expansion"] - base_outcomes["fake"]

        return base_outcomes


# ═══════════════════════════════════════════════════════════════════════
# MAIN PATTERN ENGINE
# ═══════════════════════════════════════════════════════════════════════

class GeometricPatternEngine:
    """
    Main pattern engine that orchestrates detection

    Step 1: Swing Point Detection (MANDATORY)
    Step 2: Build Structures from Swings
    Step 3: Validation Filters
    Step 4: Context Tagging
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_detector = SwingPointDetector(lookback=swing_lookback)
        self.pattern_detector = GeometricPatternDetector()
        self.validator = PatternValidator()
        self.geometry = GeometryCalculator()

    def analyze(
        self,
        df: pd.DataFrame,
        option_data: Optional[Dict] = None,
        regime: str = "NEUTRAL",
        volatility_state: str = "NORMAL",
        htf_bias: str = "NEUTRAL"
    ) -> PatternAnalysisResult:
        """
        Complete pattern analysis pipeline

        Returns all detected patterns with validation and context
        """
        # Calculate ATR first
        atr = self.geometry.calculate_atr(df)

        # STEP 1: Detect swing points
        swing_highs, swing_lows = self.swing_detector.detect_swings(df)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return PatternAnalysisResult(
                patterns=[],
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                geometric_features=GeometricFeatures()
            )

        # Extract geometric features
        slope_h, _, _ = self.geometry.calculate_slope(swing_highs[-5:])
        slope_l, _, _ = self.geometry.calculate_slope(swing_lows[-5:])

        geometric_features = GeometricFeatures(
            slope_highs=slope_h,
            slope_lows=slope_l,
            norm_slope_highs=self.geometry.normalize_slope(slope_h, atr),
            norm_slope_lows=self.geometry.normalize_slope(slope_l, atr),
            range_width=df['high'].tail(20).max() - df['low'].tail(20).min(),
            compression_ratio=self.geometry.calculate_compression_ratio(
                df['high'].tail(10).max() - df['low'].tail(10).min(), atr
            ),
            convergence_rate=self.geometry.calculate_convergence(slope_h, slope_l),
            swing_high_count=len(swing_highs),
            swing_low_count=len(swing_lows)
        )

        # STEP 2: Detect patterns
        patterns = []

        # Try each pattern detector
        detectors = [
            lambda: self.pattern_detector.detect_triangle(swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_wedge(swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_channel(swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_flag(df, swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_double_top_bottom(swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_head_shoulders(swing_highs, swing_lows, atr),
            lambda: self.pattern_detector.detect_compression(df, atr),
        ]

        for detector in detectors:
            try:
                pattern = detector()
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                continue

        # STEP 3: Validate patterns
        validated_patterns = []
        for pattern in patterns:
            validated = self.validator.validate(pattern, df, option_data)

            # Apply rejection filters
            # Reject if ATR expansion already happened (late pattern)
            recent_atr = self.geometry.calculate_atr(df.tail(5), period=5)
            if recent_atr > atr * 1.5:
                validated.geometry_score *= 0.5  # Penalize late patterns

            validated_patterns.append(validated)

        # STEP 4: Context tagging (already passed as parameters)
        for pattern in validated_patterns:
            # Tag with context - pattern without context = noise
            pattern.expected_outcomes['regime'] = regime
            pattern.expected_outcomes['volatility'] = volatility_state
            pattern.expected_outcomes['htf_bias'] = htf_bias

        # Sort by geometry score
        validated_patterns.sort(key=lambda p: p.geometry_score, reverse=True)

        # Determine dominant pattern
        dominant = validated_patterns[0] if validated_patterns else None

        # Determine overall market structure
        if dominant:
            if dominant.geometry_score > 0.7:
                market_structure = dominant.pattern_type.value
            else:
                market_structure = "FORMING"
        else:
            market_structure = "NEUTRAL"

        return PatternAnalysisResult(
            patterns=validated_patterns,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            geometric_features=geometric_features,
            dominant_pattern=dominant,
            market_structure=market_structure
        )

    def get_pattern_feature_dict(
        self,
        pattern: DetectedPattern
    ) -> Dict[str, Any]:
        """
        Convert pattern to feature dictionary for ML/display

        This is the format that feeds:
        ➡️ Pattern Memory
        ➡️ Probability Engine
        ➡️ Unified ML Signal
        """
        return {
            "pattern_type": pattern.pattern_type.value,
            "direction": pattern.direction.value,
            "geometry_score": pattern.geometry_score,
            "volume_confirmed": pattern.volume_confirmed,
            "oi_confirmed": pattern.oi_confirmed,
            "delta_confirmed": pattern.delta_confirmed,
            "fake_move_risk": pattern.fake_move_risk,
            "expected_outcomes": pattern.expected_outcomes,
            "resistance": pattern.resistance,
            "support": pattern.support,
            "target": pattern.target
        }


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def analyze_geometric_patterns(
    df: pd.DataFrame,
    option_data: Optional[Dict] = None,
    **context
) -> PatternAnalysisResult:
    """
    Quick analysis function for integration

    Usage:
        result = analyze_geometric_patterns(ohlc_df, option_data)
        for pattern in result.patterns:
            print(pattern.pattern_type, pattern.geometry_score)
    """
    engine = GeometricPatternEngine()
    return engine.analyze(df, option_data, **context)
