"""
EXPIRY DAY KILLER - Advanced False Breakout Filter

Prevents false breakout signals on expiry days by applying multiple filters:
1. Time-based filter - Block signals in last 60-90 mins on expiry
2. Volume confirmation - Breakout needs 1.5x avg volume
3. Retest validation - Wait for retest before entry
4. OI Wall alignment - Don't breakout INTO OI wall
5. Max Pain pull check - Breakout should align with max pain direction

These filters are especially critical on expiry days when:
- Gamma effects cause whipsaw moves
- Max Pain pull causes reversals
- OI walls act as magnets
- Unwinding chaos in last 90 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterResult(Enum):
    """Filter result status"""
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    WARNING = "WARNING"


@dataclass
class ExpiryKillerResult:
    """Result of Expiry Day Killer analysis"""
    # Overall result
    entry_allowed: bool
    overall_score: float  # 0-100 (higher = safer to enter)
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"

    # Individual filter results
    time_filter: FilterResult
    volume_filter: FilterResult
    retest_filter: FilterResult
    oi_wall_filter: FilterResult
    max_pain_filter: FilterResult

    # Reasons
    block_reasons: List[str]
    warning_reasons: List[str]
    passed_reasons: List[str]

    # Recommendation
    recommendation: str

    # Context
    is_expiry_day: bool
    minutes_to_close: int
    volume_ratio: float
    retest_confirmed: bool
    oi_wall_aligned: bool
    max_pain_aligned: bool


class ExpiryDayKiller:
    """
    Advanced filter system to prevent false breakout entries on expiry days

    Usage:
        killer = ExpiryDayKiller()
        result = killer.analyze(
            spot_price=24100,
            entry_type="SUPPORT",  # or "RESISTANCE"
            entry_zone={'entry_from': 24080, 'entry_to': 24100},
            df=price_df,
            oi_data={'max_call_strike': 24200, 'max_put_strike': 24000, 'max_pain': 24150}
        )

        if result.entry_allowed:
            # Safe to enter
        else:
            # Block entry, show result.block_reasons
    """

    def __init__(
        self,
        # Time filter settings
        expiry_cutoff_minutes: int = 90,  # Block signals in last N minutes
        warning_minutes: int = 120,  # Warning zone before cutoff

        # Volume filter settings
        volume_confirmation_ratio: float = 1.5,  # Require 1.5x avg volume
        volume_lookback_periods: int = 20,  # Periods to calculate avg volume

        # Retest filter settings
        retest_tolerance_points: float = 10,  # Points tolerance for retest
        retest_lookback_candles: int = 5,  # Look back N candles for retest

        # OI Wall filter settings
        oi_wall_distance_threshold: float = 50,  # Don't break INTO OI wall within N points

        # Max Pain filter settings
        max_pain_alignment_threshold: float = 30,  # Max pain should be within N points of direction
    ):
        self.expiry_cutoff_minutes = expiry_cutoff_minutes
        self.warning_minutes = warning_minutes
        self.volume_confirmation_ratio = volume_confirmation_ratio
        self.volume_lookback_periods = volume_lookback_periods
        self.retest_tolerance_points = retest_tolerance_points
        self.retest_lookback_candles = retest_lookback_candles
        self.oi_wall_distance_threshold = oi_wall_distance_threshold
        self.max_pain_alignment_threshold = max_pain_alignment_threshold

        # Market close time (IST)
        self.market_close = time(15, 30)

    def analyze(
        self,
        spot_price: float,
        entry_type: str,  # "SUPPORT" or "RESISTANCE"
        entry_zone: Dict,
        df: Optional[pd.DataFrame] = None,
        oi_data: Optional[Dict] = None,
        days_to_expiry: float = 7.0,
        current_time: Optional[datetime] = None
    ) -> ExpiryKillerResult:
        """
        Analyze if entry should be allowed based on all filters

        Args:
            spot_price: Current spot price
            entry_type: "SUPPORT" (bullish entry) or "RESISTANCE" (bearish entry)
            entry_zone: Dict with 'entry_from', 'entry_to' keys
            df: DataFrame with OHLCV data for volume/retest analysis
            oi_data: Dict with 'max_call_strike', 'max_put_strike', 'max_pain'
            days_to_expiry: Days until expiry (0 = expiry day)
            current_time: Current time (default: now)

        Returns:
            ExpiryKillerResult with all filter results
        """
        if current_time is None:
            current_time = datetime.now()

        block_reasons = []
        warning_reasons = []
        passed_reasons = []

        # Determine if it's expiry day
        is_expiry_day = days_to_expiry <= 1

        # Calculate minutes to market close
        now_time = current_time.time()
        close_datetime = datetime.combine(current_time.date(), self.market_close)
        now_datetime = datetime.combine(current_time.date(), now_time)
        minutes_to_close = int((close_datetime - now_datetime).total_seconds() / 60)
        if minutes_to_close < 0:
            minutes_to_close = 0

        # ═══════════════════════════════════════════════════════════════════
        # FILTER 1: TIME-BASED FILTER
        # Block signals in last 60-90 mins on expiry day
        # ═══════════════════════════════════════════════════════════════════
        time_filter = self._check_time_filter(
            is_expiry_day, minutes_to_close,
            block_reasons, warning_reasons, passed_reasons
        )

        # ═══════════════════════════════════════════════════════════════════
        # FILTER 2: VOLUME CONFIRMATION
        # Breakout needs 1.5x avg volume
        # ═══════════════════════════════════════════════════════════════════
        volume_filter, volume_ratio = self._check_volume_filter(
            df, block_reasons, warning_reasons, passed_reasons
        )

        # ═══════════════════════════════════════════════════════════════════
        # FILTER 3: RETEST VALIDATION
        # Wait for price to retest the broken level
        # ═══════════════════════════════════════════════════════════════════
        retest_filter, retest_confirmed = self._check_retest_filter(
            df, spot_price, entry_type, entry_zone,
            block_reasons, warning_reasons, passed_reasons
        )

        # ═══════════════════════════════════════════════════════════════════
        # FILTER 4: OI WALL ALIGNMENT
        # Don't breakout INTO massive OI wall
        # ═══════════════════════════════════════════════════════════════════
        oi_wall_filter, oi_wall_aligned = self._check_oi_wall_filter(
            spot_price, entry_type, oi_data,
            block_reasons, warning_reasons, passed_reasons
        )

        # ═══════════════════════════════════════════════════════════════════
        # FILTER 5: MAX PAIN ALIGNMENT
        # Breakout should align with max pain direction on expiry
        # ═══════════════════════════════════════════════════════════════════
        max_pain_filter, max_pain_aligned = self._check_max_pain_filter(
            spot_price, entry_type, oi_data, is_expiry_day,
            block_reasons, warning_reasons, passed_reasons
        )

        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE OVERALL SCORE
        # ═══════════════════════════════════════════════════════════════════
        overall_score = self._calculate_overall_score(
            time_filter, volume_filter, retest_filter,
            oi_wall_filter, max_pain_filter,
            is_expiry_day, volume_ratio
        )

        # Determine if entry is allowed
        entry_allowed = (
            time_filter != FilterResult.BLOCKED and
            volume_filter != FilterResult.BLOCKED and
            oi_wall_filter != FilterResult.BLOCKED and
            overall_score >= 50
        )

        # On expiry day, be more strict
        if is_expiry_day and overall_score < 70:
            entry_allowed = False
            if overall_score >= 50:
                block_reasons.append(f"Expiry day requires score >= 70 (got {overall_score:.0f})")

        # Determine risk level
        risk_level = self._determine_risk_level(overall_score, is_expiry_day)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            entry_allowed, risk_level, entry_type,
            block_reasons, warning_reasons, is_expiry_day
        )

        return ExpiryKillerResult(
            entry_allowed=entry_allowed,
            overall_score=overall_score,
            risk_level=risk_level,
            time_filter=time_filter,
            volume_filter=volume_filter,
            retest_filter=retest_filter,
            oi_wall_filter=oi_wall_filter,
            max_pain_filter=max_pain_filter,
            block_reasons=block_reasons,
            warning_reasons=warning_reasons,
            passed_reasons=passed_reasons,
            recommendation=recommendation,
            is_expiry_day=is_expiry_day,
            minutes_to_close=minutes_to_close,
            volume_ratio=volume_ratio,
            retest_confirmed=retest_confirmed,
            oi_wall_aligned=oi_wall_aligned,
            max_pain_aligned=max_pain_aligned
        )

    def _check_time_filter(
        self,
        is_expiry_day: bool,
        minutes_to_close: int,
        block_reasons: List[str],
        warning_reasons: List[str],
        passed_reasons: List[str]
    ) -> FilterResult:
        """Check time-based filter for expiry day"""

        if not is_expiry_day:
            passed_reasons.append("Not expiry day - time filter passed")
            return FilterResult.PASSED

        if minutes_to_close <= self.expiry_cutoff_minutes:
            block_reasons.append(
                f"EXPIRY DAY: Only {minutes_to_close} mins to close (cutoff: {self.expiry_cutoff_minutes})"
            )
            return FilterResult.BLOCKED

        if minutes_to_close <= self.warning_minutes:
            warning_reasons.append(
                f"EXPIRY DAY: {minutes_to_close} mins to close - entering danger zone"
            )
            return FilterResult.WARNING

        passed_reasons.append(f"Expiry day but {minutes_to_close} mins remaining - OK")
        return FilterResult.PASSED

    def _check_volume_filter(
        self,
        df: Optional[pd.DataFrame],
        block_reasons: List[str],
        warning_reasons: List[str],
        passed_reasons: List[str]
    ) -> Tuple[FilterResult, float]:
        """Check volume confirmation filter"""

        if df is None or len(df) < self.volume_lookback_periods:
            warning_reasons.append("Volume data unavailable - skipping filter")
            return FilterResult.WARNING, 1.0

        try:
            # Get volume column
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
            if vol_col not in df.columns:
                warning_reasons.append("Volume column not found")
                return FilterResult.WARNING, 1.0

            # Calculate average volume
            avg_volume = df[vol_col].iloc[-self.volume_lookback_periods:-1].mean()
            current_volume = df[vol_col].iloc[-1]

            if avg_volume <= 0:
                warning_reasons.append("Invalid average volume")
                return FilterResult.WARNING, 1.0

            volume_ratio = current_volume / avg_volume

            if volume_ratio >= self.volume_confirmation_ratio:
                passed_reasons.append(
                    f"Volume confirmed: {volume_ratio:.2f}x avg (required: {self.volume_confirmation_ratio}x)"
                )
                return FilterResult.PASSED, volume_ratio

            if volume_ratio >= 1.0:
                warning_reasons.append(
                    f"Volume weak: {volume_ratio:.2f}x avg (need {self.volume_confirmation_ratio}x for confirmation)"
                )
                return FilterResult.WARNING, volume_ratio

            block_reasons.append(
                f"Volume too low: {volume_ratio:.2f}x avg - likely false breakout"
            )
            return FilterResult.BLOCKED, volume_ratio

        except Exception as e:
            logger.debug(f"Volume filter error: {e}")
            warning_reasons.append(f"Volume check error: {str(e)}")
            return FilterResult.WARNING, 1.0

    def _check_retest_filter(
        self,
        df: Optional[pd.DataFrame],
        spot_price: float,
        entry_type: str,
        entry_zone: Dict,
        block_reasons: List[str],
        warning_reasons: List[str],
        passed_reasons: List[str]
    ) -> Tuple[FilterResult, bool]:
        """Check if price has retested the breakout level"""

        if df is None or len(df) < self.retest_lookback_candles + 2:
            warning_reasons.append("Insufficient data for retest check")
            return FilterResult.WARNING, False

        try:
            # Get price columns
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            close_col = 'Close' if 'Close' in df.columns else 'close'

            entry_from = entry_zone.get('entry_from', spot_price - 20)
            entry_to = entry_zone.get('entry_to', spot_price + 20)

            retest_confirmed = False

            if entry_type == "SUPPORT":
                # For support entry: Price should have bounced from the zone
                # Check if any recent candle's low touched the zone
                recent_lows = df[low_col].iloc[-self.retest_lookback_candles:].values
                for low in recent_lows:
                    if entry_from - self.retest_tolerance_points <= low <= entry_to + self.retest_tolerance_points:
                        # Low touched the zone - check if it bounced
                        if spot_price > low + self.retest_tolerance_points:
                            retest_confirmed = True
                            break

            else:  # RESISTANCE
                # For resistance entry: Price should have been rejected from the zone
                recent_highs = df[high_col].iloc[-self.retest_lookback_candles:].values
                for high in recent_highs:
                    if entry_from - self.retest_tolerance_points <= high <= entry_to + self.retest_tolerance_points:
                        # High touched the zone - check if it rejected
                        if spot_price < high - self.retest_tolerance_points:
                            retest_confirmed = True
                            break

            if retest_confirmed:
                passed_reasons.append(f"Retest confirmed in last {self.retest_lookback_candles} candles")
                return FilterResult.PASSED, True
            else:
                warning_reasons.append("No retest detected - waiting for confirmation")
                return FilterResult.WARNING, False

        except Exception as e:
            logger.debug(f"Retest filter error: {e}")
            warning_reasons.append(f"Retest check error: {str(e)}")
            return FilterResult.WARNING, False

    def _check_oi_wall_filter(
        self,
        spot_price: float,
        entry_type: str,
        oi_data: Optional[Dict],
        block_reasons: List[str],
        warning_reasons: List[str],
        passed_reasons: List[str]
    ) -> Tuple[FilterResult, bool]:
        """Check if breakout is INTO an OI wall (bad) or AWAY from it (good)"""

        if not oi_data:
            warning_reasons.append("OI data unavailable - skipping OI wall check")
            return FilterResult.WARNING, True

        try:
            max_call_strike = oi_data.get('max_call_strike', 0)
            max_put_strike = oi_data.get('max_put_strike', 0)

            if not max_call_strike or not max_put_strike:
                warning_reasons.append("OI wall strikes not available")
                return FilterResult.WARNING, True

            if entry_type == "SUPPORT":
                # Bullish entry - check if moving INTO CALL OI wall (resistance)
                distance_to_call_wall = max_call_strike - spot_price

                if distance_to_call_wall < 0:
                    # Price is already above CALL wall - very bullish
                    passed_reasons.append(f"Price above CALL wall ({max_call_strike}) - breakout confirmed")
                    return FilterResult.PASSED, True

                if distance_to_call_wall <= self.oi_wall_distance_threshold:
                    block_reasons.append(
                        f"Breaking INTO CALL OI Wall at {max_call_strike} (only {distance_to_call_wall:.0f} pts away)"
                    )
                    return FilterResult.BLOCKED, False

                passed_reasons.append(f"CALL wall at {max_call_strike} is {distance_to_call_wall:.0f} pts away - OK")
                return FilterResult.PASSED, True

            else:  # RESISTANCE (bearish entry)
                # Bearish entry - check if moving INTO PUT OI wall (support)
                distance_to_put_wall = spot_price - max_put_strike

                if distance_to_put_wall < 0:
                    # Price is already below PUT wall - very bearish
                    passed_reasons.append(f"Price below PUT wall ({max_put_strike}) - breakout confirmed")
                    return FilterResult.PASSED, True

                if distance_to_put_wall <= self.oi_wall_distance_threshold:
                    block_reasons.append(
                        f"Breaking INTO PUT OI Wall at {max_put_strike} (only {distance_to_put_wall:.0f} pts away)"
                    )
                    return FilterResult.BLOCKED, False

                passed_reasons.append(f"PUT wall at {max_put_strike} is {distance_to_put_wall:.0f} pts away - OK")
                return FilterResult.PASSED, True

        except Exception as e:
            logger.debug(f"OI wall filter error: {e}")
            warning_reasons.append(f"OI wall check error: {str(e)}")
            return FilterResult.WARNING, True

    def _check_max_pain_filter(
        self,
        spot_price: float,
        entry_type: str,
        oi_data: Optional[Dict],
        is_expiry_day: bool,
        block_reasons: List[str],
        warning_reasons: List[str],
        passed_reasons: List[str]
    ) -> Tuple[FilterResult, bool]:
        """Check if breakout aligns with max pain direction"""

        if not oi_data or 'max_pain' not in oi_data:
            warning_reasons.append("Max pain data unavailable")
            return FilterResult.WARNING, True

        try:
            max_pain = oi_data.get('max_pain', 0)

            if not max_pain:
                warning_reasons.append("Max pain value is zero or invalid")
                return FilterResult.WARNING, True

            distance_to_max_pain = max_pain - spot_price  # Positive = max pain is above

            if entry_type == "SUPPORT":
                # Bullish entry - max pain should be above or near current price
                if distance_to_max_pain > -self.max_pain_alignment_threshold:
                    # Max pain is above or slightly below - bullish aligned
                    passed_reasons.append(
                        f"Max Pain ({max_pain}) aligns with bullish entry (pull: {distance_to_max_pain:+.0f} pts)"
                    )
                    return FilterResult.PASSED, True
                else:
                    # Max pain is significantly below - bearish pull
                    if is_expiry_day:
                        block_reasons.append(
                            f"EXPIRY: Max Pain ({max_pain}) is {abs(distance_to_max_pain):.0f} pts BELOW - bearish pull"
                        )
                        return FilterResult.BLOCKED, False
                    else:
                        warning_reasons.append(
                            f"Max Pain ({max_pain}) is below spot - potential downside pressure"
                        )
                        return FilterResult.WARNING, False

            else:  # RESISTANCE (bearish entry)
                # Bearish entry - max pain should be below or near current price
                if distance_to_max_pain < self.max_pain_alignment_threshold:
                    # Max pain is below or slightly above - bearish aligned
                    passed_reasons.append(
                        f"Max Pain ({max_pain}) aligns with bearish entry (pull: {distance_to_max_pain:+.0f} pts)"
                    )
                    return FilterResult.PASSED, True
                else:
                    # Max pain is significantly above - bullish pull
                    if is_expiry_day:
                        block_reasons.append(
                            f"EXPIRY: Max Pain ({max_pain}) is {distance_to_max_pain:.0f} pts ABOVE - bullish pull"
                        )
                        return FilterResult.BLOCKED, False
                    else:
                        warning_reasons.append(
                            f"Max Pain ({max_pain}) is above spot - potential upside pressure"
                        )
                        return FilterResult.WARNING, False

        except Exception as e:
            logger.debug(f"Max pain filter error: {e}")
            warning_reasons.append(f"Max pain check error: {str(e)}")
            return FilterResult.WARNING, True

    def _calculate_overall_score(
        self,
        time_filter: FilterResult,
        volume_filter: FilterResult,
        retest_filter: FilterResult,
        oi_wall_filter: FilterResult,
        max_pain_filter: FilterResult,
        is_expiry_day: bool,
        volume_ratio: float
    ) -> float:
        """Calculate overall safety score (0-100)"""

        score = 100.0

        # Time filter scoring
        if time_filter == FilterResult.BLOCKED:
            score -= 40
        elif time_filter == FilterResult.WARNING:
            score -= 15

        # Volume filter scoring (most important for false breakout)
        if volume_filter == FilterResult.BLOCKED:
            score -= 35
        elif volume_filter == FilterResult.WARNING:
            score -= 10
        elif volume_filter == FilterResult.PASSED:
            # Bonus for strong volume
            if volume_ratio >= 2.0:
                score += 10

        # Retest filter scoring
        if retest_filter == FilterResult.BLOCKED:
            score -= 20
        elif retest_filter == FilterResult.WARNING:
            score -= 5
        elif retest_filter == FilterResult.PASSED:
            score += 5  # Bonus for confirmed retest

        # OI Wall filter scoring
        if oi_wall_filter == FilterResult.BLOCKED:
            score -= 30
        elif oi_wall_filter == FilterResult.WARNING:
            score -= 10

        # Max Pain filter scoring
        if max_pain_filter == FilterResult.BLOCKED:
            score -= 25
        elif max_pain_filter == FilterResult.WARNING:
            score -= 10
        elif max_pain_filter == FilterResult.PASSED:
            score += 5

        # Expiry day penalty
        if is_expiry_day:
            score -= 10  # Base penalty for expiry day

        return max(0, min(100, score))

    def _determine_risk_level(self, score: float, is_expiry_day: bool) -> str:
        """Determine risk level based on score"""
        if is_expiry_day:
            if score >= 85:
                return "LOW"
            elif score >= 70:
                return "MEDIUM"
            elif score >= 50:
                return "HIGH"
            else:
                return "EXTREME"
        else:
            if score >= 80:
                return "LOW"
            elif score >= 60:
                return "MEDIUM"
            elif score >= 40:
                return "HIGH"
            else:
                return "EXTREME"

    def _generate_recommendation(
        self,
        entry_allowed: bool,
        risk_level: str,
        entry_type: str,
        block_reasons: List[str],
        warning_reasons: List[str],
        is_expiry_day: bool
    ) -> str:
        """Generate trading recommendation"""

        if not entry_allowed:
            if block_reasons:
                return f"BLOCKED: {block_reasons[0]}"
            return "BLOCKED: Multiple filter failures"

        action = "BUY CALL" if entry_type == "SUPPORT" else "BUY PUT"

        if risk_level == "LOW":
            return f"SAFE ENTRY: {action} - All filters passed"

        if risk_level == "MEDIUM":
            if warning_reasons:
                return f"CAUTION: {action} - {warning_reasons[0]}"
            return f"PROCEED WITH CAUTION: {action}"

        if risk_level == "HIGH":
            if is_expiry_day:
                return f"HIGH RISK (EXPIRY): Consider smaller position for {action}"
            return f"HIGH RISK: Reduce position size for {action}"

        return f"EXTREME RISK: Avoid {action} - Too many concerns"


def format_killer_result(result: ExpiryKillerResult) -> str:
    """Format ExpiryKillerResult as readable string"""

    status_emoji = "ALLOWED" if result.entry_allowed else "BLOCKED"
    risk_emoji = {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
        "EXTREME": "EXTREME"
    }.get(result.risk_level, "")

    report = f"""
EXPIRY DAY KILLER ANALYSIS

ENTRY STATUS: {status_emoji}
SAFETY SCORE: {result.overall_score:.0f}/100
RISK LEVEL: {risk_emoji} {result.risk_level}

FILTER RESULTS:
  Time Filter: {result.time_filter.value}
  Volume Filter: {result.volume_filter.value} ({result.volume_ratio:.2f}x avg)
  Retest Filter: {result.retest_filter.value} ({'Confirmed' if result.retest_confirmed else 'Pending'})
  OI Wall Filter: {result.oi_wall_filter.value}
  Max Pain Filter: {result.max_pain_filter.value}

CONTEXT:
  Expiry Day: {'YES' if result.is_expiry_day else 'No'}
  Minutes to Close: {result.minutes_to_close}

"""

    if result.block_reasons:
        report += "BLOCK REASONS:\n"
        for reason in result.block_reasons:
            report += f"  {reason}\n"

    if result.warning_reasons:
        report += "\nWARNINGS:\n"
        for reason in result.warning_reasons:
            report += f"  {reason}\n"

    if result.passed_reasons:
        report += "\nPASSED CHECKS:\n"
        for reason in result.passed_reasons:
            report += f"  {reason}\n"

    report += f"\nRECOMMENDATION: {result.recommendation}"

    return report
