"""
Stop-Loss Hunt Detector - Institutional Trap Detection System
=============================================================

Detects high-probability stop-loss hunting BEFORE it happens using 4 layers:
1. Option Chain (Seller traps - OI absorption)
2. Market Depth (Fake intent / spoofing)
3. Volume + Price (Effort vs Result)
4. Chart Structure (SL cluster zones)

Key Insight: Institutions hunt SLs to ENTER positions without slippage
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import pandas as pd
import numpy as np


@dataclass
class TrapZone:
    """Represents a potential stop-loss trap zone"""
    price: float
    zone_type: str  # "CALL_SL_ZONE", "PUT_SL_ZONE", "ROUND_NUMBER", "VWAP_ZONE", "TRENDLINE"
    sl_density: float  # 0-100, how many SLs likely here
    oi_at_strike: float
    reason: str


@dataclass
class EffortResultScore:
    """Volume vs Price movement analysis"""
    effort_score: float  # Volume intensity 0-100
    result_score: float  # Price movement 0-100
    divergence: float  # effort - result (high = absorption happening)
    candle_type: str  # "SPRING_LOADING", "ABSORPTION", "NORMAL", "BREAKOUT"
    wick_ratio: float  # Long wicks = rejection


@dataclass
class OIAbsorptionScore:
    """Option chain absorption detection"""
    strike: float
    delta_oi: float  # Change in OI
    delta_premium: float  # Change in premium
    absorption_score: float  # 0-100
    trap_building: bool
    reason: str


@dataclass
class SLHuntResult:
    """Complete stop-loss hunt analysis result"""
    hunt_probability: float  # 0-100%
    hunt_likely: bool
    hunt_direction: str  # "UP" (hunting shorts), "DOWN" (hunting longs), "BOTH", "NONE"

    # Individual scores
    oi_absorption_score: float
    effort_result_score: float
    sl_cluster_score: float
    time_risk_score: float

    # Trap zones identified
    trap_zones: List[TrapZone] = field(default_factory=list)

    # Detailed analysis
    oi_analysis: Optional[OIAbsorptionScore] = None
    effort_analysis: Optional[EffortResultScore] = None

    # Action recommendation
    action: str = "WAIT"  # "WAIT", "TRADE_REVERSAL", "AVOID", "SAFE"
    reason: str = ""

    # Post-hunt entry details (if hunt confirmed)
    post_hunt_entry: Optional[Dict] = None


class SLHuntDetector:
    """
    Stop-Loss Hunt Detection System

    Detects institutional stop-loss hunting patterns using:
    1. Option chain OI absorption
    2. Volume vs Price divergence
    3. SL cluster zone mapping
    4. Time-based risk windows
    """

    def __init__(
        self,
        # Time windows (high-risk periods)
        morning_hunt_start: time = time(9, 15),
        morning_hunt_end: time = time(10, 0),
        closing_hunt_start: time = time(14, 30),
        closing_hunt_end: time = time(15, 30),

        # Thresholds
        oi_absorption_threshold: float = 70,
        effort_divergence_threshold: float = 30,
        sl_cluster_threshold: float = 60,
        hunt_probability_threshold: float = 65,

        # Round number settings
        round_number_interval: float = 50,  # NIFTY rounds to 50s
        vwap_sl_zone_pct: float = 0.5,  # 0.5% around VWAP
    ):
        self.morning_hunt_start = morning_hunt_start
        self.morning_hunt_end = morning_hunt_end
        self.closing_hunt_start = closing_hunt_start
        self.closing_hunt_end = closing_hunt_end

        self.oi_absorption_threshold = oi_absorption_threshold
        self.effort_divergence_threshold = effort_divergence_threshold
        self.sl_cluster_threshold = sl_cluster_threshold
        self.hunt_probability_threshold = hunt_probability_threshold

        self.round_number_interval = round_number_interval
        self.vwap_sl_zone_pct = vwap_sl_zone_pct

    def analyze(
        self,
        spot_price: float,
        df: Optional[pd.DataFrame] = None,
        oi_data: Optional[Dict] = None,
        vwap: Optional[float] = None,
        prev_day_high: Optional[float] = None,
        prev_day_low: Optional[float] = None,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
    ) -> SLHuntResult:
        """
        Complete stop-loss hunt analysis

        Args:
            spot_price: Current NIFTY spot price
            df: Price DataFrame with OHLCV data
            oi_data: Option chain data with OI, premium changes
            vwap: Current VWAP value
            prev_day_high: Previous day high
            prev_day_low: Previous day low
            support_levels: Known support levels
            resistance_levels: Known resistance levels

        Returns:
            SLHuntResult with complete analysis
        """

        # 1. Analyze Option Chain (OI Absorption)
        oi_score, oi_analysis = self._analyze_oi_absorption(spot_price, oi_data)

        # 2. Analyze Volume vs Price (Effort vs Result)
        effort_score, effort_analysis = self._analyze_effort_vs_result(df)

        # 3. Map SL Cluster Zones
        sl_score, trap_zones = self._map_sl_clusters(
            spot_price, oi_data, vwap, prev_day_high, prev_day_low,
            support_levels, resistance_levels
        )

        # 4. Time Risk Score
        time_score = self._calculate_time_risk()

        # 5. Calculate Combined Hunt Probability
        hunt_probability = self._calculate_hunt_probability(
            oi_score, effort_score, sl_score, time_score
        )

        # 6. Determine Hunt Direction
        hunt_direction = self._determine_hunt_direction(
            spot_price, trap_zones, oi_analysis
        )

        # 7. Generate Action Recommendation
        hunt_likely = hunt_probability >= self.hunt_probability_threshold
        action, reason = self._generate_recommendation(
            hunt_likely, hunt_probability, hunt_direction, trap_zones
        )

        # 8. Post-hunt entry setup (if hunt detected)
        post_hunt_entry = None
        if hunt_likely:
            post_hunt_entry = self._calculate_post_hunt_entry(
                spot_price, hunt_direction, trap_zones
            )

        return SLHuntResult(
            hunt_probability=hunt_probability,
            hunt_likely=hunt_likely,
            hunt_direction=hunt_direction,
            oi_absorption_score=oi_score,
            effort_result_score=effort_score,
            sl_cluster_score=sl_score,
            time_risk_score=time_score,
            trap_zones=trap_zones,
            oi_analysis=oi_analysis,
            effort_analysis=effort_analysis,
            action=action,
            reason=reason,
            post_hunt_entry=post_hunt_entry,
        )

    def _analyze_oi_absorption(
        self,
        spot_price: float,
        oi_data: Optional[Dict]
    ) -> Tuple[float, Optional[OIAbsorptionScore]]:
        """
        Layer 1: Option Chain Analysis - OI Absorption Detection

        Key Rule: Max OI strikes are the BAIT

        Red Flag Pattern:
        - Price near high OI strike
        - ΔOI increasing (sellers adding)
        - Premium NOT increasing (confident sellers)
        → TRAP BUILDING
        """
        if not oi_data:
            return 0, None

        atm_strike = round(spot_price / 50) * 50

        # Get OI data for ATM and nearby strikes
        call_oi = oi_data.get("call_oi", {})
        put_oi = oi_data.get("put_oi", {})
        call_oi_change = oi_data.get("call_oi_change", {})
        put_oi_change = oi_data.get("put_oi_change", {})
        call_premium_change = oi_data.get("call_premium_change", {})
        put_premium_change = oi_data.get("put_premium_change", {})

        # Find highest OI strike near ATM
        max_call_oi_strike = atm_strike
        max_put_oi_strike = atm_strike
        max_call_oi = 0
        max_put_oi = 0

        for strike in range(int(atm_strike - 200), int(atm_strike + 250), 50):
            strike_str = str(strike)
            if strike_str in call_oi and call_oi[strike_str] > max_call_oi:
                max_call_oi = call_oi[strike_str]
                max_call_oi_strike = strike
            if strike_str in put_oi and put_oi[strike_str] > max_put_oi:
                max_put_oi = put_oi[strike_str]
                max_put_oi_strike = strike

        # Check distance to high OI strikes
        dist_to_call_wall = max_call_oi_strike - spot_price
        dist_to_put_wall = spot_price - max_put_oi_strike

        absorption_score = 0
        trap_building = False
        analysis_strike = atm_strike
        delta_oi = 0
        delta_premium = 0
        reason = "Normal market conditions"

        # Check for CALL side trap (shorts getting hunted)
        if 0 < dist_to_call_wall <= 50:  # Price approaching CALL wall
            strike_str = str(max_call_oi_strike)
            delta_oi = call_oi_change.get(strike_str, 0)
            delta_premium = call_premium_change.get(strike_str, 0)

            # Trap pattern: OI increasing but premium NOT rising
            if delta_oi > 0 and delta_premium <= 5:
                absorption_score = min(90, 50 + (delta_oi / 1000) * 10)
                trap_building = True
                analysis_strike = max_call_oi_strike
                reason = f"CALL trap building at {max_call_oi_strike}: OI +{delta_oi:,.0f}, Premium flat"

        # Check for PUT side trap (longs getting hunted)
        elif 0 < dist_to_put_wall <= 50:  # Price approaching PUT wall
            strike_str = str(max_put_oi_strike)
            delta_oi = put_oi_change.get(strike_str, 0)
            delta_premium = put_premium_change.get(strike_str, 0)

            # Trap pattern: OI increasing but premium NOT rising
            if delta_oi > 0 and delta_premium <= 5:
                absorption_score = min(90, 50 + (delta_oi / 1000) * 10)
                trap_building = True
                analysis_strike = max_put_oi_strike
                reason = f"PUT trap building at {max_put_oi_strike}: OI +{delta_oi:,.0f}, Premium flat"

        # High OI concentration itself is a risk
        total_atm_oi = call_oi.get(str(atm_strike), 0) + put_oi.get(str(atm_strike), 0)
        if total_atm_oi > 500000:  # Very high OI concentration
            absorption_score = max(absorption_score, 40)
            if not trap_building:
                reason = f"High OI concentration at ATM: {total_atm_oi:,.0f}"

        oi_analysis = OIAbsorptionScore(
            strike=analysis_strike,
            delta_oi=delta_oi,
            delta_premium=delta_premium,
            absorption_score=absorption_score,
            trap_building=trap_building,
            reason=reason,
        )

        return absorption_score, oi_analysis

    def _analyze_effort_vs_result(
        self,
        df: Optional[pd.DataFrame]
    ) -> Tuple[float, Optional[EffortResultScore]]:
        """
        Layer 3: Volume + Price - Effort vs Result Analysis

        Fake Breakout Formula:
        - High Volume + Small Candle Body + Long Wicks = ABSORPTION

        Pre-Hunt Warning:
        - Volume rising + Price compressing = SPRING LOADING
        """
        if df is None or len(df) < 10:
            return 0, None

        # Get recent candles
        recent = df.tail(10).copy()
        latest = recent.iloc[-1]

        # Calculate metrics
        avg_volume = recent["volume"].mean()
        latest_volume = latest["volume"]

        # Body and range
        body = abs(latest["close"] - latest["open"])
        high_low_range = latest["high"] - latest["low"]

        # Wick analysis
        upper_wick = latest["high"] - max(latest["open"], latest["close"])
        lower_wick = min(latest["open"], latest["close"]) - latest["low"]
        total_wick = upper_wick + lower_wick

        # Effort Score (Volume intensity)
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
        effort_score = min(100, volume_ratio * 50)

        # Result Score (Price movement)
        avg_range = (recent["high"] - recent["low"]).mean()
        range_ratio = high_low_range / avg_range if avg_range > 0 else 1
        result_score = min(100, range_ratio * 50)

        # Divergence (Effort - Result)
        divergence = effort_score - result_score

        # Wick ratio (higher = more rejection)
        wick_ratio = (total_wick / high_low_range * 100) if high_low_range > 0 else 0

        # Classify candle type
        candle_type = "NORMAL"

        if volume_ratio > 1.5 and body < avg_range * 0.3 and wick_ratio > 60:
            candle_type = "ABSORPTION"
            divergence = max(divergence, 50)
        elif volume_ratio > 1.2 and body < avg_range * 0.5:
            candle_type = "SPRING_LOADING"
            divergence = max(divergence, 30)
        elif volume_ratio > 1.5 and body > avg_range * 0.8:
            candle_type = "BREAKOUT"
            divergence = 0  # Genuine move

        # Check for compression pattern (multiple candles)
        ranges = recent["high"] - recent["low"]
        volumes = recent["volume"]

        # Volume rising but range compressing = SPRING LOADING
        if len(ranges) >= 5:
            range_trend = ranges.iloc[-3:].mean() / ranges.iloc[:3].mean() if ranges.iloc[:3].mean() > 0 else 1
            volume_trend = volumes.iloc[-3:].mean() / volumes.iloc[:3].mean() if volumes.iloc[:3].mean() > 0 else 1

            if range_trend < 0.7 and volume_trend > 1.2:
                candle_type = "SPRING_LOADING"
                divergence = max(divergence, 40)

        effort_analysis = EffortResultScore(
            effort_score=effort_score,
            result_score=result_score,
            divergence=divergence,
            candle_type=candle_type,
            wick_ratio=wick_ratio,
        )

        return divergence, effort_analysis

    def _map_sl_clusters(
        self,
        spot_price: float,
        oi_data: Optional[Dict],
        vwap: Optional[float],
        prev_day_high: Optional[float],
        prev_day_low: Optional[float],
        support_levels: Optional[List[float]],
        resistance_levels: Optional[List[float]],
    ) -> Tuple[float, List[TrapZone]]:
        """
        Layer 4: Chart Structure - SL Cluster Zone Detection

        Common SL zones:
        - Previous day high/low
        - Round numbers
        - VWAP ±0.5%
        - Obvious trendline touches
        - Just above CALL OI / below PUT OI
        """
        trap_zones = []
        max_sl_density = 0

        # 1. Round numbers (high SL density)
        nearest_round = round(spot_price / self.round_number_interval) * self.round_number_interval
        for offset in [-self.round_number_interval, 0, self.round_number_interval]:
            round_level = nearest_round + offset
            distance = abs(spot_price - round_level)

            if distance <= 30:  # Close to round number
                sl_density = max(0, 80 - distance * 2)
                if sl_density > 30:
                    zone_type = "PUT_SL_ZONE" if round_level < spot_price else "CALL_SL_ZONE"
                    trap_zones.append(TrapZone(
                        price=round_level,
                        zone_type=zone_type,
                        sl_density=sl_density,
                        oi_at_strike=0,
                        reason=f"Round number {round_level}"
                    ))
                    max_sl_density = max(max_sl_density, sl_density)

        # 2. Previous day high/low (classic SL zones)
        if prev_day_high and abs(spot_price - prev_day_high) <= 50:
            distance = abs(spot_price - prev_day_high)
            sl_density = max(0, 90 - distance * 1.5)
            if sl_density > 30:
                trap_zones.append(TrapZone(
                    price=prev_day_high,
                    zone_type="CALL_SL_ZONE",
                    sl_density=sl_density,
                    oi_at_strike=0,
                    reason=f"Previous day high {prev_day_high}"
                ))
                max_sl_density = max(max_sl_density, sl_density)

        if prev_day_low and abs(spot_price - prev_day_low) <= 50:
            distance = abs(spot_price - prev_day_low)
            sl_density = max(0, 90 - distance * 1.5)
            if sl_density > 30:
                trap_zones.append(TrapZone(
                    price=prev_day_low,
                    zone_type="PUT_SL_ZONE",
                    sl_density=sl_density,
                    oi_at_strike=0,
                    reason=f"Previous day low {prev_day_low}"
                ))
                max_sl_density = max(max_sl_density, sl_density)

        # 3. VWAP zone
        if vwap:
            vwap_upper = vwap * (1 + self.vwap_sl_zone_pct / 100)
            vwap_lower = vwap * (1 - self.vwap_sl_zone_pct / 100)

            if vwap_lower <= spot_price <= vwap_upper:
                sl_density = 70
                trap_zones.append(TrapZone(
                    price=vwap,
                    zone_type="VWAP_ZONE",
                    sl_density=sl_density,
                    oi_at_strike=0,
                    reason=f"VWAP zone {vwap:.2f}"
                ))
                max_sl_density = max(max_sl_density, sl_density)

        # 4. Option chain based SL zones
        if oi_data:
            call_oi = oi_data.get("call_oi", {})
            put_oi = oi_data.get("put_oi", {})

            atm_strike = round(spot_price / 50) * 50

            # Find high OI strikes - these are SL magnets
            for strike in range(int(atm_strike - 150), int(atm_strike + 200), 50):
                strike_str = str(strike)

                # High CALL OI = Shorts have SL just above
                call_oi_value = call_oi.get(strike_str, 0)
                if call_oi_value > 100000 and strike > spot_price:
                    distance = strike - spot_price
                    if distance <= 100:
                        sl_density = min(95, 50 + (call_oi_value / 10000))
                        trap_zones.append(TrapZone(
                            price=strike + 10,  # SL just above strike
                            zone_type="CALL_SL_ZONE",
                            sl_density=sl_density,
                            oi_at_strike=call_oi_value,
                            reason=f"High CALL OI {call_oi_value:,.0f} at {strike}"
                        ))
                        max_sl_density = max(max_sl_density, sl_density)

                # High PUT OI = Longs have SL just below
                put_oi_value = put_oi.get(strike_str, 0)
                if put_oi_value > 100000 and strike < spot_price:
                    distance = spot_price - strike
                    if distance <= 100:
                        sl_density = min(95, 50 + (put_oi_value / 10000))
                        trap_zones.append(TrapZone(
                            price=strike - 10,  # SL just below strike
                            zone_type="PUT_SL_ZONE",
                            sl_density=sl_density,
                            oi_at_strike=put_oi_value,
                            reason=f"High PUT OI {put_oi_value:,.0f} at {strike}"
                        ))
                        max_sl_density = max(max_sl_density, sl_density)

        # 5. Support/Resistance levels (obvious SL zones)
        if support_levels:
            for support in support_levels:
                distance = abs(spot_price - support)
                if distance <= 50:
                    sl_density = max(0, 75 - distance)
                    if sl_density > 30:
                        trap_zones.append(TrapZone(
                            price=support - 5,
                            zone_type="PUT_SL_ZONE",
                            sl_density=sl_density,
                            oi_at_strike=0,
                            reason=f"Support level SL zone {support}"
                        ))
                        max_sl_density = max(max_sl_density, sl_density)

        if resistance_levels:
            for resistance in resistance_levels:
                distance = abs(spot_price - resistance)
                if distance <= 50:
                    sl_density = max(0, 75 - distance)
                    if sl_density > 30:
                        trap_zones.append(TrapZone(
                            price=resistance + 5,
                            zone_type="CALL_SL_ZONE",
                            sl_density=sl_density,
                            oi_at_strike=0,
                            reason=f"Resistance level SL zone {resistance}"
                        ))
                        max_sl_density = max(max_sl_density, sl_density)

        # Sort by SL density
        trap_zones.sort(key=lambda x: x.sl_density, reverse=True)

        return max_sl_density, trap_zones[:10]  # Top 10 zones

    def _calculate_time_risk(self) -> float:
        """
        Time-based risk scoring

        High-risk windows:
        - First 45 minutes (9:15 - 10:00)
        - Last 60 minutes (14:30 - 15:30)
        """
        now = datetime.now().time()

        # Morning hunt window
        if self.morning_hunt_start <= now <= self.morning_hunt_end:
            return 80

        # Closing hunt window
        if self.closing_hunt_start <= now <= self.closing_hunt_end:
            return 90

        # Moderate risk just outside windows
        if time(10, 0) <= now <= time(10, 30):
            return 40
        if time(14, 0) <= now <= time(14, 30):
            return 50

        return 20  # Low risk during mid-day

    def _calculate_hunt_probability(
        self,
        oi_score: float,
        effort_score: float,
        sl_score: float,
        time_score: float,
    ) -> float:
        """
        Calculate combined hunt probability

        Weights:
        - OI Absorption: 30%
        - Effort vs Result: 25%
        - SL Cluster: 25%
        - Time Risk: 20%
        """
        probability = (
            oi_score * 0.30 +
            effort_score * 0.25 +
            sl_score * 0.25 +
            time_score * 0.20
        )

        # Bonus if multiple factors align
        high_factors = sum([
            oi_score > 60,
            effort_score > 40,
            sl_score > 60,
            time_score > 60,
        ])

        if high_factors >= 3:
            probability = min(100, probability * 1.2)
        elif high_factors >= 2:
            probability = min(100, probability * 1.1)

        return round(probability, 1)

    def _determine_hunt_direction(
        self,
        spot_price: float,
        trap_zones: List[TrapZone],
        oi_analysis: Optional[OIAbsorptionScore],
    ) -> str:
        """
        Determine which direction the hunt will go

        - UP: Hunting shorts (triggering CALL SL zones)
        - DOWN: Hunting longs (triggering PUT SL zones)
        """
        if not trap_zones:
            return "NONE"

        # Count zones by type
        call_zones = [z for z in trap_zones if z.zone_type == "CALL_SL_ZONE"]
        put_zones = [z for z in trap_zones if z.zone_type == "PUT_SL_ZONE"]

        call_density = sum(z.sl_density for z in call_zones)
        put_density = sum(z.sl_density for z in put_zones)

        # Check nearest high-density zone
        nearest_call = min(call_zones, key=lambda z: abs(z.price - spot_price)) if call_zones else None
        nearest_put = min(put_zones, key=lambda z: abs(z.price - spot_price)) if put_zones else None

        # Factor in OI analysis
        oi_direction = "NONE"
        if oi_analysis and oi_analysis.trap_building:
            if "CALL" in oi_analysis.reason:
                oi_direction = "UP"
            elif "PUT" in oi_analysis.reason:
                oi_direction = "DOWN"

        # Determine direction
        if nearest_call and nearest_put:
            dist_call = nearest_call.price - spot_price
            dist_put = spot_price - nearest_put.price

            # Closer high-density zone is likely target
            if dist_call < dist_put and nearest_call.sl_density > 50:
                return "UP"
            elif dist_put < dist_call and nearest_put.sl_density > 50:
                return "DOWN"
            elif call_density > put_density * 1.5:
                return "UP"
            elif put_density > call_density * 1.5:
                return "DOWN"

        if oi_direction != "NONE":
            return oi_direction

        if call_density > put_density:
            return "UP"
        elif put_density > call_density:
            return "DOWN"

        return "BOTH"

    def _generate_recommendation(
        self,
        hunt_likely: bool,
        hunt_probability: float,
        hunt_direction: str,
        trap_zones: List[TrapZone],
    ) -> Tuple[str, str]:
        """
        Generate action recommendation
        """
        if not hunt_likely:
            return "SAFE", f"Low hunt probability ({hunt_probability}%). Normal trading conditions."

        if hunt_probability >= 80:
            if hunt_direction == "UP":
                return "WAIT", f"HIGH SL HUNT RISK ({hunt_probability}%). Expect spike UP to clear shorts, then reversal DOWN."
            elif hunt_direction == "DOWN":
                return "WAIT", f"HIGH SL HUNT RISK ({hunt_probability}%). Expect spike DOWN to clear longs, then reversal UP."
            else:
                return "AVOID", f"VERY HIGH SL HUNT RISK ({hunt_probability}%). Both sides at risk. Stay out."

        elif hunt_probability >= 65:
            if hunt_direction in ["UP", "DOWN"]:
                return "WAIT", f"Moderate hunt risk ({hunt_probability}%). Wait for hunt candle, trade reversal."
            else:
                return "WAIT", f"Moderate hunt risk ({hunt_probability}%). Multiple trap zones active."

        return "SAFE", f"Below threshold ({hunt_probability}%)."

    def _calculate_post_hunt_entry(
        self,
        spot_price: float,
        hunt_direction: str,
        trap_zones: List[TrapZone],
    ) -> Dict:
        """
        Calculate entry parameters AFTER the hunt completes

        Entry:
        - Entry after confirmation candle
        - SL = hunt candle high/low
        - Target = VWAP / opposite OI wall
        """
        if hunt_direction == "UP":
            # Hunt goes UP, we SHORT after reversal
            nearest_trap = None
            for zone in trap_zones:
                if zone.zone_type == "CALL_SL_ZONE" and zone.price > spot_price:
                    nearest_trap = zone
                    break

            if nearest_trap:
                hunt_target = nearest_trap.price
                return {
                    "direction": "SHORT",
                    "wait_for": f"Spike above {hunt_target:.0f}, then reversal candle",
                    "entry_trigger": "RED candle closing below hunt candle low",
                    "sl": f"Hunt candle high + 10 (~{hunt_target + 15:.0f})",
                    "target": "VWAP or PUT OI wall",
                    "trap_price": hunt_target,
                }

        elif hunt_direction == "DOWN":
            # Hunt goes DOWN, we LONG after reversal
            nearest_trap = None
            for zone in trap_zones:
                if zone.zone_type == "PUT_SL_ZONE" and zone.price < spot_price:
                    nearest_trap = zone
                    break

            if nearest_trap:
                hunt_target = nearest_trap.price
                return {
                    "direction": "LONG",
                    "wait_for": f"Spike below {hunt_target:.0f}, then reversal candle",
                    "entry_trigger": "GREEN candle closing above hunt candle high",
                    "sl": f"Hunt candle low - 10 (~{hunt_target - 15:.0f})",
                    "target": "VWAP or CALL OI wall",
                    "trap_price": hunt_target,
                }

        return {
            "direction": "WAIT",
            "wait_for": "Hunt completion and confirmation",
            "entry_trigger": "Reversal candle",
            "sl": "Hunt candle extreme",
            "target": "Opposite OI wall",
        }

    def detect_hunt_candle(
        self,
        df: pd.DataFrame,
        hunt_direction: str,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Detect if a stop-loss hunt candle has just occurred

        Hunt candle signature:
        - Sudden spike 10-15 points
        - Instant reversal
        - High volume
        - Long wick in hunt direction

        Returns:
            (hunt_occurred, candle_details)
        """
        if len(df) < 3:
            return False, None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Calculate metrics
        body = abs(latest["close"] - latest["open"])
        high_low_range = latest["high"] - latest["low"]

        upper_wick = latest["high"] - max(latest["open"], latest["close"])
        lower_wick = min(latest["open"], latest["close"]) - latest["low"]

        avg_range = (df["high"].tail(10) - df["low"].tail(10)).mean()
        avg_volume = df["volume"].tail(10).mean()

        volume_spike = latest["volume"] > avg_volume * 1.5

        hunt_occurred = False
        candle_details = None

        if hunt_direction == "UP":
            # Hunt UP = Long upper wick, closed lower
            if (upper_wick > body * 2 and
                upper_wick > avg_range * 0.5 and
                latest["close"] < latest["open"] and
                volume_spike):
                hunt_occurred = True
                candle_details = {
                    "type": "BEARISH_HUNT",
                    "hunt_high": latest["high"],
                    "reversal_close": latest["close"],
                    "wick_size": upper_wick,
                    "entry_below": latest["low"],
                    "sl_above": latest["high"] + 10,
                }

        elif hunt_direction == "DOWN":
            # Hunt DOWN = Long lower wick, closed higher
            if (lower_wick > body * 2 and
                lower_wick > avg_range * 0.5 and
                latest["close"] > latest["open"] and
                volume_spike):
                hunt_occurred = True
                candle_details = {
                    "type": "BULLISH_HUNT",
                    "hunt_low": latest["low"],
                    "reversal_close": latest["close"],
                    "wick_size": lower_wick,
                    "entry_above": latest["high"],
                    "sl_below": latest["low"] - 10,
                }

        return hunt_occurred, candle_details


# ============================================================================
# TELEGRAM NOTIFICATION FUNCTIONS
# ============================================================================

def format_sl_hunt_telegram(result: SLHuntResult, spot_price: float) -> str:
    """
    Format SL Hunt analysis for Telegram notification
    """
    if not result.hunt_likely:
        return ""

    msg = f"""
🎯 STOP-LOSS HUNT ALERT
══════════════════════════════
⚠️ Hunt Probability: {result.hunt_probability}%
📍 Current Price: ₹{spot_price:,.2f}
🎯 Hunt Direction: {result.hunt_direction}

📊 ANALYSIS SCORES:
├─ OI Absorption: {result.oi_absorption_score:.0f}%
├─ Effort vs Result: {result.effort_result_score:.0f}%
├─ SL Clusters: {result.sl_cluster_score:.0f}%
└─ Time Risk: {result.time_risk_score:.0f}%
"""

    if result.oi_analysis and result.oi_analysis.trap_building:
        msg += f"""
🔴 OI TRAP DETECTED:
└─ {result.oi_analysis.reason}
"""

    if result.effort_analysis:
        msg += f"""
📈 VOLUME PATTERN:
├─ Type: {result.effort_analysis.candle_type}
├─ Effort: {result.effort_analysis.effort_score:.0f}
├─ Result: {result.effort_analysis.result_score:.0f}
└─ Wick Ratio: {result.effort_analysis.wick_ratio:.0f}%
"""

    if result.trap_zones:
        msg += f"""
🎯 TOP TRAP ZONES:
"""
        for i, zone in enumerate(result.trap_zones[:5], 1):
            msg += f"  {i}. ₹{zone.price:,.0f} ({zone.zone_type}) - {zone.sl_density:.0f}% density\n"

    msg += f"""
══════════════════════════════
🚨 RECOMMENDATION: {result.action}
💡 {result.reason}
"""

    if result.post_hunt_entry:
        entry = result.post_hunt_entry
        msg += f"""
═══ POST-HUNT TRADE SETUP ═══
📍 Direction: {entry['direction']}
⏳ Wait For: {entry['wait_for']}
🔔 Entry: {entry['entry_trigger']}
🛡️ SL: {entry['sl']}
🎯 Target: {entry['target']}
"""

    return msg


def format_hunt_candle_telegram(
    candle_details: Dict,
    spot_price: float,
    hunt_direction: str,
) -> str:
    """
    Format hunt candle detection for Telegram
    """
    msg = f"""
🔥 HUNT CANDLE DETECTED!
══════════════════════════════
📍 Type: {candle_details['type']}
💰 Current: ₹{spot_price:,.2f}
"""

    if hunt_direction == "UP":
        msg += f"""
📈 Hunt High: ₹{candle_details['hunt_high']:,.2f}
📉 Reversal Close: ₹{candle_details['reversal_close']:,.2f}
📏 Wick Size: {candle_details['wick_size']:.2f} pts

🎯 TRADE SETUP:
├─ Direction: SHORT
├─ Entry Below: ₹{candle_details['entry_below']:,.2f}
└─ SL Above: ₹{candle_details['sl_above']:,.2f}
"""
    else:
        msg += f"""
📉 Hunt Low: ₹{candle_details['hunt_low']:,.2f}
📈 Reversal Close: ₹{candle_details['reversal_close']:,.2f}
📏 Wick Size: {candle_details['wick_size']:.2f} pts

🎯 TRADE SETUP:
├─ Direction: LONG
├─ Entry Above: ₹{candle_details['entry_above']:,.2f}
└─ SL Below: ₹{candle_details['sl_below']:,.2f}
"""

    msg += """
══════════════════════════════
⚡ WAIT FOR CONFIRMATION CANDLE
"""

    return msg
