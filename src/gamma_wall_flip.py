"""
Gamma Wall Flip Detector - Diamond Level Feature
=================================================
Volatility Regime Change Detection

Gamma Exposure (GEX) Analysis:
- When GEX is POSITIVE: Market makers SELL rallies, BUY dips = MEAN REVERSION
- When GEX is NEGATIVE: Market makers BUY rallies, SELL dips = TREND AMPLIFICATION

THE FLIP: When GEX changes from +ve to -ve (or vice versa):
- This is a VOLATILITY REGIME CHANGE
- Entire market behavior shifts
- This is what institutions watch for explosive moves!

Data Source: Option Chain from NIFTY Option Screener (merged_df)
- Uses OI (Open Interest) at each strike
- Calculates Gamma at each strike
- Net GEX = Call Gamma - Put Gamma weighted by OI
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st
from scipy.stats import norm
import math


@dataclass
class GammaLevel:
    """Gamma data at a specific strike"""
    strike: float
    call_gamma: float
    put_gamma: float
    call_oi: int
    put_oi: int
    net_gamma: float  # Call Gamma * OI - Put Gamma * OI
    is_wall: bool  # High gamma concentration


@dataclass
class GEXFlipSignal:
    """Gamma Exposure Flip Signal"""
    signal_type: str  # 'POSITIVE_TO_NEGATIVE', 'NEGATIVE_TO_POSITIVE', 'APPROACHING_FLIP'
    previous_gex: float
    current_gex: float
    flip_level: float  # Price level where GEX = 0
    volatility_regime: str  # 'MEAN_REVERSION', 'TREND_FOLLOWING'
    expected_behavior: str
    trade_implications: str
    strength: float  # 0-100
    urgency: str  # 'IMMEDIATE', 'DEVELOPING', 'WATCH'


@dataclass
class GammaWall:
    """Significant Gamma Wall"""
    strike: float
    gamma_value: float
    wall_type: str  # 'CALL_WALL', 'PUT_WALL'
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    price_magnet: bool  # Will price be attracted to this level?
    price_repel: bool  # Will price bounce off this level?


@dataclass
class GammaAnalysis:
    """Complete Gamma Analysis Result"""
    net_gex: float
    gex_regime: str  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    flip_signal: Optional[GEXFlipSignal]
    gamma_walls: List[GammaWall]
    call_wall: Optional[float]  # Highest call gamma strike
    put_wall: Optional[float]  # Highest put gamma strike
    gamma_neutral_level: float  # Price where GEX = 0
    volatility_expectation: str  # 'LOW', 'MODERATE', 'HIGH', 'EXPLOSIVE'
    score: float  # 0-100 composite score
    trade_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    description: str


class GammaWallFlipDetector:
    """
    Gamma Exposure (GEX) and Wall Flip Detector

    Calculates:
    - Net GEX from option chain data
    - Gamma walls (significant OI concentrations)
    - GEX flip detection (regime change)
    - Volatility expectations
    """

    def __init__(self):
        self.gex_history = []
        self.max_history = 100
        self.risk_free_rate = 0.07  # 7% for India
        self.flip_threshold = 0.1  # 10% of max GEX = flip zone

    def calculate_gamma(self, spot: float, strike: float, tte: float,
                       volatility: float, is_call: bool) -> float:
        """
        Calculate option gamma using Black-Scholes

        Gamma = N'(d1) / (S * σ * √T)
        where N'(x) = normal PDF

        Args:
            spot: Current spot price
            strike: Option strike price
            tte: Time to expiry in years
            volatility: Implied volatility
            is_call: True for call, False for put

        Returns:
            Gamma value
        """
        try:
            if tte <= 0 or volatility <= 0 or spot <= 0:
                return 0

            d1 = (math.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility ** 2) * tte) / (volatility * math.sqrt(tte))

            # Gamma is same for calls and puts
            gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(tte))

            return gamma

        except Exception:
            return 0

    def estimate_iv_from_ltp(self, spot: float, strike: float, ltp: float,
                            tte: float, is_call: bool) -> float:
        """
        Estimate IV from LTP using Newton-Raphson method
        Simplified version - uses approximation
        """
        try:
            if ltp <= 0 or spot <= 0:
                return 0.20  # Default 20% IV

            # Simple approximation: ATM options have IV ~ LTP / (0.4 * S * sqrt(T))
            moneyness = abs(spot - strike) / spot

            if moneyness < 0.02:  # ATM
                iv = ltp / (0.4 * spot * math.sqrt(tte)) if tte > 0 else 0.20
            else:
                # Adjust for moneyness
                iv = ltp / (0.3 * spot * math.sqrt(tte)) if tte > 0 else 0.20

            # Clamp to reasonable range
            return max(0.05, min(1.0, iv))

        except Exception:
            return 0.20

    def get_time_to_expiry(self) -> float:
        """
        Get time to expiry for weekly options

        Returns time in years
        """
        now = datetime.now()

        # Find next Thursday (weekly expiry)
        days_until_thursday = (3 - now.weekday()) % 7
        if days_until_thursday == 0 and now.hour >= 15:  # After 3:30 PM on Thursday
            days_until_thursday = 7

        expiry = now + timedelta(days=days_until_thursday)
        expiry = expiry.replace(hour=15, minute=30, second=0, microsecond=0)

        tte_days = (expiry - now).total_seconds() / 86400
        tte_years = max(0.001, tte_days / 365)  # Minimum 1 day

        return tte_years

    def calculate_gex_from_chain(self, merged_df: pd.DataFrame, spot_price: float) -> Tuple[List[GammaLevel], float]:
        """
        Calculate GEX from option chain data

        USES EXISTING GREEKS from option chain if available:
        - Gamma_CE, Gamma_PE (calculated Greeks)
        - Delta_CE, Delta_PE
        - IV_CE, IV_PE

        Args:
            merged_df: Option chain DataFrame with OI_CE, OI_PE, and Greeks
            spot_price: Current spot price

        Returns:
            (gamma_levels, net_gex)
        """
        gamma_levels = []
        net_gex = 0

        tte = self.get_time_to_expiry()

        # Standard contract size for NIFTY
        lot_size = 25

        # Check if Greeks are available in the dataframe
        has_greeks = 'Gamma_CE' in merged_df.columns and 'Gamma_PE' in merged_df.columns

        for idx, row in merged_df.iterrows():
            try:
                strike = row.get('Strike', row.get('strike', 0))
                call_oi = row.get('OI_CE', row.get('oi_ce', 0)) or 0
                put_oi = row.get('OI_PE', row.get('oi_pe', 0)) or 0

                if strike <= 0:
                    continue

                # USE EXISTING GREEKS if available (from NIFTY Option Screener)
                if has_greeks:
                    # Use pre-calculated Greeks from option chain
                    call_gamma = row.get('Gamma_CE', 0) or 0
                    put_gamma = row.get('Gamma_PE', 0) or 0
                    call_iv = row.get('IV_CE', 0.20) or 0.20
                    put_iv = row.get('IV_PE', 0.20) or 0.20
                else:
                    # Fallback: Calculate from LTP
                    call_ltp = row.get('LTP_CE', row.get('ltp_ce', 0)) or 0
                    put_ltp = row.get('LTP_PE', row.get('ltp_pe', 0)) or 0
                    call_iv = self.estimate_iv_from_ltp(spot_price, strike, call_ltp, tte, True)
                    put_iv = self.estimate_iv_from_ltp(spot_price, strike, put_ltp, tte, False)
                    call_gamma = self.calculate_gamma(spot_price, strike, tte, call_iv, True)
                    put_gamma = self.calculate_gamma(spot_price, strike, tte, put_iv, False)

                # GEX = Gamma * OI * Spot^2 * 0.01 * lot_size
                # Calls have positive gamma effect, puts have negative
                call_gex = call_gamma * call_oi * spot_price * spot_price * 0.01 * lot_size
                put_gex = -put_gamma * put_oi * spot_price * spot_price * 0.01 * lot_size

                strike_net_gamma = call_gex + put_gex
                net_gex += strike_net_gamma

                # Check if this is a gamma wall
                is_wall = (call_oi > 100000 or put_oi > 100000)  # Large OI = wall

                gamma_levels.append(GammaLevel(
                    strike=strike,
                    call_gamma=call_gex,
                    put_gamma=abs(put_gex),
                    call_oi=call_oi,
                    put_oi=put_oi,
                    net_gamma=strike_net_gamma,
                    is_wall=is_wall
                ))

            except Exception:
                continue

        return gamma_levels, net_gex

    def find_gamma_walls(self, gamma_levels: List[GammaLevel], spot_price: float) -> List[GammaWall]:
        """
        Find significant gamma walls

        These are strikes where:
        - Large call OI = Call wall (resistance, price magnet above)
        - Large put OI = Put wall (support, price magnet below)
        """
        walls = []

        # Sort by absolute gamma
        sorted_levels = sorted(gamma_levels, key=lambda x: abs(x.net_gamma), reverse=True)

        # Get top 5 gamma concentrations
        for level in sorted_levels[:5]:
            if level.call_gamma > level.put_gamma:
                wall_type = 'CALL_WALL'
                gamma_value = level.call_gamma
            else:
                wall_type = 'PUT_WALL'
                gamma_value = level.put_gamma

            # Determine strength
            max_gamma = max(abs(l.net_gamma) for l in gamma_levels) if gamma_levels else 1
            strength_ratio = abs(level.net_gamma) / max_gamma

            if strength_ratio > 0.7:
                strength = 'STRONG'
            elif strength_ratio > 0.4:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'

            # Price magnet if close to spot
            distance = abs(level.strike - spot_price) / spot_price
            price_magnet = distance < 0.02  # Within 2%

            # Price repel if large gamma and away from spot
            price_repel = strength == 'STRONG' and distance > 0.01

            walls.append(GammaWall(
                strike=level.strike,
                gamma_value=gamma_value,
                wall_type=wall_type,
                strength=strength,
                price_magnet=price_magnet,
                price_repel=price_repel
            ))

        return walls

    def find_gamma_neutral(self, gamma_levels: List[GammaLevel], spot_price: float) -> float:
        """
        Find the price level where net GEX = 0

        This is the flip point - volatility regime changes here
        """
        if not gamma_levels:
            return spot_price

        # Sort by strike
        sorted_levels = sorted(gamma_levels, key=lambda x: x.strike)

        # Find where net_gamma changes sign
        for i in range(len(sorted_levels) - 1):
            current = sorted_levels[i]
            next_level = sorted_levels[i + 1]

            if current.net_gamma * next_level.net_gamma < 0:
                # Sign change - interpolate
                if next_level.net_gamma - current.net_gamma != 0:
                    ratio = abs(current.net_gamma) / abs(next_level.net_gamma - current.net_gamma)
                    neutral = current.strike + ratio * (next_level.strike - current.strike)
                    return neutral

        # If no crossover found, use spot
        return spot_price

    def detect_flip(self, current_gex: float, gamma_neutral: float, spot_price: float) -> Optional[GEXFlipSignal]:
        """
        Detect GEX flip or approaching flip

        This is the DIAMOND signal - volatility regime change!
        """
        # Add to history
        self.gex_history.append({
            'timestamp': datetime.now(),
            'gex': current_gex,
            'spot': spot_price
        })

        if len(self.gex_history) > self.max_history:
            self.gex_history = self.gex_history[-self.max_history:]

        # Check if we have history to compare
        if len(self.gex_history) < 2:
            return None

        previous_gex = self.gex_history[-2]['gex']

        # Determine current regime
        if current_gex > 0:
            current_regime = 'MEAN_REVERSION'
            expected_behavior = "Market makers sell rallies, buy dips - expect range-bound trading"
        else:
            current_regime = 'TREND_FOLLOWING'
            expected_behavior = "Market makers amplify moves - expect trending/volatile trading"

        signal_type = None
        strength = 0
        urgency = 'WATCH'
        trade_implications = ""

        # FLIP DETECTED: Sign change
        if previous_gex > 0 and current_gex < 0:
            signal_type = 'POSITIVE_TO_NEGATIVE'
            strength = 95
            urgency = 'IMMEDIATE'
            trade_implications = "VOLATILITY EXPLOSION LIKELY! Prepare for trending moves. Breakouts will follow through."

        elif previous_gex < 0 and current_gex > 0:
            signal_type = 'NEGATIVE_TO_POSITIVE'
            strength = 90
            urgency = 'IMMEDIATE'
            trade_implications = "VOLATILITY COMPRESSION! Expect mean reversion. Sell extremes, buy dips."

        # APPROACHING FLIP: Close to gamma neutral
        elif gamma_neutral > 0:
            distance_to_neutral = abs(spot_price - gamma_neutral) / spot_price * 100

            if distance_to_neutral < 0.5:  # Within 0.5%
                signal_type = 'APPROACHING_FLIP'
                strength = 80
                urgency = 'DEVELOPING'
                trade_implications = f"Price approaching GEX flip level at {gamma_neutral:.0f}. Volatility regime may change soon!"

            elif distance_to_neutral < 1.0:  # Within 1%
                signal_type = 'APPROACHING_FLIP'
                strength = 60
                urgency = 'WATCH'
                trade_implications = f"Watch level {gamma_neutral:.0f} - GEX flip zone. Could trigger volatility change."

        if signal_type:
            return GEXFlipSignal(
                signal_type=signal_type,
                previous_gex=previous_gex,
                current_gex=current_gex,
                flip_level=gamma_neutral,
                volatility_regime=current_regime,
                expected_behavior=expected_behavior,
                trade_implications=trade_implications,
                strength=strength,
                urgency=urgency
            )

        return None

    def calculate_score(self, net_gex: float, flip_signal: Optional[GEXFlipSignal],
                       gamma_walls: List[GammaWall], spot_price: float) -> Tuple[float, str]:
        """
        Calculate composite gamma score and trade bias

        Returns: (score, trade_bias)
        """
        score = 50  # Start neutral

        # Factor 1: GEX Regime (+/- 15 points)
        if net_gex > 0:
            # Positive GEX = mean reversion, slightly bullish bias (buy dips)
            score += 10
            base_bias = 'NEUTRAL'
        else:
            # Negative GEX = trend following, direction matters
            score -= 5
            base_bias = 'NEUTRAL'

        # Factor 2: Flip Signal (+/- 20 points)
        if flip_signal:
            if flip_signal.signal_type == 'POSITIVE_TO_NEGATIVE':
                score -= 20  # Bearish - volatility explosion
            elif flip_signal.signal_type == 'NEGATIVE_TO_POSITIVE':
                score += 15  # Bullish - volatility compression
            elif flip_signal.signal_type == 'APPROACHING_FLIP':
                score += 0  # Neutral - uncertainty

        # Factor 3: Wall Positioning (+/- 15 points)
        call_walls_above = sum(1 for w in gamma_walls if w.wall_type == 'CALL_WALL' and w.strike > spot_price)
        put_walls_below = sum(1 for w in gamma_walls if w.wall_type == 'PUT_WALL' and w.strike < spot_price)

        if put_walls_below > call_walls_above:
            score += 10  # More support below
        elif call_walls_above > put_walls_below:
            score -= 5  # More resistance above

        # Determine trade bias
        if score > 60:
            trade_bias = 'BULLISH'
        elif score < 40:
            trade_bias = 'BEARISH'
        else:
            trade_bias = 'NEUTRAL'

        return max(0, min(100, score)), trade_bias

    def analyze(self, merged_df: pd.DataFrame = None, spot_price: float = None) -> Optional[GammaAnalysis]:
        """
        Main analysis function

        Args:
            merged_df: Option chain DataFrame (uses session_state.merged_df if None)
            spot_price: Current spot price (uses session_state if None)

        Returns:
            GammaAnalysis with all findings
        """
        try:
            # Get data from session state if not provided
            if merged_df is None:
                merged_df = st.session_state.get('merged_df')

            if merged_df is None or len(merged_df) == 0:
                return None

            if spot_price is None:
                # Try to get from session state
                spot_price = st.session_state.get('spot_price')
                if spot_price is None:
                    # Try to get from chart data
                    chart_data = st.session_state.get('chart_data')
                    if chart_data is not None and len(chart_data) > 0:
                        spot_price = chart_data['Close'].iloc[-1] if 'Close' in chart_data.columns else chart_data['close'].iloc[-1]

            if spot_price is None or spot_price <= 0:
                # Last resort: estimate from option chain
                if 'Strike' in merged_df.columns:
                    spot_price = merged_df['Strike'].median()
                else:
                    return None

            # Calculate GEX from chain
            gamma_levels, net_gex = self.calculate_gex_from_chain(merged_df, spot_price)

            if not gamma_levels:
                return None

            # Find gamma walls
            gamma_walls = self.find_gamma_walls(gamma_levels, spot_price)

            # Find call and put walls
            call_walls = [w for w in gamma_walls if w.wall_type == 'CALL_WALL']
            put_walls = [w for w in gamma_walls if w.wall_type == 'PUT_WALL']

            call_wall = max(call_walls, key=lambda x: x.gamma_value).strike if call_walls else None
            put_wall = max(put_walls, key=lambda x: x.gamma_value).strike if put_walls else None

            # Find gamma neutral level
            gamma_neutral = self.find_gamma_neutral(gamma_levels, spot_price)

            # Detect flip
            flip_signal = self.detect_flip(net_gex, gamma_neutral, spot_price)

            # Determine GEX regime
            if net_gex > 0:
                gex_regime = 'POSITIVE'
                vol_expectation = 'LOW' if net_gex > 1000000 else 'MODERATE'
            elif net_gex < 0:
                gex_regime = 'NEGATIVE'
                vol_expectation = 'HIGH' if net_gex < -1000000 else 'MODERATE'
            else:
                gex_regime = 'NEUTRAL'
                vol_expectation = 'MODERATE'

            # Check for flip - explosive volatility
            if flip_signal and flip_signal.signal_type in ['POSITIVE_TO_NEGATIVE', 'NEGATIVE_TO_POSITIVE']:
                vol_expectation = 'EXPLOSIVE'

            # Calculate score and bias
            score, trade_bias = self.calculate_score(net_gex, flip_signal, gamma_walls, spot_price)

            # Build description
            if flip_signal:
                description = flip_signal.trade_implications
            elif gex_regime == 'POSITIVE':
                description = f"Positive GEX regime - mean reversion expected. Call wall at {call_wall:.0f}, Put wall at {put_wall:.0f}" if call_wall and put_wall else "Positive GEX - expect range-bound trading"
            else:
                description = f"Negative GEX regime - trending moves expected. Flip level at {gamma_neutral:.0f}" if gamma_neutral else "Negative GEX - expect volatility"

            return GammaAnalysis(
                net_gex=net_gex,
                gex_regime=gex_regime,
                flip_signal=flip_signal,
                gamma_walls=gamma_walls,
                call_wall=call_wall,
                put_wall=put_wall,
                gamma_neutral_level=gamma_neutral,
                volatility_expectation=vol_expectation,
                score=score,
                trade_bias=trade_bias,
                description=description
            )

        except Exception as e:
            st.error(f"Gamma Analysis error: {e}")
            return None


# Singleton instance
_gamma_detector = None


def get_gamma_detector() -> GammaWallFlipDetector:
    """Get singleton Gamma Wall Flip Detector instance"""
    global _gamma_detector
    if _gamma_detector is None:
        _gamma_detector = GammaWallFlipDetector()
    return _gamma_detector


def analyze_gamma_flip(merged_df: pd.DataFrame = None, spot_price: float = None) -> Optional[GammaAnalysis]:
    """
    Convenience function to analyze gamma and detect flips

    Usage:
        from src.gamma_wall_flip import analyze_gamma_flip

        result = analyze_gamma_flip()
        if result:
            print(f"GEX Regime: {result.gex_regime}")
            print(f"Flip Signal: {result.flip_signal}")
            print(f"Call Wall: {result.call_wall}")
            print(f"Put Wall: {result.put_wall}")
    """
    detector = get_gamma_detector()
    return detector.analyze(merged_df, spot_price)
