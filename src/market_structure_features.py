"""
Market Structure Features Schema
================================
Defines the feature schema for structure-based market analysis.

Philosophy: Markets move AFTER structures are built.
This module captures PRE-MOVE features, not signals.

Features are organized by category:
1. Price Structure
2. Volume/OI Structure
3. Delta/Flow Structure
4. Volatility Structure
5. Market Depth Structure
6. Derived Structure Indicators
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS - Market Structure States
# =============================================================================

class MarketStructure(Enum):
    """High-level market structure states"""
    ACCUMULATION = "ACCUMULATION"       # Before up move - smart money buying
    DISTRIBUTION = "DISTRIBUTION"       # Before down move - smart money selling
    COMPRESSION = "COMPRESSION"         # Spring loading - explosion imminent
    EXPANSION = "EXPANSION"             # Move in progress - late entry
    MANIPULATION = "MANIPULATION"       # Liquidity engineering - fake moves
    NEUTRAL = "NEUTRAL"                 # No clear structure
    TRANSITION = "TRANSITION"           # Moving between structures


class StructurePhase(Enum):
    """Phase within a structure"""
    EARLY = "EARLY"           # Structure just forming
    DEVELOPING = "DEVELOPING" # Structure building
    MATURE = "MATURE"         # Structure complete, move imminent
    BREAKING = "BREAKING"     # Structure breaking down


class ExpectedOutcome(Enum):
    """Possible outcomes from current structure"""
    EXPANSION_UP = "EXPANSION_UP"
    EXPANSION_DOWN = "EXPANSION_DOWN"
    FAKE_BREAK_UP = "FAKE_BREAK_UP"
    FAKE_BREAK_DOWN = "FAKE_BREAK_DOWN"
    CONTINUED_RANGE = "CONTINUED_RANGE"
    SL_HUNT_ABOVE = "SL_HUNT_ABOVE"
    SL_HUNT_BELOW = "SL_HUNT_BELOW"
    GAMMA_SNAP = "GAMMA_SNAP"
    NO_MOVE = "NO_MOVE"


# =============================================================================
# DATACLASSES - Feature Schema
# =============================================================================

@dataclass
class PriceStructureFeatures:
    """
    Price-based structure features
    Captures the "shape" of recent price action
    """
    # Range Analysis
    price_range: float = 0.0                    # High - Low of period
    price_range_atr_ratio: float = 0.0          # Range / ATR (compression indicator)
    range_percentile: float = 50.0              # Where current range sits historically

    # Close Location Value (CLV)
    clv: float = 0.5                            # (Close - Low) / (High - Low)
    clv_trend: float = 0.0                      # CLV change over N periods

    # Wick Analysis
    upper_wick_ratio: float = 0.0               # Upper wick / Total range
    lower_wick_ratio: float = 0.0               # Lower wick / Total range
    wick_imbalance: float = 0.0                 # (Upper - Lower) / Total range
    rejection_score: float = 0.0                # How much price rejected from highs/lows

    # Trend Micro-Structure
    higher_highs_count: int = 0                 # Consecutive higher highs
    lower_lows_count: int = 0                   # Consecutive lower lows
    equal_highs_count: int = 0                  # Equal highs (liquidity target)
    equal_lows_count: int = 0                   # Equal lows (liquidity target)

    # Price vs Key Levels
    distance_to_resistance_pct: float = 0.0     # % distance to nearest resistance
    distance_to_support_pct: float = 0.0        # % distance to nearest support
    inside_range: bool = False                  # Price inside previous range

    # Momentum
    price_momentum_5: float = 0.0               # 5-period price change %
    price_momentum_20: float = 0.0              # 20-period price change %
    momentum_divergence: bool = False           # Short vs long momentum diverging


@dataclass
class VolumeOIStructureFeatures:
    """
    Volume and Open Interest structure features
    Key for detecting accumulation/distribution
    """
    # Volume Analysis
    volume_ratio: float = 1.0                   # Current volume / SMA(20)
    volume_trend: float = 0.0                   # Volume slope over N periods
    volume_price_divergence: bool = False       # Volume up but price flat
    volume_climax: bool = False                 # Unusually high volume
    volume_dry_up: bool = False                 # Unusually low volume

    # OI Analysis
    oi_change_pct: float = 0.0                  # OI change %
    oi_price_divergence: bool = False           # OI up but price flat (accumulation)
    oi_slope: float = 0.0                       # Rate of OI change
    oi_buildup_score: float = 0.0               # How much OI is building

    # OI Distribution
    call_oi_concentration: float = 0.0          # OI concentration in calls
    put_oi_concentration: float = 0.0           # OI concentration in puts
    oi_pcr: float = 1.0                         # Put/Call OI ratio
    oi_pcr_change: float = 0.0                  # PCR change from previous

    # Smart Money Signatures
    large_oi_additions: int = 0                 # Count of large OI additions
    oi_unwinding: bool = False                  # OI decreasing (positions closing)


@dataclass
class DeltaFlowStructureFeatures:
    """
    Order flow and delta structure features
    Reveals buying/selling pressure
    """
    # CVD (Cumulative Volume Delta)
    cvd_value: float = 0.0                      # Current CVD
    cvd_slope: float = 0.0                      # CVD trend
    cvd_price_divergence: bool = False          # CVD and price diverging

    # Delta Analysis
    delta_imbalance: float = 0.0                # Buy vs Sell imbalance
    delta_absorption: bool = False              # High delta but price not moving
    delta_exhaustion: bool = False              # Delta weakening after move

    # Order Flow
    orderflow_strength: float = 0.0             # Strength of order flow
    aggressive_buyers: float = 0.0              # Aggressive buy ratio
    aggressive_sellers: float = 0.0             # Aggressive sell ratio

    # Institutional Signatures
    institutional_sweep: bool = False           # Large sweep detected
    smart_money_direction: str = "NEUTRAL"      # LONG/SHORT/NEUTRAL
    block_trade_bias: str = "NEUTRAL"           # Accumulating/Distributing


@dataclass
class VolatilityStructureFeatures:
    """
    Volatility structure features
    Compression = explosion incoming
    """
    # ATR Analysis
    atr_current: float = 0.0                    # Current ATR
    atr_sma: float = 0.0                        # ATR moving average
    atr_ratio: float = 1.0                      # Current / SMA
    atr_percentile: float = 50.0                # Historical percentile (0-100)

    # Compression Detection
    compression_score: float = 0.0              # How "spring loaded" (0-100)
    compression_duration: int = 0               # How long compressed
    bollinger_squeeze: bool = False             # Bollinger bands squeezing

    # VIX Analysis
    vix_level: float = 15.0                     # Current VIX
    vix_percentile: float = 50.0                # VIX percentile
    vix_term_structure: str = "NORMAL"          # Contango/Backwardation

    # IV Analysis
    iv_percentile: float = 50.0                 # IV rank
    iv_rv_ratio: float = 1.0                    # IV vs Realized Vol
    iv_skew: float = 0.0                        # Put/Call IV skew


@dataclass
class MarketDepthStructureFeatures:
    """
    Market depth and order book features
    Detects manipulation and spoofing
    """
    # Bid/Ask Analysis
    bid_ask_imbalance: float = 0.0              # (Bid-Ask)/(Bid+Ask)
    bid_depth: float = 0.0                      # Total bid quantity
    ask_depth: float = 0.0                      # Total ask quantity
    spread_ratio: float = 0.0                   # Spread / Price

    # Order Book Dynamics
    order_variance: float = 0.0                 # Variance in order sizes
    large_orders_ratio: float = 0.0             # Large orders / Total orders
    cancel_rate: float = 0.0                    # Order cancellation rate

    # Spoofing Detection
    spoof_score: float = 0.0                    # Likelihood of spoofing (0-100)
    phantom_liquidity: bool = False             # Large orders that disappear

    # Walls
    bid_wall_detected: bool = False             # Large bid wall present
    ask_wall_detected: bool = False             # Large ask wall present
    wall_distance: float = 0.0                  # Distance to nearest wall


@dataclass
class GammaStructureFeatures:
    """
    Gamma and options structure features
    Key for expiry and snap moves
    """
    # GEX (Gamma Exposure)
    net_gex: float = 0.0                        # Net gamma exposure
    gex_regime: str = "NEUTRAL"                 # Positive/Negative gamma
    gex_flip_distance: float = 0.0              # Distance to gamma flip level

    # Gamma Walls
    call_wall_strike: float = 0.0               # Highest call OI strike
    put_wall_strike: float = 0.0                # Highest put OI strike
    wall_spread: float = 0.0                    # Distance between walls

    # Pin Analysis
    max_pain: float = 0.0                       # Max pain level
    max_pain_distance_pct: float = 0.0          # Distance to max pain
    pin_probability: float = 0.0                # Likelihood of pinning

    # Expiry
    days_to_expiry: float = 7.0                 # Days until expiry
    is_expiry_day: bool = False                 # Is today expiry?
    is_expiry_week: bool = False                # Is this expiry week?


@dataclass
class DerivedStructureIndicators:
    """
    Derived indicators that combine multiple features
    These are the "structure scores"
    """
    # Structure Scores (0-100)
    accumulation_score: float = 0.0             # Likelihood of accumulation
    distribution_score: float = 0.0             # Likelihood of distribution
    compression_score: float = 0.0              # Likelihood of compression
    manipulation_score: float = 0.0             # Likelihood of manipulation

    # Transition Indicators
    structure_maturity: float = 0.0             # How mature is current structure
    breakout_imminence: float = 0.0             # How close to breakout

    # Risk Indicators
    sl_hunt_probability_above: float = 0.0      # SL hunt above probability
    sl_hunt_probability_below: float = 0.0      # SL hunt below probability
    fake_breakout_probability: float = 0.0      # Fake breakout probability

    # Quality Indicators
    structure_clarity: float = 0.0              # How clear is the structure
    signal_noise_ratio: float = 0.0             # Signal quality


@dataclass
class MarketStructureSnapshot:
    """
    Complete market structure snapshot at a point in time
    This is the main output of the feature extractor
    """
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "NIFTY"
    spot_price: float = 0.0

    # Structure Classification
    primary_structure: MarketStructure = MarketStructure.NEUTRAL
    structure_confidence: float = 0.0           # 0-100
    structure_phase: StructurePhase = StructurePhase.EARLY

    # Feature Groups
    price_features: PriceStructureFeatures = field(default_factory=PriceStructureFeatures)
    volume_oi_features: VolumeOIStructureFeatures = field(default_factory=VolumeOIStructureFeatures)
    delta_flow_features: DeltaFlowStructureFeatures = field(default_factory=DeltaFlowStructureFeatures)
    volatility_features: VolatilityStructureFeatures = field(default_factory=VolatilityStructureFeatures)
    market_depth_features: MarketDepthStructureFeatures = field(default_factory=MarketDepthStructureFeatures)
    gamma_features: GammaStructureFeatures = field(default_factory=GammaStructureFeatures)
    derived_indicators: DerivedStructureIndicators = field(default_factory=DerivedStructureIndicators)

    # Expected Outcomes (from probability engine)
    outcome_probabilities: Dict[str, float] = field(default_factory=dict)
    primary_expected_outcome: ExpectedOutcome = ExpectedOutcome.NO_MOVE
    outcome_confidence: float = 0.0

    # Historical Pattern Match
    pattern_similarity: float = 0.0             # 0-1 similarity to historical
    pattern_occurrences: int = 0                # How many similar patterns found
    pattern_recency_weight: float = 1.0         # Weight for recent patterns

    # Actionable Insights
    key_levels: Dict[str, float] = field(default_factory=dict)
    risk_zones: List[Dict] = field(default_factory=list)
    recommended_action: str = "WAIT"
    action_reasoning: List[str] = field(default_factory=list)


# =============================================================================
# FEATURE EXTRACTOR CLASS
# =============================================================================

class MarketStructureFeatureExtractor:
    """
    Extracts market structure features from raw data

    This is the main interface for feature extraction.
    Takes DataFrame + option chain + market depth → MarketStructureSnapshot
    """

    def __init__(self):
        """Initialize the feature extractor"""
        self.lookback_periods = {
            'short': 5,
            'medium': 20,
            'long': 50
        }

    def extract_features(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict] = None,
        market_depth: Optional[Dict] = None,
        vix_data: Optional[Dict] = None,
        spot_price: Optional[float] = None,
        symbol: str = "NIFTY"
    ) -> MarketStructureSnapshot:
        """
        Extract complete market structure features

        Args:
            df: OHLCV DataFrame
            option_chain: Option chain data
            market_depth: Market depth data
            vix_data: VIX data
            spot_price: Current spot price
            symbol: Symbol name

        Returns:
            MarketStructureSnapshot with all features
        """
        if df is None or len(df) < 20:
            return MarketStructureSnapshot(symbol=symbol)

        # Ensure we have spot price
        if spot_price is None or spot_price <= 0:
            close_col = 'Close' if 'Close' in df.columns else 'close'
            spot_price = float(df[close_col].iloc[-1])

        # Extract all feature groups
        price_features = self._extract_price_features(df, spot_price)
        volume_oi_features = self._extract_volume_oi_features(df, option_chain)
        delta_flow_features = self._extract_delta_flow_features(df)
        volatility_features = self._extract_volatility_features(df, vix_data)
        market_depth_features = self._extract_market_depth_features(market_depth)
        gamma_features = self._extract_gamma_features(option_chain, spot_price)

        # Calculate derived indicators
        derived_indicators = self._calculate_derived_indicators(
            price_features,
            volume_oi_features,
            delta_flow_features,
            volatility_features,
            market_depth_features,
            gamma_features
        )

        # Classify structure
        primary_structure, structure_confidence = self._classify_structure(
            price_features,
            volume_oi_features,
            delta_flow_features,
            volatility_features,
            derived_indicators
        )

        # Determine structure phase
        structure_phase = self._determine_phase(
            primary_structure,
            derived_indicators
        )

        return MarketStructureSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            spot_price=spot_price,
            primary_structure=primary_structure,
            structure_confidence=structure_confidence,
            structure_phase=structure_phase,
            price_features=price_features,
            volume_oi_features=volume_oi_features,
            delta_flow_features=delta_flow_features,
            volatility_features=volatility_features,
            market_depth_features=market_depth_features,
            gamma_features=gamma_features,
            derived_indicators=derived_indicators
        )

    def extract_snapshot(
        self,
        ohlc_df: pd.DataFrame,
        option_data: Optional[Dict] = None,
        spot_price: Optional[float] = None,
        symbol: str = "NIFTY"
    ) -> MarketStructureSnapshot:
        """
        Alias for extract_features - for UI compatibility

        Args:
            ohlc_df: OHLCV DataFrame
            option_data: Option chain and related data
            spot_price: Current spot price
            symbol: Symbol name

        Returns:
            MarketStructureSnapshot with all features
        """
        # Extract option chain and market depth from option_data if provided
        option_chain = None
        market_depth = None

        if option_data:
            if 'merged_df' in option_data:
                option_chain = option_data
            elif isinstance(option_data, dict):
                option_chain = option_data
            market_depth = option_data.get('market_depth')

        return self.extract_features(
            df=ohlc_df,
            option_chain=option_chain,
            market_depth=market_depth,
            spot_price=spot_price,
            symbol=symbol
        )

    def _extract_price_features(
        self,
        df: pd.DataFrame,
        spot_price: float
    ) -> PriceStructureFeatures:
        """Extract price-based structure features"""
        features = PriceStructureFeatures()

        # Handle column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        try:
            current = df.iloc[-1]

            # Range Analysis
            features.price_range = current[high_col] - current[low_col]

            # ATR for ratio
            atr = (df[high_col] - df[low_col]).rolling(14).mean().iloc[-1]
            features.price_range_atr_ratio = features.price_range / atr if atr > 0 else 1.0

            # Range percentile
            ranges = df[high_col] - df[low_col]
            features.range_percentile = (ranges <= features.price_range).mean() * 100

            # CLV (Close Location Value)
            if features.price_range > 0:
                features.clv = (current[close_col] - current[low_col]) / features.price_range

            # CLV trend
            if len(df) >= 5:
                clvs = []
                for i in range(-5, 0):
                    h = df[high_col].iloc[i]
                    l = df[low_col].iloc[i]
                    c = df[close_col].iloc[i]
                    if h - l > 0:
                        clvs.append((c - l) / (h - l))
                if clvs:
                    features.clv_trend = clvs[-1] - clvs[0]

            # Wick Analysis
            if features.price_range > 0:
                upper_wick = current[high_col] - max(current[close_col], df['Open'].iloc[-1] if 'Open' in df.columns else current[close_col])
                lower_wick = min(current[close_col], df['Open'].iloc[-1] if 'Open' in df.columns else current[close_col]) - current[low_col]
                features.upper_wick_ratio = upper_wick / features.price_range
                features.lower_wick_ratio = lower_wick / features.price_range
                features.wick_imbalance = (upper_wick - lower_wick) / features.price_range
                features.rejection_score = max(features.upper_wick_ratio, features.lower_wick_ratio) * 100

            # Higher Highs / Lower Lows
            highs = df[high_col].tail(10).values
            lows = df[low_col].tail(10).values

            hh_count = 0
            ll_count = 0
            eh_count = 0
            el_count = 0

            for i in range(1, len(highs)):
                if highs[i] > highs[i-1] * 1.0001:
                    hh_count += 1
                elif abs(highs[i] - highs[i-1]) / highs[i-1] < 0.001:
                    eh_count += 1

                if lows[i] < lows[i-1] * 0.9999:
                    ll_count += 1
                elif abs(lows[i] - lows[i-1]) / lows[i-1] < 0.001:
                    el_count += 1

            features.higher_highs_count = hh_count
            features.lower_lows_count = ll_count
            features.equal_highs_count = eh_count
            features.equal_lows_count = el_count

            # Momentum
            if len(df) >= 5:
                features.price_momentum_5 = (df[close_col].iloc[-1] - df[close_col].iloc[-5]) / df[close_col].iloc[-5] * 100
            if len(df) >= 20:
                features.price_momentum_20 = (df[close_col].iloc[-1] - df[close_col].iloc[-20]) / df[close_col].iloc[-20] * 100

            # Momentum divergence
            if features.price_momentum_5 * features.price_momentum_20 < 0:
                features.momentum_divergence = True

        except Exception as e:
            logger.warning(f"Error extracting price features: {e}")

        return features

    def _extract_volume_oi_features(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict]
    ) -> VolumeOIStructureFeatures:
        """Extract volume and OI structure features"""
        features = VolumeOIStructureFeatures()

        try:
            # Volume features
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume' if 'volume' in df.columns else None

            if vol_col and len(df) >= 20:
                current_vol = df[vol_col].iloc[-1]
                vol_sma = df[vol_col].rolling(20).mean().iloc[-1]

                features.volume_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0

                # Volume trend
                vol_5 = df[vol_col].tail(5).mean()
                vol_20 = df[vol_col].tail(20).mean()
                features.volume_trend = (vol_5 - vol_20) / vol_20 if vol_20 > 0 else 0

                # Volume climax / dry up
                vol_percentile = (df[vol_col] <= current_vol).mean() * 100
                features.volume_climax = vol_percentile > 95
                features.volume_dry_up = vol_percentile < 10

                # Volume-Price divergence
                close_col = 'Close' if 'Close' in df.columns else 'close'
                price_change = abs(df[close_col].iloc[-1] - df[close_col].iloc[-5]) / df[close_col].iloc[-5]
                if features.volume_ratio > 1.5 and price_change < 0.005:
                    features.volume_price_divergence = True

            # OI features from option chain
            if option_chain:
                # Total OI
                call_oi = sum(option_chain.get('CE', {}).get('openInterest', [0]))
                put_oi = sum(option_chain.get('PE', {}).get('openInterest', [0]))

                if call_oi > 0:
                    features.oi_pcr = put_oi / call_oi

                # OI concentrations would need historical data
                # For now, estimate from current distribution
                total_oi = call_oi + put_oi
                if total_oi > 0:
                    features.call_oi_concentration = call_oi / total_oi
                    features.put_oi_concentration = put_oi / total_oi

        except Exception as e:
            logger.warning(f"Error extracting volume/OI features: {e}")

        return features

    def _extract_delta_flow_features(
        self,
        df: pd.DataFrame
    ) -> DeltaFlowStructureFeatures:
        """Extract delta and order flow features"""
        features = DeltaFlowStructureFeatures()

        try:
            # Approximate delta from price action
            close_col = 'Close' if 'Close' in df.columns else 'close'
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume' if 'volume' in df.columns else None

            if vol_col and len(df) >= 20:
                # Estimate buying/selling pressure from close position in range
                for i in range(-20, 0):
                    h = df[high_col].iloc[i]
                    l = df[low_col].iloc[i]
                    c = df[close_col].iloc[i]
                    v = df[vol_col].iloc[i]

                    if h - l > 0:
                        # If close near high = buying, near low = selling
                        buying_ratio = (c - l) / (h - l)
                        features.cvd_value += v * (buying_ratio - 0.5) * 2

                # CVD slope (recent trend)
                cvd_recent = 0
                for i in range(-5, 0):
                    h = df[high_col].iloc[i]
                    l = df[low_col].iloc[i]
                    c = df[close_col].iloc[i]
                    v = df[vol_col].iloc[i]
                    if h - l > 0:
                        buying_ratio = (c - l) / (h - l)
                        cvd_recent += v * (buying_ratio - 0.5) * 2

                features.cvd_slope = cvd_recent / 5 if cvd_recent != 0 else 0

                # Delta imbalance
                recent_closes = df[close_col].tail(5).values
                up_bars = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
                features.delta_imbalance = (up_bars - (len(recent_closes) - 1 - up_bars)) / (len(recent_closes) - 1)

                # Delta absorption detection
                price_change_5 = abs(df[close_col].iloc[-1] - df[close_col].iloc[-5]) / df[close_col].iloc[-5]
                if abs(features.cvd_slope) > 1000 and price_change_5 < 0.003:
                    features.delta_absorption = True

        except Exception as e:
            logger.warning(f"Error extracting delta flow features: {e}")

        return features

    def _extract_volatility_features(
        self,
        df: pd.DataFrame,
        vix_data: Optional[Dict]
    ) -> VolatilityStructureFeatures:
        """Extract volatility structure features"""
        features = VolatilityStructureFeatures()

        try:
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            close_col = 'Close' if 'Close' in df.columns else 'close'

            # ATR calculation
            tr = pd.concat([
                df[high_col] - df[low_col],
                abs(df[high_col] - df[close_col].shift()),
                abs(df[low_col] - df[close_col].shift())
            ], axis=1).max(axis=1)

            features.atr_current = tr.rolling(14).mean().iloc[-1]
            features.atr_sma = tr.rolling(50).mean().iloc[-1] if len(df) >= 50 else features.atr_current
            features.atr_ratio = features.atr_current / features.atr_sma if features.atr_sma > 0 else 1.0

            # ATR percentile
            atr_values = tr.rolling(14).mean()
            features.atr_percentile = (atr_values <= features.atr_current).mean() * 100

            # Compression detection
            if features.atr_percentile < 25:
                features.compression_score = (25 - features.atr_percentile) * 4  # Scale to 0-100

                # Count compression duration
                for i in range(-1, -min(50, len(df)), -1):
                    if (atr_values.iloc[:i] <= atr_values.iloc[i]).mean() * 100 < 25:
                        features.compression_duration += 1
                    else:
                        break

            # Bollinger squeeze detection
            if len(df) >= 20:
                sma20 = df[close_col].rolling(20).mean()
                std20 = df[close_col].rolling(20).std()
                bb_width = (4 * std20 / sma20).iloc[-1]
                bb_width_sma = (4 * std20 / sma20).rolling(50).mean().iloc[-1] if len(df) >= 50 else bb_width

                if bb_width < bb_width_sma * 0.7:
                    features.bollinger_squeeze = True

            # VIX features
            if vix_data:
                features.vix_level = vix_data.get('value', 15.0)
                features.vix_percentile = vix_data.get('percentile', 50.0)

        except Exception as e:
            logger.warning(f"Error extracting volatility features: {e}")

        return features

    def _extract_market_depth_features(
        self,
        market_depth: Optional[Dict]
    ) -> MarketDepthStructureFeatures:
        """Extract market depth structure features"""
        features = MarketDepthStructureFeatures()

        if not market_depth:
            return features

        try:
            # Bid/Ask imbalance
            bid_qty = market_depth.get('total_bid_qty', 0)
            ask_qty = market_depth.get('total_ask_qty', 0)

            if bid_qty + ask_qty > 0:
                features.bid_ask_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

            features.bid_depth = bid_qty
            features.ask_depth = ask_qty

            # Spread
            best_bid = market_depth.get('best_bid', 0)
            best_ask = market_depth.get('best_ask', 0)
            if best_bid > 0:
                features.spread_ratio = (best_ask - best_bid) / best_bid

            # Spoof detection (would need order book history)
            features.spoof_score = market_depth.get('spoof_score', 0)

        except Exception as e:
            logger.warning(f"Error extracting market depth features: {e}")

        return features

    def _extract_gamma_features(
        self,
        option_chain: Optional[Dict],
        spot_price: float
    ) -> GammaStructureFeatures:
        """Extract gamma and options structure features"""
        features = GammaStructureFeatures()

        if not option_chain:
            return features

        try:
            # Find max OI strikes (gamma walls)
            ce_data = option_chain.get('CE', {})
            pe_data = option_chain.get('PE', {})

            ce_oi = ce_data.get('openInterest', [])
            pe_oi = pe_data.get('openInterest', [])
            strikes = option_chain.get('strikes', [])

            if ce_oi and strikes and len(ce_oi) == len(strikes):
                max_call_idx = ce_oi.index(max(ce_oi))
                features.call_wall_strike = strikes[max_call_idx]

            if pe_oi and strikes and len(pe_oi) == len(strikes):
                max_put_idx = pe_oi.index(max(pe_oi))
                features.put_wall_strike = strikes[max_put_idx]

            # Wall spread
            if features.call_wall_strike > 0 and features.put_wall_strike > 0:
                features.wall_spread = features.call_wall_strike - features.put_wall_strike

            # Max pain
            features.max_pain = option_chain.get('max_pain', 0)
            if features.max_pain > 0 and spot_price > 0:
                features.max_pain_distance_pct = (spot_price - features.max_pain) / spot_price * 100

            # Expiry info
            features.days_to_expiry = option_chain.get('days_to_expiry', 7)
            features.is_expiry_day = features.days_to_expiry < 1
            features.is_expiry_week = features.days_to_expiry <= 5

            # Pin probability (closer to max pain on expiry = higher pin prob)
            if features.is_expiry_day and abs(features.max_pain_distance_pct) < 1:
                features.pin_probability = 80 - abs(features.max_pain_distance_pct) * 40
            elif features.is_expiry_week:
                features.pin_probability = max(0, 60 - abs(features.max_pain_distance_pct) * 20)

        except Exception as e:
            logger.warning(f"Error extracting gamma features: {e}")

        return features

    def _calculate_derived_indicators(
        self,
        price: PriceStructureFeatures,
        volume_oi: VolumeOIStructureFeatures,
        delta_flow: DeltaFlowStructureFeatures,
        volatility: VolatilityStructureFeatures,
        market_depth: MarketDepthStructureFeatures,
        gamma: GammaStructureFeatures
    ) -> DerivedStructureIndicators:
        """Calculate derived structure indicators"""
        derived = DerivedStructureIndicators()

        try:
            # ACCUMULATION SCORE
            # Price flat + OI increasing + Volume absorbed + Delta neutral/positive
            acc_score = 0
            if price.price_range_atr_ratio < 1.2:
                acc_score += 25
            if volume_oi.oi_price_divergence or volume_oi.volume_price_divergence:
                acc_score += 25
            if delta_flow.delta_absorption:
                acc_score += 25
            if delta_flow.delta_imbalance > 0:
                acc_score += 25
            derived.accumulation_score = min(acc_score, 100)

            # DISTRIBUTION SCORE
            # Price flat + OI increasing + Upper wicks + Delta negative
            dist_score = 0
            if price.price_range_atr_ratio < 1.2:
                dist_score += 25
            if volume_oi.oi_price_divergence:
                dist_score += 25
            if price.upper_wick_ratio > 0.4:
                dist_score += 25
            if delta_flow.delta_imbalance < 0:
                dist_score += 25
            derived.distribution_score = min(dist_score, 100)

            # COMPRESSION SCORE
            derived.compression_score = volatility.compression_score
            if volatility.bollinger_squeeze:
                derived.compression_score = min(derived.compression_score + 20, 100)
            if volume_oi.volume_dry_up:
                derived.compression_score = min(derived.compression_score + 15, 100)

            # MANIPULATION SCORE
            manip_score = 0
            if market_depth.spoof_score > 50:
                manip_score += 40
            if market_depth.phantom_liquidity:
                manip_score += 30
            if price.equal_highs_count >= 2 or price.equal_lows_count >= 2:
                manip_score += 30
            derived.manipulation_score = min(manip_score, 100)

            # STRUCTURE MATURITY
            # How long has structure been building?
            if volatility.compression_duration > 20:
                derived.structure_maturity = 100
            elif volatility.compression_duration > 10:
                derived.structure_maturity = 70
            elif volatility.compression_duration > 5:
                derived.structure_maturity = 40
            else:
                derived.structure_maturity = volatility.compression_duration * 8

            # BREAKOUT IMMINENCE
            if derived.compression_score > 70 and derived.structure_maturity > 60:
                derived.breakout_imminence = min((derived.compression_score + derived.structure_maturity) / 2, 100)

            # SL HUNT PROBABILITIES
            if price.equal_lows_count >= 2:
                derived.sl_hunt_probability_below = 60 + (price.equal_lows_count - 2) * 10
            if price.equal_highs_count >= 2:
                derived.sl_hunt_probability_above = 60 + (price.equal_highs_count - 2) * 10

            # STRUCTURE CLARITY
            # How clear/confident is the structure?
            max_score = max(derived.accumulation_score, derived.distribution_score, derived.compression_score)
            second_max = sorted([derived.accumulation_score, derived.distribution_score, derived.compression_score])[-2]
            derived.structure_clarity = max_score - second_max

            # SIGNAL NOISE RATIO
            derived.signal_noise_ratio = derived.structure_clarity / 100 if derived.structure_clarity > 0 else 0

        except Exception as e:
            logger.warning(f"Error calculating derived indicators: {e}")

        return derived

    def _classify_structure(
        self,
        price: PriceStructureFeatures,
        volume_oi: VolumeOIStructureFeatures,
        delta_flow: DeltaFlowStructureFeatures,
        volatility: VolatilityStructureFeatures,
        derived: DerivedStructureIndicators
    ) -> Tuple[MarketStructure, float]:
        """Classify the primary market structure"""

        scores = {
            MarketStructure.ACCUMULATION: derived.accumulation_score,
            MarketStructure.DISTRIBUTION: derived.distribution_score,
            MarketStructure.COMPRESSION: derived.compression_score,
            MarketStructure.MANIPULATION: derived.manipulation_score,
        }

        # Check for EXPANSION (ATR expanding)
        if volatility.atr_ratio > 1.5:
            scores[MarketStructure.EXPANSION] = volatility.atr_ratio * 30

        # Find highest score
        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        # If no clear structure, mark as NEUTRAL
        if confidence < 30:
            return MarketStructure.NEUTRAL, confidence

        # Check for TRANSITION (two structures close)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 15:
            return MarketStructure.TRANSITION, confidence

        return primary, confidence

    def _determine_phase(
        self,
        structure: MarketStructure,
        derived: DerivedStructureIndicators
    ) -> StructurePhase:
        """Determine the phase within the structure"""

        if derived.structure_maturity > 80:
            return StructurePhase.MATURE
        elif derived.structure_maturity > 50:
            return StructurePhase.DEVELOPING
        elif derived.breakout_imminence > 70:
            return StructurePhase.BREAKING
        else:
            return StructurePhase.EARLY


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def snapshot_to_dict(snapshot: MarketStructureSnapshot) -> Dict:
    """Convert snapshot to dictionary for storage/display"""
    return {
        'timestamp': snapshot.timestamp.isoformat(),
        'symbol': snapshot.symbol,
        'spot_price': snapshot.spot_price,
        'primary_structure': snapshot.primary_structure.value,
        'structure_confidence': snapshot.structure_confidence,
        'structure_phase': snapshot.structure_phase.value,
        'price_features': {
            'range_atr_ratio': snapshot.price_features.price_range_atr_ratio,
            'clv': snapshot.price_features.clv,
            'wick_imbalance': snapshot.price_features.wick_imbalance,
            'equal_highs': snapshot.price_features.equal_highs_count,
            'equal_lows': snapshot.price_features.equal_lows_count,
            'momentum_5': snapshot.price_features.price_momentum_5,
            'momentum_20': snapshot.price_features.price_momentum_20,
        },
        'volume_oi_features': {
            'volume_ratio': snapshot.volume_oi_features.volume_ratio,
            'oi_pcr': snapshot.volume_oi_features.oi_pcr,
            'volume_price_divergence': snapshot.volume_oi_features.volume_price_divergence,
        },
        'volatility_features': {
            'atr_percentile': snapshot.volatility_features.atr_percentile,
            'compression_score': snapshot.volatility_features.compression_score,
            'compression_duration': snapshot.volatility_features.compression_duration,
            'bollinger_squeeze': snapshot.volatility_features.bollinger_squeeze,
        },
        'derived_indicators': {
            'accumulation_score': snapshot.derived_indicators.accumulation_score,
            'distribution_score': snapshot.derived_indicators.distribution_score,
            'compression_score': snapshot.derived_indicators.compression_score,
            'manipulation_score': snapshot.derived_indicators.manipulation_score,
            'structure_maturity': snapshot.derived_indicators.structure_maturity,
            'breakout_imminence': snapshot.derived_indicators.breakout_imminence,
            'sl_hunt_above': snapshot.derived_indicators.sl_hunt_probability_above,
            'sl_hunt_below': snapshot.derived_indicators.sl_hunt_probability_below,
        },
        'outcome_probabilities': snapshot.outcome_probabilities,
        'primary_expected_outcome': snapshot.primary_expected_outcome.value,
    }
