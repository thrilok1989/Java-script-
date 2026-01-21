"""
Comprehensive ICT Indicator Module
Implements Order Blocks, Fair Value Gaps, Supply/Demand Zones, and Volume Profile
Based on ELFNaAan pro vip v2 Pine Script indicator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class OrderBlock:
    """Order Block data structure"""
    bar_high: float
    bar_low: float
    bar_index: int
    bar_time: pd.Timestamp
    bias: int  # 1 = bullish, -1 = bearish
    is_internal: bool
    is_mitigated: bool = False


@dataclass
class FairValueGap:
    """Fair Value Gap data structure"""
    top: float
    bottom: float
    start_index: int
    gap_type: str  # 'bullish' or 'bearish'
    is_mitigated: bool = False


@dataclass
class SupplyDemandZone:
    """Supply/Demand Zone data structure"""
    top: float
    bottom: float
    poi: float  # Point of Interest
    left_index: int
    right_index: int
    zone_type: str  # 'supply' or 'demand'
    is_broken: bool = False


@dataclass
class VolumeProfileRow:
    """Volume Profile Row data structure"""
    top: float
    bottom: float
    total_volume: float
    bull_volume: float
    bear_volume: float


class ComprehensiveICTIndicator:
    """
    Comprehensive ICT (Inner Circle Trader) Indicator

    Features:
    1. Order Blocks (Internal + Swing)
    2. Fair Value Gaps (FVG)
    3. Supply/Demand Zones
    4. Volume Profile with POC
    """

    def __init__(self,
                 # Order Block parameters
                 swing_length: int = 50,
                 internal_length: int = 5,
                 swing_ob_size: int = 10,
                 internal_ob_size: int = 10,
                 # FVG parameters
                 max_fvgs: int = 10,
                 # Supply/Demand parameters
                 sd_swing_length: int = 10,
                 sd_history: int = 20,
                 # Volume Profile parameters
                 vp_analyze_bars: int = 200,
                 vp_row_count: int = 30):
        """
        Initialize Comprehensive ICT Indicator

        Args:
            swing_length: Swing length for order blocks
            internal_length: Internal length for order blocks
            swing_ob_size: Maximum number of swing order blocks to track
            internal_ob_size: Maximum number of internal order blocks to track
            max_fvgs: Maximum number of FVGs to track
            sd_swing_length: Swing length for supply/demand zones
            sd_history: Number of supply/demand zones to keep
            vp_analyze_bars: Number of bars to analyze for volume profile
            vp_row_count: Number of rows in volume profile
        """
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.swing_ob_size = swing_ob_size
        self.internal_ob_size = internal_ob_size
        self.max_fvgs = max_fvgs
        self.sd_swing_length = sd_swing_length
        self.sd_history = sd_history
        self.vp_analyze_bars = vp_analyze_bars
        self.vp_row_count = vp_row_count

        # Storage
        self.swing_order_blocks: deque = deque(maxlen=swing_ob_size)
        self.internal_order_blocks: deque = deque(maxlen=internal_ob_size)
        self.fvgs: deque = deque(maxlen=max_fvgs)
        self.supply_zones: deque = deque(maxlen=sd_history)
        self.demand_zones: deque = deque(maxlen=sd_history)
        self.volume_profile: List[VolumeProfileRow] = []
        self.poc_price: float = 0.0

    # =========================================================================
    # ORDER BLOCKS DETECTION
    # =========================================================================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 200) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _find_swing_points(self, df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        # Initialize swing points
        swing_highs = pd.Series(np.nan, index=df.index)
        swing_lows = pd.Series(np.nan, index=df.index)

        # Find swing highs
        for i in range(length, len(df) - length):
            is_swing_high = True
            for j in range(i - length, i + length + 1):
                if j != i and high.iloc[j] >= high.iloc[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.iloc[i] = high.iloc[i]

        # Find swing lows
        for i in range(length, len(df) - length):
            is_swing_low = True
            for j in range(i - length, i + length + 1):
                if j != i and low.iloc[j] <= low.iloc[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.iloc[i] = low.iloc[i]

        return swing_highs, swing_lows

    def _detect_order_blocks(self, df: pd.DataFrame, is_internal: bool = False) -> List[OrderBlock]:
        """Detect order blocks"""
        length = self.internal_length if is_internal else self.swing_length
        swing_highs, swing_lows = self._find_swing_points(df, length)

        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        order_blocks = []

        # Detect bearish order blocks (at swing highs)
        for i in range(len(df)):
            if not pd.isna(swing_highs.iloc[i]):
                # Find the candle before the swing high with highest high
                start_idx = max(0, i - length)
                ob_idx = high.iloc[start_idx:i].idxmax()

                ob = OrderBlock(
                    bar_high=high.loc[ob_idx],
                    bar_low=low.loc[ob_idx],
                    bar_index=df.index.get_loc(ob_idx),
                    bar_time=ob_idx,
                    bias=-1,  # bearish
                    is_internal=is_internal
                )
                order_blocks.append(ob)

        # Detect bullish order blocks (at swing lows)
        for i in range(len(df)):
            if not pd.isna(swing_lows.iloc[i]):
                # Find the candle before the swing low with lowest low
                start_idx = max(0, i - length)
                ob_idx = low.iloc[start_idx:i].idxmin()

                ob = OrderBlock(
                    bar_high=high.loc[ob_idx],
                    bar_low=low.loc[ob_idx],
                    bar_index=df.index.get_loc(ob_idx),
                    bar_time=ob_idx,
                    bias=1,  # bullish
                    is_internal=is_internal
                )
                order_blocks.append(ob)

        return order_blocks

    def _check_ob_mitigation(self, df: pd.DataFrame, order_blocks: List[OrderBlock]) -> None:
        """Check if order blocks are mitigated"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        latest_high = high.iloc[-1]
        latest_low = low.iloc[-1]

        for ob in order_blocks:
            if ob.bias == -1:  # bearish
                if latest_high > ob.bar_high:
                    ob.is_mitigated = True
            elif ob.bias == 1:  # bullish
                if latest_low < ob.bar_low:
                    ob.is_mitigated = True

    # =========================================================================
    # FAIR VALUE GAPS (FVG) DETECTION
    # =========================================================================

    def _detect_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detect Fair Value Gaps"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        fvgs = []

        # Need at least 4 bars
        if len(df) < 4:
            return fvgs

        # Check for bullish FVG: high[3] < low[1]
        for i in range(3, len(df)):
            if high.iloc[i-3] < low.iloc[i-1]:
                fvg = FairValueGap(
                    top=low.iloc[i-1],
                    bottom=high.iloc[i-3],
                    start_index=i-2,
                    gap_type='bullish'
                )
                fvgs.append(fvg)

        # Check for bearish FVG: low[3] > high[1]
        for i in range(3, len(df)):
            if low.iloc[i-3] > high.iloc[i-1]:
                fvg = FairValueGap(
                    top=low.iloc[i-3],
                    bottom=high.iloc[i-1],
                    start_index=i-2,
                    gap_type='bearish'
                )
                fvgs.append(fvg)

        return fvgs

    def _check_fvg_mitigation(self, df: pd.DataFrame, fvgs: List[FairValueGap]) -> None:
        """Check if FVGs are mitigated (filled)"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        latest_high = high.iloc[-1]
        latest_low = low.iloc[-1]

        for fvg in fvgs:
            if fvg.gap_type == 'bullish':
                # Bullish FVG is fully mitigated when price goes below bottom
                if latest_low <= fvg.bottom:
                    fvg.is_mitigated = True
            elif fvg.gap_type == 'bearish':
                # Bearish FVG is fully mitigated when price goes above top
                if latest_high >= fvg.top:
                    fvg.is_mitigated = True

    # =========================================================================
    # SUPPLY/DEMAND ZONES DETECTION
    # =========================================================================

    def _detect_supply_demand_zones(self, df: pd.DataFrame) -> Tuple[List[SupplyDemandZone], List[SupplyDemandZone]]:
        """Detect supply and demand zones"""
        atr = self._calculate_atr(df, 50)
        swing_highs, swing_lows = self._find_swing_points(df, self.sd_swing_length)

        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        supply_zones = []
        demand_zones = []

        atr_buffer = atr.iloc[-1] * 2.5 / 10 if not pd.isna(atr.iloc[-1]) else 0

        # Detect supply zones (at swing highs)
        for i in range(len(df)):
            if not pd.isna(swing_highs.iloc[i]):
                zone_top = high.iloc[i]
                zone_bottom = zone_top - atr_buffer
                poi = (zone_top + zone_bottom) / 2

                supply_zone = SupplyDemandZone(
                    top=zone_top,
                    bottom=zone_bottom,
                    poi=poi,
                    left_index=i,
                    right_index=len(df) - 1,
                    zone_type='supply'
                )
                supply_zones.append(supply_zone)

        # Detect demand zones (at swing lows)
        for i in range(len(df)):
            if not pd.isna(swing_lows.iloc[i]):
                zone_bottom = low.iloc[i]
                zone_top = zone_bottom + atr_buffer
                poi = (zone_top + zone_bottom) / 2

                demand_zone = SupplyDemandZone(
                    top=zone_top,
                    bottom=zone_bottom,
                    poi=poi,
                    left_index=i,
                    right_index=len(df) - 1,
                    zone_type='demand'
                )
                demand_zones.append(demand_zone)

        return supply_zones, demand_zones

    def _check_zone_breakout(self, df: pd.DataFrame, zones: List[SupplyDemandZone]) -> None:
        """Check if supply/demand zones are broken"""
        close = df['close'] if 'close' in df.columns else df['Close']
        latest_close = close.iloc[-1]

        for zone in zones:
            if zone.zone_type == 'supply':
                if latest_close >= zone.top:
                    zone.is_broken = True
            elif zone.zone_type == 'demand':
                if latest_close <= zone.bottom:
                    zone.is_broken = True

    # =========================================================================
    # VOLUME PROFILE DETECTION
    # =========================================================================

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Tuple[List[VolumeProfileRow], float]:
        """Calculate volume profile and POC"""
        if len(df) < self.vp_analyze_bars:
            analyze_bars = len(df)
        else:
            analyze_bars = self.vp_analyze_bars

        # Get recent data
        recent_df = df.iloc[-analyze_bars:]

        high = recent_df['high'] if 'high' in recent_df.columns else recent_df['High']
        low = recent_df['low'] if 'low' in recent_df.columns else recent_df['Low']
        close = recent_df['close'] if 'close' in recent_df.columns else recent_df['Close']
        open_ = recent_df['open'] if 'open' in recent_df.columns else recent_df['Open']
        volume = recent_df['volume'] if 'volume' in recent_df.columns else recent_df['Volume']

        # Calculate price range
        price_high = high.max()
        price_low = low.min()
        price_range = price_high - price_low

        if price_range == 0:
            return [], 0.0

        # Create price levels
        step = price_range / self.vp_row_count

        rows = []
        for i in range(self.vp_row_count):
            row_top = price_high - (step * i)
            row_bottom = row_top - step

            row = VolumeProfileRow(
                top=row_top,
                bottom=row_bottom,
                total_volume=0.0,
                bull_volume=0.0,
                bear_volume=0.0
            )
            rows.append(row)

        # Distribute volume to rows
        for idx in range(len(recent_df)):
            bar_high = high.iloc[idx]
            bar_low = low.iloc[idx]
            bar_close = close.iloc[idx]
            bar_open = open_.iloc[idx]
            bar_volume = volume.iloc[idx]

            is_bullish = bar_close > bar_open

            # Find which rows this bar intersects
            for row in rows:
                # Check intersection
                if bar_low <= row.top and bar_high >= row.bottom:
                    row.total_volume += bar_volume
                    if is_bullish:
                        row.bull_volume += bar_volume
                    else:
                        row.bear_volume += bar_volume

        # Find POC (Point of Control) - highest volume row
        poc_price = 0.0
        max_volume = 0.0
        for row in rows:
            if row.total_volume > max_volume:
                max_volume = row.total_volume
                poc_price = (row.top + row.bottom) / 2

        return rows, poc_price

    # =========================================================================
    # MAIN CALCULATION METHOD
    # =========================================================================

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary containing all indicator data
        """
        # Clear previous data
        self.swing_order_blocks.clear()
        self.internal_order_blocks.clear()
        self.fvgs.clear()
        self.supply_zones.clear()
        self.demand_zones.clear()

        # Detect order blocks
        swing_obs = self._detect_order_blocks(df, is_internal=False)
        internal_obs = self._detect_order_blocks(df, is_internal=True)

        # Check mitigation
        self._check_ob_mitigation(df, swing_obs)
        self._check_ob_mitigation(df, internal_obs)

        # Keep only non-mitigated order blocks
        for ob in swing_obs:
            if not ob.is_mitigated:
                self.swing_order_blocks.append(ob)

        for ob in internal_obs:
            if not ob.is_mitigated:
                self.internal_order_blocks.append(ob)

        # Detect FVGs
        all_fvgs = self._detect_fvgs(df)
        self._check_fvg_mitigation(df, all_fvgs)

        # Keep only non-mitigated FVGs
        for fvg in all_fvgs:
            if not fvg.is_mitigated:
                self.fvgs.append(fvg)

        # Detect supply/demand zones
        supply_zones, demand_zones = self._detect_supply_demand_zones(df)
        self._check_zone_breakout(df, supply_zones)
        self._check_zone_breakout(df, demand_zones)

        # Keep only non-broken zones
        for zone in supply_zones:
            if not zone.is_broken:
                self.supply_zones.append(zone)

        for zone in demand_zones:
            if not zone.is_broken:
                self.demand_zones.append(zone)

        # Calculate volume profile
        self.volume_profile, self.poc_price = self._calculate_volume_profile(df)

        # Generate signals
        signals = self._generate_signals(df)

        return {
            'swing_order_blocks': list(self.swing_order_blocks),
            'internal_order_blocks': list(self.internal_order_blocks),
            'fvgs': list(self.fvgs),
            'supply_zones': list(self.supply_zones),
            'demand_zones': list(self.demand_zones),
            'volume_profile': self.volume_profile,
            'poc_price': self.poc_price,
            'signals': signals
        }

    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on all indicators"""
        close = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]

        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'overall_bias': 'NEUTRAL'
        }

        bullish_count = 0
        bearish_count = 0

        # Check order blocks
        for ob in self.swing_order_blocks:
            if ob.bias == 1 and close >= ob.bar_low and close <= ob.bar_high:
                signals['bullish_signals'].append({
                    'type': 'Bullish Order Block',
                    'price': f"{ob.bar_low:.2f} - {ob.bar_high:.2f}",
                    'level': 'Swing'
                })
                bullish_count += 2

        for ob in self.internal_order_blocks:
            if ob.bias == 1 and close >= ob.bar_low and close <= ob.bar_high:
                signals['bullish_signals'].append({
                    'type': 'Bullish Order Block',
                    'price': f"{ob.bar_low:.2f} - {ob.bar_high:.2f}",
                    'level': 'Internal'
                })
                bullish_count += 1

        for ob in self.swing_order_blocks:
            if ob.bias == -1 and close >= ob.bar_low and close <= ob.bar_high:
                signals['bearish_signals'].append({
                    'type': 'Bearish Order Block',
                    'price': f"{ob.bar_low:.2f} - {ob.bar_high:.2f}",
                    'level': 'Swing'
                })
                bearish_count += 2

        for ob in self.internal_order_blocks:
            if ob.bias == -1 and close >= ob.bar_low and close <= ob.bar_high:
                signals['bearish_signals'].append({
                    'type': 'Bearish Order Block',
                    'price': f"{ob.bar_low:.2f} - {ob.bar_high:.2f}",
                    'level': 'Internal'
                })
                bearish_count += 1

        # Check FVGs
        for fvg in self.fvgs:
            if fvg.gap_type == 'bullish' and close >= fvg.bottom and close <= fvg.top:
                signals['bullish_signals'].append({
                    'type': 'Bullish FVG',
                    'price': f"{fvg.bottom:.2f} - {fvg.top:.2f}"
                })
                bullish_count += 1
            elif fvg.gap_type == 'bearish' and close >= fvg.bottom and close <= fvg.top:
                signals['bearish_signals'].append({
                    'type': 'Bearish FVG',
                    'price': f"{fvg.bottom:.2f} - {fvg.top:.2f}"
                })
                bearish_count += 1

        # Check Supply/Demand zones
        for zone in self.demand_zones:
            if close >= zone.bottom and close <= zone.top:
                signals['bullish_signals'].append({
                    'type': 'Demand Zone',
                    'price': f"{zone.bottom:.2f} - {zone.top:.2f}",
                    'poi': f"{zone.poi:.2f}"
                })
                bullish_count += 1

        for zone in self.supply_zones:
            if close >= zone.bottom and close <= zone.top:
                signals['bearish_signals'].append({
                    'type': 'Supply Zone',
                    'price': f"{zone.bottom:.2f} - {zone.top:.2f}",
                    'poi': f"{zone.poi:.2f}"
                })
                bearish_count += 1

        # Check POC
        if self.poc_price > 0:
            if close > self.poc_price:
                signals['bullish_signals'].append({
                    'type': 'Above POC',
                    'price': f"{self.poc_price:.2f}"
                })
                bullish_count += 1
            elif close < self.poc_price:
                signals['bearish_signals'].append({
                    'type': 'Below POC',
                    'price': f"{self.poc_price:.2f}"
                })
                bearish_count += 1

        # Determine overall bias
        if bullish_count > bearish_count + 1:
            signals['overall_bias'] = 'BULLISH'
        elif bearish_count > bullish_count + 1:
            signals['overall_bias'] = 'BEARISH'
        else:
            signals['overall_bias'] = 'NEUTRAL'

        signals['bullish_count'] = bullish_count
        signals['bearish_count'] = bearish_count

        return signals
