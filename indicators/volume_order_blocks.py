"""
Volume Order Blocks [BigBeluga] - Python Implementation
Converted from Pine Script v6

This indicator detects Volume Order Blocks (VOB) using EMA crossovers and volume analysis.
It identifies bullish and bearish order blocks with volume-weighted importance.

Features:
- Bullish VOB: Detected on EMA crossover up at lowest point
- Bearish VOB: Detected on EMA crossover down at highest point
- Volume Collection: Aggregates volume from crossover to swing point
- Percentage Distribution: Shows each block's volume as % of total
- Overlap Detection: Removes overlapping blocks using ATR
- Mitigation Detection: Marks blocks as inactive when price crosses through

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
Original Author: BigBeluga
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class VOBLevel:
    """Represents a Volume Order Block level"""
    index: int
    upper: float
    lower: float
    mid: float
    volume: float
    start_time: any
    active: bool = True
    block_type: str = 'bullish'  # 'bullish' or 'bearish'


class VolumeOrderBlocks:
    """
    Volume Order Blocks Indicator

    Detects bullish and bearish order blocks based on EMA crossovers
    with volume analysis and percentage distribution.

    Based on BigBeluga's Pine Script indicator.
    """

    def __init__(self,
                 sensitivity: int = 5,
                 show_mid_line: bool = True,
                 max_blocks: int = 15,
                 color_bull: str = '#26ba9f',
                 color_bear: str = '#6626ba'):
        """
        Initialize Volume Order Blocks indicator

        Args:
            sensitivity: Detection sensitivity (length1 in Pine Script)
            show_mid_line: Whether to show mid-line
            max_blocks: Maximum number of blocks to keep
            color_bull: Bullish block color
            color_bear: Bearish block color
        """
        self.sensitivity = sensitivity
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.show_mid_line = show_mid_line
        self.max_blocks = max_blocks
        self.color_bull = color_bull
        self.color_bear = color_bear

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Volume Order Blocks

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            Dictionary containing:
            - bullish_blocks: List of bullish VOB dictionaries
            - bearish_blocks: List of bearish VOB dictionaries
            - ema1: Fast EMA values
            - ema2: Slow EMA values
            - ema_diff: EMA difference for trend shadow
            - total_bullish_volume: Sum of all bullish block volumes
            - total_bearish_volume: Sum of all bearish block volumes
        """
        df = df.copy()
        n = len(df)

        if n < self.length2 + 10:
            return self._empty_result()

        # Calculate EMAs
        ema1 = df['close'].ewm(span=self.length1, adjust=False).mean()
        ema2 = df['close'].ewm(span=self.length2, adjust=False).mean()

        # Detect crossovers (confirmed - not on current bar)
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_dn = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

        # Calculate rolling lowest and highest
        lowest = df['low'].rolling(window=self.length2).min()
        highest = df['high'].rolling(window=self.length2).max()

        # Calculate ATR for overlap detection and adjustment
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr_200 = tr.rolling(window=200, min_periods=1).mean()
        atr = atr_200.rolling(window=200, min_periods=1).max() * 3
        atr1 = atr_200.rolling(window=200, min_periods=1).max() * 2

        # Detect order blocks
        bullish_levels = []
        bearish_levels = []

        # Process bullish blocks (on cross up)
        for i in range(self.length2, n):
            if cross_up.iloc[i]:
                # Find the lowest point in lookback period
                lookback_start = max(0, i - self.length2)
                lookback = df.iloc[lookback_start:i]

                if len(lookback) > 0:
                    lowest_val = lookback['low'].min()
                    lowest_idx_rel = lookback['low'].idxmin()

                    # Get candle at lowest point
                    candle = df.loc[lowest_idx_rel]
                    src = min(candle['open'], candle['close'])

                    # Adjust if too close to low
                    atr1_val = atr1.iloc[i] if not pd.isna(atr1.iloc[i]) else 0
                    if (src - lowest_val) < atr1_val * 0.5:
                        src = lowest_val + atr1_val * 0.5

                    mid = (src + lowest_val) / 2

                    # Calculate volume from crossover to lowest point
                    vol = 0.0
                    idx_pos = df.index.get_loc(lowest_idx_rel) if isinstance(lowest_idx_rel, pd.Timestamp) else lowest_idx_rel
                    for k in range(idx_pos, i + 1):
                        vol += df['volume'].iloc[k]

                    level = VOBLevel(
                        index=i,
                        upper=float(src),
                        lower=float(lowest_val),
                        mid=float(mid),
                        volume=float(vol),
                        start_time=df.index[i],
                        active=True,
                        block_type='bullish'
                    )
                    bullish_levels.append(level)

        # Process bearish blocks (on cross down)
        for i in range(self.length2, n):
            if cross_dn.iloc[i]:
                # Find the highest point in lookback period
                lookback_start = max(0, i - self.length2)
                lookback = df.iloc[lookback_start:i]

                if len(lookback) > 0:
                    highest_val = lookback['high'].max()
                    highest_idx_rel = lookback['high'].idxmax()

                    # Get candle at highest point
                    candle = df.loc[highest_idx_rel]
                    src = max(candle['open'], candle['close'])

                    # Adjust if too close to high
                    atr1_val = atr1.iloc[i] if not pd.isna(atr1.iloc[i]) else 0
                    if (highest_val - src) < atr1_val * 0.5:
                        src = highest_val - atr1_val * 0.5

                    mid = (src + highest_val) / 2

                    # Calculate volume from crossover to highest point
                    vol = 0.0
                    idx_pos = df.index.get_loc(highest_idx_rel) if isinstance(highest_idx_rel, pd.Timestamp) else highest_idx_rel
                    for k in range(idx_pos, i + 1):
                        vol += df['volume'].iloc[k]

                    level = VOBLevel(
                        index=i,
                        upper=float(highest_val),
                        lower=float(src),
                        mid=float(mid),
                        volume=float(vol),
                        start_time=df.index[i],
                        active=True,
                        block_type='bearish'
                    )
                    bearish_levels.append(level)

        # Remove overlapping and crossed blocks
        current_price = df['close'].iloc[-1]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        bullish_levels = self._filter_blocks(bullish_levels, current_price, current_atr, 'bullish')
        bearish_levels = self._filter_blocks(bearish_levels, current_price, current_atr, 'bearish')

        # Keep only last max_blocks
        bullish_levels = bullish_levels[-self.max_blocks:]
        bearish_levels = bearish_levels[-self.max_blocks:]

        # Calculate total volumes for percentage
        total_bull_vol = sum(l.volume for l in bullish_levels if l.active)
        total_bear_vol = sum(l.volume for l in bearish_levels if l.active)

        # Convert to dictionaries with volume percentage
        bullish_blocks = []
        for level in bullish_levels:
            vol_pct = (level.volume / total_bull_vol * 100) if total_bull_vol > 0 else 0
            bullish_blocks.append({
                'index': level.index,
                'upper': level.upper,
                'lower': level.lower,
                'mid': level.mid,
                'volume': level.volume,
                'volume_pct': vol_pct,
                'start_time': level.start_time,
                'active': level.active,
                'type': 'bullish'
            })

        bearish_blocks = []
        for level in bearish_levels:
            vol_pct = (level.volume / total_bear_vol * 100) if total_bear_vol > 0 else 0
            bearish_blocks.append({
                'index': level.index,
                'upper': level.upper,
                'lower': level.lower,
                'mid': level.mid,
                'volume': level.volume,
                'volume_pct': vol_pct,
                'start_time': level.start_time,
                'active': level.active,
                'type': 'bearish'
            })

        # EMA difference for trend shadow
        ema_diff = ema2 - ema1

        return {
            'bullish_blocks': bullish_blocks,
            'bearish_blocks': bearish_blocks,
            'ema1': ema1.values,
            'ema2': ema2.values,
            'ema_diff': ema_diff.values,
            'total_bullish_volume': total_bull_vol,
            'total_bearish_volume': total_bear_vol,
            'current_price': current_price
        }

    def _filter_blocks(self, levels: List[VOBLevel], current_price: float,
                       atr: float, block_type: str) -> List[VOBLevel]:
        """
        Filter out overlapping and crossed blocks

        Args:
            levels: List of VOB levels
            current_price: Current close price
            atr: Current ATR value for overlap detection
            block_type: 'bullish' or 'bearish'

        Returns:
            Filtered list of VOB levels
        """
        if len(levels) == 0:
            return levels

        filtered = []

        for i, level in enumerate(levels):
            # Check if price has crossed through the block (mitigation)
            if block_type == 'bullish':
                if current_price < level.lower:
                    level.active = False
            else:  # bearish
                if current_price > level.upper:
                    level.active = False

            # Check overlap with previous level
            if i > 0 and len(filtered) > 0:
                prev_level = filtered[-1]
                if abs(level.mid - prev_level.mid) < atr:
                    # Overlap detected - keep the newer one (current)
                    # Mark previous as inactive
                    prev_level.active = False

            filtered.append(level)

        return filtered

    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'bullish_blocks': [],
            'bearish_blocks': [],
            'ema1': np.array([]),
            'ema2': np.array([]),
            'ema_diff': np.array([]),
            'total_bullish_volume': 0,
            'total_bearish_volume': 0,
            'current_price': 0
        }

    def get_htf_vob_table(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Get VOB data formatted for HTF level tabulation display

        Args:
            df: DataFrame with OHLCV data
            current_price: Current spot price

        Returns:
            Dictionary with formatted bullish and bearish VOB data for display
        """
        result = self.calculate(df)

        bullish_table = []
        bearish_table = []

        # Process bullish VOBs
        for block in result['bullish_blocks']:
            distance = current_price - block['mid']
            distance_pct = (distance / block['mid']) * 100 if block['mid'] > 0 else 0

            bullish_table.append({
                'Type': 'BULLISH VOB',
                'Upper': block['upper'],
                'Lower': block['lower'],
                'Mid': block['mid'],
                'Volume': block['volume'],
                'Volume %': block['volume_pct'],
                'Distance': distance,
                'Distance %': distance_pct,
                'Status': 'ACTIVE' if block['active'] else 'MITIGATED',
                'Proximity': 'NEAR' if abs(distance_pct) <= 0.5 and block['active'] else ''
            })

        # Process bearish VOBs
        for block in result['bearish_blocks']:
            distance = current_price - block['mid']
            distance_pct = (distance / block['mid']) * 100 if block['mid'] > 0 else 0

            bearish_table.append({
                'Type': 'BEARISH VOB',
                'Upper': block['upper'],
                'Lower': block['lower'],
                'Mid': block['mid'],
                'Volume': block['volume'],
                'Volume %': block['volume_pct'],
                'Distance': distance,
                'Distance %': distance_pct,
                'Status': 'ACTIVE' if block['active'] else 'MITIGATED',
                'Proximity': 'NEAR' if abs(distance_pct) <= 0.5 and block['active'] else ''
            })

        # Sort by distance (closest first)
        bullish_table.sort(key=lambda x: abs(x['Distance']))
        bearish_table.sort(key=lambda x: abs(x['Distance']))

        return {
            'bullish': bullish_table,
            'bearish': bearish_table,
            'total_bullish_volume': result['total_bullish_volume'],
            'total_bearish_volume': result['total_bearish_volume'],
            'bullish_count': len([b for b in result['bullish_blocks'] if b['active']]),
            'bearish_count': len([b for b in result['bearish_blocks'] if b['active']])
        }

    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signals based on VOB analysis

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        result = self.calculate(df)

        if not result['bullish_blocks'] and not result['bearish_blocks']:
            return {
                'signal': 'NEUTRAL',
                'strength': 0,
                'nearest_support': None,
                'nearest_resistance': None,
                'volume_bias': 'NEUTRAL'
            }

        current_price = result['current_price']

        # Find nearest active support (bullish VOB)
        nearest_support = None
        for block in result['bullish_blocks']:
            if block['active'] and current_price > block['upper']:
                if nearest_support is None or block['upper'] > nearest_support['upper']:
                    nearest_support = block

        # Find nearest active resistance (bearish VOB)
        nearest_resistance = None
        for block in result['bearish_blocks']:
            if block['active'] and current_price < block['lower']:
                if nearest_resistance is None or block['lower'] < nearest_resistance['lower']:
                    nearest_resistance = block

        # Determine volume bias
        total_bull = result['total_bullish_volume']
        total_bear = result['total_bearish_volume']

        if total_bull > total_bear * 1.5:
            volume_bias = 'BULLISH'
        elif total_bear > total_bull * 1.5:
            volume_bias = 'BEARISH'
        else:
            volume_bias = 'NEUTRAL'

        # Determine signal
        signal = 'NEUTRAL'
        strength = 0

        if nearest_support and nearest_resistance:
            dist_to_support = current_price - nearest_support['upper']
            dist_to_resistance = nearest_resistance['lower'] - current_price

            if dist_to_support < dist_to_resistance:
                signal = 'NEAR_SUPPORT'
                strength = min(100, int(100 - (dist_to_support / current_price * 1000)))
            else:
                signal = 'NEAR_RESISTANCE'
                strength = min(100, int(100 - (dist_to_resistance / current_price * 1000)))
        elif nearest_support:
            signal = 'ABOVE_SUPPORT'
            strength = 50
        elif nearest_resistance:
            signal = 'BELOW_RESISTANCE'
            strength = 50

        return {
            'signal': signal,
            'strength': strength,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'volume_bias': volume_bias,
            'total_bullish_volume': total_bull,
            'total_bearish_volume': total_bear
        }


def calculate_vob_for_htf(df: pd.DataFrame, timeframe: str, sensitivity: int = 5) -> Dict:
    """
    Convenience function to calculate VOB for a specific HTF timeframe

    Args:
        df: DataFrame with OHLCV data (1min or base timeframe)
        timeframe: Target timeframe string (e.g., '5T', '15T', '60T')
        sensitivity: VOB sensitivity parameter

    Returns:
        VOB calculation result for the specified timeframe
    """
    # Resample to target timeframe
    df_htf = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(df_htf) < 30:
        return {
            'bullish_blocks': [],
            'bearish_blocks': [],
            'timeframe': timeframe,
            'error': 'Insufficient data'
        }

    vob = VolumeOrderBlocks(sensitivity=sensitivity)
    result = vob.calculate(df_htf)
    result['timeframe'] = timeframe

    return result
