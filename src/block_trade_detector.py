"""
Block Trade Detector - Diamond Level Feature
=============================================
Institutional Large Trade Detection

BLOCK TRADES:
- Minimum 10,000+ contracts in a single trade
- Typically executed by institutions (FIIs, DIIs, HNIs)
- Often negotiated off-exchange, then reported
- Leave footprints in volume and OI data

DETECTION SIGNALS:
1. Volume Spike: Sudden 3x+ avg volume in single bar
2. OI Jump: Large OI change with minimal price move
3. Print Detection: Large single trades visible in tape
4. Sweep Pattern: Aggressive orders eating through levels
5. Block Time: Trades often occur at open/close

Data Sources:
- Chart data (Volume) from Advanced Chart Analysis
- Option Chain (OI changes) from NIFTY Option Screener
- Market Depth from NIFTY Option Screener
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st


@dataclass
class BlockTrade:
    """Detected Block Trade"""
    trade_type: str  # 'BUY_BLOCK', 'SELL_BLOCK', 'SWEEP_UP', 'SWEEP_DOWN'
    timestamp: datetime
    price: float
    estimated_size: int  # Estimated contract size
    direction: str  # 'BULLISH', 'BEARISH'
    strength: float  # 0-100
    confidence: float  # 0-100
    detection_method: str  # 'VOLUME_SPIKE', 'OI_CHANGE', 'DEPTH_SWEEP', 'TAPE_READ'
    description: str


@dataclass
class BlockTradePattern:
    """Pattern of multiple block trades"""
    pattern_type: str  # 'ACCUMULATION', 'DISTRIBUTION', 'MOMENTUM', 'REVERSAL'
    trades: List[BlockTrade]
    net_direction: str  # 'BULLISH', 'BEARISH', 'MIXED'
    total_size: int
    time_span_minutes: float
    interpretation: str


@dataclass
class BlockTradeAnalysis:
    """Complete Block Trade Analysis Result"""
    recent_blocks: List[BlockTrade]
    patterns: List[BlockTradePattern]
    institutional_bias: str  # 'ACCUMULATING', 'DISTRIBUTING', 'NEUTRAL'
    activity_level: str  # 'HEAVY', 'MODERATE', 'LIGHT', 'NONE'
    avg_block_size: int
    total_block_volume: int
    bullish_blocks: int
    bearish_blocks: int
    score: float  # 0-100 composite score
    description: str


class BlockTradeDetector:
    """
    Block Trade / Institutional Footprint Detector

    Detects large institutional trades through:
    - Volume analysis
    - OI change patterns
    - Market depth sweeps
    - Time-based patterns
    """

    def __init__(self):
        self.min_block_size = 10000  # Minimum 10k contracts
        self.volume_spike_threshold = 3.0  # 3x average
        self.oi_change_threshold = 50000  # 50k OI change
        self.detected_blocks = []
        self.max_history = 100

    def detect_volume_blocks(self, chart_data: pd.DataFrame) -> List[BlockTrade]:
        """
        Detect block trades from volume spikes

        Volume blocks occur when:
        - Single bar has 3x+ average volume
        - Large candle body (aggressive execution)
        """
        blocks = []

        try:
            if chart_data is None or len(chart_data) < 20:
                return blocks

            # Get volume column
            vol_col = 'Volume' if 'Volume' in chart_data.columns else 'volume'
            if vol_col not in chart_data.columns:
                return blocks

            # Calculate rolling average volume
            avg_volume = chart_data[vol_col].rolling(window=20).mean()

            # Check last 10 bars for spikes
            for i in range(-10, 0):
                try:
                    row = chart_data.iloc[i]
                    current_vol = row[vol_col]
                    avg_vol = avg_volume.iloc[i]

                    if avg_vol is None or pd.isna(avg_vol) or avg_vol == 0:
                        continue

                    volume_ratio = current_vol / avg_vol

                    if volume_ratio >= self.volume_spike_threshold:
                        # Determine direction from candle
                        open_price = row.get('Open', row.get('open', 0))
                        close = row.get('Close', row.get('close', 0))
                        high = row.get('High', row.get('high', 0))
                        low = row.get('Low', row.get('low', 0))

                        # Bullish if close > open
                        if close > open_price:
                            direction = 'BULLISH'
                            trade_type = 'BUY_BLOCK'
                        else:
                            direction = 'BEARISH'
                            trade_type = 'SELL_BLOCK'

                        # Estimate size (volume in contracts)
                        estimated_size = int(current_vol)

                        # Calculate strength
                        strength = min(100, volume_ratio * 25)

                        # Confidence based on candle body
                        body_ratio = abs(close - open_price) / (high - low) if high > low else 0
                        confidence = min(95, 50 + body_ratio * 45)

                        timestamp = chart_data.index[i] if isinstance(chart_data.index[i], datetime) else datetime.now()

                        blocks.append(BlockTrade(
                            trade_type=trade_type,
                            timestamp=timestamp,
                            price=close,
                            estimated_size=estimated_size,
                            direction=direction,
                            strength=strength,
                            confidence=confidence,
                            detection_method='VOLUME_SPIKE',
                            description=f"{volume_ratio:.1f}x volume spike at {close:.2f} - estimated {estimated_size:,} contracts"
                        ))

                except Exception:
                    continue

        except Exception:
            pass

        return blocks

    def detect_oi_blocks(self, merged_df: pd.DataFrame, spot_price: float = None) -> List[BlockTrade]:
        """
        Detect block trades from OI changes

        OI blocks occur when:
        - Large OI build-up at specific strikes
        - Sudden OI changes in single expiry
        """
        blocks = []

        try:
            if merged_df is None or len(merged_df) == 0:
                return blocks

            # Get ATM strike
            if spot_price is None:
                spot_price = merged_df['Strike'].median() if 'Strike' in merged_df.columns else 0

            # Look for large OI changes
            for idx, row in merged_df.iterrows():
                try:
                    strike = row.get('Strike', 0)
                    chg_oi_ce = abs(row.get('Chg_OI_CE', 0) or 0)
                    chg_oi_pe = abs(row.get('Chg_OI_PE', 0) or 0)
                    oi_ce = row.get('OI_CE', 0) or 0
                    oi_pe = row.get('OI_PE', 0) or 0

                    # Check for large CE OI change (block)
                    if chg_oi_ce >= self.oi_change_threshold:
                        # CE OI build-up at/above spot = bearish (resistance)
                        # CE OI build-up below spot = bullish (covered calls)
                        if strike >= spot_price:
                            direction = 'BEARISH'
                            trade_type = 'SELL_BLOCK'
                            desc = f"Large CALL OI build ({chg_oi_ce:,}) at {strike} - RESISTANCE"
                        else:
                            direction = 'BULLISH'
                            trade_type = 'BUY_BLOCK'
                            desc = f"Large CALL OI build ({chg_oi_ce:,}) at {strike} - Covered calls"

                        strength = min(100, chg_oi_ce / 10000)

                        blocks.append(BlockTrade(
                            trade_type=trade_type,
                            timestamp=datetime.now(),
                            price=strike,
                            estimated_size=int(chg_oi_ce),
                            direction=direction,
                            strength=strength,
                            confidence=min(85, strength * 0.9),
                            detection_method='OI_CHANGE',
                            description=desc
                        ))

                    # Check for large PE OI change (block)
                    if chg_oi_pe >= self.oi_change_threshold:
                        # PE OI build-up at/below spot = bullish (support)
                        # PE OI build-up above spot = bearish (aggressive puts)
                        if strike <= spot_price:
                            direction = 'BULLISH'
                            trade_type = 'BUY_BLOCK'
                            desc = f"Large PUT OI build ({chg_oi_pe:,}) at {strike} - SUPPORT"
                        else:
                            direction = 'BEARISH'
                            trade_type = 'SELL_BLOCK'
                            desc = f"Large PUT OI build ({chg_oi_pe:,}) at {strike} - Aggressive puts"

                        strength = min(100, chg_oi_pe / 10000)

                        blocks.append(BlockTrade(
                            trade_type=trade_type,
                            timestamp=datetime.now(),
                            price=strike,
                            estimated_size=int(chg_oi_pe),
                            direction=direction,
                            strength=strength,
                            confidence=min(85, strength * 0.9),
                            detection_method='OI_CHANGE',
                            description=desc
                        ))

                except Exception:
                    continue

        except Exception:
            pass

        return blocks

    def detect_sweep_patterns(self, market_depth: Dict, chart_data: pd.DataFrame = None) -> List[BlockTrade]:
        """
        Detect sweep patterns from market depth

        Sweeps occur when aggressive orders eat through multiple levels
        """
        blocks = []

        try:
            if market_depth is None:
                return blocks

            bids = market_depth.get('bids', [])
            asks = market_depth.get('asks', [])

            if not bids or not asks:
                return blocks

            # Check for thin depth (swept recently)
            total_bid_qty = sum(b.get('quantity', 0) for b in bids)
            total_ask_qty = sum(a.get('quantity', 0) for a in asks)

            if total_bid_qty == 0 or total_ask_qty == 0:
                return blocks

            # Extreme imbalance = recent sweep
            imbalance_ratio = total_bid_qty / total_ask_qty

            if imbalance_ratio > 3:  # Bids >> Asks = Ask side swept
                # Someone aggressive bought through asks
                best_ask = min(a.get('price', 0) for a in asks)

                blocks.append(BlockTrade(
                    trade_type='SWEEP_UP',
                    timestamp=datetime.now(),
                    price=best_ask,
                    estimated_size=int(total_ask_qty),
                    direction='BULLISH',
                    strength=min(100, imbalance_ratio * 20),
                    confidence=min(80, imbalance_ratio * 15),
                    detection_method='DEPTH_SWEEP',
                    description=f"Ask side swept - {imbalance_ratio:.1f}x bid/ask imbalance"
                ))

            elif imbalance_ratio < 0.33:  # Asks >> Bids = Bid side swept
                # Someone aggressive sold through bids
                best_bid = max(b.get('price', 0) for b in bids)

                blocks.append(BlockTrade(
                    trade_type='SWEEP_DOWN',
                    timestamp=datetime.now(),
                    price=best_bid,
                    estimated_size=int(total_bid_qty),
                    direction='BEARISH',
                    strength=min(100, (1/imbalance_ratio) * 20),
                    confidence=min(80, (1/imbalance_ratio) * 15),
                    detection_method='DEPTH_SWEEP',
                    description=f"Bid side swept - {1/imbalance_ratio:.1f}x ask/bid imbalance"
                ))

        except Exception:
            pass

        return blocks

    def identify_patterns(self, blocks: List[BlockTrade]) -> List[BlockTradePattern]:
        """
        Identify patterns from multiple block trades

        Patterns:
        - ACCUMULATION: Multiple buy blocks over time
        - DISTRIBUTION: Multiple sell blocks over time
        - MOMENTUM: Increasing block sizes in same direction
        - REVERSAL: Sudden switch in block direction
        """
        patterns = []

        try:
            if len(blocks) < 2:
                return patterns

            # Count directions
            bullish = [b for b in blocks if b.direction == 'BULLISH']
            bearish = [b for b in blocks if b.direction == 'BEARISH']

            # Accumulation pattern
            if len(bullish) >= 3 and len(bullish) > len(bearish) * 2:
                total_size = sum(b.estimated_size for b in bullish)
                patterns.append(BlockTradePattern(
                    pattern_type='ACCUMULATION',
                    trades=bullish,
                    net_direction='BULLISH',
                    total_size=total_size,
                    time_span_minutes=30,  # Approximate
                    interpretation=f"INSTITUTIONAL ACCUMULATION: {len(bullish)} buy blocks totaling {total_size:,} contracts"
                ))

            # Distribution pattern
            if len(bearish) >= 3 and len(bearish) > len(bullish) * 2:
                total_size = sum(b.estimated_size for b in bearish)
                patterns.append(BlockTradePattern(
                    pattern_type='DISTRIBUTION',
                    trades=bearish,
                    net_direction='BEARISH',
                    total_size=total_size,
                    time_span_minutes=30,
                    interpretation=f"INSTITUTIONAL DISTRIBUTION: {len(bearish)} sell blocks totaling {total_size:,} contracts"
                ))

            # Check for momentum (increasing sizes)
            if len(blocks) >= 3:
                sizes = [b.estimated_size for b in blocks[-5:]]
                if all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)):
                    direction = blocks[-1].direction
                    patterns.append(BlockTradePattern(
                        pattern_type='MOMENTUM',
                        trades=blocks[-5:],
                        net_direction=direction,
                        total_size=sum(sizes),
                        time_span_minutes=15,
                        interpretation=f"MOMENTUM BUILD: Increasing block sizes - {direction} pressure intensifying"
                    ))

        except Exception:
            pass

        return patterns

    def calculate_score(self, blocks: List[BlockTrade], patterns: List[BlockTradePattern]) -> Tuple[float, str]:
        """
        Calculate composite score and institutional bias

        Returns: (score, institutional_bias)
        """
        score = 50  # Start neutral

        # Factor 1: Net block direction
        bullish_strength = sum(b.strength for b in blocks if b.direction == 'BULLISH')
        bearish_strength = sum(b.strength for b in blocks if b.direction == 'BEARISH')

        net_strength = bullish_strength - bearish_strength
        score += min(25, max(-25, net_strength / 4))

        # Factor 2: Pattern strength
        for pattern in patterns:
            if pattern.net_direction == 'BULLISH':
                score += 10
            elif pattern.net_direction == 'BEARISH':
                score -= 10

        # Determine bias
        if score > 60:
            bias = 'ACCUMULATING'
        elif score < 40:
            bias = 'DISTRIBUTING'
        else:
            bias = 'NEUTRAL'

        return max(0, min(100, score)), bias

    def analyze(self, chart_data: pd.DataFrame = None, merged_df: pd.DataFrame = None,
               market_depth: Dict = None) -> Optional[BlockTradeAnalysis]:
        """
        Main analysis function

        Args:
            chart_data: OHLCV DataFrame (uses session_state.chart_data if None)
            merged_df: Option chain DataFrame (uses session_state.merged_df if None)
            market_depth: Market depth data (uses session_state.market_depth_data if None)

        Returns:
            BlockTradeAnalysis with all findings
        """
        try:
            # Get data from session state if not provided
            if chart_data is None:
                chart_data = st.session_state.get('chart_data')
                if chart_data is None:
                    chart_data = st.session_state.get('nifty_df')

            if merged_df is None:
                merged_df = st.session_state.get('merged_df')

            if market_depth is None:
                market_depth = st.session_state.get('market_depth_data')

            # Get spot price
            spot_price = None
            if chart_data is not None and len(chart_data) > 0:
                spot_price = chart_data['Close'].iloc[-1] if 'Close' in chart_data.columns else chart_data.get('close', pd.Series([0])).iloc[-1]

            # Detect all block types
            all_blocks = []

            # 1. Volume-based blocks
            if chart_data is not None:
                volume_blocks = self.detect_volume_blocks(chart_data)
                all_blocks.extend(volume_blocks)

            # 2. OI-based blocks
            if merged_df is not None:
                oi_blocks = self.detect_oi_blocks(merged_df, spot_price)
                all_blocks.extend(oi_blocks)

            # 3. Sweep patterns
            if market_depth is not None:
                sweep_blocks = self.detect_sweep_patterns(market_depth, chart_data)
                all_blocks.extend(sweep_blocks)

            # Store in history
            self.detected_blocks.extend(all_blocks)
            if len(self.detected_blocks) > self.max_history:
                self.detected_blocks = self.detected_blocks[-self.max_history:]

            # Identify patterns
            patterns = self.identify_patterns(all_blocks)

            # Calculate statistics
            bullish_blocks = sum(1 for b in all_blocks if b.direction == 'BULLISH')
            bearish_blocks = sum(1 for b in all_blocks if b.direction == 'BEARISH')
            total_volume = sum(b.estimated_size for b in all_blocks)
            avg_size = total_volume // len(all_blocks) if all_blocks else 0

            # Determine activity level
            if len(all_blocks) >= 10:
                activity_level = 'HEAVY'
            elif len(all_blocks) >= 5:
                activity_level = 'MODERATE'
            elif len(all_blocks) >= 1:
                activity_level = 'LIGHT'
            else:
                activity_level = 'NONE'

            # Calculate score and bias
            score, institutional_bias = self.calculate_score(all_blocks, patterns)

            # Build description
            if patterns:
                description = patterns[0].interpretation
            elif all_blocks:
                top_block = max(all_blocks, key=lambda x: x.strength)
                description = top_block.description
            else:
                description = "No significant block trades detected"

            return BlockTradeAnalysis(
                recent_blocks=all_blocks[-20:],  # Keep last 20
                patterns=patterns,
                institutional_bias=institutional_bias,
                activity_level=activity_level,
                avg_block_size=avg_size,
                total_block_volume=total_volume,
                bullish_blocks=bullish_blocks,
                bearish_blocks=bearish_blocks,
                score=score,
                description=description
            )

        except Exception as e:
            st.error(f"Block Trade Analysis error: {e}")
            return None


# Singleton instance
_block_trade_detector = None


def get_block_trade_detector() -> BlockTradeDetector:
    """Get singleton Block Trade Detector instance"""
    global _block_trade_detector
    if _block_trade_detector is None:
        _block_trade_detector = BlockTradeDetector()
    return _block_trade_detector


def analyze_block_trades(chart_data: pd.DataFrame = None, merged_df: pd.DataFrame = None,
                        market_depth: Dict = None) -> Optional[BlockTradeAnalysis]:
    """
    Convenience function to analyze block trades

    Usage:
        from src.block_trade_detector import analyze_block_trades

        result = analyze_block_trades()
        if result:
            print(f"Institutional Bias: {result.institutional_bias}")
            print(f"Activity Level: {result.activity_level}")
            print(f"Bullish Blocks: {result.bullish_blocks}")
            print(f"Bearish Blocks: {result.bearish_blocks}")
    """
    detector = get_block_trade_detector()
    return detector.analyze(chart_data, merged_df, market_depth)
