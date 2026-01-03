"""
Black Order Detector - Diamond Level Feature
=============================================
Hidden Institutional Order Detection from Market Depth

BLACK ORDERS (Dark Pool / Iceberg Orders):
- Large institutional orders that are PARTIALLY HIDDEN
- Only small portion visible in order book
- As one slice fills, next slice appears
- Used by institutions to hide their true size

DETECTION METHODS:
1. Order Absorption: Large orders repeatedly appear at same price
2. Hidden Size: Volume traded >> visible size
3. Bid/Ask Imbalance: Unusual depth patterns
4. Iceberg Pattern: Constant refresh at same price
5. Footprint Analysis: Large trades at specific levels

Data Source: Market Depth from NIFTY Option Screener (market_depth_data)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st


@dataclass
class BlackOrderSignal:
    """Detected Black/Hidden Order Signal"""
    signal_type: str  # 'ICEBERG_BID', 'ICEBERG_ASK', 'ABSORPTION', 'HIDDEN_BUYER', 'HIDDEN_SELLER'
    price_level: float
    visible_size: int
    estimated_hidden_size: int
    direction: str  # 'BULLISH', 'BEARISH'
    strength: float  # 0-100
    confidence: float  # 0-100
    description: str
    trade_implication: str


@dataclass
class DepthImbalance:
    """Bid/Ask Depth Imbalance Analysis"""
    total_bid_qty: int
    total_ask_qty: int
    imbalance_ratio: float  # >1 = more bids, <1 = more asks
    imbalance_percent: float  # -100 to +100
    dominant_side: str  # 'BIDS', 'ASKS', 'BALANCED'
    best_bid: float
    best_ask: float
    spread: float
    spread_percent: float


@dataclass
class BlackOrderAnalysis:
    """Complete Black Order Analysis Result"""
    signals: List[BlackOrderSignal]
    depth_imbalance: DepthImbalance
    institutional_presence: str  # 'STRONG', 'MODERATE', 'WEAK', 'NONE'
    smart_money_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    absorption_zones: List[float]  # Price levels with absorption
    iceberg_levels: List[float]  # Price levels with iceberg orders
    score: float  # 0-100 composite score
    description: str


class BlackOrderDetector:
    """
    Black Order / Iceberg Order Detector

    Uses market depth data to detect:
    - Hidden institutional orders (icebergs)
    - Order absorption patterns
    - Bid/Ask imbalances
    - Smart money footprints
    """

    def __init__(self):
        self.depth_history = []
        self.max_history = 50
        self.iceberg_threshold = 3  # Minimum refreshes to detect iceberg
        self.absorption_threshold = 2.0  # Volume/Visible size ratio
        self.imbalance_threshold = 1.5  # Bid/Ask ratio for imbalance

    def analyze_depth_imbalance(self, market_depth: Dict) -> Optional[DepthImbalance]:
        """
        Analyze bid/ask imbalance from market depth

        This reveals institutional positioning:
        - Heavy bids = institutions accumulating
        - Heavy asks = institutions distributing
        """
        try:
            bids = market_depth.get('bids', [])
            asks = market_depth.get('asks', [])

            if not bids or not asks:
                return None

            # Calculate totals
            total_bid_qty = sum(b.get('quantity', 0) for b in bids)
            total_ask_qty = sum(a.get('quantity', 0) for a in asks)

            if total_bid_qty == 0 and total_ask_qty == 0:
                return None

            # Best bid/ask
            best_bid = max(b.get('price', 0) for b in bids) if bids else 0
            best_ask = min(a.get('price', 0) for a in asks) if asks else 0

            # Spread
            spread = best_ask - best_bid if best_ask > best_bid else 0
            spread_percent = (spread / best_bid * 100) if best_bid > 0 else 0

            # Imbalance ratio
            imbalance_ratio = total_bid_qty / total_ask_qty if total_ask_qty > 0 else float('inf')

            # Imbalance percent (-100 to +100)
            total = total_bid_qty + total_ask_qty
            imbalance_percent = ((total_bid_qty - total_ask_qty) / total * 100) if total > 0 else 0

            # Determine dominant side
            if imbalance_ratio > self.imbalance_threshold:
                dominant_side = 'BIDS'
            elif imbalance_ratio < 1 / self.imbalance_threshold:
                dominant_side = 'ASKS'
            else:
                dominant_side = 'BALANCED'

            return DepthImbalance(
                total_bid_qty=total_bid_qty,
                total_ask_qty=total_ask_qty,
                imbalance_ratio=imbalance_ratio,
                imbalance_percent=imbalance_percent,
                dominant_side=dominant_side,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                spread_percent=spread_percent
            )

        except Exception:
            return None

    def detect_iceberg_orders(self, market_depth: Dict) -> List[BlackOrderSignal]:
        """
        Detect iceberg orders from market depth patterns

        Iceberg detection:
        - Same price level repeatedly appears
        - Size refreshes after fills
        - Constant quantity at same price
        """
        signals = []

        try:
            bids = market_depth.get('bids', [])
            asks = market_depth.get('asks', [])

            # Store current depth
            current_depth = {
                'timestamp': datetime.now(),
                'bids': {b.get('price', 0): b.get('quantity', 0) for b in bids},
                'asks': {a.get('price', 0): a.get('quantity', 0) for a in asks}
            }

            self.depth_history.append(current_depth)

            if len(self.depth_history) > self.max_history:
                self.depth_history = self.depth_history[-self.max_history:]

            # Need history to detect icebergs
            if len(self.depth_history) < 3:
                return signals

            # Analyze bid side for icebergs
            bid_price_counts = {}
            for snapshot in self.depth_history[-10:]:
                for price in snapshot['bids'].keys():
                    bid_price_counts[price] = bid_price_counts.get(price, 0) + 1

            # Price levels appearing consistently = potential iceberg
            for price, count in bid_price_counts.items():
                if count >= self.iceberg_threshold:
                    # Get average size at this level
                    sizes = [s['bids'].get(price, 0) for s in self.depth_history[-10:] if price in s['bids']]
                    avg_size = np.mean(sizes) if sizes else 0

                    if avg_size > 0:
                        # Estimate hidden size (typically 5-10x visible)
                        estimated_hidden = int(avg_size * 5)

                        signals.append(BlackOrderSignal(
                            signal_type='ICEBERG_BID',
                            price_level=price,
                            visible_size=int(avg_size),
                            estimated_hidden_size=estimated_hidden,
                            direction='BULLISH',
                            strength=min(100, count * 15),
                            confidence=min(90, count * 12),
                            description=f"Iceberg BID detected at {price:.2f} - appearing {count} times with avg size {avg_size:.0f}",
                            trade_implication=f"Strong institutional BUYING interest at {price:.2f}. Hidden size estimated: {estimated_hidden:,}"
                        ))

            # Analyze ask side for icebergs
            ask_price_counts = {}
            for snapshot in self.depth_history[-10:]:
                for price in snapshot['asks'].keys():
                    ask_price_counts[price] = ask_price_counts.get(price, 0) + 1

            for price, count in ask_price_counts.items():
                if count >= self.iceberg_threshold:
                    sizes = [s['asks'].get(price, 0) for s in self.depth_history[-10:] if price in s['asks']]
                    avg_size = np.mean(sizes) if sizes else 0

                    if avg_size > 0:
                        estimated_hidden = int(avg_size * 5)

                        signals.append(BlackOrderSignal(
                            signal_type='ICEBERG_ASK',
                            price_level=price,
                            visible_size=int(avg_size),
                            estimated_hidden_size=estimated_hidden,
                            direction='BEARISH',
                            strength=min(100, count * 15),
                            confidence=min(90, count * 12),
                            description=f"Iceberg ASK detected at {price:.2f} - appearing {count} times with avg size {avg_size:.0f}",
                            trade_implication=f"Strong institutional SELLING interest at {price:.2f}. Hidden size estimated: {estimated_hidden:,}"
                        ))

        except Exception:
            pass

        return signals

    def detect_absorption(self, market_depth: Dict, chart_data: pd.DataFrame = None) -> List[BlackOrderSignal]:
        """
        Detect order absorption patterns

        Absorption = Large orders getting filled without price moving
        This indicates hidden institutional activity
        """
        signals = []

        try:
            if chart_data is None or len(chart_data) < 5:
                return signals

            # Get recent volume and price action
            recent = chart_data.tail(10)
            avg_volume = recent['Volume'].mean() if 'Volume' in recent.columns else recent.get('volume', pd.Series([0])).mean()

            last_row = recent.iloc[-1]
            last_volume = last_row.get('Volume', last_row.get('volume', 0))
            last_high = last_row.get('High', last_row.get('high', 0))
            last_low = last_row.get('Low', last_row.get('low', 0))
            last_close = last_row.get('Close', last_row.get('close', 0))

            # Calculate price range
            price_range = last_high - last_low
            avg_range = (recent['High'] - recent['Low']).mean() if 'High' in recent.columns else 0

            # Absorption: High volume but small price range
            if avg_volume > 0 and avg_range > 0:
                volume_ratio = last_volume / avg_volume
                range_ratio = price_range / avg_range if avg_range > 0 else 1

                # High volume + low range = absorption
                if volume_ratio > 1.5 and range_ratio < 0.5:
                    # Determine direction from close position
                    if last_close > (last_high + last_low) / 2:
                        direction = 'BULLISH'
                        signal_type = 'HIDDEN_BUYER'
                        implication = "Large buyer absorbing all selling pressure"
                    else:
                        direction = 'BEARISH'
                        signal_type = 'HIDDEN_SELLER'
                        implication = "Large seller absorbing all buying pressure"

                    strength = min(100, volume_ratio * 30)

                    signals.append(BlackOrderSignal(
                        signal_type=signal_type,
                        price_level=last_close,
                        visible_size=int(last_volume),
                        estimated_hidden_size=int(last_volume * 3),
                        direction=direction,
                        strength=strength,
                        confidence=min(85, strength * 0.9),
                        description=f"ORDER ABSORPTION: {volume_ratio:.1f}x volume with only {range_ratio:.1f}x range",
                        trade_implication=implication
                    ))

        except Exception:
            pass

        return signals

    def detect_large_orders(self, market_depth: Dict) -> List[BlackOrderSignal]:
        """
        Detect unusually large orders in the book

        Large orders = potential institutional activity
        """
        signals = []

        try:
            bids = market_depth.get('bids', [])
            asks = market_depth.get('asks', [])

            # Calculate average order size
            all_sizes = [b.get('quantity', 0) for b in bids] + [a.get('quantity', 0) for a in asks]
            if not all_sizes:
                return signals

            avg_size = np.mean(all_sizes)
            std_size = np.std(all_sizes)

            threshold = avg_size + 2 * std_size  # 2 standard deviations

            # Check bids for large orders
            for bid in bids:
                qty = bid.get('quantity', 0)
                price = bid.get('price', 0)

                if qty > threshold and qty > 0:
                    multiple = qty / avg_size
                    signals.append(BlackOrderSignal(
                        signal_type='HIDDEN_BUYER',
                        price_level=price,
                        visible_size=qty,
                        estimated_hidden_size=int(qty * 2),  # Assume more hidden
                        direction='BULLISH',
                        strength=min(100, multiple * 20),
                        confidence=min(80, multiple * 15),
                        description=f"Large BID: {qty:,} at {price:.2f} ({multiple:.1f}x avg size)",
                        trade_implication=f"Institutional buyer protecting {price:.2f}"
                    ))

            # Check asks for large orders
            for ask in asks:
                qty = ask.get('quantity', 0)
                price = ask.get('price', 0)

                if qty > threshold and qty > 0:
                    multiple = qty / avg_size
                    signals.append(BlackOrderSignal(
                        signal_type='HIDDEN_SELLER',
                        price_level=price,
                        visible_size=qty,
                        estimated_hidden_size=int(qty * 2),
                        direction='BEARISH',
                        strength=min(100, multiple * 20),
                        confidence=min(80, multiple * 15),
                        description=f"Large ASK: {qty:,} at {price:.2f} ({multiple:.1f}x avg size)",
                        trade_implication=f"Institutional seller capping {price:.2f}"
                    ))

        except Exception:
            pass

        return signals

    def calculate_score(self, signals: List[BlackOrderSignal],
                       depth_imbalance: Optional[DepthImbalance]) -> Tuple[float, str]:
        """
        Calculate composite black order score and bias

        Returns: (score, smart_money_bias)
        """
        score = 50  # Start neutral

        # Factor 1: Signal strength and direction
        bullish_strength = sum(s.strength for s in signals if s.direction == 'BULLISH')
        bearish_strength = sum(s.strength for s in signals if s.direction == 'BEARISH')

        net_strength = bullish_strength - bearish_strength
        score += min(25, max(-25, net_strength / 4))

        # Factor 2: Depth imbalance
        if depth_imbalance:
            if depth_imbalance.dominant_side == 'BIDS':
                score += min(15, depth_imbalance.imbalance_percent / 5)
            elif depth_imbalance.dominant_side == 'ASKS':
                score -= min(15, abs(depth_imbalance.imbalance_percent) / 5)

            # Spread factor (tight spread = more institutional activity)
            if depth_imbalance.spread_percent < 0.05:
                score += 5
            elif depth_imbalance.spread_percent > 0.15:
                score -= 5

        # Determine bias
        if score > 60:
            bias = 'BULLISH'
        elif score < 40:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'

        return max(0, min(100, score)), bias

    def analyze(self, market_depth: Dict = None, chart_data: pd.DataFrame = None) -> Optional[BlackOrderAnalysis]:
        """
        Main analysis function

        Args:
            market_depth: Market depth data (uses session_state.market_depth_data if None)
            chart_data: OHLCV data (uses session_state.chart_data if None)

        Returns:
            BlackOrderAnalysis with all findings
        """
        try:
            # Get data from session state if not provided
            if market_depth is None:
                market_depth = st.session_state.get('market_depth_data')

            if market_depth is None:
                return None

            if chart_data is None:
                chart_data = st.session_state.get('chart_data')
                if chart_data is None:
                    chart_data = st.session_state.get('nifty_df')

            # Analyze depth imbalance
            depth_imbalance = self.analyze_depth_imbalance(market_depth)

            # Detect all signal types
            signals = []

            # 1. Iceberg orders
            iceberg_signals = self.detect_iceberg_orders(market_depth)
            signals.extend(iceberg_signals)

            # 2. Absorption patterns
            absorption_signals = self.detect_absorption(market_depth, chart_data)
            signals.extend(absorption_signals)

            # 3. Large orders
            large_order_signals = self.detect_large_orders(market_depth)
            signals.extend(large_order_signals)

            # Extract zones
            absorption_zones = [s.price_level for s in signals if 'ABSORPTION' in s.signal_type or 'HIDDEN' in s.signal_type]
            iceberg_levels = [s.price_level for s in signals if 'ICEBERG' in s.signal_type]

            # Determine institutional presence
            if len(signals) >= 5:
                institutional_presence = 'STRONG'
            elif len(signals) >= 3:
                institutional_presence = 'MODERATE'
            elif len(signals) >= 1:
                institutional_presence = 'WEAK'
            else:
                institutional_presence = 'NONE'

            # Calculate score and bias
            score, smart_money_bias = self.calculate_score(signals, depth_imbalance)

            # Build description
            if signals:
                top_signal = max(signals, key=lambda x: x.strength)
                description = top_signal.trade_implication
            elif depth_imbalance:
                if depth_imbalance.dominant_side == 'BIDS':
                    description = f"Bid-heavy depth ({depth_imbalance.imbalance_percent:.1f}% imbalance) - buyers in control"
                elif depth_imbalance.dominant_side == 'ASKS':
                    description = f"Ask-heavy depth ({abs(depth_imbalance.imbalance_percent):.1f}% imbalance) - sellers in control"
                else:
                    description = "Balanced depth - no clear institutional bias"
            else:
                description = "Insufficient depth data for analysis"

            return BlackOrderAnalysis(
                signals=signals,
                depth_imbalance=depth_imbalance,
                institutional_presence=institutional_presence,
                smart_money_bias=smart_money_bias,
                absorption_zones=absorption_zones,
                iceberg_levels=iceberg_levels,
                score=score,
                description=description
            )

        except Exception as e:
            st.error(f"Black Order Analysis error: {e}")
            return None


# Singleton instance
_black_order_detector = None


def get_black_order_detector() -> BlackOrderDetector:
    """Get singleton Black Order Detector instance"""
    global _black_order_detector
    if _black_order_detector is None:
        _black_order_detector = BlackOrderDetector()
    return _black_order_detector


def analyze_black_orders(market_depth: Dict = None, chart_data: pd.DataFrame = None) -> Optional[BlackOrderAnalysis]:
    """
    Convenience function to analyze black/hidden orders

    Usage:
        from src.black_order_detector import analyze_black_orders

        result = analyze_black_orders()
        if result:
            print(f"Institutional Presence: {result.institutional_presence}")
            print(f"Smart Money Bias: {result.smart_money_bias}")
            for signal in result.signals:
                print(f"  {signal.signal_type}: {signal.description}")
    """
    detector = get_black_order_detector()
    return detector.analyze(market_depth, chart_data)
