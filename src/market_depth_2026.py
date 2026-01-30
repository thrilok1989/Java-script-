"""
Market Depth 2026 Module
========================
Modern market depth analysis replacing outdated 2025 methods.

STOP TRUSTING (2025):
- Static bid/ask quantity
- Single snapshot depth
- Depth is spoofed heavily now

UPDATE TO (2026):
- Order Flow CHANGE (not size)
- Absorption Detection
- Depth Imbalance TREND

Output:
- Aggression: Buyers / Sellers / Neutral
- Absorption: Yes / No
- Spoof Risk: High / Low
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pytz

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW CHANGE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class OrderFlowChangeTracker:
    """
    Track order flow CHANGE (not absolute size).

    Key Insight:
    - Track orders appearing & disappearing
    - Repeated add-cancel behavior = spoofing
    - Real intent vs fake liquidity
    """

    def __init__(self, history_size: int = 60):
        """
        Initialize tracker.

        Args:
            history_size: Number of depth snapshots to keep
        """
        self.bid_history = deque(maxlen=history_size)
        self.ask_history = deque(maxlen=history_size)
        self.bid_qty_history = deque(maxlen=history_size)
        self.ask_qty_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.add_cancel_events = deque(maxlen=100)  # Track add-cancel patterns

    def update(
        self,
        bid_levels: List[Dict],
        ask_levels: List[Dict],
        timestamp: datetime = None
    ):
        """
        Update with new depth snapshot.

        bid_levels: [{'price': x, 'quantity': y}, ...]
        ask_levels: [{'price': x, 'quantity': y}, ...]
        """
        ts = timestamp or datetime.now(IST)

        # Store current state
        self.bid_history.append(bid_levels)
        self.ask_history.append(ask_levels)
        self.timestamps.append(ts)

        # Calculate total quantities
        total_bid = sum(level.get('quantity', 0) for level in bid_levels)
        total_ask = sum(level.get('quantity', 0) for level in ask_levels)
        self.bid_qty_history.append(total_bid)
        self.ask_qty_history.append(total_ask)

        # Detect add-cancel events
        if len(self.bid_history) >= 2:
            self._detect_add_cancel_patterns()

    def _detect_add_cancel_patterns(self):
        """
        Detect repeated add-cancel behavior (spoofing indicator).

        Spoof pattern:
        1. Large order appears at a price level
        2. Order disappears before execution
        3. Pattern repeats
        """
        prev_bids = self.bid_history[-2]
        curr_bids = self.bid_history[-1]
        prev_asks = self.ask_history[-2]
        curr_asks = self.ask_history[-1]

        # Create price-quantity maps
        prev_bid_map = {l['price']: l['quantity'] for l in prev_bids}
        curr_bid_map = {l['price']: l['quantity'] for l in curr_bids}
        prev_ask_map = {l['price']: l['quantity'] for l in prev_asks}
        curr_ask_map = {l['price']: l['quantity'] for l in curr_asks}

        # Detect significant additions then removals (potential spoofs)
        for price, qty in prev_bid_map.items():
            if qty > 1000:  # Significant size
                curr_qty = curr_bid_map.get(price, 0)
                if curr_qty < qty * 0.5:  # More than 50% removed
                    self.add_cancel_events.append({
                        'timestamp': self.timestamps[-1],
                        'side': 'BID',
                        'price': price,
                        'added_qty': qty,
                        'cancelled_qty': qty - curr_qty,
                        'type': 'CANCEL_AFTER_ADD'
                    })

        for price, qty in prev_ask_map.items():
            if qty > 1000:
                curr_qty = curr_ask_map.get(price, 0)
                if curr_qty < qty * 0.5:
                    self.add_cancel_events.append({
                        'timestamp': self.timestamps[-1],
                        'side': 'ASK',
                        'price': price,
                        'added_qty': qty,
                        'cancelled_qty': qty - curr_qty,
                        'type': 'CANCEL_AFTER_ADD'
                    })

    def get_flow_change(self, period: int = 5) -> Dict[str, Any]:
        """
        Calculate order flow change over period.

        Returns change in bid/ask quantities and interpretation.
        """
        if len(self.bid_qty_history) < period + 1:
            return {
                'bid_change': 0, 'ask_change': 0, 'net_flow': 0,
                'interpretation': 'INSUFFICIENT_DATA'
            }

        bid_change = self.bid_qty_history[-1] - self.bid_qty_history[-period-1]
        ask_change = self.ask_qty_history[-1] - self.ask_qty_history[-period-1]
        net_flow = bid_change - ask_change

        # Calculate change rate
        avg_bid = sum(list(self.bid_qty_history)[-period:]) / period
        avg_ask = sum(list(self.ask_qty_history)[-period:]) / period

        bid_change_pct = (bid_change / avg_bid * 100) if avg_bid > 0 else 0
        ask_change_pct = (ask_change / avg_ask * 100) if avg_ask > 0 else 0

        # Interpretation
        interpretation = self._interpret_flow_change(bid_change_pct, ask_change_pct)

        return {
            'bid_change': bid_change,
            'ask_change': ask_change,
            'net_flow': net_flow,
            'bid_change_pct': round(bid_change_pct, 2),
            'ask_change_pct': round(ask_change_pct, 2),
            'period': period,
            'interpretation': interpretation
        }

    def get_spoof_risk(self) -> Dict[str, Any]:
        """
        Calculate spoof risk based on add-cancel patterns.
        """
        if not self.add_cancel_events:
            return {'risk_level': 'LOW', 'recent_events': 0, 'details': []}

        # Count recent events (last 5 minutes)
        now = datetime.now(IST)
        cutoff = now - timedelta(minutes=5)
        recent_events = [
            e for e in self.add_cancel_events
            if e['timestamp'] > cutoff
        ]

        event_count = len(recent_events)

        # Calculate risk level
        if event_count >= 10:
            risk_level = 'CRITICAL'
        elif event_count >= 5:
            risk_level = 'HIGH'
        elif event_count >= 2:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'

        # Side bias (more bid spoofs or ask spoofs?)
        bid_spoofs = sum(1 for e in recent_events if e['side'] == 'BID')
        ask_spoofs = sum(1 for e in recent_events if e['side'] == 'ASK')

        if bid_spoofs > ask_spoofs * 1.5:
            spoof_bias = 'BID_HEAVY'  # Fake support being created
        elif ask_spoofs > bid_spoofs * 1.5:
            spoof_bias = 'ASK_HEAVY'  # Fake resistance being created
        else:
            spoof_bias = 'BALANCED'

        return {
            'risk_level': risk_level,
            'recent_events': event_count,
            'spoof_bias': spoof_bias,
            'bid_spoofs': bid_spoofs,
            'ask_spoofs': ask_spoofs,
            'warning': f'High spoofing activity detected ({event_count} events)' if risk_level in ['HIGH', 'CRITICAL'] else None
        }

    def _interpret_flow_change(self, bid_pct: float, ask_pct: float) -> str:
        """Interpret flow change"""
        threshold = 5  # 5% change threshold

        if bid_pct > threshold and ask_pct < -threshold:
            return 'BUYER_AGGRESSION'  # Bids increasing, asks decreasing
        elif ask_pct > threshold and bid_pct < -threshold:
            return 'SELLER_AGGRESSION'
        elif bid_pct > threshold and ask_pct > threshold:
            return 'BOTH_ADDING'  # Liquidity increasing
        elif bid_pct < -threshold and ask_pct < -threshold:
            return 'BOTH_REMOVING'  # Liquidity decreasing
        elif bid_pct > threshold:
            return 'BID_STRENGTHENING'
        elif ask_pct > threshold:
            return 'ASK_STRENGTHENING'
        elif bid_pct < -threshold:
            return 'BID_WEAKENING'
        elif ask_pct < -threshold:
            return 'ASK_WEAKENING'
        else:
            return 'STABLE'


# ═══════════════════════════════════════════════════════════════════════════════
# ABSORPTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AbsorptionDetector:
    """
    Detect absorption (large orders absorbing market orders).

    Logic:
    - Large orders absorbing market orders
    - Price not moving despite volume

    Interpretation:
    - Absorption near highs → sellers strong
    - Absorption near lows → buyers strong
    """

    def __init__(self, history_size: int = 60):
        self.price_history = deque(maxlen=history_size)
        self.volume_history = deque(maxlen=history_size)
        self.bid_qty_history = deque(maxlen=history_size)
        self.ask_qty_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

    def update(
        self,
        price: float,
        volume: float,
        total_bid_qty: float,
        total_ask_qty: float,
        timestamp: datetime = None
    ):
        """Update with new data"""
        ts = timestamp or datetime.now(IST)
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.bid_qty_history.append(total_bid_qty)
        self.ask_qty_history.append(total_ask_qty)
        self.timestamps.append(ts)

    def detect_absorption(self, period: int = 5) -> Dict[str, Any]:
        """
        Detect absorption based on volume vs price movement.

        Absorption = High volume but no price movement
        """
        if len(self.price_history) < period + 1:
            return {
                'absorption_detected': False,
                'type': None,
                'interpretation': 'INSUFFICIENT_DATA'
            }

        # Price movement
        price_change = abs(self.price_history[-1] - self.price_history[-period-1])
        price_change_pct = (price_change / self.price_history[-period-1] * 100)

        # Volume activity
        recent_vol = sum(list(self.volume_history)[-period:])
        avg_vol_per_bar = recent_vol / period

        # Calculate expected volume for price move
        # (simplified: expect 0.1% vol increase per 0.1% price move)
        expected_vol_ratio = max(1, price_change_pct * 10)

        # Compare actual volume
        if len(self.volume_history) > period * 2:
            baseline_vol = sum(list(self.volume_history)[-period*2:-period]) / period
            actual_vol_ratio = avg_vol_per_bar / baseline_vol if baseline_vol > 0 else 1
        else:
            actual_vol_ratio = 1

        # Absorption detection logic
        absorption_detected = False
        absorption_type = None

        # High volume but low price movement = absorption
        if actual_vol_ratio > 1.5 and price_change_pct < 0.2:
            absorption_detected = True

            # Determine absorption type by price position
            recent_high = max(list(self.price_history)[-period:])
            recent_low = min(list(self.price_history)[-period:])
            price_range = recent_high - recent_low
            current_price = self.price_history[-1]

            if price_range > 0:
                price_position = (current_price - recent_low) / price_range
            else:
                price_position = 0.5

            if price_position > 0.7:
                absorption_type = 'SELLING_ABSORPTION'
                interpretation = 'Sellers absorbing at highs. Resistance holding.'
            elif price_position < 0.3:
                absorption_type = 'BUYING_ABSORPTION'
                interpretation = 'Buyers absorbing at lows. Support holding.'
            else:
                absorption_type = 'NEUTRAL_ABSORPTION'
                interpretation = 'Absorption in middle range. Watch for direction.'
        else:
            interpretation = 'No absorption detected'

        return {
            'absorption_detected': absorption_detected,
            'type': absorption_type,
            'price_change_pct': round(price_change_pct, 3),
            'volume_ratio': round(actual_vol_ratio, 2),
            'interpretation': interpretation,
            'strength': self._calculate_absorption_strength(actual_vol_ratio, price_change_pct) if absorption_detected else 0
        }

    def _calculate_absorption_strength(self, vol_ratio: float, price_pct: float) -> int:
        """Calculate absorption strength 0-100"""
        # Higher vol ratio with lower price move = stronger absorption
        if price_pct == 0:
            base_strength = 100
        else:
            base_strength = min(100, int(vol_ratio / price_pct * 10))

        return min(100, max(0, base_strength))


# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH IMBALANCE TREND TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class DepthImbalanceTrendTracker:
    """
    Track depth imbalance TREND (not absolute).

    Key Question: Is bid dominance increasing or fading?

    Helps with timing entries in expansion moves.
    """

    def __init__(self, history_size: int = 60):
        self.imbalance_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

    def update(self, total_bid_qty: float, total_ask_qty: float, timestamp: datetime = None):
        """Update with new imbalance data"""
        ts = timestamp or datetime.now(IST)

        # Calculate imbalance ratio (-1 to +1 scale)
        total = total_bid_qty + total_ask_qty
        if total > 0:
            imbalance = (total_bid_qty - total_ask_qty) / total
        else:
            imbalance = 0

        self.imbalance_history.append(imbalance)
        self.timestamps.append(ts)

    def get_imbalance_trend(self, period: int = 10) -> Dict[str, Any]:
        """
        Calculate imbalance trend over period.

        Returns:
        - Current imbalance
        - Trend direction
        - Trend strength
        """
        if len(self.imbalance_history) < period + 1:
            return {
                'current_imbalance': 0,
                'trend': 'INSUFFICIENT_DATA',
                'trend_strength': 0
            }

        current = self.imbalance_history[-1]
        start = self.imbalance_history[-period-1]

        # Recent average vs older average
        recent_avg = sum(list(self.imbalance_history)[-5:]) / 5
        older_avg = sum(list(self.imbalance_history)[-period:-period+5]) / 5

        trend_change = recent_avg - older_avg

        # Determine trend
        if trend_change > 0.05:
            trend = 'BID_STRENGTHENING'
            trend_strength = min(100, int(trend_change * 200))
        elif trend_change < -0.05:
            trend = 'ASK_STRENGTHENING'
            trend_strength = min(100, int(abs(trend_change) * 200))
        else:
            trend = 'STABLE'
            trend_strength = 0

        # Current dominance
        if current > 0.1:
            dominance = 'BID_DOMINANT'
        elif current < -0.1:
            dominance = 'ASK_DOMINANT'
        else:
            dominance = 'BALANCED'

        return {
            'current_imbalance': round(current, 3),
            'current_imbalance_pct': round(current * 100, 1),
            'trend': trend,
            'trend_strength': trend_strength,
            'dominance': dominance,
            'recent_avg': round(recent_avg, 3),
            'older_avg': round(older_avg, 3),
            'interpretation': self._interpret_trend(trend, dominance)
        }

    def _interpret_trend(self, trend: str, dominance: str) -> str:
        """Interpret trend and dominance"""
        if trend == 'BID_STRENGTHENING' and dominance == 'BID_DOMINANT':
            return 'Strong bullish momentum building'
        elif trend == 'ASK_STRENGTHENING' and dominance == 'ASK_DOMINANT':
            return 'Strong bearish momentum building'
        elif trend == 'BID_STRENGTHENING' and dominance == 'ASK_DOMINANT':
            return 'Bearish weakening, potential reversal'
        elif trend == 'ASK_STRENGTHENING' and dominance == 'BID_DOMINANT':
            return 'Bullish weakening, potential reversal'
        elif trend == 'STABLE':
            if dominance == 'BID_DOMINANT':
                return 'Stable bullish bias'
            elif dominance == 'ASK_DOMINANT':
                return 'Stable bearish bias'
            else:
                return 'Balanced, no clear direction'
        else:
            return 'Mixed signals'


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED MARKET DEPTH 2026 ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDepth2026Analyzer:
    """
    Unified market depth analyzer for 2026.

    Output Format:
    - Aggression: Buyers / Sellers / Neutral
    - Absorption: Yes / No
    - Spoof Risk: High / Low
    """

    def __init__(self):
        self.flow_tracker = OrderFlowChangeTracker()
        self.absorption_detector = AbsorptionDetector()
        self.imbalance_tracker = DepthImbalanceTrendTracker()

    def update(
        self,
        bid_levels: List[Dict],
        ask_levels: List[Dict],
        price: float,
        volume: float,
        timestamp: datetime = None
    ):
        """
        Update all trackers with new depth data.

        bid_levels: [{'price': x, 'quantity': y}, ...]
        ask_levels: [{'price': x, 'quantity': y}, ...]
        """
        ts = timestamp or datetime.now(IST)

        # Calculate totals
        total_bid = sum(level.get('quantity', 0) for level in bid_levels)
        total_ask = sum(level.get('quantity', 0) for level in ask_levels)

        # Update trackers
        self.flow_tracker.update(bid_levels, ask_levels, ts)
        self.absorption_detector.update(price, volume, total_bid, total_ask, ts)
        self.imbalance_tracker.update(total_bid, total_ask, ts)

    def analyze(self, period: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive 2026 market depth analysis.

        Returns actionable output with Aggression, Absorption, Spoof Risk.
        """
        # Get all metrics
        flow_change = self.flow_tracker.get_flow_change(period)
        spoof_risk = self.flow_tracker.get_spoof_risk()
        absorption = self.absorption_detector.detect_absorption(period)
        imbalance_trend = self.imbalance_tracker.get_imbalance_trend(period * 2)

        # Determine aggression
        aggression = self._determine_aggression(flow_change, imbalance_trend)

        # Compile output
        return {
            'timestamp': datetime.now(IST).isoformat(),
            # Core 2026 metrics
            'flow_change': flow_change,
            'spoof_risk': spoof_risk,
            'absorption': absorption,
            'imbalance_trend': imbalance_trend,
            # Actionable outputs
            'aggression': aggression,
            'absorption_active': absorption.get('absorption_detected', False),
            'spoof_warning': spoof_risk.get('risk_level') in ['HIGH', 'CRITICAL'],
            # Summary for dashboard
            'summary': self._generate_summary(aggression, absorption, spoof_risk)
        }

    def _determine_aggression(
        self,
        flow_change: Dict,
        imbalance_trend: Dict
    ) -> Dict[str, Any]:
        """Determine aggression: Buyers / Sellers / Neutral"""
        flow_interp = flow_change.get('interpretation', '')
        trend = imbalance_trend.get('trend', '')
        dominance = imbalance_trend.get('dominance', '')

        # Score-based determination
        buyer_score = 0
        seller_score = 0

        # Flow change contribution
        if 'BUYER' in flow_interp or 'BID_STRENGTHENING' in flow_interp:
            buyer_score += 30
        elif 'SELLER' in flow_interp or 'ASK_STRENGTHENING' in flow_interp:
            seller_score += 30
        elif 'BID_WEAKENING' in flow_interp:
            seller_score += 15
        elif 'ASK_WEAKENING' in flow_interp:
            buyer_score += 15

        # Imbalance trend contribution
        if trend == 'BID_STRENGTHENING':
            buyer_score += 25
        elif trend == 'ASK_STRENGTHENING':
            seller_score += 25

        # Current dominance contribution
        if dominance == 'BID_DOMINANT':
            buyer_score += 20
        elif dominance == 'ASK_DOMINANT':
            seller_score += 20

        # Determine aggression
        score_diff = buyer_score - seller_score

        if score_diff > 20:
            aggression = 'BUYERS'
            confidence = min(100, 50 + score_diff)
        elif score_diff < -20:
            aggression = 'SELLERS'
            confidence = min(100, 50 + abs(score_diff))
        else:
            aggression = 'NEUTRAL'
            confidence = 50

        return {
            'aggression': aggression,
            'buyer_score': buyer_score,
            'seller_score': seller_score,
            'confidence': confidence
        }

    def _generate_summary(
        self,
        aggression: Dict,
        absorption: Dict,
        spoof: Dict
    ) -> str:
        """Generate summary string"""
        agg_emoji = '🟢' if aggression['aggression'] == 'BUYERS' else '🔴' if aggression['aggression'] == 'SELLERS' else '⚪'
        abs_emoji = '🛡️' if absorption.get('absorption_detected') else ''
        spoof_emoji = '⚠️' if spoof.get('risk_level') in ['HIGH', 'CRITICAL'] else ''

        parts = [
            f"{agg_emoji} {aggression['aggression']}"
        ]

        if absorption.get('absorption_detected'):
            parts.append(f"{abs_emoji} ABSORPTION")

        if spoof.get('risk_level') in ['HIGH', 'CRITICAL']:
            parts.append(f"{spoof_emoji} SPOOF RISK")

        return ' | '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_depth_analyzer() -> MarketDepth2026Analyzer:
    """Create a new market depth analyzer instance"""
    return MarketDepth2026Analyzer()


def quick_depth_analysis(
    bid_levels: List[Dict],
    ask_levels: List[Dict],
    price: float,
    volume: float
) -> Dict[str, Any]:
    """
    Quick one-shot depth analysis.

    Note: For best results, use the analyzer with historical updates.
    """
    analyzer = MarketDepth2026Analyzer()
    analyzer.update(bid_levels, ask_levels, price, volume)
    return analyzer.analyze()


def detect_spoofing(
    bid_levels_history: List[List[Dict]],
    ask_levels_history: List[List[Dict]]
) -> Dict[str, Any]:
    """
    Analyze history for spoofing patterns.
    """
    tracker = OrderFlowChangeTracker()
    for bids, asks in zip(bid_levels_history, ask_levels_history):
        tracker.update(bids, asks)
    return tracker.get_spoof_risk()


def detect_absorption_quick(
    price_history: List[float],
    volume_history: List[float],
    bid_qty_history: List[float],
    ask_qty_history: List[float]
) -> Dict[str, Any]:
    """
    Quick absorption detection from historical data.
    """
    detector = AbsorptionDetector()
    for p, v, b, a in zip(price_history, volume_history, bid_qty_history, ask_qty_history):
        detector.update(p, v, b, a)
    return detector.detect_absorption()


def get_imbalance_trend(
    bid_qty_history: List[float],
    ask_qty_history: List[float]
) -> Dict[str, Any]:
    """
    Get imbalance trend from historical quantities.
    """
    tracker = DepthImbalanceTrendTracker()
    for b, a in zip(bid_qty_history, ask_qty_history):
        tracker.update(b, a)
    return tracker.get_imbalance_trend()
