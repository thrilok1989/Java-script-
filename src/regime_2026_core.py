"""
2026 Regime Core Module
=======================
Central module for 2026 trading regime updates.
Implements modern market analysis logic replacing outdated 2025 methods.

Key Philosophy:
- Fewer but higher-quality decisions
- Focus on velocity/change over static values
- Time-aware trading logic
- Clear actionable outputs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-BASED TRADING LOGIC (2026)
# ═══════════════════════════════════════════════════════════════════════════════

class TimeBasedLogic2026:
    """
    Hard rules for time-based trading behavior.

    Time Zones (IST):
    - 9:15-9:45  : TRAP ZONE - High fake moves, avoid entries
    - 10:00-11:30: EXPANSION ZONE - Best for directional trades
    - 12:00-13:30: DECAY ZONE - Premium decay, range-bound
    - 14:15+     : TREND/REVERSAL ZONE - Late day moves
    """

    TIME_ZONES = {
        'trap': {'start': '09:15', 'end': '09:45', 'behavior': 'TRAPS', 'action': 'AVOID'},
        'expansion': {'start': '10:00', 'end': '11:30', 'behavior': 'EXPANSION', 'action': 'TRADE'},
        'decay': {'start': '12:00', 'end': '13:30', 'behavior': 'DECAY', 'action': 'RANGE'},
        'trend_reversal': {'start': '14:15', 'end': '15:30', 'behavior': 'TREND/REVERSAL', 'action': 'SELECTIVE'}
    }

    @classmethod
    def get_current_zone(cls) -> Dict[str, Any]:
        """Get current time zone and its characteristics"""
        now = datetime.now(IST)
        current_time = now.strftime('%H:%M')

        for zone_name, zone_info in cls.TIME_ZONES.items():
            if zone_info['start'] <= current_time <= zone_info['end']:
                return {
                    'zone': zone_name,
                    'behavior': zone_info['behavior'],
                    'action': zone_info['action'],
                    'time': current_time,
                    'warning': cls._get_zone_warning(zone_name)
                }

        # Gap periods
        if current_time < '09:15':
            return {'zone': 'pre_market', 'behavior': 'PRE-MARKET', 'action': 'WAIT', 'time': current_time, 'warning': 'Market not open'}
        elif '09:45' < current_time < '10:00':
            return {'zone': 'transition_1', 'behavior': 'TRANSITION', 'action': 'PREPARE', 'time': current_time, 'warning': 'Preparing for expansion'}
        elif '11:30' < current_time < '12:00':
            return {'zone': 'transition_2', 'behavior': 'TRANSITION', 'action': 'REDUCE', 'time': current_time, 'warning': 'Entering decay zone'}
        elif '13:30' < current_time < '14:15':
            return {'zone': 'transition_3', 'behavior': 'TRANSITION', 'action': 'WAIT', 'time': current_time, 'warning': 'Wait for late session'}
        else:
            return {'zone': 'post_market', 'behavior': 'CLOSED', 'action': 'NONE', 'time': current_time, 'warning': 'Market closed'}

    @classmethod
    def _get_zone_warning(cls, zone: str) -> str:
        """Get warning message for each zone"""
        warnings = {
            'trap': '⚠️ TRAP ZONE: High probability of fake moves. Avoid new entries.',
            'expansion': '✅ EXPANSION ZONE: Best time for directional trades.',
            'decay': '⏳ DECAY ZONE: Premium decay accelerates. Range-bound likely.',
            'trend_reversal': '🎯 TREND/REVERSAL: Late moves can be strong. Be selective.'
        }
        return warnings.get(zone, '')

    @classmethod
    def should_trade(cls, signal_type: str = 'directional') -> Tuple[bool, str]:
        """
        Check if trading is recommended in current time zone.
        Returns (should_trade, reason)
        """
        zone_info = cls.get_current_zone()
        zone = zone_info['zone']

        # Trap zone - mostly avoid
        if zone == 'trap':
            if signal_type == 'scalp':
                return False, "Trap zone: Scalps are high-risk due to fake moves"
            return False, "Trap zone: Wait for 10:00 AM for cleaner moves"

        # Expansion zone - best for trading
        if zone == 'expansion':
            return True, "Expansion zone: Good time for directional trades"

        # Decay zone - range strategies only
        if zone == 'decay':
            if signal_type == 'directional':
                return False, "Decay zone: Directional moves unlikely. Consider range strategies"
            return True, "Decay zone: Range-bound strategies may work"

        # Trend/Reversal zone - selective
        if zone == 'trend_reversal':
            return True, "Late session: Strong moves possible. Be selective with entries"

        # Transitions - wait
        if 'transition' in zone:
            return False, f"Transition period: Wait for clear zone entry"

        return False, "Market not in active trading hours"


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME CLASSIFICATION (2026)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRegime2026:
    """
    2026 Market Regime Classification

    Focus on:
    - Volatility state (not just value)
    - Writer control assessment
    - Expansion vs contraction detection
    """

    @staticmethod
    def classify_volatility_state(
        atr_current: float,
        atr_avg_20: float,
        atr_change_pct: float,
        vix: float = None
    ) -> Dict[str, Any]:
        """
        Classify volatility state based on ATR behavior.

        States:
        - COMPRESSED: ATR < 70% of avg (expansion coming)
        - NORMAL: ATR 70-130% of avg
        - ELEVATED: ATR > 130% of avg
        - EXPANDING: ATR increasing rapidly
        - CONTRACTING: ATR decreasing rapidly
        """
        atr_ratio = atr_current / atr_avg_20 if atr_avg_20 > 0 else 1.0

        # Base state from ratio
        if atr_ratio < 0.7:
            base_state = 'COMPRESSED'
        elif atr_ratio > 1.3:
            base_state = 'ELEVATED'
        else:
            base_state = 'NORMAL'

        # Velocity state
        if atr_change_pct > 15:
            velocity_state = 'EXPANDING'
        elif atr_change_pct < -15:
            velocity_state = 'CONTRACTING'
        else:
            velocity_state = 'STABLE'

        # VIX adjustment if available
        vix_state = None
        if vix is not None:
            if vix < 12:
                vix_state = 'LOW_FEAR'
            elif vix > 20:
                vix_state = 'HIGH_FEAR'
            else:
                vix_state = 'NORMAL_FEAR'

        return {
            'base_state': base_state,
            'velocity_state': velocity_state,
            'atr_ratio': round(atr_ratio, 2),
            'atr_change_pct': round(atr_change_pct, 2),
            'vix_state': vix_state,
            'trading_implication': MarketRegime2026._get_volatility_implication(base_state, velocity_state)
        }

    @staticmethod
    def _get_volatility_implication(base: str, velocity: str) -> str:
        """Get trading implication from volatility state"""
        implications = {
            ('COMPRESSED', 'STABLE'): 'Breakout imminent. Prepare for expansion.',
            ('COMPRESSED', 'EXPANDING'): 'Breakout starting. Look for directional entry.',
            ('COMPRESSED', 'CONTRACTING'): 'Extreme compression. Big move loading.',
            ('NORMAL', 'STABLE'): 'Normal conditions. Follow trend.',
            ('NORMAL', 'EXPANDING'): 'Volatility picking up. Widen stops.',
            ('NORMAL', 'CONTRACTING'): 'Settling down. Range may form.',
            ('ELEVATED', 'STABLE'): 'High vol regime. Use tight targets.',
            ('ELEVATED', 'EXPANDING'): 'Extreme volatility. Reduce size or wait.',
            ('ELEVATED', 'CONTRACTING'): 'Vol cooling off. Trend may resume.',
        }
        return implications.get((base, velocity), 'Monitor conditions')

    @staticmethod
    def assess_writer_control(
        oi_ce_change: float,
        oi_pe_change: float,
        premium_ce_change_pct: float,
        premium_pe_change_pct: float,
        price_move_pct: float
    ) -> Dict[str, Any]:
        """
        Assess if option writers are in control.

        Writer Control Strong:
        - OI increasing (writers adding)
        - Premiums decaying despite price moves
        - Range-bound behavior

        Writer Control Weak:
        - OI decreasing (writers exiting)
        - Premiums expanding with price
        - Trending behavior
        """
        # OI behavior
        total_oi_change = oi_ce_change + oi_pe_change
        oi_adding = total_oi_change > 0

        # Premium behavior
        avg_premium_change = (premium_ce_change_pct + premium_pe_change_pct) / 2
        premiums_decaying = avg_premium_change < -2  # More than 2% decay

        # Price vs Premium divergence (key 2026 signal)
        if abs(price_move_pct) > 0.3 and premiums_decaying:
            divergence = 'TRAP_WARNING'  # Price moved but premiums didn't follow
        elif abs(price_move_pct) > 0.3 and not premiums_decaying:
            divergence = 'GENUINE_MOVE'
        else:
            divergence = 'NORMAL'

        # Control assessment
        if oi_adding and premiums_decaying:
            control = 'STRONG'
            implication = 'Writers in control. Range-bound likely.'
        elif not oi_adding and not premiums_decaying:
            control = 'WEAK'
            implication = 'Writers exiting. Trend move possible.'
        else:
            control = 'MIXED'
            implication = 'Mixed signals. Wait for clarity.'

        return {
            'writer_control': control,
            'oi_behavior': 'ADDING' if oi_adding else 'EXITING',
            'premium_behavior': 'DECAYING' if premiums_decaying else 'HOLDING/EXPANDING',
            'price_premium_divergence': divergence,
            'implication': implication,
            'confidence': 'HIGH' if control in ['STRONG', 'WEAK'] else 'LOW'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2026 MARKET MOOD CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class MarketMood2026:
    """
    Classify market mood into simple actionable states.

    Moods:
    - CALM: Low activity, stable prices
    - STRESSED: High activity, writers under pressure
    - AGGRESSIVE: Strong directional move underway
    """

    @staticmethod
    def classify_mood(
        oi_velocity: float,  # OI change per minute
        premium_velocity: float,  # Premium change per minute
        price_velocity: float,  # Price change per minute
        volume_ratio: float,  # Current vol / avg vol
        absorption_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Classify market mood based on velocities and activity.
        """
        # Activity score (0-100)
        activity_score = 0

        # OI velocity contribution
        if abs(oi_velocity) > 50000:  # Large OI change per minute
            activity_score += 30
        elif abs(oi_velocity) > 20000:
            activity_score += 15

        # Premium velocity contribution
        if abs(premium_velocity) > 5:  # More than 5% per minute
            activity_score += 25
        elif abs(premium_velocity) > 2:
            activity_score += 12

        # Price velocity contribution
        if abs(price_velocity) > 0.2:  # More than 0.2% per minute
            activity_score += 25
        elif abs(price_velocity) > 0.1:
            activity_score += 12

        # Volume contribution
        if volume_ratio > 2.0:
            activity_score += 20
        elif volume_ratio > 1.5:
            activity_score += 10

        # Classify mood
        if activity_score < 30:
            mood = 'CALM'
            action = 'Range strategies or wait'
            emoji = '😌'
        elif activity_score < 60:
            if absorption_detected:
                mood = 'STRESSED'
                action = 'Watch for breakout direction'
                emoji = '😰'
            else:
                mood = 'ACTIVE'
                action = 'Follow momentum'
                emoji = '🎯'
        else:
            mood = 'AGGRESSIVE'
            action = 'Trade with trend, tight stops'
            emoji = '🔥'

        return {
            'mood': mood,
            'emoji': emoji,
            'activity_score': activity_score,
            'action': action,
            'metrics': {
                'oi_velocity': oi_velocity,
                'premium_velocity': premium_velocity,
                'price_velocity': price_velocity,
                'volume_ratio': volume_ratio
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOWED PLAYBOOK GENERATOR (2026)
# ═══════════════════════════════════════════════════════════════════════════════

class PlaybookGenerator2026:
    """
    Generate allowed playbook based on current market conditions.
    Prevents overtrading by limiting strategies to what works now.
    """

    PLAYBOOKS = {
        'BREAKOUT': {
            'description': 'Trade breakouts from compression',
            'entry': 'On break of range with volume',
            'sl': '1.5x ATR below breakout',
            'target': '2-3x ATR'
        },
        'TREND_FOLLOW': {
            'description': 'Follow established trend',
            'entry': 'Pullback to EMA20/50',
            'sl': 'Below previous swing',
            'target': 'Trail with ATR'
        },
        'RANGE_FADE': {
            'description': 'Fade range extremes',
            'entry': 'At range support/resistance',
            'sl': 'Beyond range boundary',
            'target': 'Opposite range boundary'
        },
        'PREMIUM_DECAY': {
            'description': 'Sell options for decay',
            'entry': 'ATM or slightly OTM',
            'sl': 'Based on underlying move',
            'target': 'Time decay'
        },
        'NO_TRADE': {
            'description': 'Conditions not favorable',
            'entry': 'WAIT',
            'sl': 'N/A',
            'target': 'N/A'
        }
    }

    @classmethod
    def get_allowed_playbook(
        cls,
        volatility_state: Dict,
        writer_control: Dict,
        market_mood: Dict,
        time_zone: Dict
    ) -> Dict[str, Any]:
        """
        Determine which playbook is allowed based on conditions.
        """
        allowed = []
        not_allowed = []
        primary = None

        # Time zone restrictions
        if time_zone['zone'] == 'trap':
            return {
                'primary': 'NO_TRADE',
                'allowed': [],
                'not_allowed': list(cls.PLAYBOOKS.keys()),
                'reason': 'Trap zone (9:15-9:45). Wait for cleaner conditions.',
                'playbook_details': cls.PLAYBOOKS['NO_TRADE']
            }

        vol_base = volatility_state.get('base_state', 'NORMAL')
        vol_velocity = volatility_state.get('velocity_state', 'STABLE')
        control = writer_control.get('writer_control', 'MIXED')
        mood = market_mood.get('mood', 'ACTIVE')

        # Compressed volatility + expanding = BREAKOUT
        if vol_base == 'COMPRESSED' and vol_velocity == 'EXPANDING':
            primary = 'BREAKOUT'
            allowed.append('BREAKOUT')
            not_allowed.extend(['RANGE_FADE', 'PREMIUM_DECAY'])

        # Strong writer control + decay zone = PREMIUM_DECAY
        elif control == 'STRONG' and time_zone['zone'] == 'decay':
            primary = 'PREMIUM_DECAY'
            allowed.append('PREMIUM_DECAY')
            allowed.append('RANGE_FADE')
            not_allowed.append('TREND_FOLLOW')

        # Weak writer control + expansion zone = TREND_FOLLOW
        elif control == 'WEAK' and time_zone['zone'] == 'expansion':
            primary = 'TREND_FOLLOW'
            allowed.append('TREND_FOLLOW')
            allowed.append('BREAKOUT')
            not_allowed.append('RANGE_FADE')

        # Calm mood + normal vol = RANGE_FADE
        elif mood == 'CALM' and vol_base == 'NORMAL':
            primary = 'RANGE_FADE'
            allowed.append('RANGE_FADE')
            allowed.append('PREMIUM_DECAY')
            not_allowed.append('BREAKOUT')

        # Default: TREND_FOLLOW
        else:
            primary = 'TREND_FOLLOW'
            allowed.append('TREND_FOLLOW')

        return {
            'primary': primary,
            'allowed': allowed,
            'not_allowed': not_allowed,
            'reason': cls._get_playbook_reason(primary, volatility_state, writer_control, market_mood),
            'playbook_details': cls.PLAYBOOKS.get(primary, cls.PLAYBOOKS['NO_TRADE'])
        }

    @classmethod
    def _get_playbook_reason(cls, playbook: str, vol: Dict, control: Dict, mood: Dict) -> str:
        """Generate reason for playbook selection"""
        reasons = {
            'BREAKOUT': f"Volatility compressed ({vol.get('atr_ratio', 1):.2f}x) and expanding. Breakout setup.",
            'TREND_FOLLOW': f"Writers {control.get('writer_control', 'MIXED').lower()}. Follow momentum.",
            'RANGE_FADE': f"Market {mood.get('mood', 'CALM').lower()}. Range-bound conditions.",
            'PREMIUM_DECAY': f"Writers strong. Premium decay zone. Sell options.",
            'NO_TRADE': "Conditions not favorable for any strategy."
        }
        return reasons.get(playbook, "Mixed conditions")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED 2026 ANALYSIS OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

class Regime2026Analyzer:
    """
    Main analyzer class that combines all 2026 regime logic.
    Produces unified output for dashboard display.
    """

    def __init__(self):
        self.time_logic = TimeBasedLogic2026()
        self.market_regime = MarketRegime2026()
        self.mood_classifier = MarketMood2026()
        self.playbook_generator = PlaybookGenerator2026()

    def analyze(
        self,
        price_data: pd.DataFrame,
        option_chain_data: Dict,
        market_depth_data: Dict = None,
        vix: float = None
    ) -> Dict[str, Any]:
        """
        Perform complete 2026 regime analysis.

        Returns unified output with:
        - Market Regime
        - Writer Stress
        - Volatility State
        - Allowed Playbook
        - NO TRADE warning if needed
        """
        # Time zone
        time_zone = self.time_logic.get_current_zone()

        # Extract metrics from data
        atr_current = self._calculate_atr(price_data, 14)
        atr_avg_20 = self._calculate_atr_avg(price_data, 14, 20)
        atr_change_pct = ((atr_current - atr_avg_20) / atr_avg_20 * 100) if atr_avg_20 > 0 else 0

        # Volatility state
        volatility_state = self.market_regime.classify_volatility_state(
            atr_current, atr_avg_20, atr_change_pct, vix
        )

        # Writer control (from option chain)
        oi_ce_change = option_chain_data.get('oi_ce_change', 0)
        oi_pe_change = option_chain_data.get('oi_pe_change', 0)
        premium_ce_change = option_chain_data.get('premium_ce_change_pct', 0)
        premium_pe_change = option_chain_data.get('premium_pe_change_pct', 0)
        price_move_pct = self._calculate_price_change_pct(price_data)

        writer_control = self.market_regime.assess_writer_control(
            oi_ce_change, oi_pe_change, premium_ce_change, premium_pe_change, price_move_pct
        )

        # Market mood
        oi_velocity = option_chain_data.get('oi_velocity', 0)
        premium_velocity = option_chain_data.get('premium_velocity', 0)
        price_velocity = self._calculate_price_velocity(price_data)
        volume_ratio = self._calculate_volume_ratio(price_data)
        absorption = market_depth_data.get('absorption_detected', False) if market_depth_data else False

        market_mood = self.mood_classifier.classify_mood(
            oi_velocity, premium_velocity, price_velocity, volume_ratio, absorption
        )

        # Allowed playbook
        playbook = self.playbook_generator.get_allowed_playbook(
            volatility_state, writer_control, market_mood, time_zone
        )

        # Should trade check
        should_trade, trade_reason = self.time_logic.should_trade()

        # Compile final output
        return {
            'timestamp': datetime.now(IST).isoformat(),
            'time_zone': time_zone,
            'volatility_state': volatility_state,
            'writer_control': writer_control,
            'market_mood': market_mood,
            'playbook': playbook,
            'should_trade': should_trade,
            'trade_reason': trade_reason,
            'no_trade_warning': not should_trade,
            'summary': self._generate_summary(time_zone, volatility_state, writer_control, market_mood, playbook)
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        if df is None or len(df) < period:
            return 0.0
        try:
            high = df['high'].values if 'high' in df.columns else df['High'].values
            low = df['low'].values if 'low' in df.columns else df['Low'].values
            close = df['close'].values if 'close' in df.columns else df['Close'].values

            tr1 = high - low
            tr2 = abs(high - np.roll(close, 1))
            tr3 = abs(low - np.roll(close, 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr[-period:])
            return float(atr)
        except:
            return 0.0

    def _calculate_atr_avg(self, df: pd.DataFrame, atr_period: int, avg_period: int) -> float:
        """Calculate average ATR over a period"""
        if df is None or len(df) < atr_period + avg_period:
            return self._calculate_atr(df, atr_period)
        try:
            atrs = []
            for i in range(avg_period):
                end_idx = len(df) - i
                start_idx = max(0, end_idx - atr_period)
                subset = df.iloc[start_idx:end_idx]
                atr = self._calculate_atr(subset, atr_period)
                atrs.append(atr)
            return float(np.mean(atrs))
        except:
            return self._calculate_atr(df, atr_period)

    def _calculate_price_change_pct(self, df: pd.DataFrame) -> float:
        """Calculate price change percentage"""
        if df is None or len(df) < 2:
            return 0.0
        try:
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            return float((close[-1] - close[-2]) / close[-2] * 100)
        except:
            return 0.0

    def _calculate_price_velocity(self, df: pd.DataFrame) -> float:
        """Calculate price velocity (change per minute)"""
        if df is None or len(df) < 5:
            return 0.0
        try:
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            # Last 5 minutes change
            return float((close[-1] - close[-5]) / close[-5] * 100 / 5)
        except:
            return 0.0

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume vs average"""
        if df is None or len(df) < 20:
            return 1.0
        try:
            volume = df['volume'].values if 'volume' in df.columns else df['Volume'].values
            current_vol = volume[-1]
            avg_vol = np.mean(volume[-20:])
            return float(current_vol / avg_vol) if avg_vol > 0 else 1.0
        except:
            return 1.0

    def _generate_summary(
        self,
        time_zone: Dict,
        volatility: Dict,
        writer: Dict,
        mood: Dict,
        playbook: Dict
    ) -> str:
        """Generate human-readable summary"""
        lines = []

        # Time zone
        lines.append(f"⏰ {time_zone['zone'].upper()}: {time_zone['behavior']}")

        # Volatility
        vol_emoji = '📊' if volatility['base_state'] == 'NORMAL' else '🔥' if volatility['base_state'] == 'ELEVATED' else '📉'
        lines.append(f"{vol_emoji} Volatility: {volatility['base_state']} ({volatility['velocity_state']})")

        # Writer control
        writer_emoji = '🛡️' if writer['writer_control'] == 'STRONG' else '⚔️' if writer['writer_control'] == 'WEAK' else '⚖️'
        lines.append(f"{writer_emoji} Writers: {writer['writer_control']}")

        # Mood
        lines.append(f"{mood['emoji']} Mood: {mood['mood']}")

        # Playbook
        lines.append(f"📋 Play: {playbook['primary']}")

        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime_2026_quick_analysis(
    price_df: pd.DataFrame = None,
    oi_ce_change: float = 0,
    oi_pe_change: float = 0,
    premium_ce_change: float = 0,
    premium_pe_change: float = 0,
    vix: float = None
) -> Dict[str, Any]:
    """
    Quick analysis function for easy integration.
    """
    analyzer = Regime2026Analyzer()

    option_chain_data = {
        'oi_ce_change': oi_ce_change,
        'oi_pe_change': oi_pe_change,
        'premium_ce_change_pct': premium_ce_change,
        'premium_pe_change_pct': premium_pe_change,
        'oi_velocity': (oi_ce_change + oi_pe_change) / 5,  # Assume 5 min
        'premium_velocity': (premium_ce_change + premium_pe_change) / 2 / 5
    }

    return analyzer.analyze(price_df, option_chain_data, None, vix)


def get_time_zone_warning() -> Dict[str, Any]:
    """Get current time zone and warning"""
    return TimeBasedLogic2026.get_current_zone()


def should_trade_now(signal_type: str = 'directional') -> Tuple[bool, str]:
    """Quick check if trading is recommended now"""
    return TimeBasedLogic2026.should_trade(signal_type)
