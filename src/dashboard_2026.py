"""
Dashboard 2026 Module
=====================
Streamlined dashboard showing ONLY essential information.

Home Screen MUST show:
- Market Regime
- Writer Stress
- Volatility State
- Allowed Playbook
- NO TRADE warning if needed

Everything else = collapsible.

Goal: "Fewer but higher-quality decisions"
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import pytz

# Import 2026 modules
from src.regime_2026_core import (
    Regime2026Analyzer,
    TimeBasedLogic2026,
    MarketRegime2026,
    MarketMood2026,
    PlaybookGenerator2026
)
from src.option_chain_2026 import OptionChain2026Analyzer
from src.market_depth_2026 import MarketDepth2026Analyzer
from src.chart_analysis_2026 import ChartAnalysis2026
from src.indicators_2026 import Indicators2026
from src.sector_analysis_2026 import SectorAnalysis2026

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2026 RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class Dashboard2026:
    """
    Render the 2026 regime-focused dashboard.
    """

    def __init__(self):
        self.regime_analyzer = Regime2026Analyzer()
        self.option_analyzer = OptionChain2026Analyzer()
        self.depth_analyzer = MarketDepth2026Analyzer()
        self.chart_analyzer = ChartAnalysis2026()
        self.indicator_analyzer = Indicators2026()
        self.sector_analyzer = SectorAnalysis2026()

    def render_main_dashboard(
        self,
        price_data: pd.DataFrame,
        option_chain_data: Dict,
        spot: float,
        atm_strike: float,
        strike_gap: float = 50,
        vix: float = None,
        sector_returns: Dict[str, float] = None
    ):
        """
        Render the main 2026 dashboard.

        Shows ONLY essential information on home screen.
        """
        # Get all analyses
        regime_analysis = self._get_regime_analysis(price_data, option_chain_data, vix)
        option_analysis = self._get_option_analysis(option_chain_data, spot, atm_strike, strike_gap)
        chart_analysis = self._get_chart_analysis(price_data)
        indicator_analysis = self._get_indicator_analysis(price_data)

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN DASHBOARD - TOP SECTION (ALWAYS VISIBLE)
        # ═══════════════════════════════════════════════════════════════════════

        # NO TRADE WARNING (if applicable)
        time_zone = regime_analysis.get('time_zone', {})
        should_trade = regime_analysis.get('should_trade', True)

        if not should_trade:
            st.error(f"**NO TRADE ZONE** {time_zone.get('warning', '')}")
            st.warning(f"**Reason:** {regime_analysis.get('trade_reason', 'Conditions not favorable')}")
            st.divider()

        # Main metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            self._render_market_regime(regime_analysis)

        with col2:
            self._render_writer_stress(option_analysis)

        with col3:
            self._render_volatility_state(regime_analysis, chart_analysis)

        with col4:
            self._render_market_mood(regime_analysis)

        with col5:
            self._render_time_zone(time_zone)

        st.divider()

        # Allowed Playbook (prominent display)
        self._render_playbook(regime_analysis.get('playbook', {}))

        st.divider()

        # ═══════════════════════════════════════════════════════════════════════
        # COLLAPSIBLE SECTIONS
        # ═══════════════════════════════════════════════════════════════════════

        # Option Chain Details
        with st.expander("Option Chain Analysis (OI Velocity, Premium Decay, Stress Zones)", expanded=False):
            self._render_option_details(option_analysis)

        # Chart Analysis Details
        with st.expander("Chart Analysis (Range Structure, VWAP, Volatility)", expanded=False):
            self._render_chart_details(chart_analysis)

        # Indicator Details
        with st.expander("Indicators (EMA Distance, ATR, Volume)", expanded=False):
            self._render_indicator_details(indicator_analysis)

        # Sector Analysis (if data available)
        if sector_returns:
            with st.expander("Sector Analysis (Relative Strength, Rotation)", expanded=False):
                sector_analysis = self.sector_analyzer.analyze(sector_returns)
                self._render_sector_details(sector_analysis)

    def _get_regime_analysis(self, price_data: pd.DataFrame, option_chain_data: Dict, vix: float) -> Dict:
        """Get regime analysis"""
        try:
            return self.regime_analyzer.analyze(price_data, option_chain_data, None, vix)
        except Exception as e:
            return {'error': str(e), 'should_trade': True, 'time_zone': TimeBasedLogic2026.get_current_zone()}

    def _get_option_analysis(self, option_chain_data: Dict, spot: float, atm_strike: float, strike_gap: float) -> Dict:
        """Get option chain analysis"""
        try:
            if isinstance(option_chain_data, pd.DataFrame):
                return self.option_analyzer.analyze(option_chain_data, spot, atm_strike, strike_gap)
            return option_chain_data
        except Exception as e:
            return {'error': str(e)}

    def _get_chart_analysis(self, price_data: pd.DataFrame) -> Dict:
        """Get chart analysis"""
        try:
            return self.chart_analyzer.analyze(price_data)
        except Exception as e:
            return {'error': str(e)}

    def _get_indicator_analysis(self, price_data: pd.DataFrame) -> Dict:
        """Get indicator analysis"""
        try:
            return self.indicator_analyzer.analyze(price_data)
        except Exception as e:
            return {'error': str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN METRIC RENDERERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _render_market_regime(self, analysis: Dict):
        """Render market regime card"""
        vol_state = analysis.get('volatility_state', {})
        base_state = vol_state.get('base_state', 'NORMAL')
        velocity = vol_state.get('velocity_state', 'STABLE')

        emoji = '📊'
        if base_state == 'COMPRESSED':
            emoji = '📉'
            color = 'blue'
        elif base_state == 'ELEVATED':
            emoji = '🔥'
            color = 'orange'
        else:
            color = 'gray'

        st.metric(
            label=f"{emoji} Market Regime",
            value=base_state,
            delta=velocity
        )

    def _render_writer_stress(self, analysis: Dict):
        """Render writer stress card"""
        writer_control = analysis.get('writer_control', {})
        control = writer_control.get('control', 'MIXED')

        if control == 'STRONG':
            emoji = '🛡️'
            delta = 'In Control'
        elif control == 'WEAK':
            emoji = '⚔️'
            delta = 'Losing Control'
        else:
            emoji = '⚖️'
            delta = 'Mixed'

        st.metric(
            label=f"{emoji} Writer Control",
            value=control,
            delta=delta
        )

    def _render_volatility_state(self, regime: Dict, chart: Dict):
        """Render volatility state card"""
        vol = regime.get('volatility_state', {})
        chart_vol = chart.get('volatility', {})

        atr_ratio = vol.get('atr_ratio', 1.0)
        regime_name = chart_vol.get('regime', 'NORMAL')

        if atr_ratio < 0.7:
            emoji = '😴'
            delta = 'Breakout Loading'
        elif atr_ratio > 1.3:
            emoji = '🌋'
            delta = 'High Activity'
        else:
            emoji = '📊'
            delta = 'Normal'

        st.metric(
            label=f"{emoji} Volatility",
            value=regime_name,
            delta=f"{atr_ratio:.2f}x ATR"
        )

    def _render_market_mood(self, analysis: Dict):
        """Render market mood card"""
        mood = analysis.get('market_mood', {})
        mood_name = mood.get('mood', 'CALM')
        emoji = mood.get('emoji', '😌')

        st.metric(
            label=f"{emoji} Market Mood",
            value=mood_name,
            delta=mood.get('action', '')
        )

    def _render_time_zone(self, time_zone: Dict):
        """Render time zone card"""
        zone = time_zone.get('zone', 'unknown')
        behavior = time_zone.get('behavior', '')
        action = time_zone.get('action', '')

        emoji_map = {
            'trap': '⚠️',
            'expansion': '🚀',
            'decay': '⏳',
            'trend_reversal': '🎯',
            'pre_market': '🌅',
            'post_market': '🌙'
        }
        emoji = emoji_map.get(zone, '⏰')

        st.metric(
            label=f"{emoji} Time Zone",
            value=behavior,
            delta=action
        )

    def _render_playbook(self, playbook: Dict):
        """Render allowed playbook prominently"""
        primary = playbook.get('primary', 'NO_TRADE')
        allowed = playbook.get('allowed', [])
        not_allowed = playbook.get('not_allowed', [])
        reason = playbook.get('reason', '')
        details = playbook.get('playbook_details', {})

        # Playbook header
        col1, col2 = st.columns([2, 3])

        with col1:
            emoji_map = {
                'BREAKOUT': '🔥',
                'TREND_FOLLOW': '📈',
                'RANGE_FADE': '↔️',
                'PREMIUM_DECAY': '⏱️',
                'NO_TRADE': '🚫'
            }
            emoji = emoji_map.get(primary, '📋')

            st.subheader(f"{emoji} Allowed Playbook: **{primary}**")
            st.caption(reason)

        with col2:
            if details:
                st.markdown(f"""
                **Entry:** {details.get('entry', 'N/A')}

                **Stop Loss:** {details.get('sl', 'N/A')}

                **Target:** {details.get('target', 'N/A')}
                """)

        # Show what's not allowed
        if not_allowed:
            st.caption(f"Avoid: {', '.join(not_allowed)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DETAIL RENDERERS (COLLAPSIBLE)
    # ═══════════════════════════════════════════════════════════════════════════

    def _render_option_details(self, analysis: Dict):
        """Render option chain details"""
        if analysis.get('error'):
            st.error(f"Error: {analysis['error']}")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**OI Velocity**")
            oi_vel = analysis.get('oi_velocity', {})
            st.write(f"CE: {oi_vel.get('ce_velocity_1min', 0):,.0f}/min")
            st.write(f"PE: {oi_vel.get('pe_velocity_1min', 0):,.0f}/min")
            st.caption(oi_vel.get('interpretation', ''))

        with col2:
            st.markdown("**Premium Decay**")
            decay = analysis.get('premium_decay', {})
            st.write(f"CE: {decay.get('ce_decay_pct', 0):.2f}%")
            st.write(f"PE: {decay.get('pe_decay_pct', 0):.2f}%")
            st.caption(decay.get('signal', ''))

        with col3:
            st.markdown("**Time-Weighted PCR**")
            pcr = analysis.get('tw_pcr', {})
            st.write(f"TW-PCR: {pcr.get('tw_pcr', 1.0):.3f}")
            st.write(f"Trend: {pcr.get('trend', 'N/A')}")
            st.caption(pcr.get('bias', ''))

        st.divider()

        # Gamma Zone
        gamma = analysis.get('gamma_zone', {})
        if gamma.get('in_gamma_zone'):
            st.warning(f"**IN GAMMA ZONE** ({gamma.get('lower_bound', 0)} - {gamma.get('upper_bound', 0)})")
            st.caption(gamma.get('trading_implication', ''))
        else:
            st.success(f"**ESCAPED GAMMA** - Position: {gamma.get('position', 'N/A')}")

        # Stress Zones
        st.markdown("**Writer Stress Zones**")
        stress = analysis.get('stress_zones', {})
        ce_stress = stress.get('highest_ce_stress', {})
        pe_stress = stress.get('highest_pe_stress', {})

        scol1, scol2 = st.columns(2)
        with scol1:
            if ce_stress:
                st.write(f"CE Stress: {ce_stress.get('strike', 'N/A')} ({ce_stress.get('stress_level', '')})")
        with scol2:
            if pe_stress:
                st.write(f"PE Stress: {pe_stress.get('strike', 'N/A')} ({pe_stress.get('stress_level', '')})")

        # Trap Detection
        trap = analysis.get('trap_detection', {})
        if trap.get('trap_detected'):
            st.error(f"**TRAP DETECTED:** {trap.get('trap_type', '')} - {trap.get('warning', '')}")

    def _render_chart_details(self, analysis: Dict):
        """Render chart analysis details"""
        if analysis.get('error') or not analysis.get('valid', True):
            st.warning("Chart analysis not available")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Range Structure**")
            pd_levels = analysis.get('pd_levels', {})
            or_levels = analysis.get('or_levels', {})

            if pd_levels.get('valid'):
                st.write(f"PDH: {pd_levels.get('pdh', 'N/A')}")
                st.write(f"PDL: {pd_levels.get('pdl', 'N/A')}")

            if or_levels.get('valid'):
                st.write(f"OR High: {or_levels.get('or_high', 'N/A')}")
                st.write(f"OR Low: {or_levels.get('or_low', 'N/A')}")

            range_pos = analysis.get('range_position', {})
            st.caption(f"Position: {range_pos.get('range_state', 'N/A')}")

        with col2:
            st.markdown("**Volatility Patterns**")
            vol = analysis.get('volatility', {})
            st.write(f"Regime: {vol.get('regime', 'N/A')}")
            st.write(f"ATR Ratio: {vol.get('ratio', 1.0):.2f}x")

            compression = vol.get('compression_data', {})
            if compression.get('breakout_imminent'):
                st.warning("Breakout Imminent!")

            expansion = vol.get('expansion_data', {})
            if expansion.get('expansion_detected'):
                st.success(f"Expansion: {expansion.get('current_direction', '')}")

        with col3:
            st.markdown("**VWAP Behavior**")
            vwap = analysis.get('vwap', {})
            st.write(f"Position: {vwap.get('vwap_position', 'N/A')}")
            st.write(f"Acceptance: {vwap.get('acceptance', 'N/A')}")

            signal = vwap.get('signal', {})
            st.caption(f"Bias: {signal.get('direction', 'N/A')} ({signal.get('strength', '')})")

        # State and Risk Summary
        st.divider()
        state = analysis.get('state', {})
        risk = analysis.get('risk', {})

        st.write(f"**State:** {state.get('state', 'N/A')} - {state.get('description', '')}")
        st.write(f"**Risk:** {risk.get('level', 'N/A')} - {risk.get('recommendation', '')}")

    def _render_indicator_details(self, analysis: Dict):
        """Render indicator details"""
        if not analysis.get('valid'):
            st.warning("Indicator analysis not available")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**EMA Distance**")
            ema = analysis.get('ema', {})
            st.write(f"EMA20: {ema.get('ema20', 'N/A')}")
            st.write(f"EMA50: {ema.get('ema50', 'N/A')}")
            st.write(f"Distance: {ema.get('current_distance_pct', 0):.3f}%")
            st.write(f"State: {ema.get('state', 'N/A')}")
            st.caption(ema.get('description', ''))

        with col2:
            st.markdown("**ATR Core**")
            atr = analysis.get('atr', {})
            regime = atr.get('regime', {})
            stops = atr.get('stops', {})

            st.write(f"Regime: {regime.get('regime', 'N/A')}")
            st.write(f"Current ATR: {regime.get('current_atr', 'N/A')}")
            st.write(f"Stop (1.5x): {stops.get('stops', {}).get('standard', 'N/A')}")

            chars = regime.get('characteristics', {})
            st.caption(chars.get('strategy', ''))

        with col3:
            st.markdown("**Volume Context**")
            vol = analysis.get('volume', {})
            quality = vol.get('quality', {})
            context = vol.get('context', {})

            st.write(f"Quality: {quality.get('quality', 'N/A')}")
            st.write(f"Vol Ratio: {context.get('vol_ratio', 1.0):.2f}x")
            st.write(f"Relationship: {context.get('relationship', 'N/A')}")
            st.caption(quality.get('trading_implication', ''))

        # Overall Assessment
        st.divider()
        overall = analysis.get('overall', {})
        st.write(f"**Overall:** {overall.get('bias', 'N/A')}")
        st.caption(overall.get('summary', ''))

    def _render_sector_details(self, analysis: Dict):
        """Render sector analysis details"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Leaders**")
            leaders = analysis.get('leaders', [])
            for leader in leaders[:3]:
                emoji = leader.get('emoji', '')
                st.write(f"{emoji} {leader.get('name', '')} (RS: {leader.get('rs', 0):+.2f})")

        with col2:
            st.markdown("**Laggards**")
            laggards = analysis.get('laggards', [])
            for laggard in laggards[-3:]:
                emoji = laggard.get('emoji', '')
                st.write(f"{emoji} {laggard.get('name', '')} (RS: {laggard.get('rs', 0):+.2f})")

        st.divider()

        rotation = analysis.get('rotation', {})
        speed = rotation.get('speed', {})
        phase = rotation.get('phase', {})

        st.write(f"**Rotation Speed:** {speed.get('speed', 'N/A')} ({speed.get('changes_per_hour', 0):.1f}/hr)")
        st.write(f"**Phase:** {phase.get('phase', 'N/A')} - {phase.get('description', '')}")

        # Options Activity
        options = analysis.get('options')
        if options:
            st.divider()
            st.markdown("**Sector Options Activity**")
            st.write(f"Leading: {options.get('leader', 'N/A')}")
            st.write(f"Intent: {options.get('market_intent', 'N/A')}")
            st.caption(options.get('description', ''))


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_2026_dashboard(
    price_data: pd.DataFrame,
    option_chain_df: pd.DataFrame,
    spot: float,
    atm_strike: float,
    strike_gap: float = 50,
    vix: float = None,
    sector_returns: Dict[str, float] = None
):
    """
    Main function to render the 2026 dashboard in Streamlit.

    Usage:
        from src.dashboard_2026 import render_2026_dashboard

        render_2026_dashboard(
            price_data=df,
            option_chain_df=option_df,
            spot=24850,
            atm_strike=24850,
            vix=13.5
        )
    """
    dashboard = Dashboard2026()

    # Prepare option chain data dict if DataFrame provided
    if isinstance(option_chain_df, pd.DataFrame):
        option_chain_data = option_chain_df
    else:
        option_chain_data = option_chain_df or {}

    dashboard.render_main_dashboard(
        price_data=price_data,
        option_chain_data=option_chain_data,
        spot=spot,
        atm_strike=atm_strike,
        strike_gap=strike_gap,
        vix=vix,
        sector_returns=sector_returns
    )


def render_quick_status():
    """
    Render just the quick status bar (for sidebar or minimal view).
    """
    time_zone = TimeBasedLogic2026.get_current_zone()
    should_trade, reason = TimeBasedLogic2026.should_trade()

    # Status indicator
    if not should_trade:
        st.sidebar.error(f"{time_zone['behavior']}")
        st.sidebar.caption(reason)
    else:
        st.sidebar.success(f"{time_zone['behavior']}")
        st.sidebar.caption(time_zone.get('warning', 'Good to trade'))
