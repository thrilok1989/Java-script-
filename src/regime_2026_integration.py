"""
2026 Regime Integration Module
==============================
Unified interface for all 2026 regime modules.

This module provides easy access to all 2026 analysis features.
Import this single module to get all 2026 functionality.

Usage:
    from src.regime_2026_integration import (
        analyze_market_2026,
        get_trading_decision,
        render_2026_dashboard
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pytz

# Import all 2026 modules
from src.regime_2026_core import (
    Regime2026Analyzer,
    TimeBasedLogic2026,
    MarketRegime2026,
    MarketMood2026,
    PlaybookGenerator2026,
    get_regime_2026_quick_analysis,
    get_time_zone_warning,
    should_trade_now
)

from src.option_chain_2026 import (
    OptionChain2026Analyzer,
    OIVelocityTracker,
    PremiumDecayTracker,
    WriterStressZoneDetector,
    ATMGammaZone,
    TimeWeightedPCR,
    analyze_option_chain_2026,
    get_gamma_zone,
    find_writer_stress_zones
)

from src.market_depth_2026 import (
    MarketDepth2026Analyzer,
    OrderFlowChangeTracker,
    AbsorptionDetector,
    DepthImbalanceTrendTracker,
    create_depth_analyzer,
    quick_depth_analysis,
    detect_spoofing,
    detect_absorption_quick,
    get_imbalance_trend
)

from src.chart_analysis_2026 import (
    ChartAnalysis2026,
    RangeStructureAnalyzer,
    VolatilityPatternAnalyzer,
    VWAPBehaviorAnalyzer,
    analyze_chart_2026,
    get_range_levels,
    get_volatility_regime,
    get_vwap_analysis
)

from src.indicators_2026 import (
    Indicators2026,
    EMADistanceAnalyzer,
    ATRCoreEngine,
    VolumeContextAnalyzer,
    analyze_indicators_2026,
    get_ema_compression_score,
    get_atr_stops,
    get_atr_targets,
    get_volume_context
)

from src.sector_analysis_2026 import (
    SectorAnalysis2026,
    RelativeStrengthCalculator,
    RotationSpeedTracker,
    SectorOptionsActivity,
    analyze_sectors_2026,
    get_relative_strength,
    rank_all_sectors,
    analyze_sector_options
)

from src.dashboard_2026 import (
    Dashboard2026,
    render_2026_dashboard,
    render_quick_status
)

IST = pytz.timezone('Asia/Kolkata')


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_market_2026(
    price_data: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
    spot: float = None,
    atm_strike: float = None,
    strike_gap: float = 50,
    vix: float = None,
    sector_returns: Dict[str, float] = None,
    depth_data: Dict = None
) -> Dict[str, Any]:
    """
    Comprehensive market analysis using all 2026 modules.

    Returns unified analysis covering:
    - Time zone and trading permission
    - Market regime and volatility
    - Option chain dynamics
    - Chart structure
    - Indicator signals
    - Sector rotation
    - Final trading decision

    Args:
        price_data: OHLCV DataFrame
        option_chain_df: Option chain DataFrame
        spot: Current spot price
        atm_strike: ATM strike price
        strike_gap: Gap between strikes (default 50)
        vix: India VIX value
        sector_returns: Dict of sector returns {sector: return %}
        depth_data: Market depth data dict

    Returns:
        Comprehensive analysis dict
    """
    timestamp = datetime.now(IST)
    results = {
        'timestamp': timestamp.isoformat(),
        'spot': spot,
        'atm_strike': atm_strike
    }

    # 1. Time Zone Analysis
    time_zone = TimeBasedLogic2026.get_current_zone()
    should_trade, trade_reason = TimeBasedLogic2026.should_trade()
    results['time_zone'] = time_zone
    results['should_trade'] = should_trade
    results['trade_reason'] = trade_reason

    # 2. Regime Analysis
    if price_data is not None and option_chain_df is not None:
        try:
            # Prepare option chain data for regime analyzer
            option_chain_data = {
                'oi_ce_change': option_chain_df['Chg_OI_CE'].sum() if 'Chg_OI_CE' in option_chain_df.columns else 0,
                'oi_pe_change': option_chain_df['Chg_OI_PE'].sum() if 'Chg_OI_PE' in option_chain_df.columns else 0,
                'premium_ce_change_pct': 0,  # Would need previous data
                'premium_pe_change_pct': 0,
                'oi_velocity': 0,
                'premium_velocity': 0
            }

            regime_analyzer = Regime2026Analyzer()
            regime_analysis = regime_analyzer.analyze(price_data, option_chain_data, depth_data, vix)
            results['regime'] = regime_analysis
        except Exception as e:
            results['regime'] = {'error': str(e)}

    # 3. Option Chain Analysis
    if option_chain_df is not None and spot and atm_strike:
        try:
            option_analysis = analyze_option_chain_2026(
                option_chain_df, spot, atm_strike, strike_gap
            )
            results['option_chain'] = option_analysis
        except Exception as e:
            results['option_chain'] = {'error': str(e)}

    # 4. Chart Analysis
    if price_data is not None:
        try:
            chart_analysis = analyze_chart_2026(price_data)
            results['chart'] = chart_analysis
        except Exception as e:
            results['chart'] = {'error': str(e)}

    # 5. Indicator Analysis
    if price_data is not None:
        try:
            indicator_analysis = analyze_indicators_2026(price_data)
            results['indicators'] = indicator_analysis
        except Exception as e:
            results['indicators'] = {'error': str(e)}

    # 6. Sector Analysis
    if sector_returns:
        try:
            sector_analysis = analyze_sectors_2026(sector_returns)
            results['sectors'] = sector_analysis
        except Exception as e:
            results['sectors'] = {'error': str(e)}

    # 7. Generate Final Decision
    results['decision'] = generate_trading_decision(results)

    return results


def generate_trading_decision(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final trading decision from comprehensive analysis.
    """
    decision = {
        'action': 'WAIT',
        'confidence': 0,
        'reasons': [],
        'warnings': []
    }

    # Check time zone
    if not analysis.get('should_trade', True):
        decision['action'] = 'NO_TRADE'
        decision['reasons'].append(f"Time zone: {analysis.get('trade_reason', 'Not favorable')}")
        decision['confidence'] = 100
        return decision

    # Get regime playbook
    regime = analysis.get('regime', {})
    playbook = regime.get('playbook', {})
    primary_play = playbook.get('primary', 'NO_TRADE')

    if primary_play == 'NO_TRADE':
        decision['action'] = 'WAIT'
        decision['reasons'].append(playbook.get('reason', 'Conditions not favorable'))
        decision['confidence'] = 70
        return decision

    # Build confidence score
    confidence_score = 50

    # Option chain contribution
    option_analysis = analysis.get('option_chain', {})
    mood = option_analysis.get('market_mood', {})
    if mood.get('mood') == 'CALM':
        confidence_score += 10
        decision['reasons'].append('Market calm')
    elif mood.get('mood') == 'AGGRESSIVE':
        confidence_score += 15
        decision['reasons'].append('Strong momentum')

    # Trap detection penalty
    trap = option_analysis.get('trap_detection', {})
    if trap.get('trap_detected'):
        confidence_score -= 20
        decision['warnings'].append(f"TRAP: {trap.get('trap_type', '')}")

    # Chart state contribution
    chart = analysis.get('chart', {})
    state = chart.get('state', {})
    if state.get('state') == 'EXPANSION':
        confidence_score += 15
        decision['reasons'].append('Expansion state')
    elif state.get('state') == 'BALANCE':
        if primary_play == 'RANGE_FADE':
            confidence_score += 10
        else:
            confidence_score -= 10

    # Volume confirmation
    indicators = analysis.get('indicators', {})
    volume_quality = indicators.get('volume', {}).get('quality', {}).get('quality', 'NEUTRAL')
    if volume_quality == 'EXCELLENT':
        confidence_score += 10
        decision['reasons'].append('Volume confirms')
    elif volume_quality == 'POOR':
        confidence_score -= 15
        decision['warnings'].append('Low volume confirmation')

    # Set final action
    if confidence_score >= 70:
        decision['action'] = primary_play
        decision['confidence'] = confidence_score
    elif confidence_score >= 50:
        decision['action'] = primary_play
        decision['confidence'] = confidence_score
        decision['warnings'].append('Moderate confidence - smaller size recommended')
    else:
        decision['action'] = 'WAIT'
        decision['confidence'] = confidence_score
        decision['reasons'].append('Low confidence score')

    # Add playbook details
    decision['playbook'] = playbook.get('playbook_details', {})

    return decision


def get_trading_decision(
    price_data: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
    spot: float = None,
    atm_strike: float = None,
    vix: float = None
) -> Dict[str, Any]:
    """
    Quick function to get trading decision.

    Returns simple actionable output.
    """
    analysis = analyze_market_2026(
        price_data=price_data,
        option_chain_df=option_chain_df,
        spot=spot,
        atm_strike=atm_strike,
        vix=vix
    )
    return analysis.get('decision', {'action': 'WAIT'})


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def should_i_trade() -> Tuple[bool, str, Dict]:
    """
    Quick check if trading is recommended right now.

    Returns:
        (should_trade, reason, time_zone_info)
    """
    time_zone = TimeBasedLogic2026.get_current_zone()
    should_trade, reason = TimeBasedLogic2026.should_trade()
    return should_trade, reason, time_zone


def get_atr_based_stops_targets(
    price_data: pd.DataFrame,
    stop_multiplier: float = 1.5,
    target_rr: float = 2.0
) -> Dict[str, Any]:
    """
    Get ATR-based stop and target levels.
    """
    stops = get_atr_stops(price_data, stop_multiplier)
    targets = get_atr_targets(price_data, target_rr)

    return {
        'stop_distance': stops.get('stops', {}).get('recommended', 0),
        'stop_levels': stops.get('stops', {}),
        'target_distance': targets.get('targets', {}).get('recommended', 0),
        'long_targets': targets.get('long_targets', {}),
        'short_targets': targets.get('short_targets', {}),
        'current_atr': stops.get('current_atr', 0)
    }


def get_market_summary() -> str:
    """
    Get quick market summary string.
    """
    time_zone = TimeBasedLogic2026.get_current_zone()
    should_trade, reason = TimeBasedLogic2026.should_trade()

    zone_name = time_zone.get('zone', 'unknown')
    behavior = time_zone.get('behavior', '')

    if should_trade:
        return f"Zone: {behavior} | Status: OK to trade"
    else:
        return f"Zone: {behavior} | Status: {reason}"


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main analysis functions
    'analyze_market_2026',
    'get_trading_decision',
    'generate_trading_decision',

    # Quick utilities
    'should_i_trade',
    'get_atr_based_stops_targets',
    'get_market_summary',

    # Core classes
    'Regime2026Analyzer',
    'OptionChain2026Analyzer',
    'MarketDepth2026Analyzer',
    'ChartAnalysis2026',
    'Indicators2026',
    'SectorAnalysis2026',
    'Dashboard2026',

    # Time logic
    'TimeBasedLogic2026',
    'should_trade_now',
    'get_time_zone_warning',

    # Option chain functions
    'analyze_option_chain_2026',
    'get_gamma_zone',
    'find_writer_stress_zones',

    # Chart functions
    'analyze_chart_2026',
    'get_range_levels',
    'get_volatility_regime',
    'get_vwap_analysis',

    # Indicator functions
    'analyze_indicators_2026',
    'get_ema_compression_score',
    'get_atr_stops',
    'get_atr_targets',
    'get_volume_context',

    # Sector functions
    'analyze_sectors_2026',
    'get_relative_strength',
    'rank_all_sectors',

    # Dashboard
    'render_2026_dashboard',
    'render_quick_status'
]
