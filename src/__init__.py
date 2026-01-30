"""
Signal Manager Package

Contains XGBoost ML Analyzer, Enhanced Signal Generator, Telegram Alert Manager,
and 2026 Regime Update modules.

2026 Regime Update Modules:
- regime_2026_core: Core regime analysis and time-based trading logic
- option_chain_2026: OI velocity, premium decay, writer stress, gamma zones
- market_depth_2026: Order flow, absorption detection, depth imbalance
- chart_analysis_2026: Range structure, volatility patterns, VWAP behavior
- indicators_2026: EMA distance, ATR core engine, volume context
- sector_analysis_2026: Relative strength, rotation speed, sector options
- dashboard_2026: Streamlined regime-focused dashboard
- regime_2026_integration: Unified interface for all 2026 modules
"""

__version__ = "2.0.0"  # Updated for 2026 regime changes

# 2026 Regime Modules
from src.regime_2026_integration import (
    # Main analysis
    analyze_market_2026,
    get_trading_decision,
    should_i_trade,
    get_market_summary,

    # Dashboard
    render_2026_dashboard,
    render_quick_status,

    # Time logic
    TimeBasedLogic2026,
    should_trade_now,
    get_time_zone_warning,

    # ATR utilities
    get_atr_based_stops_targets,
    get_atr_stops,
    get_atr_targets
)
