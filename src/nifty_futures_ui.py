"""
NIFTY Futures Analysis UI Module

Comprehensive Streamlit UI for NIFTY Futures Analysis including:
- Futures vs Spot Comparison
- Bias Analysis (9 indicators)
- Chart Analysis with Technical Indicators
- Cost of Carry & Arbitrage
- Rollover Analysis
- Participant Positioning
- Options Correlation
- Trading Signals

Author: Claude AI Assistant
Date: 2025-12-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from src.nifty_futures_analyzer import NiftyFuturesAnalyzer, FuturesData
from src.nifty_futures_bias_analysis import NiftyFuturesBiasAnalyzer, FuturesBiasResult

logger = logging.getLogger(__name__)


def render_nifty_futures_dashboard(
    spot_price: float,
    futures_data: Optional[Dict] = None,
    participant_data: Optional[Dict] = None,
    option_chain_data: Optional[Dict] = None,
    historical_data: Optional[pd.DataFrame] = None
):
    """
    Render the complete NIFTY Futures Analysis dashboard

    Args:
        spot_price: Current NIFTY spot price
        futures_data: Futures market data
        participant_data: FII/DII/Pro positioning
        option_chain_data: Options chain data for correlation
        historical_data: Historical price data
    """
    st.markdown("## 📈 NIFTY Futures Analysis")

    # Handle None or invalid spot_price
    if spot_price is None or spot_price == 0:
        st.warning("⚠️ **Market Data Unavailable**")
        st.info("""
        Unable to load NIFTY Futures Analysis because spot price data is not available.

        **Possible Reasons:**
        - Market is currently closed
        - API connection issues
        - Data refresh in progress

        Please try again during market hours (9:15 AM - 3:30 PM IST, Monday-Friday).
        """)
        return

    # Initialize analyzers
    futures_analyzer = NiftyFuturesAnalyzer()
    bias_analyzer = NiftyFuturesBiasAnalyzer()

    # Create demo data if not provided
    if futures_data is None:
        futures_data = _get_demo_futures_data(spot_price)

    if participant_data is None:
        participant_data = _get_demo_participant_data()

    # Create sub-tabs
    subtabs = st.tabs([
        "📊 Futures Overview",
        "🎯 Bias Analysis",
        "📈 Chart Analysis",
        "💰 Cost of Carry",
        "🔄 Rollover Analysis",
        "👥 Participant Analysis",
        "🔗 Options Correlation",
        "⚡ Trading Signals"
    ])

    # SUB-TAB 1: Futures Overview
    with subtabs[0]:
        render_futures_overview(spot_price, futures_data, futures_analyzer)

    # SUB-TAB 2: Bias Analysis
    with subtabs[1]:
        render_bias_analysis(
            futures_data,
            spot_price,
            participant_data,
            historical_data,
            bias_analyzer
        )

    # SUB-TAB 3: Chart Analysis
    with subtabs[2]:
        render_chart_analysis(futures_data, historical_data)

    # SUB-TAB 4: Cost of Carry
    with subtabs[3]:
        render_cost_of_carry(futures_data, spot_price, futures_analyzer)

    # SUB-TAB 5: Rollover Analysis
    with subtabs[4]:
        render_rollover_analysis(futures_data, futures_analyzer)

    # SUB-TAB 6: Participant Analysis
    with subtabs[5]:
        render_participant_analysis(participant_data, futures_analyzer)

    # SUB-TAB 7: Options Correlation
    with subtabs[6]:
        render_options_correlation(futures_data, option_chain_data)

    # SUB-TAB 8: Trading Signals
    with subtabs[7]:
        render_trading_signals(
            futures_data,
            spot_price,
            participant_data,
            futures_analyzer,
            bias_analyzer
        )


def render_futures_overview(
    spot_price: float,
    futures_data: Dict,
    analyzer: NiftyFuturesAnalyzer
):
    """Render Futures Overview section"""
    st.markdown("### 📊 Futures vs Spot Comparison")

    # Get futures analysis
    current_futures = futures_data.get('current_month', {})
    next_futures = futures_data.get('next_month', {})

    futures_price = current_futures.get('ltp', spot_price * 1.005)
    basis = futures_price - spot_price
    basis_pct = (basis / spot_price) * 100

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "NIFTY Spot",
            f"₹{spot_price:,.2f}",
            delta=f"{current_futures.get('spot_change_pct', 0):+.2f}%"
        )

    with col2:
        st.metric(
            "Futures (Current)",
            f"₹{futures_price:,.2f}",
            delta=f"{basis_pct:+.2f}% basis"
        )

    with col3:
        premium_discount = "Premium" if basis > 0 else "Discount"
        st.metric(
            "Basis",
            f"₹{abs(basis):.2f}",
            delta=premium_discount,
            delta_color="normal" if basis > 0 else "inverse"
        )

    with col4:
        oi_change = current_futures.get('oi_change_pct')
        if oi_change is not None:
            st.metric(
                "OI Change",
                f"{abs(oi_change):.1f}%",
                delta="Building" if oi_change > 0 else "Unwinding",
                delta_color="normal" if oi_change > 0 else "inverse"
            )
        else:
            st.metric(
                "OI Change",
                "N/A",
                delta="Data unavailable"
            )

    # Basis chart
    st.markdown("#### Basis Trend (Futures - Spot)")
    basis_chart = create_basis_chart(futures_data, spot_price)
    st.plotly_chart(basis_chart, use_container_width=True)

    # Futures contracts table
    st.markdown("#### Active Futures Contracts")

    contracts_data = []
    for contract_type, contract_data in [('Current Month', current_futures), ('Next Month', next_futures)]:
        if contract_data:
            ltp = contract_data.get('ltp', 0)
            contract_basis = ltp - spot_price
            # Handle None values gracefully
            oi = contract_data.get('oi')
            oi_change_pct = contract_data.get('oi_change_pct')
            volume = contract_data.get('volume', 0)
            volume_change_pct = contract_data.get('volume_change_pct')

            contracts_data.append({
                'Contract': contract_type,
                'Expiry': contract_data.get('expiry', 'N/A'),
                'LTP': f"₹{ltp:,.2f}",
                'Basis': f"₹{contract_basis:+.2f}",
                'Basis %': f"{(contract_basis/spot_price*100):+.2f}%",
                'OI': f"{oi:,}" if oi is not None else "N/A",
                'OI Chg': f"{oi_change_pct:+.1f}%" if oi_change_pct is not None else "N/A",
                'Volume': f"{volume:,}",
                'Vol Chg': f"{volume_change_pct:+.1f}%" if volume_change_pct is not None else "N/A"
            })

    if contracts_data:
        st.dataframe(pd.DataFrame(contracts_data), use_container_width=True, hide_index=True)


def render_bias_analysis(
    futures_data: Dict,
    spot_price: float,
    participant_data: Dict,
    historical_data: Optional[pd.DataFrame],
    bias_analyzer: NiftyFuturesBiasAnalyzer
):
    """Render Bias Analysis section with 9 indicators"""
    st.markdown("### 🎯 Futures Bias Analysis (9 Indicators)")

    # Get bias analysis
    bias_result = bias_analyzer.analyze_comprehensive_bias(
        futures_data,
        spot_price,
        participant_data,
        historical_data
    )

    # Overall bias display
    st.markdown("#### Overall Futures Bias")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        direction_color = {
            'BULLISH': '🟢',
            'BEARISH': '🔴',
            'NEUTRAL': '🟡'
        }
        st.metric(
            "Direction",
            f"{direction_color.get(bias_result.overall_direction, '🟡')} {bias_result.overall_direction}",
            delta=f"{bias_result.combined_bias:+.1f} pts"
        )

    with col2:
        st.metric(
            "Signal Strength",
            bias_result.signal_strength,
            delta=f"{bias_result.overall_confidence:.0f}% confidence"
        )

    with col3:
        st.metric(
            "Confluence",
            f"{bias_result.confluence_count}/9",
            delta="indicators agree"
        )

    with col4:
        arb_color = "normal" if bias_result.arbitrage_opportunity == "YES" else "off"
        st.metric(
            "Arbitrage",
            bias_result.arbitrage_opportunity,
            delta="opportunity" if bias_result.arbitrage_opportunity == "YES" else "none"
        )

    # Individual bias indicators
    st.markdown("#### Individual Bias Indicators")

    # Create gauge chart for combined bias
    bias_gauge = create_bias_gauge(bias_result.combined_bias)
    st.plotly_chart(bias_gauge, use_container_width=True)

    # Display all 9 bias indicators
    col1, col2, col3 = st.columns(3)

    bias_indicators = [
        ('OI Bias', bias_result.oi_bias, bias_result.oi_confidence),
        ('Volume Bias', bias_result.volume_bias, bias_result.volume_confidence),
        ('Basis Bias', bias_result.basis_bias, bias_result.basis_confidence),
        ('Rollover Bias', bias_result.rollover_bias, bias_result.rollover_confidence),
        ('Participant Bias', bias_result.participant_bias, bias_result.participant_confidence),
        ('Cost of Carry', bias_result.cost_of_carry_bias, 60),
        ('Calendar Spread', bias_result.calendar_spread_bias, 50),
        ('Momentum', bias_result.price_momentum_bias, 55),
        ('Buildup/Unwinding', bias_result.buildup_unwinding_bias, 65)
    ]

    for idx, (name, bias, conf) in enumerate(bias_indicators):
        col = [col1, col2, col3][idx % 3]
        with col:
            render_bias_indicator(name, bias, conf)

    # Bias breakdown
    st.markdown("#### Bias Breakdown")

    summary = bias_analyzer.get_bias_summary(bias_result)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Bullish Factors:**")
        bullish_factors = summary.get('top_bullish_factors', [])
        if bullish_factors:
            for factor in bullish_factors:
                st.markdown(f"- 🟢 {factor}")
        else:
            st.markdown("- None")

    with col2:
        st.markdown("**Top Bearish Factors:**")
        bearish_factors = summary.get('top_bearish_factors', [])
        if bearish_factors:
            for factor in bearish_factors:
                st.markdown(f"- 🔴 {factor}")
        else:
            st.markdown("- None")

    # Key metrics
    st.markdown("#### Key Metrics")
    metrics = summary.get('key_metrics', {})

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Futures OI", metrics.get('futures_oi_change', 'N/A'))
    with col2:
        st.metric("Futures Vol", metrics.get('futures_volume_change', 'N/A'))
    with col3:
        st.metric("Basis", metrics.get('basis', 'N/A'))
    with col4:
        st.metric("Rollover", metrics.get('rollover', 'N/A'))
    with col5:
        st.metric("FII Position", metrics.get('fii_position', 'N/A'))
    with col6:
        st.metric("Arbitrage", metrics.get('arbitrage', 'N/A'))


def render_bias_indicator(name: str, bias: float, confidence: float):
    """Render a single bias indicator"""
    # Determine color
    if bias > 20:
        color = "🟢"
        direction = "BULLISH"
    elif bias < -20:
        color = "🔴"
        direction = "BEARISH"
    else:
        color = "🟡"
        direction = "NEUTRAL"

    # Create progress bar for bias strength
    normalized_bias = (bias + 100) / 200  # Convert -100:100 to 0:1

    st.markdown(f"**{color} {name}**")
    st.progress(normalized_bias)
    st.caption(f"{direction} ({bias:+.0f} pts) • {confidence:.0f}% confidence")


def render_chart_analysis(futures_data: Dict, historical_data: Optional[pd.DataFrame]):
    """Render Chart Analysis section"""
    st.markdown("### 📈 Futures Chart Analysis")

    # Create futures price chart with technical indicators
    if historical_data is not None and not historical_data.empty:
        chart = create_futures_chart(historical_data)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Historical data not available. Showing current data only.")

    # Technical indicators
    st.markdown("#### Technical Indicators")

    current_futures = futures_data.get('current_month', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rsi = current_futures.get('rsi', 50)
        rsi_zone = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}", delta=rsi_zone)

    with col2:
        macd = current_futures.get('macd', 0)
        macd_signal = "Bullish" if macd > 0 else "Bearish"
        st.metric("MACD", f"{macd:.2f}", delta=macd_signal)

    with col3:
        bb_position = current_futures.get('bb_position', 'Middle')
        st.metric("Bollinger Band", bb_position)

    with col4:
        trend = current_futures.get('trend', 'Sideways')
        st.metric("Trend", trend)


def render_cost_of_carry(
    futures_data: Dict,
    spot_price: float,
    analyzer: NiftyFuturesAnalyzer
):
    """Render Cost of Carry analysis"""
    st.markdown("### 💰 Cost of Carry & Arbitrage")

    current_futures = futures_data.get('current_month', {})
    futures_price = current_futures.get('ltp', spot_price * 1.005)

    # Calculate cost of carry
    days_to_expiry = current_futures.get('days_to_expiry', 15)
    risk_free_rate = 0.065  # 6.5% annual
    dividend_yield = 0.012  # 1.2% annual

    coc_result = analyzer.calculate_cost_of_carry(
        spot_price=spot_price,
        futures_price=futures_price,
        days_to_expiry=days_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Theoretical Futures",
            f"₹{coc_result.theoretical_futures_price:,.2f}"
        )

    with col2:
        st.metric(
            "Actual Futures",
            f"₹{coc_result.actual_futures_price:,.2f}"
        )

    with col3:
        mispricing_color = "normal" if abs(coc_result.mispricing_percentage) > 0.3 else "off"
        st.metric(
            "Mispricing",
            f"{coc_result.mispricing_percentage:+.2f}%",
            delta="Arbitrage!" if coc_result.arbitrage_opportunity else "Fair"
        )

    with col4:
        st.metric(
            "Cost of Carry",
            f"{coc_result.cost_of_carry_rate:.2f}%"
        )

    # Arbitrage opportunity
    if coc_result.arbitrage_opportunity:
        st.success(f"""
        🎯 **Arbitrage Opportunity Detected!**

        **Strategy:** {coc_result.arbitrage_strategy}

        **Details:**
        - Expected Profit: ₹{coc_result.expected_profit_per_lot:,.2f} per lot
        - Mispricing: {coc_result.mispricing_percentage:+.2f}%
        - Days to Expiry: {days_to_expiry}

        *Note: Consider transaction costs and margin requirements*
        """)
    else:
        st.info("✅ Futures fairly priced. No arbitrage opportunity.")

    # Cost components breakdown
    st.markdown("#### Cost Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk-Free Rate", f"{risk_free_rate*100:.2f}%")
    with col2:
        st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
    with col3:
        st.metric("Net Carry", f"{(risk_free_rate - dividend_yield)*100:.2f}%")


def render_rollover_analysis(futures_data: Dict, analyzer: NiftyFuturesAnalyzer):
    """Render Rollover Analysis section"""
    st.markdown("### 🔄 Rollover Analysis")

    current_futures = futures_data.get('current_month', {})
    next_futures = futures_data.get('next_month', {})

    # Get rollover analysis
    rollover_result = analyzer.analyze_rollover(
        current_month_data=current_futures,
        next_month_data=next_futures
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rollover %",
            f"{rollover_result.rollover_percentage:.1f}%",
            delta=rollover_result.rollover_trend
        )

    with col2:
        st.metric(
            "Rollover Cost",
            f"₹{rollover_result.rollover_cost_per_lot:.2f}",
            delta="per lot"
        )

    with col3:
        bias_color = "🟢" if rollover_result.rollover_bias == "BULLISH" else "🔴" if rollover_result.rollover_bias == "BEARISH" else "🟡"
        st.metric(
            "Rollover Bias",
            f"{bias_color} {rollover_result.rollover_bias}"
        )

    with col4:
        st.metric(
            "Rollover Strength",
            f"{rollover_result.rollover_strength:.0f}/100"
        )

    # Recommendation
    if rollover_result.rollover_percentage > 70:
        st.success(f"""
        ✅ **Strong Rollover Activity**

        - {rollover_result.rollover_percentage:.1f}% positions rolled to next month
        - Indicates strong conviction in maintaining positions
        - Rollover trend: {rollover_result.rollover_trend}
        - Bias: {rollover_result.rollover_bias}
        """)
    elif rollover_result.rollover_percentage < 40:
        st.warning(f"""
        ⚠️ **Weak Rollover Activity**

        - Only {rollover_result.rollover_percentage:.1f}% positions rolled
        - Traders may be exiting positions
        - Consider cautious approach
        """)
    else:
        st.info(f"""
        ℹ️ **Moderate Rollover**

        - {rollover_result.rollover_percentage:.1f}% rollover is typical
        - Normal market activity
        """)

    # Rollover details
    st.markdown("#### Rollover Details")

    details_data = {
        'Metric': [
            'Current Month OI',
            'Next Month OI',
            'OI Rolled',
            'Rollover Cost/Lot',
            'Days to Expiry',
            'Trend'
        ],
        'Value': [
            f"{rollover_result.current_month_oi:,}",
            f"{rollover_result.next_month_oi:,}",
            f"{rollover_result.oi_rolled:,}",
            f"₹{rollover_result.rollover_cost_per_lot:.2f}",
            f"{rollover_result.days_to_expiry}",
            rollover_result.rollover_trend
        ]
    }

    st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)


def render_participant_analysis(
    participant_data: Dict,
    analyzer: NiftyFuturesAnalyzer
):
    """Render Participant Analysis section"""
    st.markdown("### 👥 Participant Positioning (FII/DII/Pro/Client)")

    # Get positioning analysis
    positioning_result = analyzer.analyze_participant_positioning(participant_data)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fii_net = positioning_result.fii_net_futures
        fii_color = "normal" if fii_net > 0 else "inverse"
        st.metric(
            "FII Net",
            f"₹{abs(fii_net):,.0f} Cr",
            delta="Long" if fii_net > 0 else "Short",
            delta_color=fii_color
        )

    with col2:
        dii_net = positioning_result.dii_net_futures
        dii_color = "normal" if dii_net > 0 else "inverse"
        st.metric(
            "DII Net",
            f"₹{abs(dii_net):,.0f} Cr",
            delta="Long" if dii_net > 0 else "Short",
            delta_color=dii_color
        )

    with col3:
        st.metric(
            "Dominant Player",
            positioning_result.dominant_participant
        )

    with col4:
        bias_color = "🟢" if positioning_result.positioning_bias == "BULLISH" else "🔴" if positioning_result.positioning_bias == "BEARISH" else "🟡"
        st.metric(
            "Positioning Bias",
            f"{bias_color} {positioning_result.positioning_bias}"
        )

    # FII vs DII chart
    st.markdown("#### FII vs DII Positioning")

    fig = go.Figure()

    categories = ['FII', 'DII', 'Pro', 'Client']
    values = [
        positioning_result.fii_net_futures,
        positioning_result.dii_net_futures,
        positioning_result.pro_net_futures,
        positioning_result.client_net_futures
    ]

    colors = ['green' if v > 0 else 'red' for v in values]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"₹{v:,.0f} Cr" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="Net Futures Positioning",
        yaxis_title="Net Position (₹ Cr)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    if positioning_result.fii_net_futures > 1000 and positioning_result.dii_net_futures > 500:
        st.success("🟢 **Bullish Confluence**: Both FII and DII are net long")
    elif positioning_result.fii_net_futures < -1000 and positioning_result.dii_net_futures < -500:
        st.error("🔴 **Bearish Confluence**: Both FII and DII are net short")
    elif abs(positioning_result.fii_net_futures) > 2000:
        direction = "Long" if positioning_result.fii_net_futures > 0 else "Short"
        st.info(f"ℹ️ **FII Dominance**: FIIs heavily {direction}")
    else:
        st.info("🟡 **Mixed Positioning**: No clear consensus")


def render_options_correlation(
    futures_data: Dict,
    option_chain_data: Optional[Dict]
):
    """Render Options Correlation section"""
    st.markdown("### 🔗 Futures-Options Correlation")

    if option_chain_data is None:
        st.info("Options chain data not available")
        return

    current_futures = futures_data.get('current_month', {})
    futures_price = current_futures.get('ltp', 0)

    # PCR analysis
    pcr = option_chain_data.get('pcr', 1.0)
    max_pain = option_chain_data.get('max_pain', futures_price)
    distance_from_max_pain = ((futures_price - max_pain) / max_pain) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Futures Price", f"₹{futures_price:,.2f}")

    with col2:
        pcr_signal = "Bullish" if pcr > 1.2 else "Bearish" if pcr < 0.8 else "Neutral"
        st.metric("PCR", f"{pcr:.2f}", delta=pcr_signal)

    with col3:
        st.metric("Max Pain", f"₹{max_pain:,.2f}")

    with col4:
        st.metric(
            "Distance from Max Pain",
            f"{abs(distance_from_max_pain):.1f}%",
            delta="Above" if distance_from_max_pain > 0 else "Below"
        )

    # Correlation insights
    st.markdown("#### Correlation Insights")

    insights = []

    if abs(distance_from_max_pain) < 1:
        insights.append("✅ Futures trading at Max Pain - expect consolidation")
    elif distance_from_max_pain > 2:
        insights.append("⚠️ Futures above Max Pain - potential pullback")
    elif distance_from_max_pain < -2:
        insights.append("⚠️ Futures below Max Pain - potential bounce")

    if pcr > 1.3:
        insights.append("🟢 High PCR indicates bullish sentiment")
    elif pcr < 0.7:
        insights.append("🔴 Low PCR indicates bearish sentiment")

    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")


def render_trading_signals(
    futures_data: Dict,
    spot_price: float,
    participant_data: Dict,
    futures_analyzer: NiftyFuturesAnalyzer,
    bias_analyzer: NiftyFuturesBiasAnalyzer
):
    """Render Trading Signals section"""
    st.markdown("### ⚡ Futures Trading Signals")

    # Generate comprehensive signal
    current_futures = futures_data.get('current_month', {})
    signal = futures_analyzer.generate_futures_signal(
        futures_data=current_futures,
        spot_price=spot_price,
        participant_data=participant_data
    )

    # Get bias analysis
    bias_result = bias_analyzer.analyze_comprehensive_bias(
        futures_data,
        spot_price,
        participant_data
    )

    # Main signal display
    signal_color = {
        'STRONG BUY': '🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG SELL': '🔴'
    }

    st.markdown(f"## {signal_color.get(signal.signal, '🟡')} {signal.signal}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Signal Strength", f"{signal.strength}/100")
    with col2:
        st.metric("Confidence", f"{signal.confidence:.0f}%")
    with col3:
        st.metric("Risk Level", signal.risk_level)

    # Signal rationale
    st.markdown("#### Signal Rationale")

    for reason in signal.reasons:
        if "bullish" in reason.lower() or "buy" in reason.lower():
            st.markdown(f"- 🟢 {reason}")
        elif "bearish" in reason.lower() or "sell" in reason.lower():
            st.markdown(f"- 🔴 {reason}")
        else:
            st.markdown(f"- 🟡 {reason}")

    # Entry/Exit levels
    st.markdown("#### Suggested Levels")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Entry", f"₹{signal.entry_price:,.2f}")
    with col2:
        st.metric("Target", f"₹{signal.target_price:,.2f}")
    with col3:
        st.metric("Stop Loss", f"₹{signal.stop_loss:,.2f}")

    # Risk-reward
    potential_gain = signal.target_price - signal.entry_price
    potential_loss = signal.entry_price - signal.stop_loss
    risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0

    st.info(f"**Risk:Reward Ratio:** 1:{risk_reward:.2f}")

    # Supporting factors
    st.markdown("#### Supporting Factors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bullish Factors:**")
        summary = bias_analyzer.get_bias_summary(bias_result)
        for factor in summary.get('top_bullish_factors', [])[:5]:
            st.markdown(f"- {factor}")

    with col2:
        st.markdown("**Bearish Factors:**")
        for factor in summary.get('top_bearish_factors', [])[:5]:
            st.markdown(f"- {factor}")

    # Cautions
    if signal.cautions:
        st.markdown("#### ⚠️ Cautions")
        for caution in signal.cautions:
            st.warning(caution)


# ============================================================================
# HELPER FUNCTIONS - CHARTS
# ============================================================================

def create_basis_chart(futures_data: Dict, spot_price: float) -> go.Figure:
    """Create basis (futures - spot) trend chart"""
    # Demo data - replace with actual historical basis data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    basis_values = np.random.normal(spot_price * 0.005, spot_price * 0.002, 30)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=basis_values,
        mode='lines',
        name='Basis',
        line=dict(color='blue', width=2),
        fill='tozeroy'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Basis")

    fig.update_layout(
        title="Futures Basis Trend (30 Days)",
        xaxis_title="Date",
        yaxis_title="Basis (₹)",
        height=300,
        hovermode='x unified'
    )

    return fig


def create_bias_gauge(combined_bias: float) -> go.Figure:
    """Create gauge chart for combined bias"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=combined_bias,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Combined Futures Bias"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -60], 'color': "darkred"},
                {'range': [-60, -20], 'color': "lightcoral"},
                {'range': [-20, 20], 'color': "lightgray"},
                {'range': [20, 60], 'color': "lightgreen"},
                {'range': [60, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': combined_bias
            }
        }
    ))

    fig.update_layout(height=300)

    return fig


def create_futures_chart(historical_data: pd.DataFrame) -> go.Figure:
    """Create futures price chart with technical indicators"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_data.index,
            open=historical_data.get('open', historical_data.get('Open', [])),
            high=historical_data.get('high', historical_data.get('High', [])),
            low=historical_data.get('low', historical_data.get('Low', [])),
            close=historical_data.get('close', historical_data.get('Close', [])),
            name='NIFTY Futures'
        ),
        row=1, col=1
    )

    # Volume
    colors = ['green' if row.get('close', 0) >= row.get('open', 0) else 'red'
              for _, row in historical_data.iterrows()]

    fig.add_trace(
        go.Bar(
            x=historical_data.index,
            y=historical_data.get('volume', historical_data.get('Volume', [])),
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="NIFTY Futures Price Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )

    return fig


# ============================================================================
# HELPER FUNCTIONS - DEMO DATA
# ============================================================================

def _get_demo_futures_data(spot_price: float) -> Dict:
    """Generate demo futures data"""
    # Handle None or invalid spot_price
    if spot_price is None or spot_price == 0:
        spot_price = 25000.0  # Default NIFTY value

    return {
        'current_month': {
            'ltp': spot_price * 1.005,
            'expiry': '2025-01-30',
            'days_to_expiry': 15,
            'oi': 15000000,
            'oi_change_pct': 5.2,
            'volume': 500000,
            'volume_change_pct': 12.5,
            'spot_change_pct': 0.8,
            'price_change_pct': 1.2,
            'price_change_5d_pct': 3.5,
            'rsi': 58,
            'macd': 0.5,
            'bb_position': 'Upper',
            'trend': 'Uptrend',
            'buildup_pattern': 'LONG_BUILDUP'
        },
        'next_month': {
            'ltp': spot_price * 1.012,
            'expiry': '2025-02-27',
            'days_to_expiry': 45,
            'oi': 8000000,
            'oi_change_pct': 15.0,
            'volume': 200000,
            'volume_change_pct': 25.0
        },
        'rollover_percentage': 68.5,
        'rollover_trend': 'MODERATE',
        'theoretical_futures_price': spot_price * 1.0048,
        'basis': spot_price * 0.005,
        'basis_pct': 0.5
    }


def _get_demo_participant_data() -> Dict:
    """Generate demo participant data"""
    return {
        'fii_net_futures': 1500,  # Cr
        'dii_net_futures': 800,
        'pro_net_futures': -500,
        'client_net_futures': -1800
    }
