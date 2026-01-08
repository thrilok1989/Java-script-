"""
Market Structure UI - Streamlit Display Component
Displays structure-based analysis without disturbing existing unified ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

# Import our structure analysis components
from src.market_structure_features import (
    MarketStructureFeatureExtractor, MarketStructure, MarketStructureSnapshot
)
from src.sequence_pattern_detector import (
    SequencePatternDetector, PatternType, SequenceAnalysisResult,
    SequencePatternDetectorAdapter, get_sequence_detector
)
from src.probability_engine import (
    ProbabilityEngine, ProbabilityAnalysis,
    ProbabilityEngineAdapter, get_probability_engine
)
from src.expiry_structure_detector import (
    ExpiryStructureDetector, ExpiryPhase, ExpiryStructureAnalysis
)
from src.geometric_pattern_engine import (
    GeometricPatternEngine, GeometricPatternType, PatternDirection,
    DetectedPattern, PatternAnalysisResult
)


# Color schemes for structures
STRUCTURE_COLORS = {
    MarketStructure.ACCUMULATION: "#00C853",      # Green
    MarketStructure.DISTRIBUTION: "#FF1744",      # Red
    MarketStructure.COMPRESSION: "#FFC107",       # Amber
    MarketStructure.EXPANSION: "#2196F3",         # Blue
    MarketStructure.MANIPULATION: "#9C27B0",      # Purple
    MarketStructure.NEUTRAL: "#757575",           # Grey
    MarketStructure.TRANSITION: "#FF9800",        # Orange
}

PATTERN_COLORS = {
    PatternType.SILENT_BUILDUP: "#4CAF50",
    PatternType.FAKE_BREAK_REVERSAL: "#F44336",
    PatternType.GAMMA_PIN_SNAP: "#9C27B0",
    PatternType.ABSORPTION_BREAK: "#2196F3",
    PatternType.SL_HUNT_SETUP: "#FF5722",
    PatternType.COMPRESSION_EXPLOSION: "#FFEB3B",
    PatternType.DELTA_DIVERGENCE: "#00BCD4",
    PatternType.OI_UNWINDING: "#795548",
}

EXPIRY_PHASE_COLORS = {
    ExpiryPhase.OPENING_HUNT: "#FF5722",
    ExpiryPhase.MORNING_TREND: "#4CAF50",
    ExpiryPhase.MIDDAY_CHOP: "#FFC107",
    ExpiryPhase.AFTERNOON_SETUP: "#2196F3",
    ExpiryPhase.GAMMA_HOUR: "#9C27B0",
    ExpiryPhase.LAST_MINUTE_CHAOS: "#F44336",
}

GEOMETRIC_PATTERN_COLORS = {
    # Triangles
    GeometricPatternType.ASCENDING_TRIANGLE: "#4CAF50",
    GeometricPatternType.DESCENDING_TRIANGLE: "#F44336",
    GeometricPatternType.SYMMETRIC_TRIANGLE: "#FFC107",
    # Channels
    GeometricPatternType.PARALLEL_CHANNEL: "#2196F3",
    GeometricPatternType.RISING_CHANNEL: "#00C853",
    GeometricPatternType.FALLING_CHANNEL: "#FF1744",
    GeometricPatternType.HORIZONTAL_RANGE: "#9E9E9E",
    # Wedges
    GeometricPatternType.RISING_WEDGE: "#E91E63",    # Bearish
    GeometricPatternType.FALLING_WEDGE: "#8BC34A",   # Bullish
    # Flags
    GeometricPatternType.BULL_FLAG: "#00E676",
    GeometricPatternType.BEAR_FLAG: "#FF5252",
    GeometricPatternType.PENNANT: "#FFAB00",
    # Tops & Bottoms
    GeometricPatternType.DOUBLE_TOP: "#D32F2F",
    GeometricPatternType.DOUBLE_BOTTOM: "#388E3C",
    GeometricPatternType.TRIPLE_TOP: "#B71C1C",
    GeometricPatternType.TRIPLE_BOTTOM: "#1B5E20",
    # H&S
    GeometricPatternType.HEAD_SHOULDERS: "#7B1FA2",
    GeometricPatternType.INVERSE_HEAD_SHOULDERS: "#00695C",
    # Compression
    GeometricPatternType.COIL_BOX: "#FF9800",
}


class MarketStructureUI:
    """Streamlit UI component for market structure analysis"""

    def __init__(self):
        self.feature_extractor = MarketStructureFeatureExtractor()
        self.pattern_detector = SequencePatternDetectorAdapter()  # Uses adapter for UI compatibility
        self.probability_engine = ProbabilityEngineAdapter()       # Uses adapter for UI compatibility
        self.expiry_detector = ExpiryStructureDetector()
        self.geometric_engine = GeometricPatternEngine()

    def render_structure_dashboard(
        self,
        ohlc_df: pd.DataFrame,
        option_data: Optional[Dict] = None,
        is_expiry: bool = False,
        spot_price: Optional[float] = None
    ):
        """Main entry point - renders full structure analysis dashboard"""

        st.markdown("### 📊 Market Structure Analysis")
        st.markdown("*Structure-based detection BEFORE price moves*")

        # Extract features
        snapshot = self._extract_current_snapshot(ohlc_df, option_data, spot_price)

        if snapshot is None:
            st.warning("Insufficient data for structure analysis")
            return

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Current Structure",
            "📐 Geometric Patterns",
            "🔄 Sequence Patterns",
            "📈 Probability Analysis",
            "⏰ Expiry Structure" if is_expiry else "📊 Structure History"
        ])

        with tab1:
            self._render_current_structure(snapshot, ohlc_df)

        with tab2:
            self._render_geometric_patterns(ohlc_df, option_data)

        with tab3:
            self._render_pattern_detection(snapshot, ohlc_df)

        with tab4:
            self._render_probability_analysis(snapshot)

        with tab5:
            if is_expiry:
                self._render_expiry_structure(ohlc_df, option_data, spot_price)
            else:
                self._render_structure_history(ohlc_df, option_data)

    def _extract_current_snapshot(
        self,
        ohlc_df: pd.DataFrame,
        option_data: Optional[Dict],
        spot_price: Optional[float]
    ) -> Optional[MarketStructureSnapshot]:
        """Extract current market structure snapshot"""
        try:
            if ohlc_df is None or len(ohlc_df) < 20:
                return None

            snapshot = self.feature_extractor.extract_snapshot(
                ohlc_df=ohlc_df,
                option_data=option_data,
                spot_price=spot_price or ohlc_df['close'].iloc[-1]
            )
            return snapshot
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None

    def _render_current_structure(self, snapshot: MarketStructureSnapshot, ohlc_df: pd.DataFrame):
        """Render current structure analysis"""

        # Main structure indicator
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            structure = snapshot.primary_structure
            color = STRUCTURE_COLORS.get(structure, "#757575")

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}44);
                border-left: 4px solid {color};
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h2 style="margin: 0; color: {color};">{structure.value}</h2>
                <p style="margin: 5px 0 0 0; color: #aaa;">Current Market Structure</p>
            </div>
            """, unsafe_allow_html=True)

            # Structure confidence
            confidence = snapshot.structure_confidence
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        with col2:
            # Price structure metrics
            st.markdown("**Price Structure**")
            price_features = snapshot.price_features

            metrics = {
                "Range/ATR": f"{price_features.price_range_atr_ratio:.2f}",
                "CLV": f"{price_features.clv:.1%}",
                "Momentum": f"{price_features.price_momentum_5:.2f}%",
            }

            for label, value in metrics.items():
                st.metric(label, value)

        with col3:
            # Volatility metrics
            st.markdown("**Volatility**")
            vol_features = snapshot.volatility_features

            st.metric("ATR Ratio", f"{vol_features.atr_ratio:.2f}")
            st.metric("VIX", f"{vol_features.vix_level:.1f}")
            st.metric("Compression", f"{vol_features.compression_score:.0f}")

        # Structure visualization chart
        st.markdown("#### Structure Overlay Chart")
        self._render_structure_chart(ohlc_df, snapshot)

        # Feature breakdown
        with st.expander("📋 Detailed Feature Breakdown", expanded=False):
            self._render_feature_breakdown(snapshot)

    def _render_structure_chart(self, ohlc_df: pd.DataFrame, snapshot: MarketStructureSnapshot):
        """Render price chart with structure overlay"""

        # Use last 100 candles
        df = ohlc_df.tail(100).copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price with Structure Zones", "Volume Profile")
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00C853',
                decreasing_line_color='#FF1744'
            ),
            row=1, col=1
        )

        # Add structure zone highlighting
        price_features = snapshot.price_features
        last_idx = len(df) - 1

        # Calculate range from recent data (last 20 bars)
        recent_df = df.tail(20)
        range_high = recent_df['high'].max()
        range_low = recent_df['low'].min()

        # Range box
        if range_high > 0 and range_low > 0:
            fig.add_hrect(
                y0=range_low,
                y1=range_high,
                fillcolor=STRUCTURE_COLORS.get(snapshot.primary_structure, "#757575"),
                opacity=0.1,
                line_width=0,
                row=1, col=1
            )

            # Range boundaries
            fig.add_hline(
                y=range_high,
                line_dash="dash",
                line_color="#FF9800",
                annotation_text="Range High",
                row=1, col=1
            )
            fig.add_hline(
                y=range_low,
                line_dash="dash",
                line_color="#2196F3",
                annotation_text="Range Low",
                row=1, col=1
            )

        # Volume bars
        colors = ['#00C853' if c >= o else '#FF1744' for c, o in zip(df['close'], df['open'])]

        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)),
                    y=df['volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.7
                ),
                row=2, col=1
            )

        # Layout
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_feature_breakdown(self, snapshot: MarketStructureSnapshot):
        """Render detailed feature breakdown"""

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Features**")
            price_dict = asdict(snapshot.price_features)
            for key, value in price_dict.items():
                if isinstance(value, float):
                    st.text(f"{key}: {value:.4f}")
                else:
                    st.text(f"{key}: {value}")

            st.markdown("**Volume/OI Features**")
            vol_oi_dict = asdict(snapshot.volume_oi_features)
            for key, value in vol_oi_dict.items():
                if isinstance(value, float):
                    st.text(f"{key}: {value:.4f}")
                else:
                    st.text(f"{key}: {value}")

        with col2:
            st.markdown("**Volatility Features**")
            vol_dict = asdict(snapshot.volatility_features)
            for key, value in vol_dict.items():
                if isinstance(value, float):
                    st.text(f"{key}: {value:.4f}")
                else:
                    st.text(f"{key}: {value}")

            st.markdown("**Derived Indicators**")
            derived_dict = asdict(snapshot.derived_indicators)
            for key, value in derived_dict.items():
                if isinstance(value, float):
                    st.text(f"{key}: {value:.4f}")
                else:
                    st.text(f"{key}: {value}")

    def _render_geometric_patterns(self, ohlc_df: pd.DataFrame, option_data: Optional[Dict]):
        """Render geometric pattern analysis - Institution-grade pattern detection"""

        st.markdown("#### 📐 Geometric Pattern Detection")
        st.markdown("*Shapes converted to numeric features ML can learn*")

        try:
            # Analyze geometric patterns
            result = self.geometric_engine.analyze(ohlc_df, option_data)

            # Display swing points info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Swing Highs", len(result.swing_highs))
            with col2:
                st.metric("Swing Lows", len(result.swing_lows))
            with col3:
                st.metric("Market Structure", result.market_structure)

            # Display geometric features
            with st.expander("📊 Geometric Features", expanded=False):
                gf = result.geometric_features
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Slope Features**")
                    st.text(f"Slope Highs: {gf.slope_highs:.6f}")
                    st.text(f"Slope Lows: {gf.slope_lows:.6f}")
                    st.text(f"Norm Slope Highs: {gf.norm_slope_highs:.6f}")
                    st.text(f"Norm Slope Lows: {gf.norm_slope_lows:.6f}")

                with col2:
                    st.markdown("**Structure Features**")
                    st.text(f"Range Width: {gf.range_width:.2f}")
                    st.text(f"Compression Ratio: {gf.compression_ratio:.2f}")
                    st.text(f"Convergence Rate: {gf.convergence_rate:.6f}")
                    st.text(f"Converging: {'Yes' if gf.is_converging else 'No'}")

            # Display detected patterns
            if not result.patterns:
                st.info("No geometric patterns detected in current price structure")
                return

            st.markdown("#### 🎯 Detected Patterns")

            for pattern in result.patterns:
                color = GEOMETRIC_PATTERN_COLORS.get(pattern.pattern_type, "#757575")
                direction_icon = "🟢" if pattern.direction == PatternDirection.BULLISH else "🔴" if pattern.direction == PatternDirection.BEARISH else "⚪"

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}11, {color}22);
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                ">
                    <h4 style="margin: 0; color: {color};">
                        {direction_icon} {pattern.pattern_type.value.replace('_', ' ')}
                    </h4>
                    <p style="margin: 5px 0; color: #bbb;">
                        Direction: {pattern.direction.value} | Geometry Score: {pattern.geometry_score:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Pattern details
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    vol_icon = "✅" if pattern.volume_confirmed else "❌"
                    st.metric("Volume", vol_icon)
                with col2:
                    oi_icon = "✅" if pattern.oi_confirmed else "❌"
                    st.metric("OI", oi_icon)
                with col3:
                    delta_icon = "✅" if pattern.delta_confirmed else "❌"
                    st.metric("Delta", delta_icon)
                with col4:
                    st.metric("Fake Risk", f"{pattern.fake_move_risk:.0%}")

                # Expected outcomes
                if pattern.expected_outcomes:
                    outcomes = pattern.expected_outcomes
                    if 'expansion' in outcomes:
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Expansion", f"{outcomes.get('expansion', 0):.0%}")
                        with cols[1]:
                            st.metric("Fake", f"{outcomes.get('fake', 0):.0%}")
                        with cols[2]:
                            st.metric("Chop", f"{outcomes.get('chop', 0):.0%}")

                # Support/Resistance levels
                if pattern.support > 0 or pattern.resistance > 0:
                    st.markdown(f"**Levels**: Support: {pattern.support:,.2f} | Resistance: {pattern.resistance:,.2f}")

                if pattern.target > 0:
                    st.markdown(f"**Target**: {pattern.target:,.2f}")

                st.markdown("---")

            # Dominant pattern highlight
            if result.dominant_pattern:
                dp = result.dominant_pattern
                color = GEOMETRIC_PATTERN_COLORS.get(dp.pattern_type, "#757575")

                st.markdown(f"""
                ### 👑 Dominant Pattern

                <div style="
                    background: {color}33;
                    border: 2px solid {color};
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                ">
                    <h2 style="margin: 0; color: {color};">{dp.pattern_type.value.replace('_', ' ')}</h2>
                    <p style="margin: 10px 0 0 0; color: #ddd;">
                        {dp.direction.value} | Score: {dp.geometry_score:.1%} | Fake Risk: {dp.fake_move_risk:.0%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Pattern chart with swing points
            st.markdown("#### 📈 Pattern Visualization")
            self._render_geometric_chart(ohlc_df, result)

        except Exception as e:
            st.error(f"Error analyzing geometric patterns: {e}")
            import traceback
            st.code(traceback.format_exc())

    def _render_geometric_chart(self, ohlc_df: pd.DataFrame, result: PatternAnalysisResult):
        """Render chart with swing points and pattern overlay"""

        df = ohlc_df.tail(100).copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75, 0.25],
            subplot_titles=("Price with Swing Points", "Volume")
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(df))),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00C853',
                decreasing_line_color='#FF1744'
            ),
            row=1, col=1
        )

        # Add swing highs
        swing_highs_in_range = [sh for sh in result.swing_highs if sh.index >= len(ohlc_df) - 100]
        if swing_highs_in_range:
            sh_x = [sh.index - (len(ohlc_df) - 100) for sh in swing_highs_in_range]
            sh_y = [sh.price for sh in swing_highs_in_range]
            fig.add_trace(
                go.Scatter(
                    x=sh_x,
                    y=sh_y,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='#FF5722'),
                    name='Swing Highs'
                ),
                row=1, col=1
            )

        # Add swing lows
        swing_lows_in_range = [sl for sl in result.swing_lows if sl.index >= len(ohlc_df) - 100]
        if swing_lows_in_range:
            sl_x = [sl.index - (len(ohlc_df) - 100) for sl in swing_lows_in_range]
            sl_y = [sl.price for sl in swing_lows_in_range]
            fig.add_trace(
                go.Scatter(
                    x=sl_x,
                    y=sl_y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='#2196F3'),
                    name='Swing Lows'
                ),
                row=1, col=1
            )

        # Add trendlines for dominant pattern
        if result.dominant_pattern and result.dominant_pattern.features:
            gf = result.geometric_features

            # High trendline
            if len(swing_highs_in_range) >= 2:
                x_line = [swing_highs_in_range[0].index - (len(ohlc_df) - 100),
                          swing_highs_in_range[-1].index - (len(ohlc_df) - 100)]
                y_line = [swing_highs_in_range[0].price,
                          swing_highs_in_range[0].price + gf.slope_highs * (x_line[1] - x_line[0])]
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='#FF9800', dash='dash', width=2),
                        name='High Trend'
                    ),
                    row=1, col=1
                )

            # Low trendline
            if len(swing_lows_in_range) >= 2:
                x_line = [swing_lows_in_range[0].index - (len(ohlc_df) - 100),
                          swing_lows_in_range[-1].index - (len(ohlc_df) - 100)]
                y_line = [swing_lows_in_range[0].price,
                          swing_lows_in_range[0].price + gf.slope_lows * (x_line[1] - x_line[0])]
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='#2196F3', dash='dash', width=2),
                        name='Low Trend'
                    ),
                    row=1, col=1
                )

        # Volume bars
        if 'volume' in df.columns:
            colors = ['#00C853' if c >= o else '#FF1744' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(df))),
                    y=df['volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.7
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=30, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_pattern_detection(self, snapshot: MarketStructureSnapshot, ohlc_df: pd.DataFrame):
        """Render pattern detection results"""

        st.markdown("#### 🔍 Detected Patterns")

        # Detect patterns
        analysis = self.pattern_detector.analyze_sequence([snapshot])

        if not analysis.detected_patterns:
            st.info("No significant patterns detected in current structure")
            return

        # Display detected patterns
        for match in analysis.detected_patterns:
            pattern = match.pattern
            color = PATTERN_COLORS.get(pattern.pattern_type, "#757575")

            with st.container():
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}11, {color}22);
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                ">
                    <h4 style="margin: 0; color: {color};">{pattern.pattern_type.value.replace('_', ' ')}</h4>
                    <p style="margin: 5px 0; color: #bbb;">{pattern.description}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Match Score", f"{match.match_score:.1%}")
                with col2:
                    st.metric("Confidence", f"{match.confidence:.1%}")
                with col3:
                    # Get the most likely outcome from expected_outcomes dict
                    outcomes = pattern.expected_outcomes if hasattr(pattern, 'expected_outcomes') else {}
                    if outcomes:
                        best_outcome = max(outcomes.items(), key=lambda x: x[1])
                        st.metric("Expected", best_outcome[0])
                    else:
                        st.metric("Expected", "N/A")
                with col4:
                    if outcomes:
                        prob = best_outcome[1]
                        st.metric("Probability", f"{prob:.0%}")
                    else:
                        st.metric("Probability", "N/A")

        # Pattern summary
        st.markdown("#### 📊 Pattern Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Dominant Structure", analysis.dominant_structure.value if analysis.dominant_structure else "N/A")
            st.metric("Total Patterns", len(analysis.detected_patterns))

        with col2:
            st.metric("Transition Probability", f"{analysis.transition_probability:.1%}")
            # Show structure probabilities
            if analysis.structure_probabilities:
                top_structure = max(analysis.structure_probabilities.items(), key=lambda x: x[1])
                st.metric("Most Likely Next", f"{top_structure[0]} ({top_structure[1]:.0%})")

    def _render_probability_analysis(self, snapshot: MarketStructureSnapshot):
        """Render probability analysis"""

        st.markdown("#### 📈 Probability Analysis")

        # Get probability analysis
        analysis = self.probability_engine.analyze(snapshot)

        # Main probability display
        col1, col2, col3 = st.columns(3)

        with col1:
            prob_up = analysis.expansion_up_probability
            color_up = "#00C853" if prob_up > 0.5 else "#757575"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {color_up}22; border-radius: 8px;">
                <h2 style="margin: 0; color: {color_up};">{prob_up:.1%}</h2>
                <p style="margin: 5px 0 0 0; color: #aaa;">Expansion UP</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            prob_down = analysis.expansion_down_probability
            color_down = "#FF1744" if prob_down > 0.5 else "#757575"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {color_down}22; border-radius: 8px;">
                <h2 style="margin: 0; color: {color_down};">{prob_down:.1%}</h2>
                <p style="margin: 5px 0 0 0; color: #aaa;">Expansion DOWN</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            prob_range = analysis.continued_range_probability
            color_range = "#FFC107" if prob_range > 0.5 else "#757575"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {color_range}22; border-radius: 8px;">
                <h2 style="margin: 0; color: {color_range};">{prob_range:.1%}</h2>
                <p style="margin: 5px 0 0 0; color: #aaa;">Range Continue</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence and sample info
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Confidence Score", f"{analysis.confidence_score:.2f}")
        with col2:
            st.metric("Sample Size", analysis.sample_size)
        with col3:
            st.metric("SL Hunt Prob", f"{analysis.sl_hunt_probability:.1%}")
        with col4:
            st.metric("Fake Break Prob", f"{analysis.fake_break_probability:.1%}")

        # All outcome probabilities
        with st.expander("📋 All Outcome Probabilities", expanded=False):
            prob_df = pd.DataFrame([
                {"Outcome": k, "Probability": f"{v:.1%}"}
                for k, v in analysis.all_probabilities.items()
            ])
            st.dataframe(prob_df, use_container_width=True)

        # Historical context
        st.markdown("#### 📚 Historical Context")
        st.info(f"""
        Based on **{analysis.sample_size}** similar historical patterns:
        - Most likely outcome: **{analysis.most_likely_outcome}** ({analysis.all_probabilities.get(analysis.most_likely_outcome, 0):.1%})
        - Warning: SL Hunt probability is **{analysis.sl_hunt_probability:.1%}**
        - Fake break probability is **{analysis.fake_break_probability:.1%}**
        """)

    def _render_expiry_structure(
        self,
        ohlc_df: pd.DataFrame,
        option_data: Optional[Dict],
        spot_price: Optional[float]
    ):
        """Render expiry-specific structure analysis"""

        st.markdown("#### ⏰ Expiry Day Structure")

        # Get expiry analysis
        analysis = self.expiry_detector.analyze(
            df=ohlc_df,
            spot_price=spot_price or ohlc_df['close'].iloc[-1],
            option_chain=option_data
        )

        # Current phase
        phase = analysis.current_phase
        phase_color = EXPIRY_PHASE_COLORS.get(phase, "#757575")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {phase_color}22, {phase_color}44);
            border-left: 4px solid {phase_color};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <h2 style="margin: 0; color: {phase_color};">{phase.value.replace('_', ' ')}</h2>
            <p style="margin: 5px 0 0 0; color: #aaa;">Current Expiry Phase</p>
        </div>
        """, unsafe_allow_html=True)

        # Phase timeline
        st.markdown("#### 📅 Expiry Day Timeline")

        phases_timeline = [
            ("9:15-10:00", "OPENING HUNT", ExpiryPhase.OPENING_HUNT),
            ("10:00-12:00", "MORNING TREND", ExpiryPhase.MORNING_TREND),
            ("12:00-14:00", "MIDDAY CHOP", ExpiryPhase.MIDDAY_CHOP),
            ("14:00-14:30", "AFTERNOON SETUP", ExpiryPhase.AFTERNOON_SETUP),
            ("14:30-15:00", "GAMMA HOUR", ExpiryPhase.GAMMA_HOUR),
            ("15:00-15:30", "LAST MINUTE CHAOS", ExpiryPhase.LAST_MINUTE_CHAOS),
        ]

        cols = st.columns(len(phases_timeline))
        for i, (time_range, label, p) in enumerate(phases_timeline):
            with cols[i]:
                is_current = p == phase
                bg_color = EXPIRY_PHASE_COLORS.get(p, "#757575") if is_current else "#333"
                border = f"3px solid {EXPIRY_PHASE_COLORS.get(p, '#757575')}" if is_current else "1px solid #555"

                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 10px 5px;
                    background: {bg_color}33;
                    border: {border};
                    border-radius: 8px;
                    min-height: 80px;
                ">
                    <small style="color: #888;">{time_range}</small>
                    <p style="margin: 5px 0; font-size: 11px; color: {'#fff' if is_current else '#aaa'};">{label}</p>
                </div>
                """, unsafe_allow_html=True)

        # Gamma and Max Pain analysis
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Gamma Pin Analysis**")
            gamma = analysis.gamma_pin_analysis
            if gamma:
                st.metric("Pin Level", f"{gamma.pin_level:,.0f}")
                st.metric("Pin Strength", f"{gamma.pin_strength:.1%}")
                st.metric("Distance to Pin", f"{gamma.distance_to_pin:.2f}%")
            else:
                st.info("No gamma pin detected")

        with col2:
            st.markdown("**💰 Max Pain Analysis**")
            max_pain = analysis.max_pain_analysis
            if max_pain:
                st.metric("Max Pain Level", f"{max_pain.max_pain_level:,.0f}")
                st.metric("Magnet Strength", f"{max_pain.magnet_strength:.1%}")
                st.metric("Distance to MP", f"{max_pain.distance_to_max_pain:.2f}%")
            else:
                st.info("No max pain data available")

        # Detected expiry patterns
        st.markdown("#### 🔍 Expiry Patterns Detected")

        if analysis.detected_patterns:
            for pattern in analysis.detected_patterns:
                st.markdown(f"""
                - **{pattern.pattern_type.value.replace('_', ' ')}**
                  (Strength: {pattern.strength:.1%}, Confidence: {pattern.confidence:.1%})
                """)
        else:
            st.info("No specific expiry patterns detected")

        # Trading recommendations
        st.markdown("#### 💡 Expiry Trading Notes")

        recommendations = analysis.trading_recommendations
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("No specific recommendations for current phase")

    def _render_structure_history(
        self,
        ohlc_df: pd.DataFrame,
        option_data: Optional[Dict]
    ):
        """Render structure history (non-expiry days)"""

        st.markdown("#### 📊 Structure Evolution")

        # Generate structure history from recent data
        if len(ohlc_df) < 50:
            st.warning("Insufficient data for structure history")
            return

        # Sample structures at intervals
        intervals = [10, 20, 30, 40, 50]
        history = []

        for i in intervals:
            if len(ohlc_df) >= i:
                df_slice = ohlc_df.tail(i)
                try:
                    snapshot = self.feature_extractor.extract_snapshot(
                        ohlc_df=df_slice,
                        option_data=option_data,
                        spot_price=df_slice['close'].iloc[-1]
                    )
                    history.append({
                        'period': f"Last {i} bars",
                        'structure': snapshot.primary_structure.value,
                        'confidence': snapshot.structure_confidence
                    })
                except:
                    pass

        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)

            # Structure transition visualization
            fig = go.Figure()

            structures = [h['structure'] for h in history]
            confidences = [h['confidence'] for h in history]
            periods = [h['period'] for h in history]

            colors = [STRUCTURE_COLORS.get(MarketStructure(s), "#757575") for s in structures]

            fig.add_trace(go.Bar(
                x=periods,
                y=confidences,
                marker_color=colors,
                text=structures,
                textposition='auto',
            ))

            fig.update_layout(
                title="Structure Confidence Over Time",
                yaxis_title="Confidence",
                template="plotly_dark",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not generate structure history")


def render_market_structure_section(
    ohlc_df: pd.DataFrame,
    option_data: Optional[Dict] = None,
    is_expiry: bool = False,
    spot_price: Optional[float] = None
):
    """
    Main entry function to render market structure analysis section
    Call this from app.py to integrate the structure analysis
    """
    ui = MarketStructureUI()
    ui.render_structure_dashboard(
        ohlc_df=ohlc_df,
        option_data=option_data,
        is_expiry=is_expiry,
        spot_price=spot_price
    )


# Compact widget for sidebar or quick view
def render_structure_widget(
    ohlc_df: pd.DataFrame,
    option_data: Optional[Dict] = None,
    spot_price: Optional[float] = None
) -> Optional[Dict]:
    """
    Compact structure widget for sidebar display
    Returns current structure info
    """
    try:
        extractor = MarketStructureFeatureExtractor()
        snapshot = extractor.extract_snapshot(
            ohlc_df=ohlc_df,
            option_data=option_data,
            spot_price=spot_price or ohlc_df['close'].iloc[-1]
        )

        structure = snapshot.primary_structure
        confidence = snapshot.structure_confidence
        color = STRUCTURE_COLORS.get(structure, "#757575")

        st.markdown(f"""
        <div style="
            background: {color}22;
            border-left: 3px solid {color};
            padding: 10px;
            border-radius: 4px;
        ">
            <strong style="color: {color};">{structure.value}</strong>
            <br/>
            <small style="color: #888;">Confidence: {confidence:.0%}</small>
        </div>
        """, unsafe_allow_html=True)

        return {
            'structure': structure.value,
            'confidence': confidence,
            'color': color
        }
    except Exception as e:
        st.warning(f"Structure: N/A")
        return None


# Quick probability display
def render_probability_widget(
    ohlc_df: pd.DataFrame,
    option_data: Optional[Dict] = None,
    spot_price: Optional[float] = None
) -> Optional[Dict]:
    """
    Compact probability widget for quick view
    """
    try:
        extractor = MarketStructureFeatureExtractor()
        snapshot = extractor.extract_snapshot(
            ohlc_df=ohlc_df,
            option_data=option_data,
            spot_price=spot_price or ohlc_df['close'].iloc[-1]
        )

        engine = ProbabilityEngine()
        analysis = engine.analyze(snapshot)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("↑ Up", f"{analysis.expansion_up_probability:.0%}")
        with col2:
            st.metric("↓ Down", f"{analysis.expansion_down_probability:.0%}")
        with col3:
            st.metric("→ Range", f"{analysis.continued_range_probability:.0%}")

        return {
            'up': analysis.expansion_up_probability,
            'down': analysis.expansion_down_probability,
            'range': analysis.continued_range_probability,
            'most_likely': analysis.most_likely_outcome
        }
    except Exception as e:
        st.warning("Probability: N/A")
        return None
