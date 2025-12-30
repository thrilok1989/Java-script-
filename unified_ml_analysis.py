"""
Unified ML Analysis Module
===========================
Combines all ML features into a single comprehensive analysis:
- ML Market Regime Detection
- XGBoost Predictions
- Volatility Regime
- OI Trap Detection
- CVD Delta Imbalance
- Liquidity Gravity
- Risk Management AI

This provides a single source of truth for all ML-based insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UnifiedMLAnalysis:
    """
    Unified ML Analysis combining all AI/ML modules
    """

    def __init__(self):
        """Initialize the Unified ML Analysis"""
        self.ml_regime_detector = None
        self.xgboost_analyzer = None
        self.volatility_detector = None
        self.oi_trap_detector = None
        self.cvd_analyzer = None
        self.liquidity_analyzer = None

        self._init_modules()

    def _init_modules(self):
        """Initialize all ML modules"""
        try:
            from src.ml_market_regime import MLMarketRegimeDetector
            self.ml_regime_detector = MLMarketRegimeDetector()
        except ImportError as e:
            logger.warning(f"ML Regime Detector not available: {e}")

        try:
            from src.xgboost_ml_analyzer import XGBoostMLAnalyzer
            self.xgboost_analyzer = XGBoostMLAnalyzer()
        except ImportError as e:
            logger.warning(f"XGBoost Analyzer not available: {e}")

        try:
            from src.volatility_regime import VolatilityRegimeDetector
            self.volatility_detector = VolatilityRegimeDetector()
        except ImportError as e:
            logger.warning(f"Volatility Detector not available: {e}")

        try:
            from src.oi_trap_detection import OITrapDetector
            self.oi_trap_detector = OITrapDetector()
        except ImportError as e:
            logger.warning(f"OI Trap Detector not available: {e}")

        try:
            from src.cvd_delta_imbalance import CVDAnalyzer
            self.cvd_analyzer = CVDAnalyzer()
        except ImportError as e:
            logger.warning(f"CVD Analyzer not available: {e}")

        try:
            from src.liquidity_gravity import LiquidityGravityAnalyzer
            self.liquidity_analyzer = LiquidityGravityAnalyzer()
        except ImportError as e:
            logger.warning(f"Liquidity Analyzer not available: {e}")

    def run_complete_analysis(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict] = None,
        vix_current: Optional[float] = None,
        spot_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run complete ML analysis using all available modules

        Args:
            df: Price DataFrame with OHLCV data
            option_chain: Option chain data (optional)
            vix_current: Current VIX value (optional)
            spot_price: Current spot price (optional)

        Returns:
            Dict with all analysis results
        """
        results = {
            'timestamp': datetime.now(),
            'success': False,
            'modules_run': [],
            'errors': []
        }

        # Ensure required columns
        if 'ATR' not in df.columns:
            df['ATR'] = df['High'] - df['Low']

        # 1. ML Market Regime Detection
        if self.ml_regime_detector:
            try:
                regime_result = self.ml_regime_detector.detect_regime(df)
                results['market_regime'] = {
                    'regime': regime_result.regime,
                    'confidence': regime_result.confidence,
                    'trend_strength': regime_result.trend_strength,
                    'volatility_state': regime_result.volatility_state,
                    'recommended_strategy': regime_result.recommended_strategy,
                    'optimal_timeframe': regime_result.optimal_timeframe,
                    'entry_signals': getattr(regime_result, 'entry_signals', []),
                    'exit_signals': getattr(regime_result, 'exit_signals', [])
                }
                results['modules_run'].append('market_regime')
            except Exception as e:
                results['errors'].append(f"Market Regime: {str(e)[:50]}")

        # 2. XGBoost Prediction
        if self.xgboost_analyzer:
            try:
                xgb_result = self.xgboost_analyzer.predict(df)
                results['xgboost'] = {
                    'prediction': xgb_result.prediction if hasattr(xgb_result, 'prediction') else 'N/A',
                    'confidence': xgb_result.confidence if hasattr(xgb_result, 'confidence') else 0,
                    'probabilities': xgb_result.probabilities if hasattr(xgb_result, 'probabilities') else {},
                    'feature_importance': xgb_result.feature_importance if hasattr(xgb_result, 'feature_importance') else {}
                }
                results['modules_run'].append('xgboost')
            except Exception as e:
                results['errors'].append(f"XGBoost: {str(e)[:50]}")

        # 3. Volatility Regime Detection
        if self.volatility_detector and vix_current:
            try:
                vol_result = self.volatility_detector.detect(
                    df=df,
                    vix_current=vix_current
                )
                results['volatility'] = {
                    'regime': vol_result.regime if hasattr(vol_result, 'regime') else 'Unknown',
                    'vix_percentile': vol_result.vix_percentile if hasattr(vol_result, 'vix_percentile') else 0,
                    'atr_percentile': vol_result.atr_percentile if hasattr(vol_result, 'atr_percentile') else 0,
                    'recommendation': vol_result.recommendation if hasattr(vol_result, 'recommendation') else ''
                }
                results['modules_run'].append('volatility')
            except Exception as e:
                results['errors'].append(f"Volatility: {str(e)[:50]}")

        # 4. OI Trap Detection
        if self.oi_trap_detector and option_chain:
            try:
                oi_result = self.oi_trap_detector.detect(option_chain, spot_price)
                results['oi_trap'] = {
                    'trap_detected': oi_result.trap_detected if hasattr(oi_result, 'trap_detected') else False,
                    'trap_type': oi_result.trap_type if hasattr(oi_result, 'trap_type') else 'None',
                    'confidence': oi_result.confidence if hasattr(oi_result, 'confidence') else 0,
                    'retail_trap_probability': oi_result.retail_trap_probability if hasattr(oi_result, 'retail_trap_probability') else 0
                }
                results['modules_run'].append('oi_trap')
            except Exception as e:
                results['errors'].append(f"OI Trap: {str(e)[:50]}")

        # 5. CVD Delta Imbalance
        if self.cvd_analyzer:
            try:
                cvd_result = self.cvd_analyzer.analyze(df)
                results['cvd'] = {
                    'delta_imbalance': cvd_result.delta_imbalance if hasattr(cvd_result, 'delta_imbalance') else 0,
                    'cumulative_delta': cvd_result.cumulative_delta if hasattr(cvd_result, 'cumulative_delta') else 0,
                    'signal': cvd_result.signal if hasattr(cvd_result, 'signal') else 'Neutral',
                    'strength': cvd_result.strength if hasattr(cvd_result, 'strength') else 0
                }
                results['modules_run'].append('cvd')
            except Exception as e:
                results['errors'].append(f"CVD: {str(e)[:50]}")

        # 6. Liquidity Gravity
        if self.liquidity_analyzer and option_chain:
            try:
                liq_result = self.liquidity_analyzer.analyze(option_chain, spot_price)
                results['liquidity'] = {
                    'gravity_center': liq_result.gravity_center if hasattr(liq_result, 'gravity_center') else 0,
                    'support_levels': liq_result.support_levels if hasattr(liq_result, 'support_levels') else [],
                    'resistance_levels': liq_result.resistance_levels if hasattr(liq_result, 'resistance_levels') else [],
                    'liquidity_score': liq_result.liquidity_score if hasattr(liq_result, 'liquidity_score') else 0
                }
                results['modules_run'].append('liquidity')
            except Exception as e:
                results['errors'].append(f"Liquidity: {str(e)[:50]}")

        # Calculate final verdict
        results['final_verdict'] = self._calculate_final_verdict(results)
        results['success'] = len(results['modules_run']) > 0

        return results

    def _calculate_final_verdict(self, results: Dict) -> Dict[str, Any]:
        """Calculate final trading verdict from all ML outputs"""

        bullish_signals = 0
        bearish_signals = 0
        total_confidence = 0
        signal_count = 0

        # Check market regime
        if 'market_regime' in results:
            regime = results['market_regime'].get('regime', '')
            conf = results['market_regime'].get('confidence', 0)

            if 'Up' in regime or 'BULLISH' in regime.upper():
                bullish_signals += 1
                total_confidence += conf
            elif 'Down' in regime or 'BEARISH' in regime.upper():
                bearish_signals += 1
                total_confidence += conf
            signal_count += 1

        # Check XGBoost prediction
        if 'xgboost' in results:
            pred = results['xgboost'].get('prediction', '')
            conf = results['xgboost'].get('confidence', 0)

            if pred == 'BUY':
                bullish_signals += 1
                total_confidence += conf
            elif pred == 'SELL':
                bearish_signals += 1
                total_confidence += conf
            signal_count += 1

        # Check CVD
        if 'cvd' in results:
            signal = results['cvd'].get('signal', '')
            strength = results['cvd'].get('strength', 0)

            if 'Bullish' in signal:
                bullish_signals += 0.5
                total_confidence += strength
            elif 'Bearish' in signal:
                bearish_signals += 0.5
                total_confidence += strength
            signal_count += 1

        # Check OI Trap
        if 'oi_trap' in results:
            trap_type = results['oi_trap'].get('trap_type', '')
            if trap_type == 'BEAR_TRAP':
                bullish_signals += 0.5
            elif trap_type == 'BULL_TRAP':
                bearish_signals += 0.5

        # Calculate final verdict
        avg_confidence = total_confidence / signal_count if signal_count > 0 else 0

        if bullish_signals > bearish_signals + 1:
            verdict = "STRONG BUY"
            action = "Consider entering long positions"
        elif bullish_signals > bearish_signals:
            verdict = "BUY"
            action = "Look for bullish setups"
        elif bearish_signals > bullish_signals + 1:
            verdict = "STRONG SELL"
            action = "Consider entering short positions"
        elif bearish_signals > bullish_signals:
            verdict = "SELL"
            action = "Look for bearish setups"
        else:
            verdict = "HOLD"
            action = "Wait for clearer signals"

        return {
            'verdict': verdict,
            'action': action,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'confidence': avg_confidence,
            'modules_analyzed': signal_count
        }


def render_unified_ml_dashboard(df: pd.DataFrame, option_chain: Dict = None,
                                 vix_current: float = None, spot_price: float = None):
    """
    Render the unified ML dashboard in Streamlit
    """
    st.subheader("🤖 Unified AI/ML Analysis")
    st.caption("Combining all ML modules for comprehensive market intelligence")

    # Initialize analyzer
    try:
        analyzer = UnifiedMLAnalysis()
    except Exception as e:
        st.error(f"Failed to initialize ML analyzer: {e}")
        return None

    # Run analysis
    with st.spinner("Running comprehensive ML analysis..."):
        try:
            results = analyzer.run_complete_analysis(
                df=df,
                option_chain=option_chain,
                vix_current=vix_current,
                spot_price=spot_price
            )
        except Exception as e:
            st.error(f"ML analysis failed: {e}")
            return None

    if not results.get('success'):
        st.warning("⚠️ No ML modules could be executed")
        if results.get('errors'):
            with st.expander("Show errors"):
                for error in results['errors']:
                    st.error(error)
        return None

    # Display Final Verdict (Large, Prominent)
    verdict_data = results.get('final_verdict', {})
    verdict = verdict_data.get('verdict', 'HOLD')

    verdict_colors = {
        'STRONG BUY': '#00FF00',
        'BUY': '#90EE90',
        'HOLD': '#FFD700',
        'SELL': '#FFA500',
        'STRONG SELL': '#FF0000'
    }

    verdict_color = verdict_colors.get(verdict, '#808080')

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {verdict_color}22, {verdict_color}44);
                border: 2px solid {verdict_color};
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;">
        <h2 style="color: {verdict_color}; margin: 0;">🎯 {verdict}</h2>
        <p style="color: #888; margin: 5px 0 0 0;">{verdict_data.get('action', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🐂 Bullish Signals",
            f"{verdict_data.get('bullish_signals', 0):.1f}"
        )

    with col2:
        st.metric(
            "🐻 Bearish Signals",
            f"{verdict_data.get('bearish_signals', 0):.1f}"
        )

    with col3:
        st.metric(
            "📊 Confidence",
            f"{verdict_data.get('confidence', 0):.1f}%"
        )

    with col4:
        st.metric(
            "🔧 Modules",
            f"{len(results.get('modules_run', []))}"
        )

    st.divider()

    # Individual Module Results
    st.markdown("### 📊 Individual Module Results")

    # Market Regime
    if 'market_regime' in results:
        regime = results['market_regime']
        with st.expander("🎯 Market Regime Detection", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            regime_name = regime.get('regime', 'Unknown')
            regime_color = '#00FF00' if 'Up' in regime_name else '#FF0000' if 'Down' in regime_name else '#FFD700'

            with col1:
                st.markdown(f"""
                <div style="background-color: {regime_color}; padding: 10px; border-radius: 8px; text-align: center;">
                    <b style="color: black;">REGIME</b><br>
                    <span style="color: black; font-weight: bold;">{regime_name}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Confidence", f"{regime.get('confidence', 0):.1f}%")

            with col3:
                st.metric("Trend Strength", f"{regime.get('trend_strength', 0):.1f}")

            with col4:
                st.metric("Volatility", regime.get('volatility_state', 'N/A'))

            st.info(f"📊 **Strategy:** {regime.get('recommended_strategy', 'N/A')}")

    # XGBoost Prediction
    if 'xgboost' in results:
        xgb = results['xgboost']
        with st.expander("🧠 XGBoost ML Prediction"):
            col1, col2, col3 = st.columns(3)

            pred = xgb.get('prediction', 'N/A')
            pred_color = '#00FF00' if pred == 'BUY' else '#FF0000' if pred == 'SELL' else '#FFD700'

            with col1:
                st.markdown(f"""
                <div style="background-color: {pred_color}; padding: 10px; border-radius: 8px; text-align: center;">
                    <b style="color: black;">PREDICTION</b><br>
                    <span style="color: black; font-weight: bold;">{pred}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Confidence", f"{xgb.get('confidence', 0):.1f}%")

            with col3:
                probs = xgb.get('probabilities', {})
                if probs:
                    st.write("**Probabilities:**")
                    for k, v in probs.items():
                        st.write(f"- {k}: {v:.1f}%")

    # Volatility Regime
    if 'volatility' in results:
        vol = results['volatility']
        with st.expander("📈 Volatility Regime"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Regime", vol.get('regime', 'Unknown'))

            with col2:
                st.metric("VIX Percentile", f"{vol.get('vix_percentile', 0):.0f}%")

            with col3:
                st.metric("ATR Percentile", f"{vol.get('atr_percentile', 0):.0f}%")

    # OI Trap Detection
    if 'oi_trap' in results:
        oi = results['oi_trap']
        with st.expander("🪤 OI Trap Detection"):
            col1, col2, col3 = st.columns(3)

            trap_detected = oi.get('trap_detected', False)

            with col1:
                if trap_detected:
                    st.error(f"⚠️ TRAP DETECTED: {oi.get('trap_type', 'Unknown')}")
                else:
                    st.success("✅ No trap detected")

            with col2:
                st.metric("Confidence", f"{oi.get('confidence', 0):.1f}%")

            with col3:
                st.metric("Retail Trap Prob", f"{oi.get('retail_trap_probability', 0):.1f}%")

    # CVD Analysis
    if 'cvd' in results:
        cvd = results['cvd']
        with st.expander("📊 CVD Delta Imbalance"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Signal", cvd.get('signal', 'Neutral'))

            with col2:
                st.metric("Delta Imbalance", f"{cvd.get('delta_imbalance', 0):.2f}")

            with col3:
                st.metric("Strength", f"{cvd.get('strength', 0):.1f}%")

    # Liquidity Analysis
    if 'liquidity' in results:
        liq = results['liquidity']
        with st.expander("💧 Liquidity Gravity"):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Gravity Center", f"₹{liq.get('gravity_center', 0):,.0f}")
                st.metric("Liquidity Score", f"{liq.get('liquidity_score', 0):.1f}")

            with col2:
                supports = liq.get('support_levels', [])
                resistances = liq.get('resistance_levels', [])

                if supports:
                    st.write("**Support Levels:**")
                    for s in supports[:3]:
                        st.write(f"- ₹{s:,.0f}")

                if resistances:
                    st.write("**Resistance Levels:**")
                    for r in resistances[:3]:
                        st.write(f"- ₹{r:,.0f}")

    # Show any errors
    if results.get('errors'):
        with st.expander("⚠️ Module Errors"):
            for error in results['errors']:
                st.warning(error)

    # Cache results in session state
    st.session_state.unified_ml_results = results

    return results


def get_unified_ml_verdict() -> Dict[str, Any]:
    """
    Get the cached unified ML verdict from session state

    Returns:
        Dict with verdict information or None
    """
    if 'unified_ml_results' in st.session_state:
        return st.session_state.unified_ml_results.get('final_verdict', {})
    return None
