"""
Unified ML Trading Signal
==========================
Combines ALL ML modules into a SINGLE trading signal

Modules Combined:
1. ML Market Regime Detection
2. XGBoost Prediction (BUY/SELL/HOLD)
3. Volatility Regime
4. OI Trap Detection
5. CVD Delta Imbalance
6. Liquidity Gravity Analysis
7. Institutional vs Retail Detection

Output: Single unified trading signal with confidence score
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSignal:
    """Unified Trading Signal"""
    signal: str  # "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
    confidence: float  # 0-100

    # Component scores
    regime_score: float  # -100 to +100
    xgboost_score: float  # -100 to +100
    volatility_score: float  # 0 to 100
    oi_trap_score: float  # -100 to +100
    cvd_score: float  # -100 to +100
    liquidity_score: float  # -100 to +100
    expiry_score: float  # 0 to 100 (risk score)

    # Details
    regime: str
    volatility_state: str
    trap_warning: str
    recommended_strategy: str

    # Expiry analysis
    days_to_expiry: float
    expiry_spike_probability: float
    expiry_spike_type: str
    expiry_warning: str

    # Risk metrics
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    position_size_multiplier: float  # 0.25 to 1.5

    # Entry/Exit levels
    entry_zone: Tuple[float, float]
    stop_loss: float
    targets: list

    # Reasoning
    bullish_reasons: list
    bearish_reasons: list

    # ATM Option LTP for trade recommendation
    atm_strike: float = 0.0
    atm_call_ltp: float = 0.0
    atm_put_ltp: float = 0.0

    timestamp: datetime = None


class UnifiedMLSignalGenerator:
    """
    Generates a unified trading signal from all ML modules
    """

    def __init__(self):
        """Initialize the signal generator"""
        self.modules_loaded = {}
        self._load_modules()

    def _load_modules(self):
        """Load all ML modules"""
        # ML Market Regime
        try:
            from src.ml_market_regime import MLMarketRegimeDetector
            self.regime_detector = MLMarketRegimeDetector()
            self.modules_loaded['regime'] = True
        except ImportError:
            self.modules_loaded['regime'] = False
            self.regime_detector = None

        # XGBoost Analyzer
        try:
            from src.xgboost_ml_analyzer import XGBoostMLAnalyzer
            self.xgboost_analyzer = XGBoostMLAnalyzer()
            self.modules_loaded['xgboost'] = True
        except ImportError:
            self.modules_loaded['xgboost'] = False
            self.xgboost_analyzer = None

        # Volatility Regime
        try:
            from src.volatility_regime import VolatilityRegimeDetector
            self.volatility_detector = VolatilityRegimeDetector()
            self.modules_loaded['volatility'] = True
        except ImportError:
            self.modules_loaded['volatility'] = False
            self.volatility_detector = None

        # OI Trap Detection
        try:
            from src.oi_trap_detection import OITrapDetector
            self.oi_trap_detector = OITrapDetector()
            self.modules_loaded['oi_trap'] = True
        except ImportError:
            self.modules_loaded['oi_trap'] = False
            self.oi_trap_detector = None

        # CVD Analyzer
        try:
            from src.cvd_delta_imbalance import CVDAnalyzer
            self.cvd_analyzer = CVDAnalyzer()
            self.modules_loaded['cvd'] = True
        except ImportError:
            self.modules_loaded['cvd'] = False
            self.cvd_analyzer = None

        # Liquidity Analyzer
        try:
            from src.liquidity_gravity import LiquidityGravityAnalyzer
            self.liquidity_analyzer = LiquidityGravityAnalyzer()
            self.modules_loaded['liquidity'] = True
        except ImportError:
            self.modules_loaded['liquidity'] = False
            self.liquidity_analyzer = None

        # Institutional Detector
        try:
            from src.institutional_retail_detector import InstitutionalRetailDetector
            self.institutional_detector = InstitutionalRetailDetector()
            self.modules_loaded['institutional'] = True
        except ImportError:
            self.modules_loaded['institutional'] = False
            self.institutional_detector = None

    def generate_signal(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict] = None,
        vix_current: Optional[float] = None,
        spot_price: Optional[float] = None,
        bias_results: Optional[Dict] = None
    ) -> UnifiedSignal:
        """
        Generate unified trading signal from all ML modules

        Args:
            df: Price DataFrame with OHLCV
            option_chain: Option chain data
            vix_current: Current VIX value
            spot_price: Current spot price
            bias_results: Results from Bias Analysis Pro

        Returns:
            UnifiedSignal with combined analysis
        """
        # Initialize scores
        scores = {
            'regime': 0,
            'xgboost': 0,
            'volatility': 50,  # Neutral default
            'oi_trap': 0,
            'cvd': 0,
            'liquidity': 0,
            'institutional': 0
        }

        bullish_reasons = []
        bearish_reasons = []

        regime_name = "Unknown"
        volatility_state = "Normal"
        trap_warning = "None"
        strategy = "Wait for confirmation"

        # Ensure ATR column exists (handle both 'High'/'Low' and 'high'/'low' columns)
        if 'ATR' not in df.columns:
            df = df.copy()
            # Check for column names (could be 'High' or 'high')
            high_col = 'High' if 'High' in df.columns else 'high' if 'high' in df.columns else None
            low_col = 'Low' if 'Low' in df.columns else 'low' if 'low' in df.columns else None
            if high_col and low_col:
                df['ATR'] = df[high_col] - df[low_col]
            else:
                df['ATR'] = 50  # Default ATR if columns missing

        # 1. ML Market Regime
        if self.regime_detector:
            try:
                regime_result = self.regime_detector.detect_regime(df)
                regime_name = regime_result.regime

                # Convert regime to score (-100 to +100)
                if 'Up' in regime_name or 'BULLISH' in regime_name.upper():
                    scores['regime'] = min(regime_result.trend_strength * 1.5, 100)
                    bullish_reasons.append(f"Regime: {regime_name} (Trend: {regime_result.trend_strength:.0f}%)")
                elif 'Down' in regime_name or 'BEARISH' in regime_name.upper():
                    scores['regime'] = -min(regime_result.trend_strength * 1.5, 100)
                    bearish_reasons.append(f"Regime: {regime_name} (Trend: {regime_result.trend_strength:.0f}%)")
                else:
                    scores['regime'] = 0

                volatility_state = regime_result.volatility_state
                strategy = regime_result.recommended_strategy

            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")

        # 2. Volatility Analysis
        if self.volatility_detector and vix_current:
            try:
                vol_result = self.volatility_detector.detect(df, vix_current)

                # Volatility affects position sizing, not direction
                if hasattr(vol_result, 'regime'):
                    vol_regime = str(vol_result.regime)
                    if 'Low' in vol_regime:
                        scores['volatility'] = 80
                        bullish_reasons.append("Low volatility - Good for trending")
                    elif 'High' in vol_regime or 'Extreme' in vol_regime:
                        scores['volatility'] = 20
                        bearish_reasons.append("High volatility - Reduce position")
                    else:
                        scores['volatility'] = 50

            except Exception as e:
                logger.warning(f"Volatility detection failed: {e}")

        # 3. OI Trap Detection
        if self.oi_trap_detector and option_chain and spot_price:
            try:
                oi_result = self.oi_trap_detector.detect(option_chain, spot_price)

                if hasattr(oi_result, 'trap_detected') and oi_result.trap_detected:
                    trap_type = getattr(oi_result, 'trap_type', 'Unknown')
                    trap_warning = f"{trap_type} detected!"

                    if trap_type == 'BEAR_TRAP':
                        scores['oi_trap'] = 50  # Bullish
                        bullish_reasons.append(f"Bear trap detected - Expect reversal up")
                    elif trap_type == 'BULL_TRAP':
                        scores['oi_trap'] = -50  # Bearish
                        bearish_reasons.append(f"Bull trap detected - Expect reversal down")
                else:
                    trap_warning = "No trap"

            except Exception as e:
                logger.warning(f"OI Trap detection failed: {e}")

        # 4. CVD Analysis
        if self.cvd_analyzer:
            try:
                cvd_result = self.cvd_analyzer.analyze(df)

                if hasattr(cvd_result, 'bias'):
                    if cvd_result.bias == 'Bullish':
                        scores['cvd'] = min(getattr(cvd_result, 'strength', 50), 100)
                        bullish_reasons.append(f"CVD Bullish (Strength: {scores['cvd']:.0f}%)")
                    elif cvd_result.bias == 'Bearish':
                        scores['cvd'] = -min(getattr(cvd_result, 'strength', 50), 100)
                        bearish_reasons.append(f"CVD Bearish (Strength: {abs(scores['cvd']):.0f}%)")

            except Exception as e:
                logger.warning(f"CVD analysis failed: {e}")

        # 5. Liquidity Analysis
        if self.liquidity_analyzer and option_chain and spot_price:
            try:
                liq_result = self.liquidity_analyzer.analyze(option_chain, spot_price)

                if hasattr(liq_result, 'gravity_center'):
                    gravity = liq_result.gravity_center
                    if gravity > spot_price:
                        scores['liquidity'] = 30  # Price likely to move up
                        bullish_reasons.append(f"Liquidity center above price (₹{gravity:,.0f})")
                    elif gravity < spot_price:
                        scores['liquidity'] = -30  # Price likely to move down
                        bearish_reasons.append(f"Liquidity center below price (₹{gravity:,.0f})")

            except Exception as e:
                logger.warning(f"Liquidity analysis failed: {e}")

        # 6. Institutional Flow
        if self.institutional_detector:
            try:
                inst_result = self.institutional_detector.detect(df)

                if hasattr(inst_result, 'dominant_participant'):
                    participant = str(inst_result.dominant_participant)
                    if 'Institutional' in participant:
                        # Follow institutional money
                        if hasattr(inst_result, 'institutional_bias'):
                            if inst_result.institutional_bias == 'Bullish':
                                scores['institutional'] = 40
                                bullish_reasons.append("Institutional buying detected")
                            elif inst_result.institutional_bias == 'Bearish':
                                scores['institutional'] = -40
                                bearish_reasons.append("Institutional selling detected")

            except Exception as e:
                logger.warning(f"Institutional detection failed: {e}")

        # 7. XGBoost Prediction (if model is trained)
        if self.xgboost_analyzer and hasattr(self.xgboost_analyzer, 'is_trained') and self.xgboost_analyzer.is_trained:
            try:
                xgb_result = self.xgboost_analyzer.predict(df, bias_results, option_chain)

                if hasattr(xgb_result, 'prediction'):
                    if xgb_result.prediction == 'BUY':
                        scores['xgboost'] = xgb_result.confidence
                        bullish_reasons.append(f"XGBoost: BUY ({xgb_result.confidence:.0f}%)")
                    elif xgb_result.prediction == 'SELL':
                        scores['xgboost'] = -xgb_result.confidence
                        bearish_reasons.append(f"XGBoost: SELL ({xgb_result.confidence:.0f}%)")

            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")

        # Calculate weighted final score
        weights = {
            'regime': 0.25,      # 25%
            'xgboost': 0.20,     # 20% (if trained)
            'volatility': 0.10,  # 10%
            'oi_trap': 0.15,     # 15%
            'cvd': 0.15,         # 15%
            'liquidity': 0.10,   # 10%
            'institutional': 0.05  # 5%
        }

        # Adjust weights if XGBoost not trained
        if not (self.xgboost_analyzer and getattr(self.xgboost_analyzer, 'is_trained', False)):
            weights['xgboost'] = 0
            weights['regime'] = 0.35
            weights['oi_trap'] = 0.20
            weights['cvd'] = 0.20

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate final score
        final_score = sum(scores[k] * weights[k] for k in scores.keys())

        # Determine signal
        if final_score >= 50:
            signal = "STRONG BUY"
        elif final_score >= 25:
            signal = "BUY"
        elif final_score <= -50:
            signal = "STRONG SELL"
        elif final_score <= -25:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Calculate confidence
        confidence = min(abs(final_score) + 50, 100)

        # Determine risk level based on volatility
        if scores['volatility'] >= 70:
            risk_level = "LOW"
            position_multiplier = 1.5
        elif scores['volatility'] >= 40:
            risk_level = "MEDIUM"
            position_multiplier = 1.0
        elif scores['volatility'] >= 20:
            risk_level = "HIGH"
            position_multiplier = 0.5
        else:
            risk_level = "EXTREME"
            position_multiplier = 0.25

        # Calculate entry/exit levels
        current_price = spot_price or (df['Close'].iloc[-1] if 'Close' in df.columns else df['close'].iloc[-1])
        # Handle both 'High'/'high' and 'Low'/'low' column names
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
        else:
            high_col = 'High' if 'High' in df.columns else 'high' if 'high' in df.columns else None
            low_col = 'Low' if 'Low' in df.columns else 'low' if 'low' in df.columns else None
            if high_col and low_col:
                atr = df[high_col].iloc[-1] - df[low_col].iloc[-1]
            else:
                atr = 50  # Default ATR if columns missing

        if 'BUY' in signal:
            entry_zone = (current_price - atr * 0.5, current_price)
            stop_loss = current_price - atr * 2
            targets = [
                current_price + atr * 1.5,
                current_price + atr * 3,
                current_price + atr * 5
            ]
        elif 'SELL' in signal:
            entry_zone = (current_price, current_price + atr * 0.5)
            stop_loss = current_price + atr * 2
            targets = [
                current_price - atr * 1.5,
                current_price - atr * 3,
                current_price - atr * 5
            ]
        else:
            entry_zone = (current_price - atr, current_price + atr)
            stop_loss = 0
            targets = []

        # Get expiry data from session state if available
        expiry_data = {}
        days_to_expiry = 7.0
        expiry_spike_prob = 0.0
        expiry_spike_type = "None"
        expiry_warning = "Normal"

        try:
            if 'expiry_spike_data' in st.session_state:
                expiry_data = st.session_state.expiry_spike_data
                # Ensure expiry_data is a dict before calling .get()
                if expiry_data and isinstance(expiry_data, dict):
                    days_to_expiry = expiry_data.get('days_to_expiry', 7.0)
                    expiry_spike_prob = expiry_data.get('probability', 0.0)
                    expiry_spike_type = expiry_data.get('type', 'None')

                if expiry_spike_prob > 60:
                    expiry_warning = f"⚠️ HIGH RISK ({expiry_spike_prob:.0f}%)"
                    bearish_reasons.append(f"Expiry spike risk: {expiry_spike_prob:.0f}%")
                elif expiry_spike_prob > 40:
                    expiry_warning = f"MODERATE ({expiry_spike_prob:.0f}%)"
                elif days_to_expiry <= 2:
                    expiry_warning = f"EXPIRY DAY ({days_to_expiry:.1f}d)"
        except:
            pass

        # Expiry score (higher = riskier)
        scores['expiry'] = min(expiry_spike_prob, 100)

        # Get ATM option LTP from session state
        atm_strike_val = 0.0
        atm_call_ltp = 0.0
        atm_put_ltp = 0.0
        try:
            if 'merged_df' in st.session_state and st.session_state.merged_df is not None:
                merged_df = st.session_state.merged_df
                if 'atm_strike' in st.session_state:
                    atm_strike_val = st.session_state.atm_strike
                elif spot_price:
                    # Calculate ATM strike from spot price
                    if 'strikePrice' in merged_df.columns:
                        atm_strike_val = min(merged_df['strikePrice'].tolist(), key=lambda x: abs(x - spot_price))

                if atm_strike_val > 0:
                    atm_row = merged_df[merged_df['strikePrice'] == atm_strike_val]
                    if not atm_row.empty:
                        atm_call_ltp = float(atm_row['LTP_CE'].iloc[0]) if 'LTP_CE' in atm_row.columns else 0.0
                        atm_put_ltp = float(atm_row['LTP_PE'].iloc[0]) if 'LTP_PE' in atm_row.columns else 0.0
        except Exception as e:
            logger.debug(f"Could not fetch ATM LTP: {e}")

        return UnifiedSignal(
            signal=signal,
            confidence=confidence,
            regime_score=scores['regime'],
            xgboost_score=scores['xgboost'],
            volatility_score=scores['volatility'],
            oi_trap_score=scores['oi_trap'],
            cvd_score=scores['cvd'],
            liquidity_score=scores['liquidity'],
            expiry_score=scores['expiry'],
            regime=regime_name,
            volatility_state=volatility_state,
            trap_warning=trap_warning,
            recommended_strategy=strategy,
            days_to_expiry=days_to_expiry,
            expiry_spike_probability=expiry_spike_prob,
            expiry_spike_type=expiry_spike_type,
            expiry_warning=expiry_warning,
            risk_level=risk_level,
            position_size_multiplier=position_multiplier,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            bullish_reasons=bullish_reasons,
            bearish_reasons=bearish_reasons,
            atm_strike=atm_strike_val,
            atm_call_ltp=atm_call_ltp,
            atm_put_ltp=atm_put_ltp,
            timestamp=datetime.now()
        )


def render_unified_signal(signal: UnifiedSignal, spot_price: float = None):
    """
    Render the unified signal in Streamlit
    """
    # Signal color
    signal_colors = {
        'STRONG BUY': '#00FF00',
        'BUY': '#90EE90',
        'HOLD': '#FFD700',
        'SELL': '#FFA500',
        'STRONG SELL': '#FF0000'
    }
    color = signal_colors.get(signal.signal, '#808080')

    # Main signal box
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}22, {color}44);
                border: 3px solid {color};
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                margin-bottom: 20px;">
        <h1 style="color: {color}; margin: 0; font-size: 2.5rem;">🎯 {signal.signal}</h1>
        <p style="color: #AAA; margin: 10px 0 0 0; font-size: 1.2rem;">
            Confidence: {signal.confidence:.0f}% | Risk: {signal.risk_level}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show ATM Option Recommendation based on signal
    if signal.signal in ['STRONG BUY', 'BUY'] and signal.atm_call_ltp > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #00FF0022, #00FF0044);
                    border: 2px solid #00FF00;
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    margin-bottom: 15px;">
            <h3 style="color: #00FF00; margin: 0;">📈 BUY ATM CALL @ ₹{signal.atm_strike:,.0f}</h3>
            <p style="color: #90EE90; margin: 5px 0 0 0; font-size: 1.5rem; font-weight: bold;">
                LTP: ₹{signal.atm_call_ltp:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif signal.signal in ['STRONG SELL', 'SELL'] and signal.atm_put_ltp > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF000022, #FF000044);
                    border: 2px solid #FF0000;
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    margin-bottom: 15px;">
            <h3 style="color: #FF6347; margin: 0;">📉 BUY ATM PUT @ ₹{signal.atm_strike:,.0f}</h3>
            <p style="color: #FFA07A; margin: 5px 0 0 0; font-size: 1.5rem; font-weight: bold;">
                LTP: ₹{signal.atm_put_ltp:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics row 1
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Regime", signal.regime, f"{signal.regime_score:+.0f}")

    with col2:
        st.metric("Volatility", signal.volatility_state, f"{signal.volatility_score:.0f}%")

    with col3:
        st.metric("OI Trap", signal.trap_warning[:12], f"{signal.oi_trap_score:+.0f}")

    with col4:
        st.metric("CVD", f"{signal.cvd_score:+.0f}", "Bull" if signal.cvd_score > 0 else "Bear" if signal.cvd_score < 0 else "-")

    with col5:
        st.metric("Expiry", f"{signal.days_to_expiry:.1f}d", signal.expiry_warning[:10])

    with col6:
        st.metric("Position", f"{signal.position_size_multiplier:.1f}x", signal.risk_level)

    st.divider()

    # Strategy recommendation
    st.info(f"📊 **Strategy:** {signal.recommended_strategy}")

    # Entry/Exit levels
    if signal.signal != "HOLD" and spot_price:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Entry Zone")
            st.write(f"**Entry:** ₹{signal.entry_zone[0]:,.0f} - ₹{signal.entry_zone[1]:,.0f}")
            st.write(f"**Stop Loss:** ₹{signal.stop_loss:,.0f}")

        with col2:
            st.markdown("### 🎯 Targets")
            for i, target in enumerate(signal.targets, 1):
                st.write(f"**T{i}:** ₹{target:,.0f}")

    # Reasons
    col1, col2 = st.columns(2)

    with col1:
        if signal.bullish_reasons:
            st.markdown("### 🐂 Bullish Signals")
            for reason in signal.bullish_reasons:
                st.success(f"✅ {reason}")

    with col2:
        if signal.bearish_reasons:
            st.markdown("### 🐻 Bearish Signals")
            for reason in signal.bearish_reasons:
                st.error(f"⚠️ {reason}")

    # Component scores chart
    with st.expander("📊 Component Scores Breakdown"):
        scores_df = pd.DataFrame({
            'Component': ['Regime', 'XGBoost', 'Volatility', 'OI Trap', 'CVD', 'Liquidity'],
            'Score': [
                signal.regime_score,
                signal.xgboost_score,
                signal.volatility_score - 50,  # Normalize to -50 to +50
                signal.oi_trap_score,
                signal.cvd_score,
                signal.liquidity_score
            ]
        })

        import plotly.express as px
        fig = px.bar(
            scores_df,
            x='Component',
            y='Score',
            color='Score',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[-100, 100]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def get_cached_unified_signal() -> Optional[UnifiedSignal]:
    """Get cached unified signal from session state"""
    return st.session_state.get('unified_ml_signal')
