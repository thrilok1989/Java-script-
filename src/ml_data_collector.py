"""
ML Data Collector - Real Trade Signal Recording
================================================
Collects REAL signals and outcomes for ML training

This module:
1. Records every signal generated with full feature set
2. Tracks price movement after signal (5min, 15min, 30min, 1hr)
3. Labels signals as WIN/LOSS based on actual outcome
4. Stores data for ML model training

Data stored:
- All features at signal time
- Entry price, exit prices at intervals
- P&L outcome (WIN/LOSS/BREAKEVEN)
- Market conditions
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Data storage path
DATA_DIR = "ml_training_data"
SIGNALS_FILE = "recorded_signals.json"
OUTCOMES_FILE = "signal_outcomes.json"


@dataclass
class RecordedSignal:
    """A recorded trading signal with all features"""
    signal_id: str
    timestamp: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    spot_price: float
    atm_strike: float

    # Core features
    regime_score: float
    volatility_score: float
    oi_trap_score: float
    cvd_score: float
    liquidity_score: float
    institutional_score: float
    spike_score: float

    # Diamond features
    cvd_diamond_score: float
    gamma_flip_score: float
    black_order_score: float
    block_trade_score: float

    # Option chain features
    pcr: float
    total_ce_oi: int
    total_pe_oi: int
    atm_iv_ce: float
    atm_iv_pe: float
    max_pain: float

    # Technical features
    rsi: float
    macd_signal: float
    vwap_position: float
    atr: float

    # Market state
    vix: float
    is_expiry_day: bool
    time_to_expiry_days: float

    # Outcome tracking (filled later)
    entry_price: float = 0.0
    price_5min: float = 0.0
    price_15min: float = 0.0
    price_30min: float = 0.0
    price_1hr: float = 0.0
    outcome: str = "PENDING"  # WIN, LOSS, BREAKEVEN, PENDING
    pnl_percent: float = 0.0
    outcome_recorded: bool = False


class MLDataCollector:
    """
    Collects real trading signals and outcomes for ML training
    """

    def __init__(self):
        self.data_dir = DATA_DIR
        self.signals_file = os.path.join(DATA_DIR, SIGNALS_FILE)
        self.outcomes_file = os.path.join(DATA_DIR, OUTCOMES_FILE)
        self.pending_signals: List[RecordedSignal] = []

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data
        self.recorded_signals = self._load_signals()

    def _load_signals(self) -> List[Dict]:
        """Load previously recorded signals"""
        try:
            if os.path.exists(self.signals_file):
                with open(self.signals_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load signals: {e}")
        return []

    def _save_signals(self):
        """Save signals to file"""
        try:
            with open(self.signals_file, 'w') as f:
                json.dump(self.recorded_signals, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save signals: {e}")

    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

    def extract_features_from_session(self) -> Dict[str, Any]:
        """
        Extract all current features from session state
        """
        features = {}

        # Core ML scores
        features['regime_score'] = 0
        features['volatility_score'] = 50
        features['oi_trap_score'] = 0
        features['cvd_score'] = 0
        features['liquidity_score'] = 0
        features['institutional_score'] = 0
        features['spike_score'] = 0

        # Get ML regime result
        regime_result = st.session_state.get('ml_regime_result', {})
        if regime_result:
            features['regime_score'] = regime_result.get('trend_strength', 0)
            if 'BULLISH' in str(regime_result.get('regime', '')).upper():
                features['regime_score'] = abs(features['regime_score'])
            elif 'BEARISH' in str(regime_result.get('regime', '')).upper():
                features['regime_score'] = -abs(features['regime_score'])

        # Diamond features
        cvd_result = st.session_state.get('cvd_diamond_result', {})
        features['cvd_diamond_score'] = cvd_result.get('score', 50)

        gamma_result = st.session_state.get('gamma_flip_result', {})
        features['gamma_flip_score'] = gamma_result.get('score', 50)

        black_result = st.session_state.get('black_order_result', {})
        features['black_order_score'] = black_result.get('score', 50)

        block_result = st.session_state.get('block_trade_result', {})
        features['block_trade_score'] = block_result.get('score', 50)

        # Option chain features
        merged_df = st.session_state.get('merged_df')
        if merged_df is not None and len(merged_df) > 0:
            features['total_ce_oi'] = int(merged_df.get('OI_CE', pd.Series([0])).sum())
            features['total_pe_oi'] = int(merged_df.get('OI_PE', pd.Series([0])).sum())
            features['pcr'] = features['total_pe_oi'] / max(features['total_ce_oi'], 1)
            features['atm_iv_ce'] = float(merged_df.get('IV_CE', pd.Series([0.2])).median())
            features['atm_iv_pe'] = float(merged_df.get('IV_PE', pd.Series([0.2])).median())
        else:
            features['total_ce_oi'] = 0
            features['total_pe_oi'] = 0
            features['pcr'] = 1.0
            features['atm_iv_ce'] = 0.2
            features['atm_iv_pe'] = 0.2

        # Technical features from chart data
        chart_data = st.session_state.get('chart_data')
        if chart_data is not None and len(chart_data) > 0:
            close = chart_data['Close'].iloc[-1] if 'Close' in chart_data.columns else chart_data.get('close', pd.Series([0])).iloc[-1]

            # RSI calculation
            if len(chart_data) >= 14:
                delta = chart_data['Close'].diff() if 'Close' in chart_data.columns else chart_data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.0001)
                features['rsi'] = float(100 - (100 / (1 + rs.iloc[-1]))) if not pd.isna(rs.iloc[-1]) else 50
            else:
                features['rsi'] = 50

            # ATR
            if 'High' in chart_data.columns and 'Low' in chart_data.columns:
                features['atr'] = float((chart_data['High'] - chart_data['Low']).rolling(14).mean().iloc[-1])
            else:
                features['atr'] = 50

            # VWAP position
            if 'Volume' in chart_data.columns:
                vwap = (chart_data['Close'] * chart_data['Volume']).cumsum() / chart_data['Volume'].cumsum()
                features['vwap_position'] = float((close - vwap.iloc[-1]) / vwap.iloc[-1] * 100) if vwap.iloc[-1] > 0 else 0
            else:
                features['vwap_position'] = 0

            features['macd_signal'] = 0  # Simplified
        else:
            features['rsi'] = 50
            features['atr'] = 50
            features['vwap_position'] = 0
            features['macd_signal'] = 0

        # Market state
        features['vix'] = st.session_state.get('vix_current', 15.0)
        features['spot_price'] = st.session_state.get('nifty_spot', 0)

        # Expiry info
        now = datetime.now()
        days_to_thursday = (3 - now.weekday()) % 7
        if days_to_thursday == 0 and now.hour >= 15:
            days_to_thursday = 7
        features['is_expiry_day'] = days_to_thursday == 0
        features['time_to_expiry_days'] = max(0.1, days_to_thursday)

        # Max pain (simplified)
        features['max_pain'] = features['spot_price']  # Would need proper calculation

        return features

    def record_signal(self, signal_type: str, confidence: float,
                     atm_strike: float = 0, entry_price: float = 0) -> str:
        """
        Record a new trading signal with all features

        Args:
            signal_type: BUY, SELL, or HOLD
            confidence: 0-100 confidence score
            atm_strike: ATM strike price
            entry_price: Entry price (spot or option)

        Returns:
            signal_id: Unique ID for tracking
        """
        try:
            features = self.extract_features_from_session()

            signal_id = self._generate_signal_id()

            recorded = RecordedSignal(
                signal_id=signal_id,
                timestamp=datetime.now().isoformat(),
                signal_type=signal_type,
                confidence=confidence,
                spot_price=features.get('spot_price', 0),
                atm_strike=atm_strike,

                # Core features
                regime_score=features.get('regime_score', 0),
                volatility_score=features.get('volatility_score', 50),
                oi_trap_score=features.get('oi_trap_score', 0),
                cvd_score=features.get('cvd_score', 0),
                liquidity_score=features.get('liquidity_score', 0),
                institutional_score=features.get('institutional_score', 0),
                spike_score=features.get('spike_score', 0),

                # Diamond features
                cvd_diamond_score=features.get('cvd_diamond_score', 50),
                gamma_flip_score=features.get('gamma_flip_score', 50),
                black_order_score=features.get('black_order_score', 50),
                block_trade_score=features.get('block_trade_score', 50),

                # Option chain
                pcr=features.get('pcr', 1.0),
                total_ce_oi=features.get('total_ce_oi', 0),
                total_pe_oi=features.get('total_pe_oi', 0),
                atm_iv_ce=features.get('atm_iv_ce', 0.2),
                atm_iv_pe=features.get('atm_iv_pe', 0.2),
                max_pain=features.get('max_pain', 0),

                # Technical
                rsi=features.get('rsi', 50),
                macd_signal=features.get('macd_signal', 0),
                vwap_position=features.get('vwap_position', 0),
                atr=features.get('atr', 50),

                # Market state
                vix=features.get('vix', 15),
                is_expiry_day=features.get('is_expiry_day', False),
                time_to_expiry_days=features.get('time_to_expiry_days', 7),

                # Entry
                entry_price=entry_price if entry_price > 0 else features.get('spot_price', 0)
            )

            # Add to pending for outcome tracking
            self.pending_signals.append(recorded)

            # Save to recorded signals
            self.recorded_signals.append(asdict(recorded))
            self._save_signals()

            logger.info(f"Recorded signal {signal_id}: {signal_type} @ {recorded.entry_price}")

            return signal_id

        except Exception as e:
            logger.error(f"Failed to record signal: {e}")
            return ""

    def update_outcome(self, signal_id: str, current_price: float,
                      time_elapsed_minutes: int) -> bool:
        """
        Update signal outcome based on current price

        Args:
            signal_id: Signal to update
            current_price: Current market price
            time_elapsed_minutes: Minutes since signal

        Returns:
            True if updated successfully
        """
        try:
            for i, signal in enumerate(self.recorded_signals):
                if signal.get('signal_id') == signal_id:
                    entry = signal.get('entry_price', 0)
                    if entry <= 0:
                        continue

                    # Update price at interval
                    if time_elapsed_minutes <= 5:
                        signal['price_5min'] = current_price
                    elif time_elapsed_minutes <= 15:
                        signal['price_15min'] = current_price
                    elif time_elapsed_minutes <= 30:
                        signal['price_30min'] = current_price
                    elif time_elapsed_minutes <= 60:
                        signal['price_1hr'] = current_price

                        # Calculate final outcome at 1hr
                        pnl_percent = (current_price - entry) / entry * 100

                        if signal['signal_type'] == 'SELL':
                            pnl_percent = -pnl_percent  # Invert for SELL

                        signal['pnl_percent'] = pnl_percent

                        # Determine outcome
                        if pnl_percent > 0.3:  # >0.3% profit
                            signal['outcome'] = 'WIN'
                        elif pnl_percent < -0.3:  # >0.3% loss
                            signal['outcome'] = 'LOSS'
                        else:
                            signal['outcome'] = 'BREAKEVEN'

                        signal['outcome_recorded'] = True

                    self.recorded_signals[i] = signal
                    self._save_signals()
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            return False

    def get_training_data(self) -> pd.DataFrame:
        """
        Get recorded signals with outcomes as training DataFrame

        Returns:
            DataFrame with features and labels
        """
        # Filter only signals with recorded outcomes
        completed = [s for s in self.recorded_signals if s.get('outcome_recorded', False)]

        if not completed:
            logger.warning("No completed signals for training")
            return pd.DataFrame()

        df = pd.DataFrame(completed)

        # Create label column
        df['label'] = df['outcome'].map({'WIN': 1, 'LOSS': 0, 'BREAKEVEN': 2})

        return df

    def get_statistics(self) -> Dict:
        """Get statistics about recorded signals"""
        total = len(self.recorded_signals)
        completed = [s for s in self.recorded_signals if s.get('outcome_recorded', False)]

        wins = sum(1 for s in completed if s.get('outcome') == 'WIN')
        losses = sum(1 for s in completed if s.get('outcome') == 'LOSS')
        breakeven = sum(1 for s in completed if s.get('outcome') == 'BREAKEVEN')

        return {
            'total_signals': total,
            'completed_signals': len(completed),
            'pending_signals': total - len(completed),
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'win_rate': wins / max(len(completed), 1) * 100,
            'avg_pnl': np.mean([s.get('pnl_percent', 0) for s in completed]) if completed else 0
        }


# Singleton instance
_data_collector = None

def get_data_collector() -> MLDataCollector:
    """Get singleton data collector instance"""
    global _data_collector
    if _data_collector is None:
        _data_collector = MLDataCollector()
    return _data_collector


def record_trading_signal(signal_type: str, confidence: float,
                         atm_strike: float = 0, entry_price: float = 0) -> str:
    """
    Convenience function to record a signal

    Usage:
        from src.ml_data_collector import record_trading_signal

        signal_id = record_trading_signal('BUY', 75.5, atm_strike=24000)
    """
    collector = get_data_collector()
    return collector.record_signal(signal_type, confidence, atm_strike, entry_price)
