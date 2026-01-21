"""
Smart Alert System - Platinum Level Feature
============================================
Intelligent alert system that monitors market conditions and sends Telegram notifications

Alert Types:
1. Price Alerts - When price hits S/R levels
2. OI Alerts - Large OI buildups or breakdowns
3. Gamma Flip Alert - GEX regime change
4. CVD Divergence Alert - Hidden buying/selling
5. Block Trade Alert - Institutional activity detected
6. SL Hunt Alert - Stop loss hunt detected
7. Expiry Alert - High probability expiry moves
8. FII/DII Alert - Institutional position changes

Features:
- Customizable alert thresholds
- Duplicate alert prevention
- Priority-based notifications
- Alert history tracking
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import os
import logging

logger = logging.getLogger(__name__)

# Alert storage
ALERTS_DIR = "alerts"
ALERTS_FILE = os.path.join(ALERTS_DIR, "alert_history.json")
ALERT_CONFIG_FILE = os.path.join(ALERTS_DIR, "alert_config.json")


@dataclass
class Alert:
    """Single alert"""
    alert_id: str
    alert_type: str
    priority: str  # HIGH, MEDIUM, LOW
    title: str
    message: str
    value: float
    threshold: float
    direction: str  # BULLISH, BEARISH, NEUTRAL
    timestamp: str
    sent: bool = False
    acknowledged: bool = False


@dataclass
class AlertConfig:
    """Alert configuration"""
    # Price alerts
    price_alert_enabled: bool = True
    price_support_threshold: float = 0.3  # % distance from support
    price_resistance_threshold: float = 0.3  # % distance from resistance

    # OI alerts
    oi_alert_enabled: bool = True
    oi_change_threshold: int = 100000  # Minimum OI change to alert

    # Gamma alerts
    gamma_flip_enabled: bool = True
    gex_flip_alert: bool = True

    # CVD alerts
    cvd_alert_enabled: bool = True
    cvd_divergence_threshold: float = 30  # Score threshold

    # Block trade alerts
    block_trade_enabled: bool = True
    block_size_threshold: int = 50000  # Minimum block size

    # SL Hunt alerts
    sl_hunt_enabled: bool = True
    sl_hunt_probability_threshold: float = 60  # Minimum probability

    # Expiry alerts
    expiry_alert_enabled: bool = True
    expiry_spike_threshold: float = 50  # Minimum probability

    # FII/DII alerts
    fii_dii_enabled: bool = True
    fii_change_threshold: int = 50000  # Net change threshold

    # Cooldown (prevent duplicate alerts)
    cooldown_minutes: int = 15  # Minutes between same type alerts


class SmartAlertSystem:
    """
    Smart Alert System with Telegram integration
    """

    def __init__(self):
        os.makedirs(ALERTS_DIR, exist_ok=True)
        self.config = self._load_config()
        self.alert_history = self._load_history()
        self.last_alert_times: Dict[str, datetime] = {}

    def _load_config(self) -> AlertConfig:
        """Load alert configuration"""
        try:
            if os.path.exists(ALERT_CONFIG_FILE):
                with open(ALERT_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return AlertConfig(**data)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        return AlertConfig()

    def _save_config(self):
        """Save alert configuration"""
        try:
            with open(ALERT_CONFIG_FILE, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save config: {e}")

    def _load_history(self) -> List[Dict]:
        """Load alert history"""
        try:
            if os.path.exists(ALERTS_FILE):
                with open(ALERTS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
        return []

    def _save_history(self):
        """Save alert history"""
        try:
            # Keep only last 100 alerts
            self.alert_history = self.alert_history[-100:]
            with open(ALERTS_FILE, 'w') as f:
                json.dump(self.alert_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save history: {e}")

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import random
        return f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if we can send this type of alert (cooldown)"""
        if alert_type in self.last_alert_times:
            last_time = self.last_alert_times[alert_type]
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            if datetime.now() - last_time < cooldown:
                return False
        return True

    def _record_alert_sent(self, alert_type: str):
        """Record that alert was sent"""
        self.last_alert_times[alert_type] = datetime.now()

    def send_telegram(self, message: str, priority: str = "MEDIUM") -> Tuple[bool, str]:
        """
        Send alert to Telegram

        Args:
            message: Alert message
            priority: HIGH, MEDIUM, LOW

        Returns:
            (success, response_message)
        """
        try:
            # Get Telegram credentials
            try:
                bot_token = st.secrets["TELEGRAM"]["BOT_TOKEN"]
                chat_id = st.secrets["TELEGRAM"]["CHAT_ID"]
            except:
                bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
                chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

            if not bot_token or not chat_id:
                return False, "Telegram not configured"

            # Add priority emoji
            if priority == "HIGH":
                message = f"🚨🚨 *HIGH PRIORITY* 🚨🚨\n\n{message}"
            elif priority == "MEDIUM":
                message = f"⚠️ *ALERT* ⚠️\n\n{message}"
            else:
                message = f"ℹ️ *INFO*\n\n{message}"

            # Send message
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                return True, "Sent successfully"
            else:
                return False, f"Telegram API error: {response.status_code}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def check_price_alerts(self) -> List[Alert]:
        """Check for price-based alerts"""
        alerts = []

        if not self.config.price_alert_enabled:
            return alerts

        try:
            spot_price = st.session_state.get('nifty_spot', 0)
            if spot_price <= 0:
                return alerts

            # Get S/R levels from session state
            sr_levels = st.session_state.get('sr_levels', {})
            support = sr_levels.get('support', 0)
            resistance = sr_levels.get('resistance', 0)

            # Check support proximity
            if support > 0:
                distance_pct = (spot_price - support) / spot_price * 100
                if 0 < distance_pct < self.config.price_support_threshold:
                    if self._can_send_alert('price_support'):
                        alerts.append(Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type='PRICE_SUPPORT',
                            priority='HIGH',
                            title='Price Near Support',
                            message=f"NIFTY at ₹{spot_price:,.0f} is {distance_pct:.2f}% from support ₹{support:,.0f}",
                            value=spot_price,
                            threshold=support,
                            direction='BULLISH',
                            timestamp=datetime.now().isoformat()
                        ))

            # Check resistance proximity
            if resistance > 0:
                distance_pct = (resistance - spot_price) / spot_price * 100
                if 0 < distance_pct < self.config.price_resistance_threshold:
                    if self._can_send_alert('price_resistance'):
                        alerts.append(Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type='PRICE_RESISTANCE',
                            priority='HIGH',
                            title='Price Near Resistance',
                            message=f"NIFTY at ₹{spot_price:,.0f} is {distance_pct:.2f}% from resistance ₹{resistance:,.0f}",
                            value=spot_price,
                            threshold=resistance,
                            direction='BEARISH',
                            timestamp=datetime.now().isoformat()
                        ))

        except Exception as e:
            logger.warning(f"Price alert check failed: {e}")

        return alerts

    def check_gamma_flip_alerts(self) -> List[Alert]:
        """Check for Gamma/GEX flip alerts"""
        alerts = []

        if not self.config.gamma_flip_enabled:
            return alerts

        try:
            gamma_result = st.session_state.get('gamma_flip_result', {})

            if gamma_result:
                flip_signal = gamma_result.get('flip_signal')
                if flip_signal and self._can_send_alert('gamma_flip'):
                    gex_regime = gamma_result.get('gex_regime', 'UNKNOWN')
                    vol_expectation = gamma_result.get('volatility_expectation', 'UNKNOWN')

                    direction = 'BEARISH' if 'NEGATIVE' in str(flip_signal) else 'BULLISH'

                    alerts.append(Alert(
                        alert_id=self._generate_alert_id(),
                        alert_type='GAMMA_FLIP',
                        priority='HIGH',
                        title='🔄 GAMMA FLIP DETECTED!',
                        message=f"GEX Regime: {gex_regime}\nFlip Type: {flip_signal}\nVolatility: {vol_expectation}\n\n⚠️ VOLATILITY REGIME CHANGE - Expect trending moves!",
                        value=gamma_result.get('score', 0),
                        threshold=0,
                        direction=direction,
                        timestamp=datetime.now().isoformat()
                    ))

        except Exception as e:
            logger.warning(f"Gamma flip alert check failed: {e}")

        return alerts

    def check_cvd_alerts(self) -> List[Alert]:
        """Check for CVD divergence alerts"""
        alerts = []

        if not self.config.cvd_alert_enabled:
            return alerts

        try:
            cvd_result = st.session_state.get('cvd_diamond_result', {})

            if cvd_result:
                score = cvd_result.get('score', 50)
                institutional_activity = cvd_result.get('institutional_activity', 'NONE')
                smart_money_direction = cvd_result.get('smart_money_direction', 'UNKNOWN')

                # Alert on significant divergence
                if institutional_activity in ['ACCUMULATION', 'DISTRIBUTION']:
                    if self._can_send_alert('cvd_divergence'):
                        direction = 'BULLISH' if institutional_activity == 'ACCUMULATION' else 'BEARISH'

                        # Cross-reference with option block activity if available
                        block_result = st.session_state.get('block_trade_result', {})
                        option_context = ""
                        if block_result:
                            call_blocks = block_result.get('call_blocks', 0)
                            put_blocks = block_result.get('put_blocks', 0)
                            if call_blocks > 0 or put_blocks > 0:
                                # Check if options confirm or contradict CVD
                                if direction == 'BULLISH' and call_blocks > put_blocks:
                                    option_context = f"\n✓ Confirmed by CALL blocks ({call_blocks})"
                                elif direction == 'BULLISH' and put_blocks > call_blocks:
                                    option_context = f"\n⚠ Contradicted by PUT blocks ({put_blocks})"
                                elif direction == 'BEARISH' and put_blocks > call_blocks:
                                    option_context = f"\n✓ Confirmed by PUT blocks ({put_blocks})"
                                elif direction == 'BEARISH' and call_blocks > put_blocks:
                                    option_context = f"\n⚠ Contradicted by CALL blocks ({call_blocks})"

                        alerts.append(Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type='CVD_DIVERGENCE',
                            priority='MEDIUM',
                            title=f'💎 CVD: {institutional_activity}',
                            message=f"Institutional {institutional_activity} detected!\nCVD Score: {score:.0f}\nSmart Money: {smart_money_direction}{option_context}",
                            value=score,
                            threshold=self.config.cvd_divergence_threshold,
                            direction=direction,
                            timestamp=datetime.now().isoformat()
                        ))

        except Exception as e:
            logger.warning(f"CVD alert check failed: {e}")

        return alerts

    def check_block_trade_alerts(self) -> List[Alert]:
        """Check for block trade alerts"""
        alerts = []

        if not self.config.block_trade_enabled:
            return alerts

        try:
            block_result = st.session_state.get('block_trade_result', {})

            if block_result:
                activity_level = block_result.get('activity_level', 'NONE')
                institutional_bias = block_result.get('institutional_bias', 'NEUTRAL')
                call_blocks = block_result.get('call_blocks', 0)
                put_blocks = block_result.get('put_blocks', 0)
                bullish_blocks = block_result.get('bullish_blocks', 0)
                bearish_blocks = block_result.get('bearish_blocks', 0)

                if activity_level in ['HEAVY', 'MODERATE'] and self._can_send_alert('block_trade'):
                    # Determine direction based on CALL vs PUT accumulation
                    # CALL accumulation = BULLISH, PUT accumulation = BEARISH
                    if call_blocks > put_blocks:
                        direction = 'BULLISH'
                        dominant_type = 'CALL'
                    elif put_blocks > call_blocks:
                        direction = 'BEARISH'
                        dominant_type = 'PUT'
                    else:
                        # If equal or no option type data, use institutional bias
                        direction = 'BULLISH' if institutional_bias == 'ACCUMULATING' else 'BEARISH'
                        dominant_type = 'MIXED'

                    # Build message with CALL/PUT breakdown
                    if call_blocks > 0 or put_blocks > 0:
                        option_info = f"\nCALL Blocks: {call_blocks}\nPUT Blocks: {put_blocks}"
                    else:
                        option_info = f"\nBullish Blocks: {bullish_blocks}\nBearish Blocks: {bearish_blocks}"

                    alerts.append(Alert(
                        alert_id=self._generate_alert_id(),
                        alert_type='BLOCK_TRADE',
                        priority='MEDIUM',
                        title=f'📦 Block Trade: {activity_level}',
                        message=f"Institutional {institutional_bias}{option_info}\nTotal Volume: {block_result.get('total_volume', 0):,}",
                        value=block_result.get('score', 50),
                        threshold=50,
                        direction=direction,
                        timestamp=datetime.now().isoformat()
                    ))

        except Exception as e:
            logger.warning(f"Block trade alert check failed: {e}")

        return alerts

    def check_sl_hunt_alerts(self) -> List[Alert]:
        """Check for SL hunt alerts"""
        alerts = []

        if not self.config.sl_hunt_enabled:
            return alerts

        try:
            sl_result = st.session_state.get('sl_hunt_result', {})

            if sl_result:
                hunt_probability = sl_result.get('hunt_probability', 0)

                if hunt_probability >= self.config.sl_hunt_probability_threshold:
                    if self._can_send_alert('sl_hunt'):
                        trap_direction = sl_result.get('trap_direction', 'UNKNOWN')
                        direction = 'BEARISH' if trap_direction == 'BULL_TRAP' else 'BULLISH'

                        alerts.append(Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type='SL_HUNT',
                            priority='HIGH',
                            title=f'🎯 SL HUNT DETECTED!',
                            message=f"Hunt Probability: {hunt_probability:.0f}%\nTrap Type: {trap_direction}\n\n⚠️ Stop losses may be targeted!",
                            value=hunt_probability,
                            threshold=self.config.sl_hunt_probability_threshold,
                            direction=direction,
                            timestamp=datetime.now().isoformat()
                        ))

        except Exception as e:
            logger.warning(f"SL hunt alert check failed: {e}")

        return alerts

    def check_expiry_alerts(self) -> List[Alert]:
        """Check for expiry day alerts"""
        alerts = []

        if not self.config.expiry_alert_enabled:
            return alerts

        try:
            killer_result = st.session_state.get('last_killer_result', {})

            if killer_result:
                spike_probability = killer_result.get('spike_probability', 0)

                if spike_probability >= self.config.expiry_spike_threshold:
                    if self._can_send_alert('expiry_spike'):
                        breakout_direction = killer_result.get('breakout_direction', 'UNKNOWN')
                        direction = 'BULLISH' if breakout_direction == 'UP' else 'BEARISH'

                        alerts.append(Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type='EXPIRY_SPIKE',
                            priority='HIGH',
                            title=f'⚡ EXPIRY SPIKE ALERT!',
                            message=f"Spike Probability: {spike_probability:.0f}%\nExpected Direction: {breakout_direction}\n\n🗓️ High probability expiry move expected!",
                            value=spike_probability,
                            threshold=self.config.expiry_spike_threshold,
                            direction=direction,
                            timestamp=datetime.now().isoformat()
                        ))

        except Exception as e:
            logger.warning(f"Expiry alert check failed: {e}")

        return alerts

    def check_all_alerts(self) -> List[Alert]:
        """
        Check all alert conditions

        Returns list of triggered alerts
        """
        all_alerts = []

        # Check each alert type
        all_alerts.extend(self.check_price_alerts())
        all_alerts.extend(self.check_gamma_flip_alerts())
        all_alerts.extend(self.check_cvd_alerts())
        all_alerts.extend(self.check_block_trade_alerts())
        all_alerts.extend(self.check_sl_hunt_alerts())
        all_alerts.extend(self.check_expiry_alerts())

        return all_alerts

    def process_alerts(self, send_telegram: bool = True) -> Dict:
        """
        Check all alerts and send notifications

        Args:
            send_telegram: Whether to send Telegram notifications

        Returns:
            Summary of alerts processed
        """
        alerts = self.check_all_alerts()
        sent_count = 0
        failed_count = 0

        # Check if we have both CVD and Block Trade alerts to combine
        cvd_alert = next((a for a in alerts if a.alert_type == 'CVD_DIVERGENCE'), None)
        block_alert = next((a for a in alerts if a.alert_type == 'BLOCK_TRADE'), None)

        # If both CVD and Block Trade alerts exist, combine them
        if cvd_alert and block_alert:
            # Remove individual alerts from list
            alerts = [a for a in alerts if a.alert_type not in ['CVD_DIVERGENCE', 'BLOCK_TRADE']]

            # Determine overall direction (prioritize when both agree)
            if cvd_alert.direction == block_alert.direction:
                combined_direction = cvd_alert.direction
                confidence_indicator = "🔥 STRONG CONFLUENCE"
            else:
                combined_direction = cvd_alert.direction  # CVD takes precedence
                confidence_indicator = "⚠️ MIXED SIGNALS"

            # Build combined message
            combined_message = f"{confidence_indicator}\n\n"
            combined_message += f"💎 *CVD: {cvd_alert.title.split(': ')[1]}*\n"
            combined_message += f"Score: {cvd_alert.value:.0f} | {cvd_alert.message.split('Smart Money: ')[1].split('\\n')[0] if 'Smart Money:' in cvd_alert.message else 'N/A'}\n\n"
            combined_message += f"📦 *Block Trade: {block_alert.title.split(': ')[1]}*\n"

            # Extract block data
            block_msg_lines = block_alert.message.split('\n')
            for line in block_msg_lines:
                if 'CALL Blocks:' in line or 'PUT Blocks:' in line or 'Total Volume:' in line or 'Bullish Blocks:' in line or 'Bearish Blocks:' in line:
                    combined_message += f"{line}\n"

            combined_message += f"\n📊 Direction: *{combined_direction}*"

            # Create combined alert
            combined_alert = Alert(
                alert_id=self._generate_alert_id(),
                alert_type='COMBINED_INSTITUTIONAL',
                priority='MEDIUM',
                title='💎📦 Institutional Activity',
                message=combined_message,
                value=(cvd_alert.value + block_alert.value) / 2,
                threshold=50,
                direction=combined_direction,
                timestamp=datetime.now().isoformat()
            )

            # Add to history
            self.alert_history.append(asdict(combined_alert))

            # Send combined alert
            if send_telegram:
                success, msg = self.send_telegram(
                    combined_message,
                    priority=combined_alert.priority
                )

                if success:
                    combined_alert.sent = True
                    self._record_alert_sent('CVD_DIVERGENCE')
                    self._record_alert_sent('BLOCK_TRADE')
                    sent_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to send combined alert: {msg}")

        # Process remaining alerts individually
        for alert in alerts:
            # Add to history
            self.alert_history.append(asdict(alert))

            # Send Telegram if enabled
            if send_telegram:
                success, msg = self.send_telegram(
                    f"*{alert.title}*\n\n{alert.message}\n\n📊 Direction: {alert.direction}",
                    priority=alert.priority
                )

                if success:
                    alert.sent = True
                    self._record_alert_sent(alert.alert_type)
                    sent_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to send alert: {msg}")

        # Save history
        self._save_history()

        return {
            'total_alerts': len(alerts),
            'sent': sent_count,
            'failed': failed_count,
            'alerts': [asdict(a) for a in alerts]
        }


# Singleton
_alert_system = None


def get_alert_system() -> SmartAlertSystem:
    """Get singleton alert system"""
    global _alert_system
    if _alert_system is None:
        _alert_system = SmartAlertSystem()
    return _alert_system


def check_and_send_alerts() -> Dict:
    """
    Check all alerts and send Telegram notifications

    Usage:
        from src.smart_alert_system import check_and_send_alerts

        result = check_and_send_alerts()
        print(f"Sent {result['sent']} alerts")
    """
    system = get_alert_system()
    return system.process_alerts(send_telegram=True)


def send_custom_alert(title: str, message: str, priority: str = "MEDIUM") -> Tuple[bool, str]:
    """
    Send a custom alert to Telegram

    Usage:
        from src.smart_alert_system import send_custom_alert

        success, msg = send_custom_alert(
            title="Custom Alert",
            message="Your custom message here",
            priority="HIGH"
        )
    """
    system = get_alert_system()
    full_message = f"*{title}*\n\n{message}"
    return system.send_telegram(full_message, priority)
