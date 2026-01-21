import requests
import os
import aiohttp
import html
import streamlit as st
from config import get_telegram_credentials, IST, get_current_time_ist
from datetime import datetime
from typing import Dict, Any, Optional

class TelegramBot:
    def __init__(self):
        """Initialize Telegram bot"""
        creds = get_telegram_credentials()
        self.enabled = creds['enabled']
        
        if self.enabled:
            self.bot_token = creds['bot_token']
            self.chat_id = creds['chat_id']
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send Telegram message"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def send_message_async(self, message: str, parse_mode: str = "HTML"):
        """Send Telegram message asynchronously"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    return response.status == 200
        except:
            return False
    
    def send_signal_ready(self, setup: dict):
        """Send signal ready alert"""
        message = f"""
🎯 <b>SIGNAL READY</b>

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}

<b>Support:</b> {setup.get('vob_support', 'N/A')}
<b>Resistance:</b> {setup.get('vob_resistance', 'N/A')}

<b>Status:</b> ✅ Ready to Trade
<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

📱 Open app to execute trade
        """
        return self.send_message(message.strip())

    def send_order_placed(self, setup: dict, order_id: str, strike: int,
                         sl: float, target: float):
        """Send order placed confirmation"""
        message = f"""
✅ <b>ORDER PLACED SUCCESSFULLY</b>

<b>Order ID:</b> {order_id}

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}
<b>Strike:</b> {strike}

<b>Stop Loss:</b> {sl}
<b>Target:</b> {target}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

📊 Monitor position in app
        """
        return self.send_message(message.strip())
    
    def send_order_failed(self, setup: dict, error: str):
        """Send order failure alert"""
        message = f"""
❌ <b>ORDER PLACEMENT FAILED</b>

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}

<b>Error:</b> {error}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

⚠️ Check app for details
        """
        return self.send_message(message.strip())
    
    def send_position_exit(self, order_id: str, pnl: float):
        """Send position exit alert"""
        pnl_emoji = "💰" if pnl > 0 else "📉"
        message = f"""
{pnl_emoji} <b>POSITION EXITED</b>

<b>Order ID:</b> {order_id}
<b>P&L:</b> ₹{pnl:,.2f}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}
        """
        return self.send_message(message.strip())

    def send_classic_entry_alert(self, signal_type: str, entry_zone: tuple, stop_loss: float,
                                 targets: dict, current_price: float, source: str,
                                 confirmations: dict, range_zones: dict = None) -> bool:
        """
        Send CLASSIC entry alert (simple VOB-based)

        Args:
            signal_type: "LONG" or "SHORT"
            entry_zone: (lower, upper) tuple
            stop_loss: SL price
            targets: {'t1': price, 't2': price}
            current_price: Current market price
            source: Entry source (e.g., "VOB Resistance")
            confirmations: Dict with regime, atm_bias, volume, price_action status
            range_zones: Optional dict with {'low': price, 'mid': price, 'high': price}
        """
        signal_emoji = "🟢" if signal_type == "LONG" else "🔴"
        direction_label = "LONG" if signal_type == "LONG" else "SHORT"

        # Calculate points
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        sl_points = abs(stop_loss - entry_mid)
        t1_points = abs(targets.get('t1', entry_mid) - entry_mid)
        t2_points = abs(targets.get('t2', entry_mid) - entry_mid)

        # Count confirmations
        confirmed_count = sum(1 for v in confirmations.values() if '✅' in str(v))
        total_checks = len(confirmations)

        # Format range zones if provided
        range_info = ""
        if range_zones:
            range_info = f"""
📊 <b>Range Zones:</b>
   🟢 <b>Low (BUY):</b> ₹{range_zones.get('low', 0):,.0f}
   ⚪ <b>Mid (AVOID):</b> ₹{range_zones.get('mid', 0):,.0f}
   🔴 <b>High (SELL):</b> ₹{range_zones.get('high', 0):,.0f}

"""

        message = f"""
{signal_emoji} <b>CLASSIC {direction_label} SIGNAL - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Entry:</b> ₹{entry_zone[0]:,.0f} - ₹{entry_zone[1]:,.0f}
<b>Source:</b> {source}

🛑 <b>SL:</b> ₹{stop_loss:,.0f} ({'+' if signal_type == 'SHORT' else '-'}{sl_points:.0f}pts)
🎯 <b>T1:</b> ₹{targets.get('t1', 0):,.0f} ({'+' if signal_type == 'LONG' else '-'}{t1_points:.0f}pts)
🎯 <b>T2:</b> ₹{targets.get('t2', 0):,.0f} ({'+' if signal_type == 'LONG' else '-'}{t2_points:.0f}pts)
{range_info}✅ <b>Confirmations: {confirmed_count}/{total_checks}</b>
• <b>Regime:</b> {confirmations.get('regime', 'N/A')}
• <b>ATM Bias:</b> {confirmations.get('atm_bias', 'N/A')}
• <b>Volume:</b> {confirmations.get('volume', 'N/A')}
• <b>Price Action:</b> {confirmations.get('price_action', 'N/A')}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}
📍 <b>Price:</b> ₹{current_price:,.0f}
        """
        return self.send_message(message.strip())

    def send_advanced_entry_alert(self, signal_type: str, pattern_type: str, entry_zone: tuple,
                                  smart_sl: dict, smart_targets: dict, confluence: dict,
                                  current_price: float, pattern_details: dict = None,
                                  range_zones: dict = None) -> bool:
        """
        Send ADVANCED entry alert (pattern-based with full confluence analysis)

        Args:
            signal_type: "LONG" or "SHORT"
            pattern_type: Pattern name (e.g., "Head & Shoulders Neckline")
            entry_zone: (lower, upper) tuple
            smart_sl: {'price': float, 'reason': str, 'risk_points': float, 'risk_percent': float,
                       'invalidation_triggers': list}
            smart_targets: {'t1': {...}, 't2': {...}, 't3': {...}} with price, confluence, sources
            confluence: {'score': float, 'confirmed': int, 'total': int, 'checks': {...}}
            current_price: Current market price
            pattern_details: Optional pattern metadata
            range_zones: Optional dict with {'low': price, 'mid': price, 'high': price}
        """
        signal_emoji = "🚀" if signal_type == "LONG" else "🔴"
        direction_label = "LONG" if signal_type == "LONG" else "SHORT"

        # Calculate points
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        t1_points = abs(smart_targets['t1']['price'] - entry_mid)
        t2_points = abs(smart_targets['t2']['price'] - entry_mid)
        t3_points = abs(smart_targets['t3']['price'] - entry_mid)

        # Format confluence checks
        checks = confluence.get('checks', {})
        check_lines = []
        for key, value in checks.items():
            if isinstance(value, dict):
                status = value.get('status', '⚠️')
                detail = value.get('detail', '')
                check_lines.append(f"   {status} {key.replace('_', ' ').title()}: {detail}")
            else:
                check_lines.append(f"   {value}")

        checks_text = '\n'.join(check_lines[:8])  # Limit to 8 checks

        # Pattern details if available
        pattern_info = ""
        if pattern_details:
            pattern_info = f"""
📐 <b>Pattern Details:</b>
   Left Shoulder: ₹{pattern_details.get('left_shoulder', 0):,.0f}
   Head: ₹{pattern_details.get('head', 0):,.0f}
   Right Shoulder: ₹{pattern_details.get('right_shoulder', 0):,.0f}
   Neckline: ₹{pattern_details.get('neckline', 0):,.0f}

"""

        # Format range zones if provided
        range_info = ""
        if range_zones:
            range_info = f"""
📊 <b>Range Zones:</b>
   🟢 <b>Low (BUY):</b> ₹{range_zones.get('low', 0):,.0f}
   ⚪ <b>Mid (AVOID):</b> ₹{range_zones.get('mid', 0):,.0f}
   🔴 <b>High (SELL):</b> ₹{range_zones.get('high', 0):,.0f}

"""

        message = f"""
🚀 <b>ADVANCED {direction_label} SIGNAL - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Pattern:</b> {pattern_type.upper()}
<b>Entry:</b> ₹{entry_zone[0]:,.0f} - ₹{entry_zone[1]:,.0f}

🛑 <b>Smart SL:</b> ₹{smart_sl['price']:,.0f} (+{smart_sl['risk_points']:.0f}pts)
   • {smart_sl['reason']}
   • <b>Risk:</b> {smart_sl['risk_percent']:.1f}%

🎯 <b>Smart Targets:</b>
   <b>T1:</b> ₹{smart_targets['t1']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t1_points:.0f}pts)
      └─ {smart_targets['t1']['confluence']} ({smart_targets['t1']['source_count']} sources)

   <b>T2:</b> ₹{smart_targets['t2']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t2_points:.0f}pts) ⭐
      └─ {smart_targets['t2']['confluence']} ({smart_targets['t2']['source_count']} sources)

   <b>T3:</b> ₹{smart_targets['t3']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t3_points:.0f}pts)
      └─ {smart_targets['t3']['confluence']} ({smart_targets['t3']['source_count']} sources)
{range_info}{pattern_info}🔍 <b>Confluence:</b> {confluence['score']:.0f}% ({confluence['confirmed']}/{confluence['total']} confirmations)
{checks_text}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}
📍 <b>Price:</b> ₹{current_price:,.0f}
        """
        return self.send_message(message.strip())

    def send_oi_unwinding_alert(self, position_type: str, entry_strike: int,
                               oi_check_result: dict, current_price: float) -> bool:
        """
        Send OI unwinding exit alert

        Args:
            position_type: "LONG" or "SHORT"
            entry_strike: Entry strike price
            oi_check_result: Result from OIShiftMonitor.check_oi_shift()
            current_price: Current market price
        """
        alert_emoji = "🚨" if oi_check_result['action'] == 'EXIT_ALL' else "⚠️"

        message = f"""
{alert_emoji} <b>OI UNWINDING ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type} from ₹{entry_strike:,}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>OI Change:</b>
{oi_check_result['details']}
<b>Change:</b> {oi_check_result['oi_change_pct']:.1f}% {alert_emoji}

⚠️ <b>SUPPORT/RESISTANCE WALL COLLAPSING!</b>

<b>Action:</b> {oi_check_result['action'].replace('_', ' ')}
<b>Reason:</b> {oi_check_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

🚨 Your entry level S/R is disappearing!
        """
        return self.send_message(message.strip())

    def send_opposite_oi_buildup_alert(self, position_type: str, barrier_result: dict,
                                      current_price: float) -> bool:
        """
        Send alert when fresh OI builds on opposite side

        Args:
            position_type: "LONG" or "SHORT"
            barrier_result: Result from OIShiftMonitor.check_opposite_side_buildup()
            current_price: Current market price
        """
        alert_emoji = "🔴" if barrier_result['alert_priority'] == 'CRITICAL' else "⚠️"

        # Format barrier details
        barriers_text = ""
        for barrier in barrier_result.get('barriers', [])[:3]:  # Top 3
            barriers_text += f"\n• ₹{barrier['strike']:,} (+{barrier['oi_change']:,} OI, {barrier['distance']}pts away)"

        message = f"""
{alert_emoji} <b>NEW BARRIER FORMING - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>Fresh OI Detected:</b>{barriers_text}

⚠️ <b>NEW {'RESISTANCE' if position_type == 'LONG' else 'SUPPORT'} FORMING!</b>

<b>Action:</b> {barrier_result['action'].replace('_', ' ')}
<b>Reason:</b> {barrier_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

💡 Consider tightening SL or partial exit
        """
        return self.send_message(message.strip())

    def send_volume_spike_alert(self, position_type: str, volume_check: dict,
                               current_price: float) -> bool:
        """
        Send volume spike exit alert

        Args:
            position_type: "LONG" or "SHORT"
            volume_check: Result from VolumeSpikeMonitor.check_volume_spike()
            current_price: Current market price
        """
        alert_emoji = "🚨" if volume_check['severity'] == 'CRITICAL' else "⚠️"

        message = f"""
{alert_emoji} <b>VOLUME SPIKE ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>VOLUME SPIKE DETECTED:</b>
<b>Volume Ratio:</b> {volume_check['volume_ratio']:.1f}x average
<b>Buy Volume:</b> {volume_check['buy_volume']:,} ({volume_check['buy_pct']:.0f}%)
<b>Sell Volume:</b> {volume_check['sell_volume']:,} ({volume_check['sell_pct']:.0f}%)
<b>Delta:</b> {volume_check['delta']:,}

⚠️ <b>INSTITUTIONAL MOVE DETECTED!</b>

<b>Action:</b> {volume_check['action'].replace('_', ' ')}
<b>Reason:</b> {volume_check['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

🚨 Market shifting against you!
        """
        return self.send_message(message.strip())

    def send_volume_absorption_alert(self, position_type: str, absorption_result: dict,
                                    current_price: float) -> bool:
        """
        Send volume absorption exit alert

        Args:
            position_type: "LONG" or "SHORT"
            absorption_result: Result from VolumeSpikeMonitor.detect_absorption()
            current_price: Current market price
        """
        sr_type = "RESISTANCE" if position_type == "LONG" else "SUPPORT"

        message = f"""
⚠️ <b>VOLUME ABSORPTION ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>Absorption Detected:</b>
<b>Total Volume:</b> {absorption_result['total_volume']:,}
<b>Price Change:</b> Only {absorption_result['price_change_pct']:.2f}%

⚠️ <b>{sr_type} DEFENDING STRONGLY!</b>

<b>Action:</b> {absorption_result['action'].replace('_', ' ')}
<b>Reason:</b> {absorption_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

💡 High volume but price not breaking
Sellers/buyers absorbing all pressure
        """
        return self.send_message(message.strip())

    async def send_ai_market_alert(self, report: Dict[str, Any], confidence_thresh: float = 0.60) -> bool:
        """
        Send AI market alert to Telegram
        """
        if not report:
            return False
        
        confidence = float(report.get("confidence", 0.0) or 0.0)
        if confidence < confidence_thresh:
            return False
        
        label = report.get("label", "UNKNOWN")
        rec = report.get("recommendation", "HOLD")
        tech = report.get("technical_score", 0.0)
        news = report.get("news_score", 0.0)
        ai_score = report.get("ai_score", 0.0)
        reasons = report.get("ai_reasons", []) or []
        ai_summary = report.get("ai_summary", "") or ""

        # Format top reasons
        top_reasons = "\n".join(f"{i+1}. {html.escape(str(r))}" for i, r in enumerate(reasons[:4]))
        
        # Format technical contributions
        tech_lines = []
        contribs = report.get("technical_contributions", {}) or {}
        for k, v in contribs.items():
            sign = "Bullish" if v > 0 else "Bearish" if v < 0 else "Neutral"
            tech_lines.append(f"{html.escape(k)}: {sign} ({v:.3f})")
        
        tech_block = "\n".join(tech_lines[:6]) or "N/A"
        
        # Determine emoji based on bias
        if "BULLISH" in label.upper():
            bias_emoji = "🟢"
        elif "BEARISH" in label.upper():
            bias_emoji = "🔴"
        else:
            bias_emoji = "⚪"

        # Create the message
        text = f"""
{bias_emoji} <b>🤖 AI MARKET REPORT</b>

<b>Market:</b> {html.escape(str(report.get('market','')))}
<b>Bias:</b> <b>{bias_emoji} {html.escape(label)}</b>
<b>Recommendation:</b> <b>{html.escape(rec)}</b>

📊 <b>SCORES</b>
<b>Confidence:</b> {confidence:.2f}
<b>AI Score:</b> {ai_score:.3f}
<b>Technical Score:</b> {tech:.3f}
<b>News Score:</b> {news:.3f}

⚙️ <b>TECHNICAL SUMMARY</b>
{tech_block}

📰 <b>NEWS SUMMARY</b>
{html.escape(str(report.get('news_summary',''))[:600])}

🧠 <b>AI REASONING</b>
{top_reasons}

📋 <b>SUMMARY</b>
{html.escape(ai_summary[:800])}

⏰ <b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}
        """
        
        return await self.send_message_async(text.strip())

    def send_ict_indicator_alert(self, symbol: str, ict_signals: dict, current_price: float) -> bool:
        """
        Send ICT Comprehensive Indicator alert with all detected signals

        Args:
            symbol: Trading symbol (e.g., "NIFTY 50")
            ict_signals: Dictionary with signal data from ICT indicator
            current_price: Current market price
        """
        if not self.enabled:
            return False

        overall_bias = ict_signals.get('overall_bias', 'NEUTRAL')
        bullish_signals = ict_signals.get('bullish_signals', [])
        bearish_signals = ict_signals.get('bearish_signals', [])
        bullish_count = ict_signals.get('bullish_count', 0)
        bearish_count = ict_signals.get('bearish_count', 0)

        # Determine emoji based on bias
        bias_emoji = "🟢" if overall_bias == "BULLISH" else "🔴" if overall_bias == "BEARISH" else "⚪"

        # Format bullish signals
        bullish_text = ""
        if bullish_signals:
            bullish_text = "\n<b>🟢 BULLISH SIGNALS:</b>\n"
            for signal in bullish_signals[:5]:  # Limit to 5 signals
                sig_type = signal.get('type', 'Unknown')
                sig_price = signal.get('price', 'N/A')
                sig_level = signal.get('level', '')
                poi = signal.get('poi', '')

                level_badge = f" [{sig_level}]" if sig_level else ""
                poi_badge = f" POI: {poi}" if poi else ""

                bullish_text += f"  • {sig_type}{level_badge}: {sig_price}{poi_badge}\n"

        # Format bearish signals
        bearish_text = ""
        if bearish_signals:
            bearish_text = "\n<b>🔴 BEARISH SIGNALS:</b>\n"
            for signal in bearish_signals[:5]:  # Limit to 5 signals
                sig_type = signal.get('type', 'Unknown')
                sig_price = signal.get('price', 'N/A')
                sig_level = signal.get('level', '')
                poi = signal.get('poi', '')

                level_badge = f" [{sig_level}]" if sig_level else ""
                poi_badge = f" POI: {poi}" if poi else ""

                bearish_text += f"  • {sig_type}{level_badge}: {sig_price}{poi_badge}\n"

        message = f"""
{bias_emoji} <b>ICT INDICATOR ALERT - {symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Overall Bias:</b> {overall_bias}
<b>Current Price:</b> ₹{current_price:,.2f}

<b>Signal Strength:</b>
🟢 Bullish: {bullish_count}
🔴 Bearish: {bearish_count}
{bullish_text}{bearish_text}
<b>Components:</b>
• Order Blocks (Swing & Internal)
• Fair Value Gaps (FVG)
• Supply/Demand Zones
• Volume Profile POC

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}
📊 Open app for full chart visualization
        """
        return self.send_message(message.strip())


def send_test_message():
    """Send test message to verify Telegram setup"""
    bot = TelegramBot()
    if bot.enabled:
        message = """
✅ <b>Telegram Connected!</b>

Your trading alerts are now active.

<b>Test Time (IST):</b> """ + get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')
        return bot.send_message(message.strip())
    return False


# Async wrapper function for backward compatibility
async def send_ai_market_alert_async(report: Dict[str, Any], confidence_thresh: float = 0.60) -> bool:
    """
    Async wrapper function for sending AI market alerts
    """
    bot = TelegramBot()
    if not bot.enabled:
        return False
    return await bot.send_ai_market_alert(report, confidence_thresh)


# Alias for backward compatibility
TelegramAlerts = TelegramBot
