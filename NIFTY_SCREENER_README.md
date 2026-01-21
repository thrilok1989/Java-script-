# 🎯 NIFTY Option Screener v7.0 - Standalone App

A dedicated, standalone application for NIFTY options analysis with 100% Seller's Perspective, ATM Bias Analysis, Moment Detection, and Expiry Spike Detection.

## 📋 Overview

This is a **standalone version** of the NIFTY Option Screener extracted from the main trading application. It can run independently without the other tabs and features of the main app.

### Key Features

- **100% Seller's Perspective**: All analysis from option writers' viewpoint
- **ATM Bias Analyzer**: 12 comprehensive metrics for at-the-money analysis
- **Moment Detector**: Real-time momentum burst and orderbook pressure detection
- **Expiry Spike Detector**: Special alerts when ≤5 days to expiry
- **Enhanced OI/PCR Analytics**: Deep dive into Open Interest and Put-Call Ratio
- **Telegram Integration**: Automated signal notifications
- **Live Market Data**: Real-time data from Dhan API
- **Greeks Calculation**: Delta, Gamma, Theta, Vega for all strikes
- **Auto-Refresh**: Dynamic intervals based on market hours

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Dhan trading account with API access
- Telegram bot (optional, for alerts)

### Installation

1. **Clone or navigate to the repository:**
```bash
cd /path/to/Java-script-
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure credentials:**

Create or edit `.streamlit/secrets.toml`:
```toml
[DHAN]
CLIENT_ID = "your_dhan_client_id"
ACCESS_TOKEN = "your_dhan_access_token"

[TELEGRAM]
BOT_TOKEN = "your_telegram_bot_token"
CHAT_ID = "your_telegram_chat_id"
```

### Running the App

```bash
streamlit run nifty_screener_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Features in Detail

### 1. ATM Bias Analyzer (12 Metrics)

Analyzes the at-the-money (ATM) strike with comprehensive metrics:

1. **OI Bias**: Open Interest comparison between CALL and PUT
2. **Change in OI Bias**: Momentum of OI changes
3. **Volume Bias**: Trading volume comparison
4. **Delta Bias**: Combined delta exposure
5. **Gamma Bias**: Gamma concentration analysis
6. **Premium Bias**: Premium differential
7. **IV Bias**: Implied volatility comparison
8. **Delta Exposure Bias**: Market-maker positioning
9. **Gamma Exposure Bias**: Gamma risk assessment
10. **IV Skew Bias**: Volatility skew analysis
11. **OI Change Bias**: Recent OI momentum
12. **Combined Score**: Weighted aggregate of all metrics

### 2. Moment Detector

Real-time detection of significant market moments:

- **Momentum Burst**: Sudden spikes in Volume × IV × ΔOI
- **Orderbook Pressure**: Bid/Ask depth imbalances
- **Gamma Cluster**: ATM gamma concentration
- **OI Velocity & Acceleration**: Speed of OI changes

### 3. Expiry Spike Detector

Activates when ≤5 days to expiry, detecting:

- ATM OI concentration
- Distance from Max Pain
- PCR extremes (>1.5 or <0.7)
- Massive OI walls
- Gamma flip risk
- Rapid unwinding activity

### 4. Enhanced OI/PCR Analytics

Comprehensive Open Interest and Put-Call Ratio analysis:

- Total CALL vs PUT OI
- PCR interpretation and sentiment
- OI concentration metrics
- ITM/OTM distribution
- Max OI strikes identification
- Historical PCR context

### 5. Seller's Perspective

All metrics interpreted from option seller/market maker viewpoint:

- **CALL Building** (↑ CALL OI) = **BEARISH** (sellers expect price to stay below)
- **PUT Building** (↑ PUT OI) = **BULLISH** (sellers expect price to stay above)
- **CALL Unwinding** (↓ CALL OI) = **BULLISH** (sellers covering bearish bets)
- **PUT Unwinding** (↓ PUT OI) = **BEARISH** (sellers covering bullish bets)

## 🔄 Auto-Refresh System

The app automatically refreshes based on market conditions:

- **During Trading Hours** (9:15 AM - 3:30 PM IST): 10 seconds
- **Outside Trading Hours**: 60 seconds
- **Late Night** (11 PM - 6 AM IST): 5 minutes

## 📱 Telegram Signal Alerts

Automated signals sent when:

- Position ≠ NEUTRAL
- Confidence ≥ 40%
- Signal conditions met

Signals include:
- Position (BULLISH/BEARISH)
- Confidence percentage
- Key metrics and reasoning
- Moment detector alerts
- Expiry spike warnings

## 📁 File Structure

```
Java-script-/
├── nifty_screener_app.py          # Standalone app entry point
├── NiftyOptionScreener.py          # Core screener logic (8,967 lines)
├── config.py                       # Configuration & credentials
├── dhan_api.py                     # Dhan API integration
├── market_hours_scheduler.py       # Market hours tracking
├── telegram_alerts.py              # Telegram bot integration
├── market_depth_advanced.py        # Advanced market depth analysis
├── option_chain_table.py           # Option chain display
├── NIFTY_SCREENER_README.md       # This file
└── .streamlit/
    └── secrets.toml                # Credentials (create this)
```

## 🔧 Dependencies

Key Python packages required:

- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Statistical functions (Black-Scholes)
- `plotly` - Interactive charts
- `requests` - API calls
- `pytz` - Timezone handling
- `streamlit-autorefresh` - Auto-refresh functionality
- `supabase` - Database (optional)
- `python-dotenv` - Environment variables

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Use Cases

### For Day Traders
- Real-time ATM bias analysis
- Quick entry/exit signals
- Moment detection for scalping opportunities

### For Options Sellers
- Comprehensive seller's perspective
- Greeks analysis for premium collection
- Risk assessment via Gamma/Delta exposure

### For Swing Traders
- Multi-day bias trends
- Support/Resistance identification
- Expiry spike warnings

### For Market Analysts
- Institutional positioning insights
- Market maker activity detection
- Deep OI/PCR analytics

## ⚙️ Configuration Options

### Refresh Intervals

Modify in `nifty_screener_app.py`:
```python
def get_auto_refresh_interval():
    # Customize intervals based on your needs
    if trading_hours:
        return 10000  # 10 seconds
    else:
        return 60000  # 60 seconds
```

### Telegram Settings

Configure in `.streamlit/secrets.toml`:
```toml
[TELEGRAM]
BOT_TOKEN = "your_bot_token"
CHAT_ID = "your_chat_id"
ENABLED = true
```

## 🐛 Troubleshooting

### Common Issues

**1. "Module not found" errors:**
```bash
pip install -r requirements.txt --upgrade
```

**2. "Missing credentials" error:**
- Ensure `.streamlit/secrets.toml` exists and is properly configured
- Check that DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are valid

**3. "API rate limit exceeded":**
- Dhan API limits: 25 req/sec, 250 req/min, 7000 req/day
- Increase refresh interval during testing

**4. Blank screen or loading issues:**
- Check browser console for JavaScript errors
- Try clearing browser cache
- Use Chrome/Firefox for best compatibility

**5. Telegram alerts not working:**
- Verify BOT_TOKEN and CHAT_ID are correct
- Test bot with `/start` command
- Check bot has permission to send messages

## 📈 Performance Tips

1. **During Market Hours**: The app is optimized for real-time analysis
2. **Outside Market Hours**: Data updates are less frequent
3. **Mobile Usage**: Best viewed on desktop, but mobile-responsive
4. **Multiple Tabs**: Avoid opening multiple instances (API limits)

## 🔐 Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Keep API credentials secure
- Use environment variables in production
- Regenerate tokens if compromised

## 📊 Data Sources

- **Primary**: Dhan API (option chain, market depth, NIFTY spot)
- **Backup**: Session state caching for continuity
- **Historical**: Optional Supabase integration

## 🆚 Differences from Main App

**Standalone App:**
- Single-purpose: Only NIFTY Option Screener
- Faster startup: No other tabs to load
- Independent: Doesn't require main app
- Dedicated: Full screen for screener
- Simplified: Focused UI without navigation

**Main App:**
- Multi-tab: 5 different analysis tools
- Comprehensive: Multiple market instruments
- Integrated: Shared data between tabs
- Feature-rich: More tools and utilities

## 📝 License

This is part of the Java-script- trading application repository.

## 🤝 Contributing

If you find bugs or have feature requests, please create an issue in the main repository.

## ⚠️ Disclaimer

**Important**: This tool is for educational and analytical purposes only. Trading in options involves substantial risk and is not suitable for all investors. Past performance is not indicative of future results. Always:

- Do your own research
- Understand the risks
- Never invest more than you can afford to lose
- Consult with a financial advisor
- Use proper risk management

The developers are not responsible for any trading losses incurred using this tool.

## 📞 Support

For issues specific to the standalone app:
1. Check this README first
2. Review console/terminal logs
3. Verify credentials configuration
4. Check Dhan API status
5. Create an issue in the repository

## 🎓 Learning Resources

To better understand the concepts:

- **Options Trading**: Learn about CALL/PUT options, Greeks, OI, PCR
- **Seller's Perspective**: Understand option writing strategies
- **Technical Analysis**: Support/Resistance, momentum indicators
- **Risk Management**: Position sizing, stop losses

---

**Built with ❤️ for the trading community**

**Happy Trading! 🚀📈**
