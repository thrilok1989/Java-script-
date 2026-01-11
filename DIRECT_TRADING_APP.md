# ⚡ Direct Trading App

A lightweight, fast trading platform for NIFTY and SENSEX options with ATM ± 5 strike selection.

## 🎯 Features

- **ATM ± 5 Strike Grid**: Quick access to 11 strikes around ATM
- **One-Click Trading**: Buy CE/PE with a single click
- **Market & Limit Orders**: Flexible order types
- **Live Positions**: Track open positions in sidebar
- **Order Book**: View today's orders
- **Real-time Data**: Uses same cache as main app

## 🚀 Quick Start

### Run the Trading App

```bash
# Make sure you're in the project directory
cd /path/to/Java-script-

# Run the trading app on port 8502 (different from main app)
streamlit run direct_trading_app.py --server.port 8502
```

### Access the App

Open your browser to: **http://localhost:8502**

## 📋 Prerequisites

1. **Dhan API Credentials**: Configure in `.streamlit/secrets.toml`
2. **Main App Data Cache**: The trading app reuses data from the main app's cache

## 🔧 How It Works

### Data Sharing

The trading app uses the **same data cache** as the main app:

```
Main App (port 8501)         Trading App (port 8502)
     ↓                              ↓
     └──────→ Data Cache ←──────────┘
              (Shared)
```

**Benefits:**
- No duplicate API calls
- Instant data access if main app is running
- Lower API rate limit usage
- Faster performance

### Running Both Apps

You can run both apps simultaneously:

```bash
# Terminal 1: Main app
streamlit run app.py --server.port 8501

# Terminal 2: Trading app
streamlit run direct_trading_app.py --server.port 8502
```

**Main App**: Use for analysis, signals, charts, AI
**Trading App**: Use for fast order execution

## 🎨 Interface

### Strike Grid Layout

```
📈 CALL (CE)              📉 PUT (PE)
═════════════            ═════════════
BUY 24,750 (ATM-5) CE    BUY 24,750 (ATM-5) PE
BUY 24,800 (ATM-4) CE    BUY 24,800 (ATM-4) PE
BUY 24,850 (ATM-3) CE    BUY 24,850 (ATM-3) PE
BUY 24,900 (ATM-2) CE    BUY 24,900 (ATM-2) PE
BUY 24,950 (ATM-1) CE    BUY 24,950 (ATM-1) PE
BUY 25,000 (ATM) CE ⭐   BUY 25,000 (ATM) PE ⭐
BUY 25,050 (ATM+1) CE    BUY 25,050 (ATM+1) PE
BUY 25,100 (ATM+2) CE    BUY 25,100 (ATM+2) PE
BUY 25,150 (ATM+3) CE    BUY 25,150 (ATM+3) PE
BUY 25,200 (ATM+4) CE    BUY 25,200 (ATM+4) PE
BUY 25,250 (ATM+5) CE    BUY 25,250 (ATM+5) PE
```

### Order Flow

1. **Click Strike**: Click any CE/PE button
2. **Enter Details**: Set lots and order type
3. **Place Order**: Confirm and execute
4. **Track**: View in sidebar positions/orders

## 📊 Sidebar Features

### Open Positions
- Live P&L tracking
- Current LTP
- Average price
- Quantity

### Order Book
- Last 5 orders
- Order status
- Quick reference

## ⚙️ Configuration

The trading app uses the same configuration as the main app:

### Required in `.streamlit/secrets.toml`:
```toml
[DHAN]
CLIENT_ID = "your_dhan_client_id"
ACCESS_TOKEN = "your_dhan_access_token"
```

### Strike Intervals (from `config.py`):
- NIFTY: 50 points
- SENSEX: 100 points

### Lot Sizes (from `config.py`):
- NIFTY: 75
- SENSEX: 30

## 🔥 Use Cases

### 1. Scalping
- Quick entry/exit with one-click buttons
- Minimal UI distraction
- Fast execution

### 2. Hedging
- Quick hedge placement
- Easy strike selection
- Live position monitoring

### 3. Dedicated Trading Screen
- Run on second monitor
- Main app for analysis
- Trading app for execution

### 4. Mobile Trading
- Simplified interface
- Larger buttons
- Touch-friendly

## 🚨 Important Notes

1. **Data Dependency**: Trading app reads from cache. If no data, run main app first or wait for cache to populate.

2. **API Credentials**: Must be configured in `.streamlit/secrets.toml`

3. **Market Hours**: Best performance during market hours (9:15 AM - 3:30 PM IST)

4. **Order Validation**: Always verify strike, expiry, and quantity before placing orders

5. **Risk Warning**: Real money trading - use at your own risk

## 🐛 Troubleshooting

### "Loading data from cache..."

**Solution**:
- Run the main app first to populate cache
- OR wait 10-30 seconds for background data load

### "Security ID not found"

**Solution**:
- Check if option chain data is available
- Verify strike price exists for current expiry
- Ensure market is open or data is cached

### "Order failed"

**Solution**:
- Check Dhan API credentials
- Verify sufficient margin
- Check order parameters (quantity, price)
- Review API error message

## 📈 Performance

**Load Time**: < 2 seconds (using cached data)
**Order Execution**: < 1 second
**Memory Usage**: ~100 MB (vs 300 MB main app)
**CPU Usage**: Minimal (no heavy analytics)

## 🔮 Future Enhancements

- [ ] Option Greeks display
- [ ] Quick position exit buttons
- [ ] Bracket order support
- [ ] Custom strike ranges
- [ ] Keyboard shortcuts
- [ ] Price alerts
- [ ] Multi-leg strategies

## 📝 License

Same as main repository

## 🤝 Contributing

Improvements welcome! Submit PRs to main repository.

---

**Happy Trading! 🚀**
