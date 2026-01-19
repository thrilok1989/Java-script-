import streamlit as st
import time
from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from pytz import timezone as pytz_timezone
import plotly.graph_objects as go
import io
import requests
from streamlit_autorefresh import st_autorefresh
import os
import asyncio
import logging

# Import modules
from config import *
from market_data import *
from market_hours_scheduler import scheduler, is_within_trading_hours, should_run_app
from signal_manager import SignalManager
from strike_calculator import calculate_strike, calculate_levels
from trade_executor import TradeExecutor
from telegram_alerts import TelegramBot, send_test_message
from dhan_api import check_dhan_connection
from bias_analysis import BiasAnalysisPro
from advanced_chart_analysis import AdvancedChartAnalysis
from overall_market_sentiment import render_overall_market_sentiment, calculate_overall_sentiment, run_ai_analysis, shutdown_ai_engine
from advanced_proximity_alerts import get_proximity_alert_system
from data_cache_manager import (
    get_cache_manager,
    preload_all_data,
    get_cached_nifty_data,
    get_cached_sensex_data,
    get_cached_bias_analysis_results
)
from ai_tab_integration import render_master_ai_analysis_tab, render_advanced_analytics_tab
from src.market_structure_ui import render_market_structure_section, render_structure_widget

# Configure logging
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# GET SPOT PRICE FOR BROWSER TAB TITLE
# ═══════════════════════════════════════════════════════════════════════
# Get spot price from session state or cache for dynamic page title
page_title = "NIFTY/SENSEX Trader"
if 'last_spot_price' in st.session_state and st.session_state.last_spot_price:
    spot = st.session_state.last_spot_price
    page_title = f"NIFTY ₹{spot:,.2f} | Trader"

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & PERFORMANCE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=page_title,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS - PREVENT BLUR/LOADING OVERLAY DURING REFRESH
# ═══════════════════════════════════════════════════════════════════════
# This CSS prevents the app from showing blur/white screen during refresh
# allowing users to continue viewing data while refresh happens in background

st.markdown("""
<style>
    /* Hide the Streamlit loading spinner and blur overlay */
    .stApp > div[data-testid="stAppViewContainer"] > div:first-child {
        display: none !important;
    }

    /* Prevent blur overlay during rerun */
    .stApp [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    /* Hide the "Running..." indicator in top right */
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
    }

    /* Hide all spinner elements */
    .stSpinner {
        display: none !important;
    }

    /* Hide loading indicator */
    div[data-testid="stLoadingIndicator"] {
        display: none !important;
    }

    /* Keep app content visible during refresh - no opacity change */
    .element-container {
        opacity: 1 !important;
        transition: none !important;
    }

    .stMarkdown {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Prevent white flash during page reload */
    body {
        background-color: #0E1117;
        transition: none !important;
    }

    /* Smooth transitions for dynamic content - DISABLED to prevent blur */
    .stApp {
        transition: none !important;
    }

    /* Hide loading overlay completely */
    div[data-testid="stAppViewContainer"] > div[style*="position: absolute"] {
        display: none !important;
    }

    /* Ensure dataframes remain visible during refresh */
    .dataframe {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Keep charts visible during refresh */
    .stPlotlyChart {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Hide the app header spinner */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Prevent flickering on dynamic updates */
    section[data-testid="stSidebar"],
    section[data-testid="stMain"] {
        transition: none !important;
    }

    /* Keep all content visible - override any opacity changes */
    [data-testid="stVerticalBlock"] {
        opacity: 1 !important;
    }

    /* Remove blur filter if applied */
    * {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    /* ═══════════════════════════════════════════════════════════════════ */
    /* MOBILE & DESKTOP RESPONSIVE IMPROVEMENTS */
    /* ═══════════════════════════════════════════════════════════════════ */

    /* Ensure tabs are scrollable on mobile */
    [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }

    /* Make tabs touch-friendly on mobile */
    [data-baseweb="tab"] {
        min-height: 48px !important;
        padding: 12px 16px !important;
    }

    /* Responsive font sizes for mobile */
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }

        /* Reduce padding on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Make tables scrollable on mobile */
        .dataframe-container {
            overflow-x: auto !important;
        }

        /* Stack columns on mobile */
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }

    /* Improve touch targets for buttons */
    button {
        min-height: 44px !important;
        min-width: 44px !important;
    }

    /* Smooth scrolling for better UX */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# Performance optimization: Reduce widget refresh overhead
# This improves app responsiveness and reduces lag
if 'performance_mode' not in st.session_state:
    st.session_state.performance_mode = True

# ═══════════════════════════════════════════════════════════════════════
# AI MARKET ANALYSIS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Get API keys from Streamlit secrets
NEWSDATA_API_KEY = st.secrets.get("NEWSDATA_API_KEY", "")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")

# Initialize AI analysis tracking
if 'last_ai_analysis_time' not in st.session_state:
    st.session_state.last_ai_analysis_time = 0

if 'ai_analysis_interval' not in st.session_state:
    # Run AI analysis every 30 minutes (1800 seconds)
    st.session_state.ai_analysis_interval = 1800

if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = None

async def run_ai_market_analysis():
    """Run AI market analysis and send alerts"""
    try:
        # Get overall market sentiment from cached sentiment
        overall_market = "NEUTRAL"
        if st.session_state.cached_sentiment:
            sentiment_map = {
                'BULLISH': 'BULL',
                'BEARISH': 'BEAR',
                'NEUTRAL': 'NEUTRAL'
            }
            overall_market = sentiment_map.get(st.session_state.cached_sentiment.get('overall_sentiment', 'NEUTRAL'), 'NEUTRAL')
        
        # Calculate module biases from various indicators
        module_biases = {
            "overall_sentiment": 0.5,
            "option_chain": 0.5,
            "proximity_alerts": 0.5,
        }
        
        # Try to get more accurate biases from actual data
        try:
            # Get bias analysis results if available
            if st.session_state.bias_analysis_results and st.session_state.bias_analysis_results.get('success'):
                bias_results = st.session_state.bias_analysis_results
                overall_score = bias_results.get('overall_score', 0)
                
                # Convert score to 0-1 bias
                if overall_score > 0:
                    module_biases["overall_sentiment"] = min(1.0, 0.5 + (overall_score / 200))
                elif overall_score < 0:
                    module_biases["overall_sentiment"] = max(0.0, 0.5 + (overall_score / 200))
        except:
            pass
        
        # Market metadata
        nifty_data = get_cached_nifty_data()
        market_meta = {
            "volatility": 0.15,  # Default
            "volume_change": 0.05,
            "query": "NSE India market",
            "current_price": nifty_data.get('spot_price', 0) if nifty_data else 0,
            "market_status": get_market_status().get('session', 'unknown')
        }
        
        # Run AI analysis with save and telegram send
        report = await run_ai_analysis(
            overall_market,
            module_biases,
            market_meta,
            news_api_key=NEWSDATA_API_KEY,  # Uses the variable from secrets
            perplexity_api_key=PERPLEXITY_API_KEY,  # Uses the variable from secrets
            save_report=True,
            telegram_send=True
        )
        
        if not report.get("triggered"):
            logger.info("AI not triggered: %s", report.get("reason"))
            return None
        
        logger.info("AI Market Report: label=%s confidence=%.2f recommendation=%s", 
                   report.get("label"), report.get("confidence"), report.get("recommendation"))
        
        return report
        
    except Exception as e:
        logger.error(f"Error in AI market analysis: {e}")
        return None

# Function to check and run AI analysis if needed
def check_and_run_ai_analysis():
    """Check if it's time to run AI analysis and run it if needed"""
    current_time = time.time()
    
    # Check if enough time has passed since last analysis
    if current_time - st.session_state.last_ai_analysis_time > st.session_state.ai_analysis_interval:
        # Only run during market hours for more relevant analysis
        market_status = get_market_status()
        if market_status.get('open', False) and market_status.get('session') == 'regular':
            # Update last analysis time
            st.session_state.last_ai_analysis_time = current_time
            
            # Run AI analysis asynchronously
            try:
                # We'll run it in a separate thread to not block the main app
                import threading
                
                async def run_async():
                    report = await run_ai_market_analysis()
                    if report:
                        st.session_state.ai_analysis_results = report
                
                # Start the async task in a new thread
                thread = threading.Thread(
                    target=lambda: asyncio.run(run_async()),
                    daemon=True
                )
                thread.start()
                
                return True
            except Exception as e:
                logger.error(f"Failed to start AI analysis thread: {e}")
    
    return False

# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if 'signal_manager' not in st.session_state:
    st.session_state.signal_manager = SignalManager()

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if 'active_setup_id' not in st.session_state:
    st.session_state.active_setup_id = None

# Lazy initialization - only create these objects when needed (on tab access)
# This significantly reduces initial load time
def get_bias_analyzer():
    """Lazy load bias analyzer"""
    if 'bias_analyzer' not in st.session_state:
        st.session_state.bias_analyzer = BiasAnalysisPro()
    return st.session_state.bias_analyzer

def get_advanced_chart_analyzer():
    """Lazy load advanced chart analyzer"""
    if 'advanced_chart_analyzer' not in st.session_state:
        st.session_state.advanced_chart_analyzer = AdvancedChartAnalysis()
    return st.session_state.advanced_chart_analyzer

if 'bias_analysis_results' not in st.session_state:
    st.session_state.bias_analysis_results = None

if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# Initialize background data loading
if 'data_preloaded' not in st.session_state:
    st.session_state.data_preloaded = False
    # Start background data loading on first run
    preload_all_data()
    st.session_state.data_preloaded = True

# NSE Options Analyzer - Initialize instruments session state
NSE_INSTRUMENTS = {
    'indices': {
        'NIFTY': {'lot_size': 75, 'zone_size': 20, 'atm_range': 200},
        'BANKNIFTY': {'lot_size': 25, 'zone_size': 100, 'atm_range': 500},
        'SENSEX': {'lot_size': 10, 'zone_size': 50, 'atm_range': 300},
        'NIFTY IT': {'lot_size': 50, 'zone_size': 50, 'atm_range': 300},
        'NIFTY AUTO': {'lot_size': 50, 'zone_size': 50, 'atm_range': 300}
    },
    'stocks': {
        'TCS': {'lot_size': 150, 'zone_size': 30, 'atm_range': 150},
        'RELIANCE': {'lot_size': 250, 'zone_size': 40, 'atm_range': 200},
        'HDFCBANK': {'lot_size': 550, 'zone_size': 50, 'atm_range': 250}
    }
}

# Initialize session states for all NSE instruments
for category in NSE_INSTRUMENTS:
    for instrument in NSE_INSTRUMENTS[category]:
        if f'{instrument}_price_data' not in st.session_state:
            st.session_state[f'{instrument}_price_data'] = pd.DataFrame(columns=["Time", "Spot"])

        if f'{instrument}_trade_log' not in st.session_state:
            st.session_state[f'{instrument}_trade_log'] = []

        if f'{instrument}_call_log_book' not in st.session_state:
            st.session_state[f'{instrument}_call_log_book'] = []

        if f'{instrument}_support_zone' not in st.session_state:
            st.session_state[f'{instrument}_support_zone'] = (None, None)

        if f'{instrument}_resistance_zone' not in st.session_state:
            st.session_state[f'{instrument}_resistance_zone'] = (None, None)

# Initialize overall option chain data
if 'overall_option_data' not in st.session_state:
    st.session_state['overall_option_data'] = {}

# ═══════════════════════════════════════════════════════════════════════
# AUTO REFRESH & PERFORMANCE OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════════════
# Optimized for fast loading and refresh:
# - Chart data cached for 60 seconds
# - Signal checks reduced to 30-second intervals
# - Lazy loading for tab-specific data
# - Streamlit caching for expensive computations

# Auto-refresh every 1 minute (configurable via AUTO_REFRESH_INTERVAL)
# This ensures the app stays updated with latest market data
# The refresh is seamless - no blur/flash thanks to custom CSS above
refresh_count = st_autorefresh(interval=AUTO_REFRESH_INTERVAL * 1000, key="data_refresh")

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Check and run AI analysis if needed
check_and_run_ai_analysis()

# ═══════════════════════════════════════════════════════════════════════
# MARKET HOURS WARNING BANNER
# ═══════════════════════════════════════════════════════════════════════

if MARKET_HOURS_ENABLED:
    should_run, reason = should_run_app()

    if not should_run:
        # Display prominent warning banner when market is closed
        st.error(f"""
        ⚠️ **MARKET CLOSED - APP RUNNING IN LIMITED MODE**

        **Reason:** {reason}

        **Trading Hours:** 8:30 AM - 3:45 PM IST (Monday - Friday, excluding holidays)

        The app will automatically resume full operation during market hours.
        Background data refresh is paused to conserve API quota.
        """)

        # Show next market open time if available
        market_status = get_market_status()
        if 'next_open' in market_status:
            st.info(f"📅 **Next Market Open:** {market_status['next_open']}")
    else:
        # Show market session info when market is open
        market_status = get_market_status()
        session = market_status.get('session', 'unknown')

        if session == 'pre_market':
            st.info(f"⏰ **{reason}** - Limited liquidity expected")
        elif session == 'post_market':
            st.warning(f"⏰ **{reason}** - Trading session ending soon")
        # Don't show banner during regular market hours to save space

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR - SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ System Status")
    
    # Market status
    market_status = get_market_status()
    if market_status['open']:
        st.success(f"{market_status['message']} | {market_status['time']}")
    else:
        st.error(market_status['message'])
    
    st.divider()
    
    # DhanHQ connection
    st.subheader("🔌 DhanHQ API")
    if DEMO_MODE:
        st.info("🧪 DEMO MODE Active")
    else:
        if check_dhan_connection():
            st.success("✅ Connected")
        else:
            st.error("❌ Connection Failed")

    st.divider()

    # API Rate Limiter Status
    st.subheader("📊 API Rate Limiter")
    try:
        from api_request_limiter import global_rate_limiter
        rate_status = global_rate_limiter.get_status()

        if rate_status['healthy']:
            st.success(f"✅ Healthy | {rate_status['total_requests']} requests")
        else:
            st.error(f"⚠️ {rate_status['message']}")
            for cb in rate_status['circuit_breakers_active']:
                st.warning(f"⏳ {cb['api_type']}: {cb['remaining_seconds']}s remaining")

            # Reset button
            if st.button("🔄 Reset Rate Limiter", key="reset_rate_limiter"):
                global_rate_limiter.reset_circuit_breaker()
                st.success("✅ Rate limiter reset!")
                st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Status unavailable: {e}")

    st.divider()

    # Telegram status
    st.subheader("📱 Telegram Alerts")
    telegram_creds = get_telegram_credentials()
    if telegram_creds['enabled']:
        st.success("✅ Connected")
        if st.button("Send Test Message"):
            if send_test_message():
                st.success("Test message sent!")
            else:
                st.error("Failed to send")
    else:
        st.warning("⚠️ Not Configured")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    st.write(f"**Auto Refresh:** {AUTO_REFRESH_INTERVAL}s")
    st.write(f"**NIFTY Lot Size:** {LOT_SIZES['NIFTY']}")
    st.write(f"**SENSEX Lot Size:** {LOT_SIZES['SENSEX']}")
    st.write(f"**SL Offset:** {STOP_LOSS_OFFSET} points")
    
    st.divider()
    
    # AI Analysis Status
    st.subheader("🤖 AI Market Analysis (Powered by Perplexity AI)")
    if NEWSDATA_API_KEY and PERPLEXITY_API_KEY:
        st.success("✅ API Keys Configured (Perplexity AI)")
        
        # Show last AI analysis time
        if st.session_state.last_ai_analysis_time > 0:
            last_time = datetime.fromtimestamp(st.session_state.last_ai_analysis_time)
            time_str = last_time.strftime("%H:%M:%S")
            time_ago = int(time.time() - st.session_state.last_ai_analysis_time)
            
            if time_ago < 60:
                st.info(f"Last analysis: {time_ago}s ago")
            elif time_ago < 3600:
                st.info(f"Last analysis: {time_ago//60}m ago")
            else:
                st.info(f"Last analysis: {time_ago//3600}h ago")
        else:
            st.info("⏳ Never run")
        
        # Manual trigger button
        if st.button("Run AI Analysis Now", key="run_ai_analysis"):
            with st.spinner("Running AI market analysis..."):
                try:
                    # Run AI analysis
                    async def run_ai():
                        report = await run_ai_market_analysis()
                        if report:
                            st.session_state.ai_analysis_results = report
                            st.session_state.last_ai_analysis_time = time.time()
                            st.success(f"✅ AI Analysis Complete: {report.get('label')}")
                        else:
                            st.warning("⚠️ AI analysis not triggered (not enough signals)")
                    
                    # Run async in current event loop
                    import asyncio
                    asyncio.run(run_ai())
                except Exception as e:
                    st.error(f"❌ AI analysis failed: {e}")
    else:
        st.warning("⚠️ API Keys Required")
        st.caption("Set NEWSDATA_API_KEY and PERPLEXITY_API_KEY in Streamlit secrets (.streamlit/secrets.toml)")
    
    st.divider()

    # Background Data Loading Status
    st.subheader("🔄 Data Loading")
    cache_manager = get_cache_manager()

    # Check cache status for each data type
    data_status = {
        'Market Data': cache_manager.is_valid('nifty_data'),
        'Bias Analysis': cache_manager.is_valid('bias_analysis'),
    }

    for name, is_valid in data_status.items():
        if is_valid:
            st.success(f"✅ {name}")
        else:
            st.info(f"⏳ {name} Loading...")

    st.caption("🔄 Auto-refreshing every 60-120 seconds (optimized for performance)")

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════

# Get NIFTY data from cache (loaded in background)
# PERFORMANCE OPTIMIZED: Never blocks - shows loading message if data not ready
nifty_data = get_cached_nifty_data()

if not nifty_data or not nifty_data.get('success'):
    # Show loading message (non-blocking)
    # Background thread will load data and it will appear on next refresh
    st.info("⏳ Loading NIFTY data in background... Please wait a moment and the data will appear automatically.")

    # Check if there's an error message to display
    if nifty_data and nifty_data.get('error'):
        st.error(f"⚠️ **Error:** {nifty_data['error']}")

        # Show help message if it's a credentials error
        if 'credentials' in nifty_data['error'].lower() or 'secrets.toml' in nifty_data['error'].lower():
            st.warning("""
            **Setup Required:**
            1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
            2. Fill in your DhanHQ API credentials
            3. Restart the application
            """)
    else:
        st.info("💡 **Performance Note:** Tab and button clicks should be instant now. Data loads in background without blocking UI.")

    # Use a placeholder/default data structure to prevent errors
    nifty_data = {
        'success': False,
        'spot_price': None,
        'atm_strike': None,
        'open': None,
        'high': None,
        'low': None,
        'close': None,
        'expiry_dates': [],
        'current_expiry': 'N/A',
        'chart_data': None,
        'error': nifty_data.get('error', 'Loading...') if nifty_data else 'Loading...',
        'timestamp': None
    }
    # Note: We don't stop() here - let the app continue and show what it can

# Store spot price in session state for browser tab title (used at top of script)
if nifty_data and nifty_data.get('success') and nifty_data.get('spot_price'):
    st.session_state.last_spot_price = nifty_data['spot_price']
    # Update browser tab title with current spot price using JavaScript
    spot_price = nifty_data['spot_price']
    st.markdown(f"""
    <script>
        // Update browser tab title to always show spot price
        document.title = "NIFTY ₹{spot_price:,.2f} | Trader";
    </script>
    """, unsafe_allow_html=True)
else:
    # Use default title if spot price unavailable
    st.markdown("""
    <script>
        document.title = "NIFTY/SENSEX Trader";
    </script>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CACHED CHART DATA FETCHER (Performance Optimization)
# ═══════════════════════════════════════════════════════════════════════
# Cache chart data for 60 seconds to avoid repeated API calls
if 'chart_data_cache' not in st.session_state:
    st.session_state.chart_data_cache = {}
if 'chart_data_cache_time' not in st.session_state:
    st.session_state.chart_data_cache_time = {}

@st.cache_data(ttl=60, show_spinner=False)  # 60 seconds (1 minute)
def get_cached_chart_data(symbol, period, interval):
    """Cached chart data fetcher - reduces API calls"""
    chart_analyzer = AdvancedChartAnalysis()
    return chart_analyzer.fetch_intraday_data(symbol, period=period, interval=interval)

@st.cache_data(ttl=60, show_spinner=False)  # 60 seconds (1 minute)
def calculate_sentiment():
    """Cached sentiment calculation"""
    try:
        return calculate_overall_sentiment()
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════
# SENTIMENT CACHE (Optimized)
# ═══════════════════════════════════════════════════════════════════════
# Initialize sentiment cache in session state
if 'cached_sentiment' not in st.session_state:
    st.session_state.cached_sentiment = None
if 'sentiment_cache_time' not in st.session_state:
    st.session_state.sentiment_cache_time = 0

# Calculate sentiment once every 120 seconds and cache it (increased from 60s for better performance)
current_time = time.time()
if current_time - st.session_state.sentiment_cache_time > 120:
    sentiment_result = calculate_sentiment()
    if sentiment_result:
        st.session_state.cached_sentiment = sentiment_result
        st.session_state.sentiment_cache_time = current_time

# ═══════════════════════════════════════════════════════════════════════
# MARKET STATUS
# ═══════════════════════════════════════════════════════════════════════
market_status = get_market_status()

# Display market data - Use session state fallbacks from NIFTY Option Screener
col1, col2, col3, col4 = st.columns(4)

# Get fallback values from session state (set by NIFTY Option Screener)
fallback_spot = None
fallback_atm = None
fallback_expiry = None

# Check nifty_spot directly from screener
if 'nifty_spot' in st.session_state and st.session_state['nifty_spot']:
    fallback_spot = st.session_state['nifty_spot']

# Check overall_option_data for additional fallback
if 'overall_option_data' in st.session_state:
    option_data = st.session_state.get('overall_option_data', {}).get('NIFTY', {})
    if option_data.get('spot') and not fallback_spot:
        fallback_spot = option_data['spot']
    if option_data.get('atm_strike'):
        fallback_atm = option_data['atm_strike']

if 'atm_strike' in st.session_state and st.session_state['atm_strike']:
    fallback_atm = st.session_state['atm_strike']

if 'current_expiry' in st.session_state and st.session_state.get('current_expiry'):
    fallback_expiry = st.session_state['current_expiry']

# Determine actual values to display (prefer nifty_data, fallback to session state)
display_spot = nifty_data.get('spot_price') if nifty_data.get('spot_price') not in [None, 0] else fallback_spot
display_atm = nifty_data.get('atm_strike') if nifty_data.get('atm_strike') not in [None, 0] else fallback_atm
display_expiry = nifty_data.get('current_expiry') if nifty_data.get('current_expiry') not in [None, 'N/A'] else fallback_expiry

with col1:
    if display_spot is not None and display_spot != 0:
        st.metric(
            "NIFTY Spot",
            f"₹{display_spot:,.2f}",
            delta=None
        )
    else:
        st.metric(
            "NIFTY Spot",
            "N/A",
            delta=None
        )
        if nifty_data.get('error'):
            st.error(f"⚠️ {nifty_data['error']}")

with col2:
    if display_atm is not None and display_atm != 0:
        st.metric(
            "ATM Strike",
            f"{display_atm}"
        )
    else:
        st.metric(
            "ATM Strike",
            "N/A"
        )

with col3:
    st.metric(
        "Current Expiry",
        display_expiry if display_expiry else 'N/A'
    )

with col4:
    # Show data freshness status
    cache_manager = get_cache_manager()
    nifty_cache_time = cache_manager._cache_timestamps.get('nifty_data', 0)
    if nifty_cache_time > 0:
        age_seconds = int(time.time() - nifty_cache_time)
        if age_seconds < 15:
            st.success(f"🟢 Live ({age_seconds}s ago)")
        elif age_seconds < 60:
            st.info(f"🔵 Fresh ({age_seconds}s ago)")
        else:
            st.warning(f"🟡 Updating...")
    else:
        st.info("📅 Loading...")

st.divider()

# VOB/HTF sections removed for performance - app loads faster now


# ═══════════════════════════════════════════════════════════════════════
# AI MARKET ANALYSIS RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════

# Display AI analysis results if available
if st.session_state.ai_analysis_results:
    report = st.session_state.ai_analysis_results
    
    st.markdown("### 🤖 AI Market Analysis")
    
    # Determine color based on label
    label = report.get('label', 'NEUTRAL')
    if label == 'BULL':
        label_color = "#4caf50"
        label_emoji = "🐂"
    elif label == 'BEAR':
        label_color = "#f44336"
        label_emoji = "🐻"
    else:
        label_color = "#ff9800"
        label_emoji = "⚖️"
    
    # Confidence level
    confidence = report.get('confidence', 0)
    if confidence >= 80:
        confidence_color = "#4caf50"
        confidence_text = "High"
    elif confidence >= 60:
        confidence_color = "#ff9800"
        confidence_text = "Moderate"
    else:
        confidence_color = "#f44336"
        confidence_text = "Low"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<h2 style='color:{label_color}; text-align: center;'>{label_emoji} {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Market Direction</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h2 style='color:{confidence_color}; text-align: center;'>{confidence:.0f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence ({confidence_text})</p>", unsafe_allow_html=True)
    
    with col3:
        # Show report age
        if 'timestamp' in report:
            try:
                report_time = datetime.fromisoformat(str(report['timestamp']).replace('Z', '+00:00'))
                now = datetime.now(report_time.tzinfo) if report_time.tzinfo else datetime.now()
                age_minutes = int((now - report_time).total_seconds() / 60)
                
                if age_minutes < 1:
                    age_text = "Just now"
                elif age_minutes == 1:
                    age_text = "1 minute ago"
                else:
                    age_text = f"{age_minutes} minutes ago"
                
                st.markdown(f"<h4 style='text-align: center;'>{age_text}</h4>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Last Updated</p>", unsafe_allow_html=True)
            except:
                st.markdown("<p style='text-align: center;'>Time unknown</p>", unsafe_allow_html=True)
    
    # Recommendation
    st.markdown("#### 💡 AI Recommendation")
    recommendation = report.get('recommendation', 'No specific recommendation')
    st.info(recommendation)
    
    # Key Findings
    if 'key_findings' in report and report['key_findings']:
        with st.expander("📊 Key Findings", expanded=False):
            for finding in report['key_findings'][:5]:  # Show top 5 findings
                st.write(f"• {finding}")
    
    # Risk Assessment
    if 'risk_assessment' in report:
        with st.expander("⚠️ Risk Assessment", expanded=False):
            risk = report['risk_assessment']
            st.write(f"**Level:** {risk.get('level', 'Medium')}")
            st.write(f"**Score:** {risk.get('score', 50)}/100")
            if 'factors' in risk:
                st.write("**Factors:**")
                for factor in risk.get('factors', []):
                    st.write(f"  - {factor}")
    
    # View full report button
    if st.button("📄 View Full AI Report", key="view_full_ai_report"):
        # Show the full report in a modal or expander
        with st.expander("Full AI Market Analysis Report", expanded=True):
            st.json(report)
    
    st.divider()

# ═══════════════════════════════════════════════════════════════════════
# PRE-LOAD OPTION SCREENER DATA (for SL Hunt Detector and other ML modules)
# ═══════════════════════════════════════════════════════════════════════
# Load option screener data silently if not already loaded
if st.session_state.get('merged_df') is None:
    try:
        from NiftyOptionScreener import load_option_screener_data_silently
        load_option_screener_data_silently()
    except Exception as e:
        pass  # Silently fail - will retry in tab

# ═══════════════════════════════════════════════════════════════════════
# TABS - USING NATIVE STREAMLIT TABS FOR BETTER UX
# ═══════════════════════════════════════════════════════════════════════

# Native tabs - work seamlessly on mobile and desktop, no multiple clicks needed
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🌟 Overall Market Sentiment",
    "🎯 Trade Setup",
    "📊 Active Signals & Structure",
    "📈 Positions",
    "🎲 Bias Analysis Pro",
    "📉 Advanced Chart Analysis",
    "🎯 NIFTY Option Screener v7.0",
    "🌐 Enhanced Market Data",
    "🔍 NSE Stock Screener",
    "📈 NIFTY Futures Analysis",
    "🤖 AI Training & Models"
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: OVERALL MARKET SENTIMENT (AUTO-LOAD WITH SMART CACHING)
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    render_overall_market_sentiment(NSE_INSTRUMENTS)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: TRADE SETUP
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🎯 Create New Trade Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_index = st.selectbox(
            "Select Index",
            ["NIFTY", "SENSEX"],
            key="setup_index"
        )
    
    with col2:
        selected_direction = st.selectbox(
            "Select Direction",
            ["CALL", "PUT"],
            key="setup_direction"
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Safe default value handling for VOB support level
        try:
            default_support = max(0.0, float(nifty_data['spot_price']) - 50.0) if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A'] else 25000.0
        except (TypeError, ValueError):
            default_support = 25000.0

        vob_support = st.number_input(
            "Support Level",
            min_value=0.0,
            value=default_support,
            step=10.0,
            key="vob_support"
        )

    with col2:
        # Safe default value handling for VOB resistance level
        try:
            default_resistance = max(0.0, float(nifty_data['spot_price']) + 50.0) if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A'] else 25100.0
        except (TypeError, ValueError):
            default_resistance = 25100.0

        vob_resistance = st.number_input(
            "Resistance Level",
            min_value=0.0,
            value=default_resistance,
            step=10.0,
            key="vob_resistance"
        )
    
    st.divider()
    
    # Preview calculated levels
    st.subheader("📋 Preview Trade Levels")
    
    levels = calculate_levels(
        selected_index,
        selected_direction,
        vob_support,
        vob_resistance,
        STOP_LOSS_OFFSET
    )

    # Safe handling for spot price in strike calculation
    safe_spot_price = 25000.0  # Default fallback value
    if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A']:
        try:
            safe_spot_price = float(nifty_data['spot_price'])
        except (TypeError, ValueError):
            safe_spot_price = 25000.0

    strike_info = calculate_strike(
        selected_index,
        safe_spot_price,
        selected_direction,
        nifty_data.get('current_expiry', 'N/A')
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entry Level", f"{levels['entry_level']:.2f}")
    
    with col2:
        st.metric("Stop Loss", f"{levels['sl_level']:.2f}")
    
    with col3:
        st.metric("Target", f"{levels['target_level']:.2f}")
    
    with col4:
        st.metric("Risk:Reward", f"1:{levels['rr_ratio']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Strike:** {strike_info['strike']} {strike_info['option_type']} ({strike_info['strike_type']})")
    
    with col2:
        lot_size = LOT_SIZES[selected_index]
        st.info(f"**Quantity:** {lot_size} ({selected_index} lot size)")
    
    st.divider()
    
    # Create setup button
    if st.button("✅ Create Signal Setup", type="primary", use_container_width=True):
        signal_id = st.session_state.signal_manager.create_setup(
            selected_index,
            selected_direction,
            vob_support,
            vob_resistance
        )
        st.session_state.active_setup_id = signal_id
        st.success(f"✅ Signal setup created! ID: {signal_id[:20]}...")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: ACTIVE SIGNALS
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📊 Active Signal Setups")

    active_setups = st.session_state.signal_manager.get_active_setups()

    if not active_setups:
        st.info("No active signal setups. Create one in the Trade Setup tab.")
    else:
        for signal_id, setup in active_setups.items():
            with st.container():
                st.subheader(f"{setup['index']} {setup['direction']}")

                # Signal count display
                signal_count = setup['signal_count']
                signals_display = "⭐" * signal_count + "☆" * (3 - signal_count)

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**Signals:** {signals_display} ({signal_count}/3)")
                    st.write(f"**VOB Support:** {setup['vob_support']}")
                    st.write(f"**VOB Resistance:** {setup['vob_resistance']}")

                with col2:
                    if signal_count < 3:
                        if st.button(f"➕ Add Signal", key=f"add_{signal_id}"):
                            st.session_state.signal_manager.add_signal(signal_id)

                            # Check if ready and send Telegram
                            updated_setup = st.session_state.signal_manager.get_setup(signal_id)
                            if updated_setup['status'] == 'ready':
                                telegram = TelegramBot()
                                telegram.send_signal_ready(updated_setup)

                            st.rerun()

                    if signal_count > 0:
                        if st.button(f"➖ Remove Signal", key=f"remove_{signal_id}"):
                            st.session_state.signal_manager.remove_signal(signal_id)
                            st.rerun()

                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{signal_id}"):
                        st.session_state.signal_manager.delete_setup(signal_id)
                        st.rerun()

                # NEW FEATURE: Trade Execution Button with Index Selection
                if setup['status'] == 'ready' or signal_count >= 3:
                    st.markdown("---")
                    st.markdown("### 🚀 Execute Trade")

                    # Index Selection
                    trade_col1, trade_col2 = st.columns([1, 2])

                    with trade_col1:
                        selected_index = st.radio(
                            "Select Index to Trade:",
                            ["NIFTY", "SENSEX"],
                            key=f"index_select_{signal_id}",
                            horizontal=True
                        )

                    # Get data based on selected index
                    if selected_index == "NIFTY":
                        index_data = nifty_data
                    else:
                        index_data = get_cached_sensex_data()
                        if not index_data:
                            st.error("❌ SENSEX data not available")
                            continue

                    current_price = index_data['spot_price']

                    # Display current market status
                    with trade_col2:
                        st.metric(
                            label=f"{selected_index} Spot Price",
                            value=f"{current_price:.2f}",
                            delta=f"Support: {setup['vob_support']:.2f} | Resistance: {setup['vob_resistance']:.2f}"
                        )

                    # Calculate if signal conditions are met
                    if setup['direction'] == 'CALL':
                        vob_touched = check_vob_touch(current_price, setup['vob_support'], VOB_TOUCH_TOLERANCE)
                        vob_type = "Support"
                        vob_level = setup['vob_support']
                    else:
                        vob_touched = check_vob_touch(current_price, setup['vob_resistance'], VOB_TOUCH_TOLERANCE)
                        vob_type = "Resistance"
                        vob_level = setup['vob_resistance']

                    # Show signal status
                    if vob_touched:
                        st.success(f"✅ {vob_type} LEVEL TOUCHED! Ready to execute.")
                    else:
                        st.info(f"⏳ Waiting for {vob_type} touch | Current: {current_price:.2f} | Target: {vob_level:.2f} | Distance: {abs(current_price - vob_level):.2f} pts")

                    # Execute Button (always available when signal is ready)
                    exec_col1, exec_col2, exec_col3 = st.columns([1, 1, 1])

                    with exec_col2:
                        if st.button(
                            f"🚀 EXECUTE {selected_index} TRADE",
                            key=f"execute_{signal_id}",
                            type="primary",
                            use_container_width=True
                        ):
                            with st.spinner(f"Executing {selected_index} trade..."):
                                # Update setup with selected index
                                trade_setup = setup.copy()
                                trade_setup['index'] = selected_index

                                executor = TradeExecutor()
                                result = executor.execute_trade(
                                    trade_setup,
                                    current_price,
                                    index_data['current_expiry']
                                )

                                if result['success']:
                                    st.success(f"✅ {result['message']}")
                                    st.success(f"**Order ID:** {result['order_id']}")

                                    # Mark as executed
                                    st.session_state.signal_manager.mark_executed(signal_id, result['order_id'])

                                    # Store position in session state for monitoring
                                    if 'active_positions' not in st.session_state:
                                        st.session_state.active_positions = {}

                                    details = result['order_details']
                                    st.session_state.active_positions[result['order_id']] = {
                                        'order_id': result['order_id'],
                                        'index': selected_index,
                                        'direction': setup['direction'],
                                        'strike': details['strike'],
                                        'option_type': details['option_type'],
                                        'quantity': details['quantity'],
                                        'entry': details['entry_level'],
                                        'sl': details['sl_price'],
                                        'target': details['target_price'],
                                        'timestamp': get_current_time_ist().isoformat(),
                                        'status': 'active'
                                    }

                                    # Display order details
                                    st.write("**Order Details:**")
                                    st.write(f"- Index: {selected_index}")
                                    st.write(f"- Strike: {details['strike']} {details['option_type']} ({details['strike_type']})")
                                    st.write(f"- Quantity: {details['quantity']}")
                                    st.write(f"- Entry: {details['entry_level']:.2f}")
                                    st.write(f"- Stop Loss: {details['sl_price']:.2f} (Support/Resistance {'-' if setup['direction'] == 'CALL' else '+'} 8 points)")
                                    st.write(f"- Target: {details['target_price']:.2f} (Opposite level)")
                                    st.write(f"- R:R Ratio: 1:{details['rr_ratio']}")

                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                                    if 'error' in result:
                                        st.error(f"Error: {result['error']}")

                st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # MARKET STRUCTURE ANALYSIS SECTION (Inside Active Signals Tab)
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("## 🧠 Market Structure Analysis")
    st.caption("Structure-based detection BEFORE price moves | Probability Engine | Expiry Patterns")

    # Load Market Structure Analysis
    try:
        # Get OHLC data from multiple sources (prioritize freshest)
        ohlc_df = None
        option_data = None
        spot_price = None
        is_expiry = False
        nifty_data_struct = None  # Initialize to avoid reference errors

        # SOURCE 1: PRIMARY - st.session_state.chart_data (used by Unified ML - MOST RELIABLE)
        if 'chart_data' in st.session_state:
            chart_data = st.session_state.get('chart_data')
            if isinstance(chart_data, pd.DataFrame) and len(chart_data) > 0:
                ohlc_df = chart_data.copy()

        # SOURCE 2: data_df from Option Screener
        if ohlc_df is None and 'data_df' in st.session_state:
            chart_data = st.session_state.get('data_df')
            if isinstance(chart_data, pd.DataFrame) and len(chart_data) > 0:
                ohlc_df = chart_data.copy()

        # SOURCE 3: nifty_df alternative
        if ohlc_df is None and 'nifty_df' in st.session_state:
            chart_data = st.session_state.get('nifty_df')
            if isinstance(chart_data, pd.DataFrame) and len(chart_data) > 0:
                ohlc_df = chart_data.copy()

        # SOURCE 4: Try data cache manager
        if ohlc_df is None:
            nifty_data_struct = get_cached_nifty_data()
            if nifty_data_struct and 'chart_data' in nifty_data_struct:
                chart_data = nifty_data_struct['chart_data']
                if isinstance(chart_data, pd.DataFrame) and len(chart_data) > 0:
                    ohlc_df = chart_data

        # Get spot price from multiple sources
        if 'nifty_spot' in st.session_state and st.session_state['nifty_spot']:
            spot_price = st.session_state['nifty_spot']
        elif 'last_spot_price' in st.session_state and st.session_state.get('last_spot_price'):
            spot_price = st.session_state.last_spot_price
        elif nifty_data_struct and 'spot_price' in nifty_data_struct:
            spot_price = nifty_data_struct['spot_price']
        elif ohlc_df is not None and len(ohlc_df) > 0 and 'close' in ohlc_df.columns:
            spot_price = float(ohlc_df['close'].iloc[-1])

        # Get option data from multiple sources
        if 'merged_df' in st.session_state:
            merged = st.session_state.get('merged_df')
            if isinstance(merged, pd.DataFrame) and len(merged) > 0:
                option_data = {'merged_df': merged, 'source': 'option_screener'}

        if option_data is None and 'overall_option_data' in st.session_state:
            screener_data = st.session_state.get('overall_option_data', {}).get('NIFTY', {})
            if screener_data:
                option_data = screener_data

        if option_data is None:
            option_data = {}

        # Add extras
        if 'atm_strike' in st.session_state:
            option_data['atm_strike'] = st.session_state['atm_strike']
        if 'market_depth_data' in st.session_state:
            option_data['market_depth'] = st.session_state['market_depth_data']

        # Check expiry day
        today = datetime.now()
        is_expiry = today.weekday() == 3  # Thursday

        if ohlc_df is not None and len(ohlc_df) > 20:
            render_market_structure_section(
                ohlc_df=ohlc_df,
                option_data=option_data,
                is_expiry=is_expiry,
                spot_price=spot_price
            )
        else:
            st.info("📊 Load chart data from **Overall Market Sentiment** or **NIFTY Option Screener** tab to see Market Structure Analysis.")

    except Exception as e:
        st.warning(f"Market Structure module loading: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: POSITIONS
# ═══════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📈 Active Positions & Monitoring")

    # Initialize active_positions in session state if not exists
    if 'active_positions' not in st.session_state:
        st.session_state.active_positions = {}

    # Get current spot prices for monitoring - with safe null handling
    nifty_spot = 0
    if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 'N/A']:
        try:
            nifty_spot = float(nifty_data['spot_price'])
        except (TypeError, ValueError):
            nifty_spot = 0

    sensex_data_obj = get_cached_sensex_data()
    sensex_spot = 0
    if sensex_data_obj and sensex_data_obj.get('spot_price') and sensex_data_obj['spot_price'] not in [None, 'N/A']:
        try:
            sensex_spot = float(sensex_data_obj['spot_price'])
        except (TypeError, ValueError):
            sensex_spot = 0

    # Check for auto-exit conditions
    positions_to_exit = []
    for order_id, pos in st.session_state.active_positions.items():
        if pos['status'] != 'active':
            continue

        # Get current spot price based on index
        current_spot = nifty_spot if pos['index'] == 'NIFTY' else sensex_spot

        # Check if target or SL is hit
        if pos['direction'] == 'CALL':
            # For CALL: Target is resistance, SL is support - 8
            target_hit = current_spot >= pos['target']
            sl_hit = current_spot <= pos['sl']
        else:  # PUT
            # For PUT: Target is support, SL is resistance + 8
            target_hit = current_spot <= pos['target']
            sl_hit = current_spot >= pos['sl']

        if target_hit:
            st.success(f"🎯 TARGET HIT for {pos['index']} {pos['direction']} | Order ID: {order_id}")
            positions_to_exit.append((order_id, 'target'))
        elif sl_hit:
            st.error(f"🛑 STOP LOSS HIT for {pos['index']} {pos['direction']} | Order ID: {order_id}")
            positions_to_exit.append((order_id, 'stoploss'))

    # Display tracked positions
    if not st.session_state.active_positions:
        st.info("No active positions tracked. Execute a trade from the Active Signals tab.")
    else:
        active_count = sum(1 for p in st.session_state.active_positions.values() if p['status'] == 'active')
        st.info(f"📊 Tracking {active_count} active position(s)")

        for order_id, pos in st.session_state.active_positions.items():
            if pos['status'] != 'active':
                continue

            with st.container():
                # Position header
                st.subheader(f"{pos['index']} {pos['direction']} - {pos['option_type']}")

                # Get current spot price
                current_spot = nifty_spot if pos['index'] == 'NIFTY' else sensex_spot

                # Calculate distance to target and SL
                if pos['direction'] == 'CALL':
                    dist_to_target = pos['target'] - current_spot
                    dist_to_sl = current_spot - pos['sl']
                else:
                    dist_to_target = current_spot - pos['target']
                    dist_to_sl = pos['sl'] - current_spot

                # Create columns for display
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.metric(
                        label="Current Spot",
                        value=f"{current_spot:.2f}",
                        delta=f"Entry: {pos['entry']:.2f}"
                    )
                    st.write(f"**Strike:** {pos['strike']} {pos['option_type']}")
                    st.write(f"**Quantity:** {pos['quantity']}")

                with col2:
                    target_color = "green" if dist_to_target >= 0 else "orange"
                    st.metric(
                        label="🎯 Target",
                        value=f"{pos['target']:.2f}",
                        delta=f"{dist_to_target:+.2f} pts away"
                    )
                    if pos['direction'] == 'CALL':
                        st.caption("Trigger: When spot reaches resistance (VOB/HTF)")
                    else:
                        st.caption("Trigger: When spot reaches support (VOB/HTF)")

                with col3:
                    sl_color = "red" if dist_to_sl <= 0 else "green"
                    st.metric(
                        label="🛑 Stop Loss",
                        value=f"{pos['sl']:.2f}",
                        delta=f"{dist_to_sl:+.2f} pts away"
                    )
                    if pos['direction'] == 'CALL':
                        st.caption("Support - 8 points")
                    else:
                        st.caption("Resistance + 8 points")

                with col4:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("❌ Exit", key=f"exit_{order_id}", type="secondary"):
                        # Exit position
                        if DEMO_MODE:
                            st.session_state.active_positions[order_id]['status'] = 'closed'
                            st.success("✅ Position exited (DEMO MODE)")
                            time.sleep(1)
                            st.rerun()
                        else:
                            from dhan_api import DhanAPI
                            dhan = DhanAPI()
                            result = dhan.exit_position(order_id)
                            if result['success']:
                                st.session_state.active_positions[order_id]['status'] = 'closed'
                                st.success("✅ Position exited successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Exit failed: {result.get('error', 'Unknown error')}")

                # Progress bar for position
                entry_to_target = abs(pos['target'] - pos['entry'])
                current_progress = abs(current_spot - pos['entry'])
                progress = min(current_progress / entry_to_target, 1.0) if entry_to_target > 0 else 0

                st.progress(progress, text=f"Progress to Target: {progress*100:.1f}%")

                # Status indicators
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if dist_to_target <= 5:
                        st.warning(f"⚠️ Near Target ({dist_to_target:.2f} pts)")
                with status_col2:
                    if dist_to_sl <= 5:
                        st.error(f"⚠️ Near Stop Loss ({dist_to_sl:.2f} pts)")

                st.divider()

    # Auto-exit positions
    if positions_to_exit:
        for order_id, exit_reason in positions_to_exit:
            st.info(f"Auto-exiting position {order_id} due to {exit_reason}...")

            if not DEMO_MODE:
                from dhan_api import DhanAPI
                dhan = DhanAPI()
                result = dhan.exit_position(order_id)

                if result['success']:
                    st.session_state.active_positions[order_id]['status'] = 'closed'
                    st.success(f"✅ Position auto-exited: {exit_reason}")
                else:
                    st.error(f"❌ Auto-exit failed: {result.get('error')}")
            else:
                st.session_state.active_positions[order_id]['status'] = 'closed'
                st.success(f"✅ Position auto-exited (DEMO): {exit_reason}")

        time.sleep(2)
        st.rerun()

    # Show API positions (if not DEMO_MODE)
    if not DEMO_MODE:
        st.markdown("---")
        st.subheader("📡 Live Positions from Dhan API")

        from dhan_api import DhanAPI
        try:
            dhan = DhanAPI()
            positions_result = dhan.get_positions()

            if positions_result['success']:
                positions = positions_result['positions']

                if not positions:
                    st.info("No live positions from API")
                else:
                    for idx, pos in enumerate(positions):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])

                            with col1:
                                st.write(f"**Symbol:** {pos.get('tradingSymbol', 'N/A')}")
                                st.write(f"**Quantity:** {pos.get('netQty', 0)}")

                            with col2:
                                pnl = pos.get('unrealizedProfit', 0)
                                pnl_color = "green" if pnl > 0 else "red"
                                st.markdown(f"**P&L:** <span style='color:{pnl_color}'>₹{pnl:,.2f}</span>", unsafe_allow_html=True)

                            with col3:
                                # Use index and trading symbol for unique key to avoid duplicate key errors
                                unique_key = f"exit_api_{idx}_{pos.get('tradingSymbol', 'pos')}"
                                if st.button("❌ Exit", key=unique_key):
                                    result = dhan.exit_position(pos.get('orderId'))
                                    if result['success']:
                                        st.success("Position exited!")
                                        st.rerun()
                                    else:
                                        st.error("Exit failed")

                            st.divider()
            else:
                st.error(f"Failed to fetch positions: {positions_result.get('error')}")

        except Exception as e:
            st.error(f"Error fetching API positions: {e}")
    else:
        st.info("🧪 DEMO MODE - Connect to Dhan API to see live positions")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: BIAS ANALYSIS PRO
# ═══════════════════════════════════════════════════════════════════════

with tab5:
    st.header("🎯 Comprehensive Bias Analysis Pro")
    st.caption("13 Bias Indicators with Adaptive Weighted Scoring | 🔄 Auto-refreshing every 60 seconds")

    # Auto-load cached results if not already in session state
    if not st.session_state.bias_analysis_results:
        cached_results = get_cached_bias_analysis_results()
        if cached_results:
            st.session_state.bias_analysis_results = cached_results

    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        bias_symbol = st.selectbox(
            "Select Market for Bias Analysis",
            ["^NSEI (NIFTY 50)", "^BSESN (SENSEX)", "^DJI (DOW JONES)"],
            key="bias_analysis_symbol"
        )
        symbol_code = bias_symbol.split()[0]

    with col2:
        if st.button("🔍 Analyze All Bias", type="primary", use_container_width=True):
            with st.spinner("Analyzing bias indicators..."):
                try:
                    bias_analyzer = get_bias_analyzer()
                    results = bias_analyzer.analyze_all_bias_indicators(symbol_code)
                    st.session_state.bias_analysis_results = results
                    # Update cache
                    cache_manager = get_cache_manager()
                    cache_manager.set('bias_analysis', results)
                    if results['success']:
                        st.success("✅ Bias analysis completed!")
                    else:
                        st.error(f"❌ Analysis failed: {results.get('error')}")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

    with col3:
        if st.session_state.bias_analysis_results:
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.bias_analysis_results = None
                st.rerun()

    st.divider()

    # Display results if available
    if st.session_state.bias_analysis_results and st.session_state.bias_analysis_results.get('success'):
        results = st.session_state.bias_analysis_results

        # =====================================================================
        # OVERALL BIAS SUMMARY
        # =====================================================================
        st.subheader("📊 Overall Market Bias")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Current Price",
                f"₹{results['current_price']:,.2f}"
            )

        with col2:
            overall_bias = results['overall_bias']
            bias_emoji = "🐂" if overall_bias == "BULLISH" else "🐻" if overall_bias == "BEARISH" else "⚖️"
            bias_color = "green" if overall_bias == "BULLISH" else "red" if overall_bias == "BEARISH" else "gray"

            st.markdown(f"<h3 style='color:{bias_color};'>{bias_emoji} {overall_bias}</h3>",
                       unsafe_allow_html=True)
            st.caption("Overall Market Bias")

        with col3:
            score = results['overall_score']
            score_color = "green" if score > 0 else "red" if score < 0 else "gray"
            st.markdown(f"<h3 style='color:{score_color};'>{score:.1f}</h3>",
                       unsafe_allow_html=True)
            st.caption("Overall Score")

        with col4:
            confidence = results['overall_confidence']
            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

        with col5:
            st.metric(
                "Total Indicators",
                results['total_indicators']
            )

        # Bias distribution
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🐂 Bullish Signals", results['bullish_count'])

        with col2:
            st.metric("🐻 Bearish Signals", results['bearish_count'])

        with col3:
            st.metric("⚖️ Neutral Signals", results['neutral_count'])

        st.divider()

        # =====================================================================
        # DETAILED BIAS BREAKDOWN TABLE
        # =====================================================================
        st.subheader("📋 Detailed Bias Breakdown")

        # Convert bias results to DataFrame
        bias_df = pd.DataFrame(results['bias_results'])

        # Function to color code bias
        def color_bias(val):
            if 'BULLISH' in str(val):
                return 'background-color: #26a69a; color: white;'
            elif 'BEARISH' in str(val):
                return 'background-color: #ef5350; color: white;'
            else:
                return 'background-color: #78909c; color: white;'

        # Function to color code scores
        def color_score(val):
            try:
                score = float(val)
                if score > 50:
                    return 'background-color: #1b5e20; color: white; font-weight: bold;'
                elif score > 0:
                    return 'background-color: #4caf50; color: white;'
                elif score < -50:
                    return 'background-color: #b71c1c; color: white; font-weight: bold;'
                elif score < 0:
                    return 'background-color: #f44336; color: white;'
                else:
                    return 'background-color: #616161; color: white;'
            except:
                return ''

        # Create styled dataframe
        styled_df = bias_df.style.applymap(color_bias, subset=['bias']) \
                                 .applymap(color_score, subset=['score']) \
                                 .format({'score': '{:.2f}', 'weight': '{:.1f}'})

        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)

        st.divider()

        # =====================================================================
        # VISUAL SCORE REPRESENTATION
        # =====================================================================
        st.subheader("📊 Visual Bias Representation")

        # Create a chart showing each indicator's contribution
        chart_data = pd.DataFrame({
            'Indicator': [b['indicator'] for b in results['bias_results']],
            'Weighted Score': [b['score'] * b['weight'] for b in results['bias_results']]
        })

        # Sort by weighted score
        chart_data = chart_data.sort_values('Weighted Score', ascending=True)

        # Display bar chart
        st.bar_chart(chart_data.set_index('Indicator'))

        st.divider()

        # =====================================================================
        # BIAS CATEGORY BREAKDOWN
        # =====================================================================
        st.subheader("📈 Bias by Category")

        # Display mode info
        if 'mode' in results:
            mode_color = "🔄" if results['mode'] == "REVERSAL" else "📊"
            st.info(f"{mode_color} **Mode:** {results['mode']} | Fast: {results.get('fast_bull_pct', 0):.0f}% Bull | Slow: {results.get('slow_bull_pct', 0):.0f}% Bull")

        col1, col2, col3 = st.columns(3)

        # Optimize: Create DataFrame once and filter using vectorized operations
        all_bias_df = pd.DataFrame(results['bias_results'])

        if not all_bias_df.empty:
            # Pre-compute bullish/bearish flags for all rows
            is_bullish = all_bias_df['bias'].str.contains('BULLISH', na=False)
            is_bearish = all_bias_df['bias'].str.contains('BEARISH', na=False)

            with col1:
                st.markdown("**⚡ Fast Indicators (8)**")
                fast_df = all_bias_df[all_bias_df['category'] == 'fast']
                if not fast_df.empty:
                    fast_bull = is_bullish[fast_df.index].sum()
                    fast_bear = is_bearish[fast_df.index].sum()
                    fast_neutral = len(fast_df) - fast_bull - fast_bear

                    st.write(f"🐂 {fast_bull} | 🐻 {fast_bear} | ⚖️ {fast_neutral}")
                    st.dataframe(fast_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**📊 Medium Indicators (0)**")
                med_df = all_bias_df[all_bias_df['category'] == 'medium']
                if not med_df.empty:
                    med_bull = is_bullish[med_df.index].sum()
                    med_bear = is_bearish[med_df.index].sum()
                    med_neutral = len(med_df) - med_bull - med_bear

                    st.write(f"🐂 {med_bull} | 🐻 {med_bear} | ⚖️ {med_neutral}")
                    st.dataframe(med_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)
                else:
                    st.info("No medium indicators configured")

            with col3:
                st.markdown("**🐢 Slow Indicators (0)**")
                slow_df = all_bias_df[all_bias_df['category'] == 'slow']
                if not slow_df.empty:
                    slow_bull = is_bullish[slow_df.index].sum()
                    slow_bear = is_bearish[slow_df.index].sum()
                    slow_neutral = len(slow_df) - slow_bull - slow_bear

                    st.write(f"🐂 {slow_bull} | 🐻 {slow_bear} | ⚖️ {slow_neutral}")
                    st.dataframe(slow_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)
                else:
                    st.info("No slow indicators configured")

        st.divider()

        # =====================================================================
        # TRADING RECOMMENDATION
        # =====================================================================
        st.subheader("💡 Trading Recommendation")

        overall_bias = results['overall_bias']
        overall_score = results['overall_score']
        confidence = results['overall_confidence']

        if overall_bias == "BULLISH" and confidence > 70:
            st.success("### 🐂 STRONG BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for LONG entries on dips
            - ✅ Wait for support levels or VOB support touch
            - ✅ Set stop loss below recent swing low
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        elif overall_bias == "BULLISH" and confidence >= 50:
            st.success("### 🐂 MODERATE BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider LONG entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at resistance levels
            - ⚠️ Monitor for trend confirmation
            """)
        elif overall_bias == "BEARISH" and confidence > 70:
            st.error("### 🐻 STRONG BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for SHORT entries on rallies
            - ✅ Wait for resistance levels or VOB resistance touch
            - ✅ Set stop loss above recent swing high
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        elif overall_bias == "BEARISH" and confidence >= 50:
            st.error("### 🐻 MODERATE BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider SHORT entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at support levels
            - ⚠️ Monitor for trend reversal
            """)
        else:
            st.warning("### ⚖️ NEUTRAL / NO CLEAR SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - 🔄 Stay out of the market or use range trading
            - 🔄 Wait for clearer bias formation
            - 🔄 Monitor key support/resistance levels
            - 🔄 Reduce position sizes if trading
            """)

        # Key levels for entry
        st.divider()
        st.subheader("🎯 Key Considerations")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Bullish Signals Count**")
            st.markdown(f"<h2 style='color:green;'>{results['bullish_count']}/{results['total_indicators']}</h2>",
                       unsafe_allow_html=True)

        with col2:
            st.markdown("**Bearish Signals Count**")
            st.markdown(f"<h2 style='color:red;'>{results['bearish_count']}/{results['total_indicators']}</h2>",
                       unsafe_allow_html=True)

        with col3:
            st.markdown("**Confidence Level**")
            confidence_color = "green" if confidence > 70 else "orange" if confidence > 50 else "red"
            st.markdown(f"<h2 style='color:{confidence_color};'>{confidence:.1f}%</h2>",
                       unsafe_allow_html=True)

        # Timestamp
        st.caption(f"Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        st.info("👆 Click 'Analyze All Bias' button to start comprehensive bias analysis")

        st.markdown("""
        ### About Bias Analysis Pro

        This comprehensive bias analyzer evaluates **13 bias indicators** matching Pine Script EXACTLY:

        #### ⚡ Fast Indicators (8)
        - **Volume Delta** (Up Vol - Down Vol)
        - **HVP** (High Volume Pivots)
        - **VOB** (Volume Order Blocks)
        - **Order Blocks** (EMA 5/18 Crossover)
        - **RSI** (Relative Strength Index)
        - **DMI** (Directional Movement Index)
        - **VIDYA** (Variable Index Dynamic Average)
        - **MFI** (Money Flow Index)

        #### 📊 Medium Indicators (2)
        - **Close vs VWAP** (Price above/below VWAP)
        - **Price vs VWAP** (Position relative to VWAP)

        #### 🐢 Slow Indicators (3)
        - **Weighted Stocks (Daily)** (Top 9 NSE stocks)
        - **Weighted Stocks (15m)** (Intraday trend)
        - **Weighted Stocks (1h)** (Higher timeframe trend)

        #### 🎯 Adaptive Scoring System
        - **Normal Mode:** Fast (2x), Medium (3x), Slow (5x) weights
        - **Reversal Mode:** Fast (5x), Medium (3x), Slow (2x) weights
        - Mode switches automatically when divergence detected
        - Scores range from **-100 (Strong Bearish)** to **+100 (Strong Bullish)**
        - Overall bias requires **60%+ strength** for directional bias

        #### ✅ How to Use
        1. Select the market (NIFTY, SENSEX, or DOW)
        2. Click "Analyze All Bias" button
        3. Review comprehensive bias breakdown by category
        4. Check for REVERSAL mode warnings
        5. Use signals to inform your trading decisions

        **Note:** This tool is converted from the Pine Script "Smart Trading Dashboard - Adaptive + VOB" indicator with EXACT matching logic.
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 6: ADVANCED CHART ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab6:
    st.header("📈 Advanced Chart Analysis")
    st.caption("TradingView-style Chart with Advanced Indicators: Volume Bars, Volume Order Blocks, HTF Support/Resistance (3min, 5min, 10min, 15min levels), Volume Footprint (1D timeframe, 10 bins, Dynamic POC), Ultimate RSI, OM Indicator (Order Flow & Momentum), Advanced Price Action (BOS, CHOCH, Fibonacci, Geometric Patterns)")

    # Chart controls
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    with col1:
        chart_symbol = st.selectbox(
            "Select Market",
            ["^NSEI (NIFTY 50)", "^NSEBANK (BANK NIFTY)", "^BSESN (SENSEX)", "^DJI (DOW JONES)"],
            key="chart_symbol"
        )
        symbol_code = chart_symbol.split()[0]

    with col2:
        chart_period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo"],
            index=0,
            key="chart_period"
        )

    with col3:
        chart_interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "1h"],
            index=0,
            key="chart_interval"
        )

    with col4:
        chart_auto_refresh = st.selectbox(
            "Auto Refresh",
            ["Off", "30s", "60s", "2m", "5m"],
            index=2,
            key="chart_auto_refresh",
            help="Automatically refresh chart at selected interval"
        )

    with col5:
        if st.button("🔄 Refresh", type="primary", use_container_width=True, key="manual_refresh_chart"):
            st.session_state.chart_needs_refresh = True

    st.divider()

    # Initialize chart session state variables
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'chart_needs_refresh' not in st.session_state:
        st.session_state.chart_needs_refresh = True
    if 'last_chart_params' not in st.session_state:
        st.session_state.last_chart_params = None
    if 'last_chart_update' not in st.session_state:
        st.session_state.last_chart_update = None

    # Check if chart parameters changed
    current_params = (symbol_code, chart_period, chart_interval)
    if st.session_state.last_chart_params != current_params:
        st.session_state.chart_needs_refresh = True
        st.session_state.last_chart_params = current_params

    # Handle auto-refresh timing
    should_auto_refresh = False
    if chart_auto_refresh != "Off" and st.session_state.last_chart_update is not None:
        # Convert refresh interval to seconds
        refresh_seconds = {
            "30s": 30,
            "60s": 60,
            "2m": 120,
            "5m": 300
        }.get(chart_auto_refresh, 60)

        # Check if enough time has passed
        from datetime import timedelta
        time_since_update = (get_current_time_ist() - st.session_state.last_chart_update).total_seconds()
        if time_since_update >= refresh_seconds:
            should_auto_refresh = True
            st.session_state.chart_needs_refresh = True

    # Auto-load chart data on first load or when refresh is needed
    if st.session_state.chart_needs_refresh:
        with st.spinner("Loading chart data and calculating indicators..."):
            try:
                # Fetch data using cached function (60s cache)
                df = get_cached_chart_data(symbol_code, chart_period, chart_interval)

                if df is not None and len(df) > 0:
                    st.session_state.chart_data = df
                    st.session_state.last_chart_update = get_current_time_ist()
                    st.success(f"✅ Loaded {len(df)} candles | Last updated (IST): {st.session_state.last_chart_update.strftime('%H:%M:%S %Z')}")
                else:
                    st.error("❌ Failed to fetch data. Try a different period or interval.")
                    st.session_state.chart_data = None

            except Exception as e:
                st.error(f"❌ Error loading chart: {e}")
                st.session_state.chart_data = None

        st.session_state.chart_needs_refresh = False

    # Show auto-refresh countdown
    if chart_auto_refresh != "Off" and st.session_state.chart_data is not None and st.session_state.last_chart_update is not None:
        refresh_seconds = {
            "30s": 30,
            "60s": 60,
            "2m": 120,
            "5m": 300
        }.get(chart_auto_refresh, 60)

        time_since_update = (get_current_time_ist() - st.session_state.last_chart_update).total_seconds()
        time_until_refresh = max(0, refresh_seconds - time_since_update)

        if time_until_refresh <= 0:
            # Time for refresh
            st.session_state.chart_needs_refresh = True
            st.rerun()
        else:
            st.info(f"⏱️ Next auto-refresh in {int(time_until_refresh)} seconds (auto-refresh enabled)")

    # ═══════════════════════════════════════════════════════════════════════
    # ML MARKET REGIME ANALYSIS (NEW)
    # ═══════════════════════════════════════════════════════════════════════
    if st.session_state.chart_data is not None and len(st.session_state.chart_data) > 0:
        try:
            from src.ml_market_regime import MLMarketRegimeDetector

            df_for_ml = st.session_state.chart_data.copy()

            # Add ATR if not present
            if 'ATR' not in df_for_ml.columns:
                df_for_ml['ATR'] = df_for_ml['High'] - df_for_ml['Low']

            detector = MLMarketRegimeDetector()
            ml_result = detector.detect_regime(df_for_ml)

            # Display ML Regime in a prominent section
            st.subheader("🤖 ML Market Regime Analysis")

            col1, col2, col3, col4, col5 = st.columns(5)

            # Color-code regime
            regime_colors = {
                'Trending Up': '#00FF00',
                'TRENDING_UP': '#00FF00',
                'Trending Down': '#FF0000',
                'TRENDING_DOWN': '#FF0000',
                'Range Bound': '#FFD700',
                'RANGE_BOUND': '#FFD700',
                'Volatile Breakout': '#FF6600',
                'VOLATILE': '#FF6600',
                'Consolidation': '#808080'
            }
            regime_color = regime_colors.get(ml_result.regime, '#FFFFFF')

            with col1:
                st.markdown(f"""
                <div style="background-color: {regime_color}; padding: 10px; border-radius: 8px; text-align: center;">
                    <b style="color: black; font-size: 14px;">REGIME</b><br>
                    <span style="color: black; font-size: 16px; font-weight: bold;">{ml_result.regime}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Confidence", f"{ml_result.confidence:.1f}%")

            with col3:
                st.metric("Trend Strength", f"{ml_result.trend_strength:.1f}")

            with col4:
                st.metric("Volatility", ml_result.volatility_state)

            with col5:
                st.metric("Optimal TF", ml_result.optimal_timeframe)

            # Show strategy recommendation
            st.info(f"📊 **Strategy:** {ml_result.recommended_strategy}")

            # Cache the ML result for other tabs
            st.session_state.ml_regime_result = {
                'regime': ml_result.regime,
                'confidence': ml_result.confidence,
                'trend_strength': ml_result.trend_strength,
                'volatility_state': ml_result.volatility_state,
                'recommended_strategy': ml_result.recommended_strategy,
                'optimal_timeframe': ml_result.optimal_timeframe
            }

        except Exception as e:
            st.warning(f"⚠️ ML Regime Analysis unavailable: {str(e)[:100]}")

    st.divider()

    # Indicator toggles
    st.subheader("🔧 Indicator Settings")

    col1, col2 = st.columns(2)

    with col1:
        show_rsi = st.checkbox("📈 Ultimate RSI", value=True, key="show_rsi")
        show_volume = st.checkbox("📊 Volume Bars", value=True, key="show_volume")

    with col2:
        show_om = st.checkbox("🎯 OM Indicator", value=True, key="show_om")

    # Volume Profile Indicators
    st.markdown("**📊 Volume Profile Indicators**")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_liquidity_profile = st.checkbox("💧 Liquidity Sentiment Profile", value=False, key="show_liquidity_profile")

    with col2:
        show_money_flow_profile = st.checkbox("💰 Money Flow Profile", value=True, key="show_money_flow_profile")

    with col3:
        show_deltaflow_profile = st.checkbox("⚡ DeltaFlow Profile", value=True, key="show_deltaflow_profile")

    # Advanced Price Action Indicators
    st.markdown("**🎯 Advanced Price Action**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_bos = st.checkbox("🔺 BOS (Break of Structure)", value=True, key="show_bos")

    with col2:
        show_choch = st.checkbox("🔄 CHOCH (Change of Character)", value=True, key="show_choch")

    with col3:
        show_fibonacci = st.checkbox("📐 Fibonacci Levels", value=True, key="show_fibonacci")

    with col4:
        show_patterns = st.checkbox("📊 Geometric Patterns", value=True, key="show_patterns")

    # New Advanced Indicators (LuxAlgo & BigBeluga)
    st.markdown("**🎯 Advanced Reversal & Volume Analysis**")
    col1, col2 = st.columns(2)

    with col1:
        show_reversal_zones = st.checkbox("🎯 Reversal Probability Zones (LuxAlgo)", value=True, key="show_reversal_zones", help="Statistical reversal prediction with probability targets")

    st.divider()

    # Indicator Configuration Section
    st.subheader("⚙️ Indicator Configuration")
    st.caption("Configure each indicator's parameters below")

    # Ultimate RSI Settings
    if show_rsi:
        with st.expander("📈 Ultimate RSI Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                rsi_length = st.slider(
                    "RSI Length",
                    min_value=5,
                    max_value=30,
                    value=14,
                    step=1,
                    help="RSI calculation period",
                    key="rsi_length"
                )
                rsi_method = st.selectbox(
                    "Calculation Method",
                    options=["RMA", "EMA", "SMA", "TMA"],
                    index=0,
                    help="Moving average method for RSI",
                    key="rsi_method"
                )

            with col2:
                rsi_smooth = st.slider(
                    "Signal Smoothing",
                    min_value=5,
                    max_value=30,
                    value=14,
                    step=1,
                    help="Signal line smoothing period",
                    key="rsi_smooth"
                )
                rsi_signal_method = st.selectbox(
                    "Signal Method",
                    options=["EMA", "SMA", "RMA", "TMA"],
                    index=0,
                    help="Signal line calculation method",
                    key="rsi_signal_method"
                )

            with col3:
                rsi_ob_level = st.slider(
                    "Overbought Level",
                    min_value=70,
                    max_value=90,
                    value=80,
                    step=5,
                    help="Overbought threshold",
                    key="rsi_ob_level"
                )
                rsi_os_level = st.slider(
                    "Oversold Level",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=5,
                    help="Oversold threshold",
                    key="rsi_os_level"
                )

    # Liquidity Sentiment Profile Settings
    if show_liquidity_profile:
        with st.expander("💧 Liquidity Sentiment Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                lsp_anchor_period = st.selectbox(
                    "Anchor Period",
                    options=["Auto", "Session", "Day", "Week", "Month", "Quarter", "Year"],
                    index=0,
                    key="lsp_anchor_period"
                )
                lsp_num_rows = st.slider(
                    "Number of Rows",
                    min_value=10,
                    max_value=100,
                    value=25,
                    step=5,
                    key="lsp_num_rows"
                )

            with col2:
                lsp_profile_width = st.slider(
                    "Profile Width %",
                    min_value=10,
                    max_value=50,
                    value=50,
                    step=5,
                    key="lsp_profile_width"
                ) / 100.0
                lsp_show_liquidity = st.checkbox("Show Liquidity Profile", value=True, key="lsp_show_liquidity")
                lsp_show_sentiment = st.checkbox("Show Sentiment Profile", value=True, key="lsp_show_sentiment")

            with col3:
                lsp_show_poc = st.checkbox("Show Level of Significance", value=False, key="lsp_show_poc")
                lsp_show_price_levels = st.checkbox("Show Price Levels", value=False, key="lsp_show_price_levels")
                lsp_show_range_bg = st.checkbox("Show Range Background", value=True, key="lsp_show_range_bg")

            st.markdown("**Volume Thresholds**")
            col1, col2 = st.columns(2)

            with col1:
                lsp_hv_threshold = st.slider(
                    "High Volume Threshold %",
                    min_value=50,
                    max_value=99,
                    value=73,
                    step=1,
                    key="lsp_hv_threshold"
                ) / 100.0

            with col2:
                lsp_lv_threshold = st.slider(
                    "Low Volume Threshold %",
                    min_value=10,
                    max_value=40,
                    value=21,
                    step=1,
                    key="lsp_lv_threshold"
                ) / 100.0

    # OM Indicator Settings
    if show_om:
        with st.expander("🎯 OM Indicator Settings", expanded=False):
            st.markdown("**High Volume Pivots (HVP) Module**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                om_show_hvp = st.checkbox("Show HVP", value=True, key="om_show_hvp")
            with col2:
                om_hvp_left_bars = st.number_input("Left Bars", min_value=5, max_value=30, value=15, step=1, key="om_hvp_left_bars")
            with col3:
                om_hvp_right_bars = st.number_input("Right Bars", min_value=5, max_value=30, value=15, step=1, key="om_hvp_right_bars")
            with col4:
                om_hvp_volume_filter = st.number_input("Volume Filter", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="om_hvp_volume_filter")

            st.markdown("**Delta Module**")
            col1, col2 = st.columns(2)
            with col1:
                om_delta_length = st.slider("Delta Length", min_value=5, max_value=30, value=10, step=1, key="om_delta_length")
            with col2:
                om_delta_threshold = st.slider("Delta Threshold", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="om_delta_threshold")

            st.markdown("**VIDYA Module**")
            col1, col2, col3 = st.columns(3)
            with col1:
                om_vidya_length = st.slider("VIDYA Length", min_value=5, max_value=30, value=10, step=1, key="om_vidya_length")
            with col2:
                om_vidya_momentum = st.slider("Momentum Period", min_value=10, max_value=40, value=20, step=5, key="om_vidya_momentum")
            with col3:
                om_band_distance = st.slider("Band Distance", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="om_band_distance")

    # Money Flow Profile Settings
    if show_money_flow_profile:
        with st.expander("💰 Money Flow Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                mfp_lookback = st.slider(
                    "Lookback Length",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Lookback period for profile calculation",
                    key="mfp_lookback"
                )
                mfp_num_rows = st.slider(
                    "Number of Rows",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Number of price bins (default: 10)",
                    key="mfp_num_rows"
                )

            with col2:
                mfp_source = st.selectbox(
                    "Profile Source",
                    options=["Volume", "Money Flow"],
                    index=0,
                    help="Volume or Money Flow (volume × price)",
                    key="mfp_source"
                )
                mfp_sentiment_method = st.selectbox(
                    "Sentiment Method",
                    options=["Bar Polarity", "Bar Buying/Selling Pressure"],
                    index=0,
                    help="Method to determine bullish vs bearish bars",
                    key="mfp_sentiment_method"
                )

            with col3:
                mfp_show_poc = st.selectbox(
                    "POC Display",
                    options=["Last(Zone)", "Last(Line)", "Developing", "None"],
                    index=0,
                    help="Point of Control display mode",
                    key="mfp_show_poc"
                )
                mfp_show_consolidation = st.checkbox(
                    "Show Consolidation Zones",
                    value=True,
                    help="Highlight high volume consolidation areas",
                    key="mfp_show_consolidation"
                )

            st.markdown("**Volume Thresholds**")
            col1, col2, col3 = st.columns(3)

            with col1:
                mfp_hv_threshold = st.slider(
                    "High Volume %",
                    min_value=50,
                    max_value=99,
                    value=53,
                    step=1,
                    key="mfp_hv_threshold"
                ) / 100.0

            with col2:
                mfp_lv_threshold = st.slider(
                    "Low Volume %",
                    min_value=10,
                    max_value=40,
                    value=37,
                    step=1,
                    key="mfp_lv_threshold"
                ) / 100.0

            with col3:
                mfp_consolidation_threshold = st.slider(
                    "Consolidation %",
                    min_value=0,
                    max_value=100,
                    value=25,
                    step=5,
                    key="mfp_consolidation_threshold"
                ) / 100.0

    # DeltaFlow Profile Settings
    if show_deltaflow_profile:
        with st.expander("⚡ DeltaFlow Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                dfp_lookback = st.slider(
                    "Lookback Length",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Lookback period for profile calculation",
                    key="dfp_lookback"
                )
                dfp_bins = st.slider(
                    "Number of Bins",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    help="Number of price bins for delta analysis",
                    key="dfp_bins"
                )

            with col2:
                dfp_show_poc = st.checkbox(
                    "Show POC Line",
                    value=True,
                    help="Display Point of Control line",
                    key="dfp_show_poc"
                )
                dfp_show_delta_heatmap = st.checkbox(
                    "Show Delta Heatmap",
                    value=True,
                    help="Color bins by delta strength",
                    key="dfp_show_delta_heatmap"
                )

            with col3:
                dfp_show_delta_display = st.checkbox(
                    "Show Delta Labels",
                    value=True,
                    help="Display delta percentage per level",
                    key="dfp_show_delta_display"
                )
                dfp_show_volume_bars = st.checkbox(
                    "Show Volume Bars",
                    value=True,
                    help="Display buy/sell volume bars per bin",
                    key="dfp_show_volume_bars"
                )

    # Reversal Probability Zones Settings
    if show_reversal_zones:
        with st.expander("🎯 Reversal Probability Zones Settings (LuxAlgo)", expanded=False):
            st.markdown("**Reversal Analysis Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                rpz_swing_length = st.slider(
                    "Swing Length",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Lookback period for swing detection",
                    key="rpz_swing_length"
                )

            with col2:
                rpz_max_reversals = st.slider(
                    "Max Historical Samples",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                    help="Number of historical reversals to analyze",
                    key="rpz_max_reversals"
                )

            with col3:
                rpz_normalize = st.checkbox(
                    "Normalize Data",
                    value=False,
                    help="Use percentage-based analysis",
                    key="rpz_normalize"
                )

            st.markdown("**Probability Levels to Show**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                rpz_show_25 = st.checkbox("25th Percentile", value=True, key="rpz_show_25")

            with col2:
                rpz_show_50 = st.checkbox("50th Percentile", value=True, key="rpz_show_50")

            with col3:
                rpz_show_75 = st.checkbox("75th Percentile", value=True, key="rpz_show_75")

            with col4:
                rpz_show_90 = st.checkbox("90th Percentile", value=True, key="rpz_show_90")

    st.divider()

    # Display chart if data is available
    if st.session_state.chart_data is not None:
        try:
            with st.spinner("Rendering chart with indicators..."):
                # Prepare indicator parameters
                rsi_params = {
                    'length': rsi_length if show_rsi else 14,
                    'smooth': rsi_smooth if show_rsi else 14,
                    'method': rsi_method if show_rsi else 'RMA',
                    'signal_method': rsi_signal_method if show_rsi else 'EMA',
                    'ob_level': rsi_ob_level if show_rsi else 80,
                    'os_level': rsi_os_level if show_rsi else 20
                } if show_rsi else None

                om_params = {
                    'hvp_left_bars': om_hvp_left_bars if show_om else 15,
                    'hvp_right_bars': om_hvp_right_bars if show_om else 15,
                    'hvp_volume_filter': om_hvp_volume_filter if show_om else 2.0,
                    'delta_length': om_delta_length if show_om else 10,
                    'delta_threshold': om_delta_threshold if show_om else 1.5,
                    'vidya_length': om_vidya_length if show_om else 10,
                    'vidya_momentum': om_vidya_momentum if show_om else 20,
                    'band_distance': om_band_distance if show_om else 2.0,
                    'show_hvp': om_show_hvp if show_om else True
                } if show_om else None

                liquidity_params = {
                    'anchor_period': lsp_anchor_period if show_liquidity_profile else 'Auto',
                    'num_rows': lsp_num_rows if show_liquidity_profile else 25,
                    'profile_width': lsp_profile_width if show_liquidity_profile else 0.50,
                    'show_liquidity_profile': lsp_show_liquidity if show_liquidity_profile else True,
                    'show_sentiment_profile': lsp_show_sentiment if show_liquidity_profile else True,
                    'show_poc': lsp_show_poc if show_liquidity_profile else False,
                    'show_price_levels': lsp_show_price_levels if show_liquidity_profile else False,
                    'show_range_bg': lsp_show_range_bg if show_liquidity_profile else True,
                    'hv_threshold': lsp_hv_threshold if show_liquidity_profile else 0.73,
                    'lv_threshold': lsp_lv_threshold if show_liquidity_profile else 0.21
                } if show_liquidity_profile else None

                money_flow_params = {
                    'lookback': mfp_lookback if show_money_flow_profile else 200,
                    'num_rows': mfp_num_rows if show_money_flow_profile else 10,
                    'profile_source': mfp_source if show_money_flow_profile else 'Volume',
                    'sentiment_method': mfp_sentiment_method if show_money_flow_profile else 'Bar Polarity',
                    'show_poc': mfp_show_poc if show_money_flow_profile else 'Last(Zone)',
                    'show_consolidation': mfp_show_consolidation if show_money_flow_profile else True,
                    'hv_threshold': mfp_hv_threshold if show_money_flow_profile else 0.53,
                    'lv_threshold': mfp_lv_threshold if show_money_flow_profile else 0.37,
                    'consolidation_threshold': mfp_consolidation_threshold if show_money_flow_profile else 0.25
                } if show_money_flow_profile else None

                deltaflow_params = {
                    'lookback': dfp_lookback if show_deltaflow_profile else 200,
                    'bins': dfp_bins if show_deltaflow_profile else 30,
                    'show_poc': dfp_show_poc if show_deltaflow_profile else True,
                    'show_delta_heatmap': dfp_show_delta_heatmap if show_deltaflow_profile else True,
                    'show_delta_display': dfp_show_delta_display if show_deltaflow_profile else True,
                    'show_volume_bars': dfp_show_volume_bars if show_deltaflow_profile else True
                } if show_deltaflow_profile else None

                reversal_zones_params = {
                    'swing_length': rpz_swing_length if show_reversal_zones else 20,
                    'max_reversals': rpz_max_reversals if show_reversal_zones else 1000,
                    'normalize_data': rpz_normalize if show_reversal_zones else False,
                    'percentile_25': rpz_show_25 if show_reversal_zones else True,
                    'percentile_50': rpz_show_50 if show_reversal_zones else True,
                    'percentile_75': rpz_show_75 if show_reversal_zones else True,
                    'percentile_90': rpz_show_90 if show_reversal_zones else True
                } if show_reversal_zones else None

                # Create chart with selected indicators
                chart_analyzer = get_advanced_chart_analyzer()
                fig = chart_analyzer.create_advanced_chart(
                    st.session_state.chart_data,
                    symbol_code,
                    show_vob=False,
                    show_htf_sr=False,
                    show_footprint=False,
                    show_rsi=show_rsi,
                    show_om=show_om,
                    show_volume=show_volume,
                    show_liquidity_profile=show_liquidity_profile,
                    show_money_flow_profile=show_money_flow_profile,
                    show_deltaflow_profile=show_deltaflow_profile,
                    show_bos=show_bos,
                    show_choch=show_choch,
                    show_fibonacci=show_fibonacci,
                    show_patterns=show_patterns,
                    show_reversal_zones=show_reversal_zones,
                    vob_params=None,
                    htf_params=None,
                    footprint_params=None,
                    rsi_params=rsi_params,
                    om_params=om_params,
                    liquidity_params=liquidity_params,
                    money_flow_params=money_flow_params,
                    deltaflow_params=deltaflow_params,
                    reversal_zones_params=reversal_zones_params
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True)

                # Chart statistics
                st.subheader("📊 Chart Statistics")

                col1, col2, col3, col4, col5 = st.columns(5)

                df_stats = st.session_state.chart_data

                with col1:
                    st.metric("Current Price", f"₹{df_stats['close'].iloc[-1]:,.2f}")

                with col2:
                    price_change = df_stats['close'].iloc[-1] - df_stats['close'].iloc[0]
                    price_change_pct = (price_change / df_stats['close'].iloc[0]) * 100
                    st.metric("Change", f"₹{price_change:,.2f}", delta=f"{price_change_pct:.2f}%")

                with col3:
                    st.metric("High", f"₹{df_stats['high'].max():,.2f}")

                with col4:
                    st.metric("Low", f"₹{df_stats['low'].min():,.2f}")

                with col5:
                    avg_volume = df_stats['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")

                # Indicator Data Tables
                st.divider()
                st.subheader("📊 Indicator Data Tables")
                st.caption("Detailed data from all enabled indicators")

                # Create tabs for each indicator category
                indicator_tabs = []
                # Always show Market Regime tab
                indicator_tabs.append("🎯 Market Regime")
                if show_rsi:
                    indicator_tabs.append("📈 Ultimate RSI")
                if show_om:
                    indicator_tabs.append("🎯 OM Indicator")
                if show_liquidity_profile:
                    indicator_tabs.append("💧 Liquidity Profile")
                if show_money_flow_profile:
                    indicator_tabs.append("💰 Money Flow Profile")
                if show_deltaflow_profile:
                    indicator_tabs.append("⚡ DeltaFlow Profile")
                if show_bos or show_choch or show_fibonacci or show_patterns:
                    indicator_tabs.append("🎯 Price Action")
                if show_reversal_zones:
                    indicator_tabs.append("🎯 Reversal Zones")

                if indicator_tabs:
                    tabs = st.tabs(indicator_tabs)
                    tab_idx = 0

                    # Market Regime Dashboard (Always first tab) - EXPANDED ML VERSION
                    with tabs[tab_idx]:
                        from src.ml_market_regime import MLMarketRegimeDetector
                        from enhanced_market_data import EnhancedMarketData
                        from bias_analysis import BiasAnalysisPro

                        st.markdown("### 🎯 ML Market Regime Analysis")
                        st.caption("🚀 AI-powered market regime detection using ALL available data sources")

                        # Collect indicator data for regime detection
                        regime_indicator_data = {}

                        # BOS data
                        if show_bos:
                            from indicators.advanced_price_action import AdvancedPriceAction
                            pa_indicator = AdvancedPriceAction()
                            regime_indicator_data['bos'] = pa_indicator.detect_bos(df_stats)
                            regime_indicator_data['choch'] = pa_indicator.detect_choch(df_stats)
                        else:
                            regime_indicator_data['bos'] = []
                            regime_indicator_data['choch'] = []

                        # RSI data
                        if show_rsi:
                            from indicators.ultimate_rsi import UltimateRSI
                            rsi_for_regime = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                            regime_indicator_data['rsi'] = rsi_for_regime.get_signals(df_stats)

                        # Money Flow Profile data
                        if show_money_flow_profile:
                            from indicators.money_flow_profile import MoneyFlowProfile
                            mfp_for_regime = MoneyFlowProfile(**money_flow_params) if money_flow_params else MoneyFlowProfile(num_rows=10)
                            regime_indicator_data['money_flow_profile'] = mfp_for_regime.get_signals(df_stats)

                        # DeltaFlow Profile data
                        if show_deltaflow_profile:
                            from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile
                            dfp_for_regime = DeltaFlowVolumeProfile(**deltaflow_params) if deltaflow_params else DeltaFlowVolumeProfile(bins=30)
                            regime_indicator_data['deltaflow_profile'] = dfp_for_regime.get_signals(df_stats)

                        # Reversal Probability Zones data
                        reversal_zones_result = None
                        if show_reversal_zones:
                            from indicators.reversal_probability_zones import ReversalProbabilityZones
                            rpz_for_regime = ReversalProbabilityZones(**reversal_zones_params) if reversal_zones_params else ReversalProbabilityZones()
                            reversal_zones_result = rpz_for_regime.calculate(df_stats)

                        # Fetch additional data sources
                        with st.spinner("Fetching comprehensive market data..."):
                            # Enhanced market data (sector rotation, VIX, gamma squeeze)
                            enhanced_data_fetcher = EnhancedMarketData()

                            sector_rotation_data = enhanced_data_fetcher.analyze_sector_rotation()
                            india_vix_data = enhanced_data_fetcher.fetch_india_vix()
                            gamma_squeeze_data = enhanced_data_fetcher.detect_gamma_squeeze('NIFTY')

                            # Bias analysis
                            bias_analyzer = BiasAnalysisPro()
                            bias_analysis_data = bias_analyzer.analyze_all_bias_indicators(data=df_stats)

                            # Option chain data (from session state)
                            option_chain_data = None
                            if 'overall_option_data' in st.session_state:
                                option_chain_data = st.session_state.overall_option_data.get('NIFTY')

                        # Detect regime using ML with ALL data sources
                        regime_detector = MLMarketRegimeDetector()
                        regime_result = regime_detector.detect_regime(
                            df_stats,
                            cvd_result=None,
                            volatility_result=None,
                            oi_trap_result=None,
                            option_chain_data=option_chain_data,
                            sector_rotation_data=sector_rotation_data,
                            bias_analysis_data=bias_analysis_data,
                            india_vix_data=india_vix_data,
                            gamma_squeeze_data=gamma_squeeze_data,
                            advanced_chart_indicators=regime_indicator_data,
                            reversal_zones_data=reversal_zones_result,
                            volume_footprint_data=None
                        )

                        # ═══════════════════════════════════════════════════════════════
                        # TRADING SENTIMENT - PROMINENT DISPLAY (LONG/SHORT)
                        # ═══════════════════════════════════════════════════════════════
                        sentiment = regime_result.trading_sentiment
                        sentiment_conf = regime_result.sentiment_confidence
                        sentiment_score = regime_result.sentiment_score

                        # Determine color based on sentiment
                        if "STRONG LONG" in sentiment:
                            sentiment_color = "#00ff88"  # Bright green
                            bg_color = "rgba(0, 255, 136, 0.15)"
                        elif "LONG" in sentiment:
                            sentiment_color = "#00cc66"  # Green
                            bg_color = "rgba(0, 204, 102, 0.15)"
                        elif "STRONG SHORT" in sentiment:
                            sentiment_color = "#ff4444"  # Bright red
                            bg_color = "rgba(255, 68, 68, 0.15)"
                        elif "SHORT" in sentiment:
                            sentiment_color = "#ff6666"  # Red
                            bg_color = "rgba(255, 102, 102, 0.15)"
                        else:  # NEUTRAL
                            sentiment_color = "#66b3ff"  # Blue
                            bg_color = "rgba(102, 179, 255, 0.15)"

                        # Big bold sentiment display
                        st.markdown(f"""
                        <div style="
                            background: {bg_color};
                            padding: 30px;
                            border-radius: 15px;
                            border-left: 8px solid {sentiment_color};
                            text-align: center;
                            margin-bottom: 20px;
                        ">
                            <div style='font-size: 1.2rem; color:#cccccc; margin-bottom:10px;'>
                                📊 MARKET SENTIMENT (Based on ALL Indicators)
                            </div>
                            <div style='font-size: 4rem; color:{sentiment_color}; font-weight:900; margin:10px 0;'>
                                {sentiment}
                            </div>
                            <div style='font-size: 1.5rem; color:{sentiment_color}; margin-top:10px;'>
                                Confidence: {sentiment_conf:.1f}% | Score: {sentiment_score:+.1f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.divider()

                        # Display regime info
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            regime = regime_result.regime
                            regime_emoji = {
                                'Trending Up': '🚀',
                                'Trending Down': '📉',
                                'Range Bound': '↔️',
                                'Volatile Breakout': '⚡',
                                'Consolidation': '🔄'
                            }
                            st.metric(
                                "Current Regime",
                                f"{regime_emoji.get(regime, '🎯')} {regime}",
                                delta=None
                            )

                        with col2:
                            confidence_pct = regime_result.confidence
                            confidence_color = '🟢' if confidence_pct > 70 else '🟡' if confidence_pct > 50 else '🔴'
                            st.metric(
                                "Confidence",
                                f"{confidence_color} {confidence_pct:.1f}%",
                                delta=None
                            )

                        with col3:
                            st.metric(
                                "Volatility",
                                f"📊 {regime_result.volatility_state}",
                                delta=None
                            )

                        st.divider()

                        # Display Support/Resistance Levels
                        if regime_result.support_resistance:
                            try:
                                st.markdown("### 🎯 Support & Resistance Levels")

                                sr_levels = regime_result.support_resistance

                                # Validate current_price is a number
                                current_price = sr_levels.get('current_price')
                                if not isinstance(current_price, (int, float)) or current_price is None or current_price == 0:
                                    st.warning("⚠️ Unable to calculate S/R levels - invalid current price")
                                else:
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("#### 📈 Resistance Levels")
                                        resistance_data = []

                                        near_res = sr_levels.get('near_resistance')
                                        if near_res and isinstance(near_res, (int, float)):
                                            resistance_data.append({
                                                'Type': 'Near Resistance',
                                                'Level': f"₹{near_res:.2f}",
                                                'Distance': f"{((near_res - current_price) / current_price * 100):.2f}%"
                                            })

                                        major_res = sr_levels.get('major_resistance')
                                        if major_res and isinstance(major_res, (int, float)):
                                            resistance_data.append({
                                                'Type': 'Major Resistance',
                                                'Level': f"₹{major_res:.2f}",
                                                'Distance': f"{((major_res - current_price) / current_price * 100):.2f}%"
                                            })

                                        for i, r in enumerate(sr_levels.get('all_resistances', [])[:3], 1):
                                            if isinstance(r, (int, float)):
                                                resistance_data.append({
                                                    'Type': f'R{i}',
                                                    'Level': f"₹{r:.2f}",
                                                    'Distance': f"{((r - current_price) / current_price * 100):.2f}%"
                                                })

                                        if resistance_data:
                                            st.dataframe(pd.DataFrame(resistance_data), use_container_width=True, hide_index=True)
                                        else:
                                            st.info("No resistance levels detected")

                                    with col2:
                                        st.markdown("#### 📉 Support Levels")
                                        support_data = []

                                        near_sup = sr_levels.get('near_support')
                                        if near_sup and isinstance(near_sup, (int, float)):
                                            support_data.append({
                                                'Type': 'Near Support',
                                                'Level': f"₹{near_sup:.2f}",
                                                'Distance': f"{((current_price - near_sup) / current_price * 100):.2f}%"
                                            })

                                        major_sup = sr_levels.get('major_support')
                                        if major_sup and isinstance(major_sup, (int, float)):
                                            support_data.append({
                                                'Type': 'Major Support',
                                                'Level': f"₹{major_sup:.2f}",
                                                'Distance': f"{((current_price - major_sup) / current_price * 100):.2f}%"
                                            })

                                        for i, s in enumerate(sr_levels.get('all_supports', [])[:3], 1):
                                            if isinstance(s, (int, float)):
                                                support_data.append({
                                                    'Type': f'S{i}',
                                                    'Level': f"₹{s:.2f}",
                                                    'Distance': f"{((current_price - s) / current_price * 100):.2f}%"
                                                })

                                        if support_data:
                                            st.dataframe(pd.DataFrame(support_data), use_container_width=True, hide_index=True)
                                        else:
                                            st.info("No support levels detected")

                                    st.divider()

                            except Exception as e:
                                st.error(f"⚠️ Error displaying S/R levels: {e}")
                                logger.error(f"S/R display error: {e}", exc_info=True)

                        # Display Entry/Exit Signals
                        if regime_result.entry_exit_signals:
                            st.markdown("### 🎯 Entry/Exit Trading Signals")

                            entry_exit = regime_result.entry_exit_signals

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                action_emoji = {
                                    'BUY_ON_PULLBACK': '📈',
                                    'SELL_ON_RALLY': '📉',
                                    'BUY_ON_BREAK': '🚀',
                                    'SELL_ON_BREAK': '💥',
                                    'RANGE_TRADE': '↔️',
                                    'WAIT': '⏸️',
                                    'WAIT_FOR_CONFIRMATION': '⏳'
                                }
                                st.metric(
                                    "Action",
                                    f"{action_emoji.get(entry_exit['action'], '🎯')} {entry_exit['action'].replace('_', ' ').title()}",
                                    delta=None
                                )

                            with col2:
                                direction_emoji = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪', 'BOTH': '🔵'}
                                st.metric(
                                    "Direction",
                                    f"{direction_emoji.get(entry_exit['direction'], '⚪')} {entry_exit['direction']}",
                                    delta=None
                                )

                            with col3:
                                if entry_exit.get('risk_reward'):
                                    rr_color = '🟢' if entry_exit['risk_reward'] > 2 else '🟡' if entry_exit['risk_reward'] > 1.5 else '🔴'
                                    st.metric(
                                        "Risk:Reward",
                                        f"{rr_color} 1:{entry_exit['risk_reward']:.2f}",
                                        delta=None
                                    )
                                else:
                                    st.metric("Risk:Reward", "N/A", delta=None)

                            # Trading Levels
                            if entry_exit.get('entry_level') and isinstance(entry_exit['entry_level'], (int, float)):
                                st.markdown("#### 📊 Trading Levels")
                                levels_data = []
                                levels_data.append({
                                    'Level': 'Entry',
                                    'Price': f"₹{entry_exit['entry_level']:.2f}"
                                })
                                if entry_exit.get('stop_loss'):
                                    levels_data.append({
                                        'Level': 'Stop Loss',
                                        'Price': f"₹{entry_exit['stop_loss']:.2f}"
                                    })
                                if entry_exit.get('target_1'):
                                    levels_data.append({
                                        'Level': 'Target 1',
                                        'Price': f"₹{entry_exit['target_1']:.2f}"
                                    })
                                if entry_exit.get('target_2'):
                                    levels_data.append({
                                        'Level': 'Target 2',
                                        'Price': f"₹{entry_exit['target_2']:.2f}"
                                    })
                                st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)

                            # Reasoning
                            if entry_exit.get('reasoning'):
                                st.markdown("#### 💡 Trading Rationale")
                                for reason in entry_exit['reasoning']:
                                    st.markdown(f"• {reason}")

                            st.divider()

                        # Regime Details
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 📊 Regime Indicators")
                            indicator_data = []
                            indicator_data.append({
                                'Indicator': 'Trend Strength',
                                'Value': f"{regime_result.trend_strength:.1f}%"
                            })
                            indicator_data.append({
                                'Indicator': 'Market Phase',
                                'Value': regime_result.market_phase
                            })
                            indicator_data.append({
                                'Indicator': 'Optimal Timeframe',
                                'Value': regime_result.optimal_timeframe
                            })

                            st.dataframe(pd.DataFrame(indicator_data), use_container_width=True, hide_index=True)

                        with col2:
                            st.markdown("#### 💡 Trading Strategy")
                            st.info(regime_result.recommended_strategy)

                        # All Signals
                        if regime_result.signals:
                            st.markdown("#### 🎯 Market Signals")
                            signal_cols = st.columns(2)
                            mid = len(regime_result.signals) // 2

                            with signal_cols[0]:
                                for signal in regime_result.signals[:mid]:
                                    st.markdown(f"• {signal}")

                            with signal_cols[1]:
                                for signal in regime_result.signals[mid:]:
                                    st.markdown(f"• {signal}")

                    tab_idx += 1

                    # Ultimate RSI Data
                    if show_rsi:
                        with tabs[tab_idx]:
                            from indicators.ultimate_rsi import UltimateRSI
                            rsi_indicator = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                            rsi_signals = rsi_indicator.get_signals(df_stats)

                            st.markdown("#### 📈 Ultimate RSI History")

                            # Last 20 values
                            rsi_table = []
                            for i in range(max(0, len(df_stats) - 20), len(df_stats)):
                                rsi_table.append({
                                    'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                    'Price': f"₹{df_stats['close'].iloc[i]:.2f}",
                                    'RSI': f"{rsi_signals['ultimate_rsi'][i]:.2f}",
                                    'Signal': f"{rsi_signals['signal'][i]:.2f}",
                                    'Status': '🔴 OB' if rsi_signals['ultimate_rsi'][i] > (rsi_params.get('ob_level', 80) if rsi_params else 80) else '🟢 OS' if rsi_signals['ultimate_rsi'][i] < (rsi_params.get('os_level', 20) if rsi_params else 20) else '⚪ Neutral'
                                })
                            st.dataframe(pd.DataFrame(rsi_table), use_container_width=True, hide_index=True)

                            # Current values
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current RSI", f"{rsi_signals['ultimate_rsi'][-1]:.2f}")
                            with col2:
                                st.metric("Signal Line", f"{rsi_signals['signal'][-1]:.2f}")
                            with col3:
                                divergence = rsi_signals['ultimate_rsi'][-1] - rsi_signals['signal'][-1]
                                st.metric("Divergence", f"{divergence:.2f}")
                        tab_idx += 1

                    # OM Indicator Data
                    if show_om:
                        with tabs[tab_idx]:
                            from indicators.om_indicator import OMIndicator
                            om_indicator = OMIndicator(**om_params) if om_params else OMIndicator()
                            om_results = om_indicator.calculate(df_stats)

                            st.markdown("#### 🎯 OM Indicator Modules")

                            # VOB Module
                            if 'vob' in om_results:
                                st.markdown("**📦 Volume Order Blocks Module**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if om_results['vob'].get('bullish_blocks'):
                                        st.markdown("🟢 **Bullish Blocks**")
                                        vob_bull = []
                                        for i, block in enumerate(om_results['vob']['bullish_blocks'][:5]):
                                            vob_bull.append({
                                                '#': i + 1,
                                                'Lower': f"₹{block['lower']:.2f}",
                                                'Upper': f"₹{block['upper']:.2f}",
                                                'Active': '✅' if block['active'] else '❌'
                                            })
                                        st.dataframe(pd.DataFrame(vob_bull), use_container_width=True, hide_index=True)
                                with col2:
                                    if om_results['vob'].get('bearish_blocks'):
                                        st.markdown("🔴 **Bearish Blocks**")
                                        vob_bear = []
                                        for i, block in enumerate(om_results['vob']['bearish_blocks'][:5]):
                                            vob_bear.append({
                                                '#': i + 1,
                                                'Lower': f"₹{block['lower']:.2f}",
                                                'Upper': f"₹{block['upper']:.2f}",
                                                'Active': '✅' if block['active'] else '❌'
                                            })
                                        st.dataframe(pd.DataFrame(vob_bear), use_container_width=True, hide_index=True)
                                st.divider()

                            # Delta Module
                            if 'delta' in om_results:
                                st.markdown("**📊 Delta Module (Buy/Sell Pressure)**")
                                delta_data = []
                                delta_values = om_results['delta']
                                for i in range(max(0, len(df_stats) - 10), len(df_stats)):
                                    delta_data.append({
                                        'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                        'Delta': f"{delta_values[i]:.2f}" if i < len(delta_values) else 'N/A',
                                        'Signal': '🟢 Bullish' if i < len(delta_values) and delta_values[i] > 0 else '🔴 Bearish' if i < len(delta_values) else 'N/A'
                                    })
                                st.dataframe(pd.DataFrame(delta_data), use_container_width=True, hide_index=True)
                                st.divider()

                            # VIDYA Module
                            if 'vidya' in om_results:
                                st.markdown("**📈 VIDYA Trend Module**")
                                vidya_data = []
                                vidya_values = om_results['vidya']
                                for i in range(max(0, len(df_stats) - 10), len(df_stats)):
                                    vidya_data.append({
                                        'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                        'Price': f"₹{df_stats['close'].iloc[i]:.2f}",
                                        'VIDYA': f"₹{vidya_values[i]:.2f}" if i < len(vidya_values) else 'N/A',
                                    })
                                st.dataframe(pd.DataFrame(vidya_data), use_container_width=True, hide_index=True)
                                st.divider()

                            # HVP Module
                            if 'hvp' in om_results and om_results['hvp'].get('pivots'):
                                st.markdown("**🎯 High Volume Pivots**")
                                hvp_data = []
                                for i, pivot in enumerate(om_results['hvp']['pivots'][:10]):
                                    hvp_data.append({
                                        '#': i + 1,
                                        'Price': f"₹{pivot['price']:.2f}",
                                        'Type': pivot['type'].upper(),
                                        'Volume': f"{pivot['volume']:,.0f}"
                                    })
                                st.dataframe(pd.DataFrame(hvp_data), use_container_width=True, hide_index=True)

                        tab_idx += 1

                    # Liquidity Sentiment Profile Data
                    if show_liquidity_profile:
                        with tabs[tab_idx]:
                            from indicators.liquidity_sentiment_profile import LiquiditySentimentProfile
                            lsp_indicator = LiquiditySentimentProfile(**liquidity_params) if liquidity_params else LiquiditySentimentProfile()
                            lsp_data = lsp_indicator.calculate(df_stats)

                            st.markdown("#### 💧 Liquidity & Sentiment Profile")

                            if 'price_levels' in lsp_data and len(lsp_data['price_levels']) > 0:
                                # Profile data table
                                profile_data = []
                                for level in lsp_data['price_levels']:
                                    profile_data.append({
                                        'Price': f"₹{level['price']:.2f}",
                                        'Total Volume': f"{level.get('total_volume', 0):,.0f}",
                                        'Buy Volume': f"{level.get('buy_volume', 0):,.0f}",
                                        'Sell Volume': f"{level.get('sell_volume', 0):,.0f}",
                                        'Sentiment': '🟢 Bullish' if level.get('buy_volume', 0) > level.get('sell_volume', 0) else '🔴 Bearish',
                                        'Activity': '🔥' * min(5, int(level.get('total_volume', 0) / max(1, lsp_data.get('avg_volume', 1))))
                                    })
                                st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True, height=400)

                                # Key levels
                                st.divider()
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if 'poc' in lsp_data:
                                        st.metric("Point of Control", f"₹{lsp_data['poc']:.2f}")
                                with col2:
                                    if 'high_volume_node' in lsp_data:
                                        st.metric("High Volume Node", f"₹{lsp_data['high_volume_node']:.2f}")
                                with col3:
                                    if 'low_volume_node' in lsp_data:
                                        st.metric("Low Volume Node", f"₹{lsp_data['low_volume_node']:.2f}")
                            else:
                                st.info("No liquidity profile data available")
                        tab_idx += 1

                    # Money Flow Profile Data
                    if show_money_flow_profile:
                        with tabs[tab_idx]:
                            from indicators.money_flow_profile import MoneyFlowProfile
                            mfp_indicator = MoneyFlowProfile(**money_flow_params) if money_flow_params else MoneyFlowProfile(num_rows=10)
                            mfp_signals = mfp_indicator.get_signals(df_stats)

                            st.markdown("#### 💰 Money Flow Profile Analysis")

                            if mfp_signals.get('success'):
                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Sentiment", mfp_signals['sentiment'])
                                with col2:
                                    st.metric("POC Price", f"₹{mfp_signals['poc_price']:.2f}")
                                with col3:
                                    st.metric("Bullish Volume", f"{mfp_signals['bullish_volume_pct']:.1f}%")
                                with col4:
                                    st.metric("Bearish Volume", f"{mfp_signals['bearish_volume_pct']:.1f}%")

                                st.divider()

                                # Price position
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Current Price",
                                        f"₹{mfp_signals['current_price']:.2f}",
                                        delta=f"{mfp_signals['distance_from_poc']:+.2f} from POC"
                                    )
                                with col2:
                                    st.metric("Position", mfp_signals['price_position'])

                                # Volume levels
                                st.divider()
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**🔥 High Volume Levels (Consolidation)**")
                                    if mfp_signals['high_volume_levels']:
                                        hv_data = []
                                        for level in mfp_signals['high_volume_levels']:
                                            hv_data.append({
                                                'Price': f"₹{level:.2f}",
                                                'Type': '🟡 Value Area'
                                            })
                                        st.dataframe(pd.DataFrame(hv_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No high volume levels detected")

                                with col2:
                                    st.markdown("**⚡ Low Volume Levels (Supply/Demand)**")
                                    if mfp_signals['low_volume_levels']:
                                        lv_data = []
                                        for level in mfp_signals['low_volume_levels']:
                                            lv_data.append({
                                                'Price': f"₹{level:.2f}",
                                                'Type': '🔵 Breakout Zone'
                                            })
                                        st.dataframe(pd.DataFrame(lv_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No low volume levels detected")

                                # Consolidation zones
                                if mfp_signals.get('consolidation_zones'):
                                    st.divider()
                                    st.markdown("**📊 Consolidation Zones**")
                                    cz_data = []
                                    for zone in mfp_signals['consolidation_zones']:
                                        cz_data.append({
                                            'Lower': f"₹{zone['lower']:.2f}",
                                            'Upper': f"₹{zone['upper']:.2f}",
                                            'Range': f"₹{zone['upper'] - zone['lower']:.2f}"
                                        })
                                    st.dataframe(pd.DataFrame(cz_data), use_container_width=True, hide_index=True)

                                # Trading insights
                                st.divider()
                                st.markdown("**💡 Trading Insights**")
                                insights = []
                                if mfp_signals['sentiment'] == 'BULLISH':
                                    insights.append("🟢 Strong bullish volume dominance - Look for dip buying opportunities")
                                elif mfp_signals['sentiment'] == 'BEARISH':
                                    insights.append("🔴 Strong bearish volume dominance - Look for short opportunities")
                                else:
                                    insights.append("🟡 Neutral sentiment - Wait for clearer direction")

                                if mfp_signals['price_position'] == 'Above POC':
                                    insights.append("📈 Price above POC - Bulls in control, POC acts as support")
                                elif mfp_signals['price_position'] == 'Below POC':
                                    insights.append("📉 Price below POC - Bears in control, POC acts as resistance")

                                for insight in insights:
                                    st.info(insight)

                            else:
                                st.error(f"Error: {mfp_signals.get('error', 'Unknown error')}")
                        tab_idx += 1

                    # DeltaFlow Profile Data
                    if show_deltaflow_profile:
                        with tabs[tab_idx]:
                            from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile
                            dfp_indicator = DeltaFlowVolumeProfile(**deltaflow_params) if deltaflow_params else DeltaFlowVolumeProfile(bins=30)
                            dfp_signals = dfp_indicator.get_signals(df_stats)

                            st.markdown("#### ⚡ DeltaFlow Volume Profile Analysis")

                            if dfp_signals.get('success'):
                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    sentiment_emoji = "🟢" if "BULLISH" in dfp_signals['sentiment'] else "🔴" if "BEARISH" in dfp_signals['sentiment'] else "🟡"
                                    st.metric("Sentiment", f"{sentiment_emoji} {dfp_signals['sentiment']}")
                                with col2:
                                    delta_color = "normal" if abs(dfp_signals['overall_delta']) < 10 else "inverse" if dfp_signals['overall_delta'] < 0 else "off"
                                    st.metric("Overall Delta", f"{dfp_signals['overall_delta']:+.1f}%", delta_color=delta_color)
                                with col3:
                                    st.metric("Buy Volume", f"{dfp_signals['overall_bull_pct']:.1f}%")
                                with col4:
                                    st.metric("Sell Volume", f"{dfp_signals['overall_bear_pct']:.1f}%")

                                st.divider()

                                # POC and position
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "POC Price",
                                        f"₹{dfp_signals['poc_price']:.2f}",
                                        help="Point of Control - Highest volume price level"
                                    )
                                with col2:
                                    st.metric(
                                        "Distance from POC",
                                        f"{dfp_signals['distance_from_poc_pct']:+.2f}%",
                                        help=f"Position: {dfp_signals['price_position']}"
                                    )

                                # Delta levels
                                st.divider()
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**🟢 Strong Buy Levels (Delta > +30%)**")
                                    if dfp_signals['strong_buy_levels']:
                                        buy_data = []
                                        for level in dfp_signals['strong_buy_levels']:
                                            buy_data.append({
                                                'Price': f"₹{level['price']:.2f}",
                                                'Delta': f"{level['delta']:+.1f}%",
                                                'Strength': '🟢' * min(5, int(abs(level['delta']) / 10))
                                            })
                                        st.dataframe(pd.DataFrame(buy_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No strong buy levels detected")

                                with col2:
                                    st.markdown("**🔴 Strong Sell Levels (Delta < -30%)**")
                                    if dfp_signals['strong_sell_levels']:
                                        sell_data = []
                                        for level in dfp_signals['strong_sell_levels']:
                                            sell_data.append({
                                                'Price': f"₹{level['price']:.2f}",
                                                'Delta': f"{level['delta']:+.1f}%",
                                                'Strength': '🔴' * min(5, int(abs(level['delta']) / 10))
                                            })
                                        st.dataframe(pd.DataFrame(sell_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No strong sell levels detected")

                                # Absorption zones
                                if dfp_signals['absorption_zones']:
                                    st.divider()
                                    st.markdown("**🛡️ Absorption Zones (High Volume, Low Delta)**")
                                    st.caption("Areas where large orders are being absorbed - potential reversal zones")
                                    az_data = []
                                    for zone in dfp_signals['absorption_zones']:
                                        az_data.append({
                                            'Lower': f"₹{zone['lower']:.2f}",
                                            'Upper': f"₹{zone['upper']:.2f}",
                                            'Volume': f"{zone['volume']:,.0f}",
                                            'Significance': '🛡️🛡️🛡️'
                                        })
                                    st.dataframe(pd.DataFrame(az_data), use_container_width=True, hide_index=True)

                                # Delta distribution summary
                                dfp_summary = dfp_indicator.get_delta_levels_summary(df_stats)
                                if dfp_summary.get('success'):
                                    st.divider()
                                    st.markdown("**📊 Delta Distribution Across Price Levels**")
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    with col1:
                                        st.metric("Strong Buy Bins", f"{dfp_summary['strong_buy']}")
                                    with col2:
                                        st.metric("Moderate Buy", f"{dfp_summary['moderate_buy']}")
                                    with col3:
                                        st.metric("Neutral", f"{dfp_summary['neutral']}")
                                    with col4:
                                        st.metric("Moderate Sell", f"{dfp_summary['moderate_sell']}")
                                    with col5:
                                        st.metric("Strong Sell Bins", f"{dfp_summary['strong_sell']}")

                                # Trading insights
                                st.divider()
                                st.markdown("**💡 Trading Insights**")
                                insights = []

                                if dfp_signals['sentiment'] == 'STRONG BULLISH':
                                    insights.append("🟢🟢 Extremely strong buying pressure - Momentum likely to continue")
                                elif dfp_signals['sentiment'] == 'BULLISH':
                                    insights.append("🟢 Bullish delta - Buyers in control")
                                elif dfp_signals['sentiment'] == 'STRONG BEARISH':
                                    insights.append("🔴🔴 Extremely strong selling pressure - Downtrend likely")
                                elif dfp_signals['sentiment'] == 'BEARISH':
                                    insights.append("🔴 Bearish delta - Sellers in control")
                                else:
                                    insights.append("🟡 Neutral delta - Balanced orderflow, wait for confirmation")

                                if dfp_signals['strong_buy_levels']:
                                    insights.append(f"📍 {len(dfp_signals['strong_buy_levels'])} strong buy level(s) detected - Potential support zones")

                                if dfp_signals['strong_sell_levels']:
                                    insights.append(f"📍 {len(dfp_signals['strong_sell_levels'])} strong sell level(s) detected - Potential resistance zones")

                                if dfp_signals['absorption_zones']:
                                    insights.append(f"🛡️ {len(dfp_signals['absorption_zones'])} absorption zone(s) - Watch for reversals at these levels")

                                for insight in insights:
                                    st.info(insight)

                            else:
                                st.error(f"Error: {dfp_signals.get('error', 'Unknown error')}")
                        tab_idx += 1

                    # Price Action Data
                    if show_bos or show_choch or show_fibonacci or show_patterns:
                        with tabs[tab_idx]:
                            from indicators.advanced_price_action import AdvancedPriceAction
                            pa_indicator = AdvancedPriceAction()

                            st.markdown("#### 🎯 Advanced Price Action Signals")

                            # BOS Detection
                            if show_bos:
                                st.markdown("**🔺 Break of Structure (BOS)**")
                                bos_data = pa_indicator.detect_bos(df_stats)
                                if bos_data and len(bos_data) > 0:
                                    bos_table = []
                                    for bos in bos_data[-10:]:
                                        bos_table.append({
                                            'Index': bos.get('index', 'N/A'),
                                            'Type': '🟢 Bullish' if bos['type'] == 'bullish' else '🔴 Bearish',
                                            'Price': f"₹{bos['price']:.2f}",
                                            'Strength': '🔴' * min(5, int(bos.get('strength', 1)))
                                        })
                                    st.dataframe(pd.DataFrame(bos_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No BOS signals detected")
                                st.divider()

                            # CHOCH Detection
                            if show_choch:
                                st.markdown("**🔄 Change of Character (CHOCH)**")
                                choch_data = pa_indicator.detect_choch(df_stats)
                                if choch_data and len(choch_data) > 0:
                                    choch_table = []
                                    for choch in choch_data[-10:]:
                                        choch_table.append({
                                            'Index': choch.get('index', 'N/A'),
                                            'Type': '🟢 Bullish' if choch['type'] == 'bullish' else '🔴 Bearish',
                                            'Price': f"₹{choch['price']:.2f}",
                                        })
                                    st.dataframe(pd.DataFrame(choch_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No CHOCH signals detected")
                                st.divider()

                            # Fibonacci Levels
                            if show_fibonacci:
                                st.markdown("**📐 Fibonacci Retracement Levels**")
                                fib_levels = pa_indicator.calculate_fibonacci(df_stats)
                                if fib_levels:
                                    fib_table = []
                                    for level_name, level_price in fib_levels.items():
                                        fib_table.append({
                                            'Level': level_name,
                                            'Price': f"₹{level_price:.2f}"
                                        })
                                    st.dataframe(pd.DataFrame(fib_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No Fibonacci levels calculated")
                                st.divider()

                            # Patterns
                            if show_patterns:
                                st.markdown("**📊 Geometric Patterns**")
                                patterns = pa_indicator.detect_patterns(df_stats)
                                if patterns and len(patterns) > 0:
                                    pattern_table = []
                                    for pattern in patterns[-10:]:
                                        pattern_table.append({
                                            'Pattern': pattern['name'],
                                            'Type': pattern['type'],
                                            'Start': pattern.get('start_idx', 'N/A'),
                                            'End': pattern.get('end_idx', 'N/A'),
                                        })
                                    st.dataframe(pd.DataFrame(pattern_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No patterns detected")

                        tab_idx += 1

                    # Reversal Probability Zones Data (LuxAlgo)
                    if show_reversal_zones:
                        with tabs[tab_idx]:
                            st.markdown("#### 🎯 Reversal Probability Zones Analysis (LuxAlgo)")
                            st.caption("Statistical reversal prediction with probability targets")

                            if reversal_zones_result and reversal_zones_result.get('success'):
                                zone = reversal_zones_result.get('zone')
                                current_price = reversal_zones_result.get('current_price', 0)

                                # Zone Direction and Base Info
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    direction = "🟢 Bullish" if zone.is_bullish else "🔴 Bearish"
                                    st.metric("Expected Reversal", direction)

                                with col2:
                                    st.metric("Swing Price", f"₹{zone.price:.2f}")

                                with col3:
                                    st.metric("Current Price", f"₹{current_price:.2f}")

                                st.divider()

                                # Probability Targets
                                st.markdown("**📊 Probability Targets**")

                                targets_table = []
                                if zone.percentile_25_price:
                                    distance_pct = ((zone.percentile_25_price - current_price) / current_price) * 100
                                    targets_table.append({
                                        'Probability': '25%',
                                        'Target Price': f"₹{zone.percentile_25_price:.2f}",
                                        'Distance': f"{distance_pct:+.2f}%",
                                        'Bars Expected': zone.percentile_25_bars
                                    })

                                if zone.percentile_50_price:
                                    distance_pct = ((zone.percentile_50_price - current_price) / current_price) * 100
                                    targets_table.append({
                                        'Probability': '50%',
                                        'Target Price': f"₹{zone.percentile_50_price:.2f}",
                                        'Distance': f"{distance_pct:+.2f}%",
                                        'Bars Expected': zone.percentile_50_bars
                                    })

                                if zone.percentile_75_price:
                                    distance_pct = ((zone.percentile_75_price - current_price) / current_price) * 100
                                    targets_table.append({
                                        'Probability': '75%',
                                        'Target Price': f"₹{zone.percentile_75_price:.2f}",
                                        'Distance': f"{distance_pct:+.2f}%",
                                        'Bars Expected': zone.percentile_75_bars
                                    })

                                if zone.percentile_90_price:
                                    distance_pct = ((zone.percentile_90_price - current_price) / current_price) * 100
                                    targets_table.append({
                                        'Probability': '90%',
                                        'Target Price': f"₹{zone.percentile_90_price:.2f}",
                                        'Distance': f"{distance_pct:+.2f}%",
                                        'Bars Expected': zone.percentile_90_bars
                                    })

                                if targets_table:
                                    st.dataframe(pd.DataFrame(targets_table), use_container_width=True, hide_index=True)

                                st.divider()

                                # Historical Sample Info
                                st.markdown("**📈 Historical Analysis**")
                                col1, col2 = st.columns(2)

                                with col1:
                                    total_bullish = reversal_zones_result.get('total_bullish_samples', 0)
                                    st.metric("Bullish Reversals Analyzed", total_bullish)

                                with col2:
                                    total_bearish = reversal_zones_result.get('total_bearish_samples', 0)
                                    st.metric("Bearish Reversals Analyzed", total_bearish)

                                # Recent Swings
                                st.divider()
                                st.markdown("**🔄 Recent Swing Points**")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Swing Highs**")
                                    swing_highs = reversal_zones_result.get('swing_highs', [])
                                    if swing_highs:
                                        highs_table = []
                                        for sh in swing_highs[-5:]:
                                            highs_table.append({
                                                'Bar': sh.get('bar', 'N/A'),
                                                'Price': f"₹{sh.get('price', 0):.2f}"
                                            })
                                        st.dataframe(pd.DataFrame(highs_table), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No swing highs detected")

                                with col2:
                                    st.markdown("**Swing Lows**")
                                    swing_lows = reversal_zones_result.get('swing_lows', [])
                                    if swing_lows:
                                        lows_table = []
                                        for sl in swing_lows[-5:]:
                                            lows_table.append({
                                                'Bar': sl.get('bar', 'N/A'),
                                                'Price': f"₹{sl.get('price', 0):.2f}"
                                            })
                                        st.dataframe(pd.DataFrame(lows_table), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No swing lows detected")

                            else:
                                error_msg = reversal_zones_result.get('error', 'Unknown error') if reversal_zones_result else 'No data available'
                                st.warning(f"⚠️ Unable to calculate reversal zones: {error_msg}")

                # Trading signals based on indicators
                st.divider()
                st.subheader("🎯 Trading Signals")

                if show_rsi:
                    from indicators.ultimate_rsi import UltimateRSI
                    rsi_indicator = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                    rsi_signals = rsi_indicator.get_signals(df_stats)

                    latest_rsi = rsi_signals['ultimate_rsi'][-1]
                    latest_signal = rsi_signals['signal'][-1]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Ultimate RSI Analysis**")
                        ob_threshold = rsi_params['ob_level'] if rsi_params else 80
                        os_threshold = rsi_params['os_level'] if rsi_params else 20
                        rsi_state = "Overbought" if latest_rsi > ob_threshold else "Oversold" if latest_rsi < os_threshold else "Neutral"
                        rsi_color = "red" if latest_rsi > ob_threshold else "green" if latest_rsi < os_threshold else "gray"

                        st.markdown(f"Current RSI: <span style='color:{rsi_color}; font-size:24px;'>{latest_rsi:.2f}</span>", unsafe_allow_html=True)
                        st.write(f"Signal Line: {latest_signal:.2f}")
                        st.write(f"State: **{rsi_state}**")

                    with col2:
                        st.markdown("**RSI Trading Recommendation**")
                        if latest_rsi > ob_threshold:
                            st.warning("⚠️ **OVERBOUGHT** - Consider taking profits or waiting for pullback")
                        elif latest_rsi < os_threshold:
                            st.success("✅ **OVERSOLD** - Potential buying opportunity")
                        elif latest_rsi > latest_signal:
                            st.info("📈 **BULLISH** - RSI above signal line")
                        elif latest_rsi < latest_signal:
                            st.info("📉 **BEARISH** - RSI below signal line")
                        else:
                            st.info("⏸ **NEUTRAL** - No clear signal")

                # Market Context Section
                st.divider()

                try:
                    # Get proximity alert system
                    proximity_system = get_proximity_alert_system(cooldown_minutes=10)

                    # Get current price
                    current_price = df_stats['close'].iloc[-1]

                    # Determine symbol for alerts
                    alert_symbol = "NIFTY" if "NSEI" in symbol_code else "SENSEX" if "BSESN" in symbol_code else symbol_code

                    # Display current price
                    st.metric("Current Price", f"₹{current_price:,.2f}")

                    # Gather and display comprehensive market context
                    market_context = proximity_system._gather_market_context()

                    # Display market context in an expander
                    with st.expander("📊 **Comprehensive Market Context**", expanded=True):
                        st.markdown("### Overall Market Sentiment")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Overall sentiment
                            sentiment = market_context['overall_sentiment']
                            if sentiment == 'BULLISH':
                                st.success(f"🐂 **{sentiment}**")
                            elif sentiment == 'BEARISH':
                                st.error(f"🐻 **{sentiment}**")
                            else:
                                st.info(f"⚖️ **{sentiment}**")

                        with col2:
                            st.metric("Overall Score", f"{market_context['overall_score']:.1f}")

                        st.divider()

                        # Enhanced Market Analysis
                        st.markdown("### 📈 Enhanced Market Analysis")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['technical_indicators_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['technical_indicators_score']:.1f}")

                        # PCR Analysis
                        st.markdown("### 📉 PCR Analysis (Put-Call Ratio)")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['pcr_analysis_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['pcr_analysis_score']:.1f}")

                        # NIFTY ATM Zone
                        st.markdown("### 🎯 NIFTY ATM Zone Summary")
                        verdict = market_context['nifty_atm_verdict']
                        if 'Bullish' in verdict:
                            st.success(f"🐂 **{verdict}**")
                        elif 'Bearish' in verdict:
                            st.error(f"🐻 **{verdict}**")
                        else:
                            st.info(f"⚖️ **{verdict}**")

                        # Option Chain Analysis
                        st.markdown("### 🔗 Option Chain ATM Zone Analysis")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['option_chain_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['option_chain_score']:.1f}")

                    # Show active alerts
                    if all_alerts:
                        st.markdown("**Active Proximity Alerts:**")
                        for alert in all_alerts[:5]:  # Show top 5 alerts
                            if alert.alert_type == 'VOB':
                                emoji = "🟢" if "Bull" in alert.level_type else "🔴"
                                st.write(f"{emoji} **{alert.alert_type}** {alert.level_type}: "
                                        f"₹{alert.level:.2f} "
                                        f"({alert.distance:.2f} pts away)")
                            else:  # HTF
                                emoji = "🟢" if alert.level_type == 'Support' else "🔴"
                                tf_readable = alert.timeframe.replace('T', 'm') if alert.timeframe else ''
                                st.write(f"{emoji} **{alert.alert_type}** {alert.level_type} ({tf_readable}): "
                                        f"₹{alert.level:.2f} "
                                        f"({alert.distance:.2f} pts away)")

                        if notifications_sent > 0:
                            st.success(f"📱 Sent {notifications_sent} Telegram notification(s) with comprehensive market context")
                    else:
                        st.info("ℹ️ No proximity alerts at current price")

                except Exception as e:
                    st.warning(f"⚠️ Proximity alerts unavailable: {str(e)}")

                # Data table
                st.divider()
                st.subheader("📋 Recent Candles Data")

                # Show last 20 candles
                display_df = df_stats.tail(20).copy()
                display_df = display_df.reset_index()

                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                elif display_df.index.name == 'Datetime':
                    display_df['Time'] = display_df.index.strftime('%Y-%m-%d %H:%M')

                st.dataframe(display_df[['open', 'high', 'low', 'close', 'volume']].tail(20),
                           use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error rendering chart: {e}")
            st.write("Error details:", str(e))

    else:
        st.info("⏳ Loading chart... The chart will automatically load and refresh based on your settings")

        st.markdown("""
        ### About Advanced Chart Analysis

        This advanced charting module provides professional-grade technical analysis with 6 powerful indicators:

        #### 📊 Volume Bars
        - TradingView-style volume histogram
        - Green bars for bullish candles (close > open)
        - Red bars for bearish candles (close < open)
        - Essential for confirming price movements and identifying volume spikes

        #### 📦 Volume Order Blocks (BigBeluga)
        - Detects institutional order blocks based on volume and EMA crossovers
        - Shows bullish (support) and bearish (resistance) zones
        - Helps identify high-probability entry/exit zones

        #### 📊 HTF Support/Resistance (BigBeluga)
        - Multi-timeframe pivot analysis (4H, 12H, Daily, Weekly)
        - Identifies key support and resistance levels
        - Non-repainting pivot detection

        #### 👣 Real-Time HTF Volume Footprint (BigBeluga)
        - Volume distribution across price levels
        - Point of Control (POC) - highest volume traded price
        - Value Area - where 70% of volume occurred

        #### 📈 Ultimate RSI (LuxAlgo)
        - Enhanced RSI using price range instead of just price change
        - More responsive to market conditions
        - Signal line for trend confirmation
        - Overbought/Oversold detection

        #### 🎯 OM Indicator (Order Flow & Momentum)
        - **VWAP**: Volume Weighted Average Price for intraday trading
        - **VOB**: Volume Order Blocks with EMA-based detection
        - **HVP**: High Volume Pivots marking significant support/resistance
        - **Delta Module**: Buy/Sell pressure analysis with spike detection
        - **VIDYA**: Variable Index Dynamic Average with trend detection
        - **LTP Trap**: Last Traded Price trap signals for reversal detection
        - Comprehensive order flow analysis combining 6 sub-indicators

        #### 🎯 How to Use
        1. Select market (NIFTY, SENSEX, or DOW)
        2. Choose period and interval (default: 1 day, 1 minute)
        3. Chart loads automatically - no need to click any button!
        4. Set auto-refresh interval (default: 60s) or click Refresh for manual update
        5. Toggle indicators on/off as needed
        6. Analyze chart and trading signals
        7. Use signals to inform your trading decisions

        **Note:** All indicators are converted from Pine Script with high accuracy and optimized for Python/Plotly.
        """)

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME TREND & SUPPORT/RESISTANCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("📊 Multi-Timeframe Trend & Support/Resistance Analysis")
    st.caption("Analyze trend direction and key S/R levels across multiple timeframes")

    try:
        from src.multi_timeframe_analysis import MultiTimeframeAnalyzer, create_mtf_summary_table

        # Initialize analyzer
        mtf_analyzer = MultiTimeframeAnalyzer()

        # Fetch data for different timeframes
        with st.spinner("Fetching multi-timeframe data..."):
            try:
                from advanced_chart_analysis import AdvancedChartAnalysis
                chart_analyzer = AdvancedChartAnalysis()

                # Fetch data for each timeframe
                data_15m = chart_analyzer.fetch_intraday_data(symbol_code, period='5d', interval='15m')
                data_1h = chart_analyzer.fetch_intraday_data(symbol_code, period='5d', interval='1h')
                data_4h = chart_analyzer.fetch_intraday_data(symbol_code, period='1mo', interval='1h')  # Use 1h data for 4h approximation
                data_1d = chart_analyzer.fetch_intraday_data(symbol_code, period='6mo', interval='1d')

                # Analyze all timeframes
                mtf_results = mtf_analyzer.analyze_all_timeframes(
                    data_1d=data_1d,
                    data_4h=data_4h,
                    data_1h=data_1h,
                    data_15m=data_15m
                )

                if mtf_results:
                    # Create summary table
                    summary_table = create_mtf_summary_table(mtf_results)

                    # Display the table
                    st.markdown("#### Trend & S/R Summary Table")
                    st.dataframe(summary_table, use_container_width=True, hide_index=True)

                    # Display detailed analysis for each timeframe
                    st.markdown("#### Detailed Analysis by Timeframe")

                    # Create tabs for each timeframe
                    mtf_tabs = st.tabs(["15 Min", "1 Hour", "4 Hour", "1 Day"])

                    timeframe_map = {
                        0: '15m',
                        1: '1h',
                        2: '4h',
                        3: '1d'
                    }

                    for idx, (tab_idx, timeframe) in enumerate(timeframe_map.items()):
                        if timeframe in mtf_results:
                            result = mtf_results[timeframe]

                            with mtf_tabs[tab_idx]:
                                # Trend overview
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    trend_color = {
                                        'UPTREND': '🟢',
                                        'DOWNTREND': '🔴',
                                        'SIDEWAYS': '🟡'
                                    }.get(result.trend_direction, '⚪')

                                    st.metric(
                                        "Trend Direction",
                                        f"{trend_color} {result.trend_direction}",
                                        delta=f"{result.trend_strength:.0f}% strength"
                                    )

                                with col2:
                                    rsi_zone = "Overbought" if result.rsi > 70 else "Oversold" if result.rsi < 30 else "Neutral"
                                    st.metric("RSI", f"{result.rsi:.1f}", delta=rsi_zone)

                                with col3:
                                    macd_signal = "Bullish" if result.macd > result.macd_signal else "Bearish"
                                    st.metric("MACD", f"{result.macd:.2f}", delta=macd_signal)

                                # Support and Resistance levels
                                st.markdown("##### Support & Resistance Levels")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Support Levels:**")
                                    if result.support_1 > 0:
                                        dist_s1 = result.distance_from_support1_pct
                                        st.markdown(f"- S1: ₹{result.support_1:,.2f} ({dist_s1:.1f}% below)")
                                    if result.support_2 > 0:
                                        st.markdown(f"- S2: ₹{result.support_2:,.2f}")
                                    if result.support_3 > 0:
                                        st.markdown(f"- S3: ₹{result.support_3:,.2f}")

                                with col2:
                                    st.markdown("**Resistance Levels:**")
                                    if result.resistance_1 > 0:
                                        dist_r1 = result.distance_from_resistance1_pct
                                        st.markdown(f"- R1: ₹{result.resistance_1:,.2f} ({dist_r1:.1f}% above)")
                                    if result.resistance_2 > 0:
                                        st.markdown(f"- R2: ₹{result.resistance_2:,.2f}")
                                    if result.resistance_3 > 0:
                                        st.markdown(f"- R3: ₹{result.resistance_3:,.2f}")

                                # Moving Averages
                                st.markdown("##### Moving Averages")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    if result.sma_20 > 0:
                                        above_below = "above" if result.current_price > result.sma_20 else "below"
                                        st.metric(
                                            "SMA 20",
                                            f"₹{result.sma_20:,.2f}",
                                            delta=f"Price {abs(result.distance_from_sma20_pct):.1f}% {above_below}"
                                        )

                                with col2:
                                    if result.sma_50 > 0:
                                        above_below = "above" if result.current_price > result.sma_50 else "below"
                                        st.metric("SMA 50", f"₹{result.sma_50:,.2f}", delta=f"{above_below}")

                                with col3:
                                    if result.sma_200 > 0:
                                        above_below = "above" if result.current_price > result.sma_200 else "below"
                                        st.metric("SMA 200", f"₹{result.sma_200:,.2f}", delta=f"{above_below}")

                                # Price Structure
                                st.markdown("##### Price Structure")

                                structure_text = []
                                if result.higher_highs and result.higher_lows:
                                    structure_text.append("🟢 **Strong Uptrend**: Higher Highs + Higher Lows")
                                elif result.lower_highs and result.lower_lows:
                                    structure_text.append("🔴 **Strong Downtrend**: Lower Highs + Lower Lows")
                                elif result.higher_highs:
                                    structure_text.append("🟢 Higher Highs detected")
                                elif result.lower_lows:
                                    structure_text.append("🔴 Lower Lows detected")
                                else:
                                    structure_text.append("🟡 **Choppy/Ranging Market**")

                                for text in structure_text:
                                    st.markdown(text)

                else:
                    st.warning("No multi-timeframe data available")

            except Exception as e:
                st.error(f"Error in multi-timeframe analysis: {e}")
                import traceback
                st.code(traceback.format_exc())

    except ImportError as e:
        st.warning(f"Multi-timeframe analysis module not available: {e}")
        st.info("Install the module: src/multi_timeframe_analysis.py")
    except Exception as e:
        st.error(f"Error loading multi-timeframe analysis: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 7: NIFTY OPTION SCREENER V7.0 (AUTO-LOAD WITH SMART CACHING)
# ═══════════════════════════════════════════════════════════════════════

with tab7:
    st.header("🎯 NIFTY Option Screener v7.0")
    st.caption("100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS")

    # Auto-load the Option Screener
    try:
        from NiftyOptionScreener import render_nifty_option_screener
        render_nifty_option_screener()
    except ImportError as e:
        st.error(f"❌ Failed to load Nifty Option Screener v7.0: {e}")
        st.info("Please ensure NiftyOptionScreener.py is in the project directory")
    except Exception as e:
        st.error(f"❌ Error rendering Nifty Option Screener: {e}")
        st.exception(e)

    # Force clean tab completion to ensure tabs 8 & 9 render
    st.write("")

# ═══════════════════════════════════════════════════════════════════════
# TAB 8: ENHANCED MARKET DATA
# ═══════════════════════════════════════════════════════════════════════

with tab8:
    st.markdown("# 🌐 Enhanced Market Data Analysis")
    st.markdown("### ✅ Tab 8 LOADED")
    st.write("=" * 50)
    st.info("If you see this, tab 8 is working!")
    st.write("=" * 50)

    try:
        st.caption("Comprehensive market data from Dhan API + Yahoo Finance | India VIX, Sector Rotation, Global Markets, Intermarket Data, Gamma Squeeze, Intraday Timing")

        # DISABLED AUTO-FETCH - Only fetch when user clicks Refresh
        # This prevents tab from hanging when Dhan API is down (holidays, weekends)
        # Auto-fetch code commented out to prevent blank tabs

        # Control buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True, key="refresh_enhanced_data_btn"):
                with st.spinner("Refreshing market data..."):
                    try:
                        from enhanced_market_data import get_enhanced_market_data
                        enhanced_data = get_enhanced_market_data()
                        st.session_state.enhanced_market_data = enhanced_data
                        st.success("✅ Data refreshed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to refresh data: {e}")

        with col2:
            if 'enhanced_market_data' in st.session_state:
                data = st.session_state.enhanced_market_data
                st.caption(f"📅 Last Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S IST')}")

        # Display enhanced market data if available
        if 'enhanced_market_data' in st.session_state:
            try:
                from enhanced_market_display import render_enhanced_market_data_tab
                render_enhanced_market_data_tab(st.session_state.enhanced_market_data)
            except Exception as e:
                st.error(f"❌ Error displaying enhanced data: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.info("""
            ℹ️ Enhanced market data will auto-load on first visit and refresh every 5 minutes.

            **Data Sources:**
            - 📊 **Dhan API:** India VIX, All Sector Indices (IT, Auto, Pharma, Metal, FMCG, Realty, Energy)
            - 🌍 **Yahoo Finance:** Global Markets (S&P 500, Nasdaq, Dow, Nikkei, Hang Seng, etc.)
            - 💰 **Intermarket:** USD Index, Crude Oil, Gold, USD/INR, US 10Y Treasury, Bitcoin

            **Advanced Analysis:**
            - ⚡ **India VIX Analysis:** Fear & Greed Index with sentiment scoring
            - 🏢 **Sector Rotation Model:** Identify market leadership and rotation patterns
            - 🎯 **Gamma Squeeze Detection:** Option market makers hedging analysis
            - ⏰ **Intraday Seasonality:** Time-based trading recommendations
            - 🌐 **Global Correlation:** How worldwide markets affect Indian markets

            **All data is presented in comprehensive tables with bias scores and trading insights!**
            """)
    except Exception as e:
        st.error(f"❌ Critical error in Enhanced Market Data tab: {e}")
        import traceback
        st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════
# TAB 9: NSE STOCK SCREENER
# ═══════════════════════════════════════════════════════════════════════

with tab9:
    st.markdown("# 🔍 NSE Stock Screener")

    # LAZY LOADING - Only load when user clicks button (performance fix)
    if st.button("🔄 Load Stock Screener", type="primary", key="load_screener_btn"):
        st.session_state.load_stock_screener = True

    if st.session_state.get('load_stock_screener', False):
        try:
            with st.spinner("Loading stock screener..."):
                from nse_stock_screener_dhan import render_nse_stock_screener_tab
                render_nse_stock_screener_tab()
        except Exception as e:
            st.error(f"❌ Error loading NSE Stock Screener: {e}")
            st.info("Ensure nse_stock_screener_dhan.py exists and all dependencies are installed.")
    else:
        st.info("👆 Click 'Load Stock Screener' to start screening NSE stocks.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 10: NIFTY FUTURES ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab10:
    st.markdown("# 📈 NIFTY Futures Analysis")

    # LAZY LOADING - Only fetch when user clicks button (performance fix)
    col1, col2 = st.columns([1, 3])
    with col1:
        load_futures = st.button("🔄 Load Futures Data", type="primary", key="load_futures_btn")
    with col2:
        if 'futures_data_cache' in st.session_state:
            st.caption("✅ Data loaded - Click to refresh")

    if load_futures or 'futures_data_cache' in st.session_state:
        try:
            from src.nifty_futures_ui import render_nifty_futures_dashboard

            spot_price = nifty_data.get('spot_price', 25000.0)
            futures_data = None

            # Only fetch if button clicked (not from cache)
            if load_futures:
                with st.spinner("Loading futures data..."):
                    try:
                        from dhan_data_fetcher import get_nifty_futures_data
                        futures_result = get_nifty_futures_data()

                        if futures_result.get('spot_price'):
                            spot_price = futures_result['spot_price']

                        if futures_result.get('success'):
                            futures_data = {
                                'current_month': futures_result.get('current_month', {}),
                                'next_month': futures_result.get('next_month', {}),
                                'data_source': futures_result.get('data_source', 'unknown')
                            }
                            st.session_state.futures_data_cache = futures_data
                            st.session_state.futures_spot_cache = spot_price
                            st.success("✅ Futures data loaded!")
                    except Exception as e:
                        st.warning(f"Could not fetch futures data: {e}")
            else:
                # Use cached data
                futures_data = st.session_state.get('futures_data_cache')
                spot_price = st.session_state.get('futures_spot_cache', spot_price)

            option_chain_data = st.session_state.get('option_chain')

            render_nifty_futures_dashboard(
                spot_price=spot_price,
                futures_data=futures_data,
                participant_data=None,
                option_chain_data=option_chain_data,
                historical_data=None
            )

        except ImportError as e:
            st.error(f"❌ Error importing NIFTY Futures UI: {e}")
            st.info("Required: src/nifty_futures_ui.py, src/nifty_futures_analyzer.py")
        except Exception as e:
            st.error(f"❌ Error loading NIFTY Futures Analysis: {e}")
    else:
        st.info("👆 Click 'Load Futures Data' to view NIFTY Futures analysis.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 11: AI TRAINING & MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

with tab11:
    st.markdown("# 🤖 AI Training & Model Management")
    st.caption("Train XGBoost models on real trading data | Track predictions & outcomes | Manage model versions")

    # LAZY LOADING - Only load when user clicks button (performance fix)
    if st.button("🔄 Load AI Training Dashboard", type="primary", key="load_ai_training_btn"):
        st.session_state.load_ai_training = True

    if st.session_state.get('load_ai_training', False):
        try:
            with st.spinner("Loading AI Training dashboard..."):
                from src.ai_training_ui import render_ai_training_dashboard
                render_ai_training_dashboard()
        except ImportError as e:
            st.error(f"❌ AI Training module not available: {e}")
            st.info("Required: src/ai_training_ui.py, src/training_data_collector.py, src/model_trainer_pipeline.py")
        except Exception as e:
            st.error(f"❌ Error loading AI Training: {e}")
    else:
        st.info("👆 Click 'Load AI Training Dashboard' to access AI model training and management.")

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.divider()
st.caption(f"Last Updated (IST): {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')} | Auto-refresh: {AUTO_REFRESH_INTERVAL}s")
st.caption(f"🤖 AI Market Analysis: Runs every 30 minutes during market hours | Last AI analysis: {datetime.fromtimestamp(st.session_state.last_ai_analysis_time).strftime('%H:%M:%S') if st.session_state.last_ai_analysis_time > 0 else 'Never'}")
