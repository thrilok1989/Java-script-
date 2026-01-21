"""
🎯 NIFTY Option Screener v7.0 - Standalone Application
100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR

This is a standalone version of the NIFTY Option Screener extracted from the main trading app.
Run this app separately: streamlit run nifty_screener_app.py
"""

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

# Import configuration
from config import *

# Import market hours scheduler
from market_hours_scheduler import is_within_trading_hours

# Import Dhan API
from dhan_api import check_dhan_connection

# Import the main Nifty Option Screener module
try:
    from NiftyOptionScreener import render_nifty_option_screener
    NIFTY_SCREENER_AVAILABLE = True
except Exception as e:
    NIFTY_SCREENER_AVAILABLE = False
    print(f"Error: Could not import NiftyOptionScreener: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# GET SPOT PRICE FOR BROWSER TAB TITLE
# ═══════════════════════════════════════════════════════════════════════
page_title = "NIFTY Option Screener v7.0"
if 'last_spot_price' in st.session_state and st.session_state.last_spot_price:
    spot = st.session_state.last_spot_price
    page_title = f"NIFTY ₹{spot:,.2f} | Option Screener"

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

    /* Compact layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }

    /* Style for IST time display */
    .ist-time {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f77b4;
        background: #f0f2f6;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Seller perspective styling */
    .seller-explanation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .seller-explanation h3 {
        color: white;
        margin-top: 0;
    }

    .seller-bullish {
        color: #00ff00;
        font-weight: bold;
    }

    .seller-bearish {
        color: #ff6b6b;
        font-weight: bold;
    }

    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# AUTO-REFRESH LOGIC - DYNAMIC INTERVALS
# ═══════════════════════════════════════════════════════════════════════

def get_auto_refresh_interval():
    """
    Determines auto-refresh interval based on market hours:
    - During trading hours (9:15 AM - 3:30 PM IST): 10 seconds
    - Outside trading hours: 60 seconds
    - Late night (11 PM - 6 AM IST): 300 seconds (5 minutes)
    """
    try:
        from pytz import timezone
        ist = timezone('Asia/Kolkata')
        now = datetime.now(ist)
        hour = now.hour
        minute = now.minute

        # Trading hours: 9:15 AM - 3:30 PM IST
        if (hour == 9 and minute >= 15) or (10 <= hour < 15) or (hour == 15 and minute <= 30):
            return 10000  # 10 seconds during market hours

        # Late night: 11 PM - 6 AM IST
        elif hour >= 23 or hour < 6:
            return 300000  # 5 minutes late night

        # Outside trading hours
        else:
            return 60000  # 1 minute

    except Exception as e:
        logger.error(f"Error calculating refresh interval: {e}")
        return 60000  # Default to 1 minute

# Initialize auto-refresh
refresh_interval = get_auto_refresh_interval()
count = st_autorefresh(interval=refresh_interval, limit=None, key="nifty_screener_refresh")

# ═══════════════════════════════════════════════════════════════════════
# HEADER & TITLE
# ═══════════════════════════════════════════════════════════════════════

st.title("🎯 NIFTY Option Screener v7.0")
st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h4 style='color: #667eea;'>100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR</h4>
</div>
""", unsafe_allow_html=True)

# Display refresh status
refresh_status = "🟢 Live (10s refresh)" if refresh_interval == 10000 else "🟡 Monitoring (60s refresh)" if refresh_interval == 60000 else "🔵 Low Activity (5m refresh)"
st.caption(f"{refresh_status} | Auto-refresh enabled | Count: {count}")

# ═══════════════════════════════════════════════════════════════════════
# CONNECTION CHECK
# ═══════════════════════════════════════════════════════════════════════

# Check Dhan API connection
try:
    dhan_status = check_dhan_connection()
    if not dhan_status.get('success', False):
        st.warning("⚠️ Dhan API connection issue. Some features may be limited.")
        if 'error' in dhan_status:
            st.caption(f"Error: {dhan_status['error']}")
except Exception as e:
    st.warning(f"⚠️ Could not verify Dhan connection: {e}")

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT - RENDER NIFTY OPTION SCREENER
# ═══════════════════════════════════════════════════════════════════════

if NIFTY_SCREENER_AVAILABLE:
    try:
        render_nifty_option_screener()
    except Exception as e:
        st.error(f"❌ Error rendering NIFTY Option Screener: {e}")
        st.info("An error occurred while rendering the screener. Please check the logs for details.")
        import traceback
        with st.expander("📋 Error Details"):
            st.code(traceback.format_exc())

        # Provide troubleshooting tips
        st.markdown("""
        ### 🔧 Troubleshooting Tips:
        1. **Check Credentials**: Ensure DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are configured in `.streamlit/secrets.toml`
        2. **Network Issues**: Verify your internet connection
        3. **API Limits**: Check if you've exceeded Dhan API rate limits
        4. **Market Hours**: Some features work best during trading hours (9:15 AM - 3:30 PM IST)
        5. **Dependencies**: Ensure all required Python packages are installed

        **Need help?** Check the console/terminal for detailed error messages.
        """)
else:
    st.error("❌ NIFTY Option Screener module not available")
    st.info("The NiftyOptionScreener.py module failed to load. Please check the console/terminal for import errors.")

    st.markdown("""
    ### 🔧 Setup Instructions:

    1. **Ensure all dependencies are installed:**
    ```bash
    pip install -r requirements.txt
    ```

    2. **Configure credentials in `.streamlit/secrets.toml`:**
    ```toml
    [DHAN]
    CLIENT_ID = "your_client_id"
    ACCESS_TOKEN = "your_access_token"

    [TELEGRAM]
    BOT_TOKEN = "your_telegram_bot_token"
    CHAT_ID = "your_telegram_chat_id"
    ```

    3. **Run the app:**
    ```bash
    streamlit run nifty_screener_app.py
    ```
    """)

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🎯 NIFTY Option Screener v7.0</strong> - Standalone Application</p>
    <p>Built with Streamlit | Data from Dhan API | Real-time Market Analysis</p>
    <p><em>Disclaimer: This tool is for educational purposes only. Trading in options involves risk.</em></p>
</div>
""", unsafe_allow_html=True)
