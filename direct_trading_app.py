"""
Direct Trading App - Lightweight Trading Platform
==================================================

A focused trading interface for NIFTY and SENSEX options with ATM ± 5 strikes.
Fetches its own data using the same infrastructure as the main app.

Features:
- ATM ± 5 strike selection
- Quick order placement (Market/Limit)
- Live positions tracking
- Order book management
- Real-time P&L

Run this app:
    streamlit run direct_trading_app.py --server.port 8502
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import requests

# Import shared modules from main app
from config import *
from dhan_data_fetcher import DhanDataFetcher, get_nifty_data, get_sensex_data
from market_hours_scheduler import is_within_trading_hours, get_market_status

# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING - Independent cache for this app
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_nifty_data():
    """Fetch fresh NIFTY data"""
    return get_nifty_data()

@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_sensex_data():
    """Fetch fresh SENSEX data"""
    return get_sensex_data()

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Direct Trading - NIFTY/SENSEX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Compact header */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Large strike price buttons */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: 600;
    }

    /* Highlight ATM strike */
    .atm-strike {
        background-color: #ffd700 !important;
        color: #000 !important;
    }

    /* Buy button */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #089981;
        border-color: #089981;
    }

    /* Sell button */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f23645;
        border-color: #f23645;
        color: white;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.title("⚡ Direct Trading")

with col2:
    market_status = get_market_status()
    status_emoji = "🟢" if market_status.get('is_open') else "🔴"
    st.markdown(f"### {status_emoji} {market_status.get('session', 'Unknown')}")

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# MAIN TRADING INTERFACE
# ═══════════════════════════════════════════════════════════════════════

# Index Selection
col1, col2 = st.columns([1, 3])

with col1:
    trade_index = st.selectbox(
        "Select Index",
        ["NIFTY", "SENSEX"],
        key="trade_index"
    )

# Get index data - Fetch fresh data with Streamlit caching
with st.spinner(f"Loading {trade_index} data..."):
    if trade_index == "NIFTY":
        index_data = fetch_nifty_data()
        strike_gap = STRIKE_INTERVALS.get("NIFTY", 50)
    else:
        index_data = fetch_sensex_data()
        strike_gap = STRIKE_INTERVALS.get("SENSEX", 100)

# Check if data is available
if not index_data or not index_data.get('success'):
    st.error(f"❌ Failed to load {trade_index} data")

    error_msg = index_data.get('error', 'Unknown error') if index_data else 'No data returned'
    st.error(f"**Error:** {error_msg}")

    # Show helpful messages
    if 'credentials' in error_msg.lower() or 'secrets' in error_msg.lower():
        st.info("""
        **Dhan API Setup Required:**
        1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
        2. Add your Dhan API credentials
        3. Restart the app
        """)
    else:
        st.info("""
        **Troubleshooting:**
        - Check internet connection
        - Verify Dhan API credentials in `.streamlit/secrets.toml`
        - Wait a moment and click the Refresh button
        - Check if market is open (data works better during trading hours)
        """)

    with st.expander("🔍 Debug Info"):
        st.json({
            'data_exists': index_data is not None,
            'success': index_data.get('success') if index_data else None,
            'error': error_msg,
            'spot_price': index_data.get('spot_price') if index_data else None,
            'atm_strike': index_data.get('atm_strike') if index_data else None
        })

    st.stop()

# Extract data
spot_price = index_data.get('spot_price', 0)
atm_strike = index_data.get('atm_strike', 0)
current_expiry = index_data.get('current_expiry', 'N/A')

# Display key metrics
with col2:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Spot Price", f"₹{spot_price:,.2f}")

    with metric_col2:
        st.metric("ATM Strike", f"{int(atm_strike):,}")

    with metric_col3:
        st.metric("Strike Gap", f"{strike_gap}")

    with metric_col4:
        st.metric("Expiry", current_expiry)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# ATM ± 5 STRIKE GRID
# ═══════════════════════════════════════════════════════════════════════

st.subheader("🎯 ATM ± 5 Strikes")

# Generate strikes
strikes = []
for i in range(-5, 6):  # -5 to +5
    strike = atm_strike + (i * strike_gap)
    strikes.append({
        'strike': int(strike),
        'offset': i,
        'is_atm': i == 0
    })

# Create two columns for CE and PE
col_ce, col_pe = st.columns(2)

with col_ce:
    st.markdown("### 📈 CALL (CE)")

with col_pe:
    st.markdown("### 📉 PUT (PE)")

# Display strikes in a grid
for strike_info in strikes:
    strike = strike_info['strike']
    offset = strike_info['offset']
    is_atm = strike_info['is_atm']

    col_ce, col_pe = st.columns(2)

    # Strike label
    if offset == 0:
        label = f"{strike:,} (ATM)"
    elif offset > 0:
        label = f"{strike:,} (ATM+{offset})"
    else:
        label = f"{strike:,} (ATM{offset})"

    with col_ce:
        if st.button(
            f"BUY {label} CE",
            key=f"ce_{strike}",
            type="primary" if is_atm else "secondary",
            use_container_width=True
        ):
            st.session_state['selected_strike'] = strike
            st.session_state['selected_type'] = 'CE'
            st.session_state['show_order_dialog'] = True

    with col_pe:
        if st.button(
            f"BUY {label} PE",
            key=f"pe_{strike}",
            type="primary" if is_atm else "secondary",
            use_container_width=True
        ):
            st.session_state['selected_strike'] = strike
            st.session_state['selected_type'] = 'PE'
            st.session_state['show_order_dialog'] = True

# ═══════════════════════════════════════════════════════════════════════
# ORDER PLACEMENT DIALOG
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.get('show_order_dialog', False):
    st.divider()
    st.subheader("📋 Order Details")

    selected_strike = st.session_state.get('selected_strike')
    selected_type = st.session_state.get('selected_type')

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.info(f"**Strike:** {selected_strike:,} {selected_type}")

    with col2:
        lots = st.number_input(
            "Lots",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="order_lots"
        )

    with col3:
        order_type = st.selectbox(
            "Order Type",
            ["MARKET", "LIMIT"],
            key="order_type"
        )

    lot_size = LOT_SIZES.get(trade_index, 50)
    total_quantity = lots * lot_size

    if order_type == "LIMIT":
        limit_price = st.number_input(
            "Limit Price",
            min_value=0.05,
            value=50.0,
            step=0.05,
            key="limit_price"
        )

    st.markdown(f"**Total Quantity:** {total_quantity} ({lots} lots × {lot_size})")

    # Order buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("✅ Place Order", type="primary", use_container_width=True):
            try:
                # Fetch option chain to get security ID
                fetcher = DhanDataFetcher()
                option_chain_result = fetcher.fetch_option_chain(trade_index, current_expiry)

                if not option_chain_result.get('success'):
                    st.error(f"❌ Failed to fetch option chain: {option_chain_result.get('error')}")
                else:
                    # Find security ID
                    option_data = option_chain_result.get('data', {})
                    option_list = option_data.get(selected_type, [])

                    security_id = None
                    for option in option_list:
                        if option.get('strike_price') == selected_strike:
                            security_id = option.get('security_id')
                            break

                    if not security_id:
                        st.error(f"❌ Security ID not found for {trade_index} {selected_strike} {selected_type}")
                    else:
                        # Place order
                        creds = get_dhan_credentials()

                        order_data = {
                            "dhanClientId": creds['client_id'],
                            "transactionType": "BUY",
                            "exchangeSegment": "NSE_FNO" if trade_index == "NIFTY" else "BSE_FNO",
                            "productType": "INTRADAY",
                            "orderType": order_type,
                            "validity": "DAY",
                            "securityId": str(security_id),
                            "quantity": int(total_quantity),
                            "disclosedQuantity": 0,
                            "price": float(limit_price) if order_type == "LIMIT" else 0.0,
                            "triggerPrice": 0.0,
                            "afterMarketOrder": False
                        }

                        headers = {
                            'Content-Type': 'application/json',
                            'access-token': creds['access_token']
                        }

                        response = requests.post(
                            "https://api.dhan.co/v2/orders",
                            json=order_data,
                            headers=headers,
                            timeout=10
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Order placed!\n\nOrder ID: {result.get('orderId')}\nStatus: {result.get('orderStatus')}")
                            st.session_state['show_order_dialog'] = False
                        else:
                            st.error(f"❌ Order failed: {response.text}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state['show_order_dialog'] = False
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# POSITIONS & ORDERS (SIDEBAR)
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📊 Positions & Orders")

    # Get positions
    try:
        creds = get_dhan_credentials()
        if creds:
            headers = {
                'Content-Type': 'application/json',
                'access-token': creds['access_token']
            }

            # Fetch positions
            pos_response = requests.get(
                "https://api.dhan.co/v2/positions",
                headers=headers,
                timeout=10
            )

            if pos_response.status_code == 200:
                positions = pos_response.json()

                if positions:
                    st.subheader("📈 Open Positions")
                    for pos in positions:
                        with st.expander(f"{pos.get('tradingSymbol', 'N/A')}"):
                            st.write(f"Qty: {pos.get('quantity', 0)}")
                            st.write(f"Avg: ₹{pos.get('avgPrice', 0):.2f}")
                            st.write(f"LTP: ₹{pos.get('ltp', 0):.2f}")
                            st.write(f"P&L: ₹{pos.get('realizedProfit', 0):.2f}")
                else:
                    st.info("No open positions")

            # Fetch orders
            ord_response = requests.get(
                "https://api.dhan.co/v2/orders",
                headers=headers,
                timeout=10
            )

            if ord_response.status_code == 200:
                orders = ord_response.json()

                if orders:
                    st.subheader("📋 Today's Orders")
                    for order in orders[-5:]:  # Last 5 orders
                        status = order.get('orderStatus', 'N/A')
                        symbol = order.get('tradingSymbol', 'N/A')
                        st.write(f"{symbol}: {status}")
                else:
                    st.info("No orders today")

    except Exception as e:
        st.warning(f"Unable to fetch positions/orders: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.divider()
st.caption("⚡ Direct Trading App | Powered by Dhan API")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
