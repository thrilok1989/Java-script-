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

# Updated lot sizes
LOT_SIZES_TRADING = {
    "NIFTY": 60,
    "SENSEX": 20
}

@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_nifty_data():
    """Fetch fresh NIFTY data"""
    return get_nifty_data()

@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_sensex_data():
    """Fetch fresh SENSEX data"""
    return get_sensex_data()

@st.cache_data(ttl=5)  # Cache for 5 seconds (more frequent for LTP)
def fetch_option_chain(index, expiry):
    """Fetch option chain for LTP data"""
    fetcher = DhanDataFetcher()
    return fetcher.fetch_option_chain(index, expiry)

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
lot_size = LOT_SIZES_TRADING.get(trade_index, 50)

with col2:
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric("Spot Price", f"₹{spot_price:,.2f}")

    with metric_col2:
        st.metric("ATM Strike", f"{int(atm_strike):,}")

    with metric_col3:
        st.metric("Strike Gap", f"{strike_gap}")

    with metric_col4:
        st.metric("Lot Size", f"{lot_size}")

    with metric_col5:
        st.metric("Expiry", current_expiry)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# FETCH OPTION CHAIN FOR LTP DATA
# ═══════════════════════════════════════════════════════════════════════

# Fetch option chain to get LTP
option_chain_data = fetch_option_chain(trade_index, current_expiry)

# Parse option chain to get LTP for each strike
ltp_map = {'CE': {}, 'PE': {}}

if option_chain_data and option_chain_data.get('success'):
    option_data = option_chain_data.get('data', {})

    # Dhan API returns data in 'oc' (option chain) structure
    # Structure: {"oc": {"25000": {"ce": {...}, "pe": {...}}, "25050": {...}}}
    oc_data = option_data.get('oc', {})

    if oc_data:
        # Parse each strike
        for strike_str, strike_data in oc_data.items():
            try:
                strike = int(float(strike_str))
            except (ValueError, TypeError):
                continue

            # Parse CE (Call) data
            ce_data = strike_data.get('ce')
            if ce_data:
                ltp = ce_data.get('last_price', 0) or ce_data.get('ltp', 0)
                if ltp:
                    ltp_map['CE'][strike] = float(ltp)

            # Parse PE (Put) data
            pe_data = strike_data.get('pe')
            if pe_data:
                ltp = pe_data.get('last_price', 0) or pe_data.get('ltp', 0)
                if ltp:
                    ltp_map['PE'][strike] = float(ltp)
    else:
        # Show debug info if no option chain data
        with st.expander("⚠️ No Option Chain Data - Click to Debug"):
            st.warning("Option chain data structure is empty or unexpected.")
            st.json({
                'success': option_chain_data.get('success'),
                'data_keys': list(option_data.keys()) if option_data else [],
                'sample_data': str(option_data)[:500] if option_data else 'No data'
            })
else:
    st.warning("⚠️ Unable to fetch option chain data. LTP will not be available.")
    if option_chain_data:
        st.error(f"Error: {option_chain_data.get('error', 'Unknown error')}")

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

# Create columns for headers
col_ce_btn, col_ce_ltp, col_pe_ltp, col_pe_btn = st.columns([3, 1, 1, 3])

with col_ce_btn:
    st.markdown("### 📈 CALL (CE)")

with col_ce_ltp:
    st.markdown("**LTP**")

with col_pe_ltp:
    st.markdown("**LTP**")

with col_pe_btn:
    st.markdown("### 📉 PUT (PE)")

# Display strikes in a grid with LTP
for strike_info in strikes:
    strike = strike_info['strike']
    offset = strike_info['offset']
    is_atm = strike_info['is_atm']

    # Get LTP values
    ce_ltp = ltp_map['CE'].get(strike, 0)
    pe_ltp = ltp_map['PE'].get(strike, 0)

    # 4 columns: CE Button | CE LTP | PE LTP | PE Button
    col_ce_btn, col_ce_ltp, col_pe_ltp, col_pe_btn = st.columns([3, 1, 1, 3])

    # Strike label
    if offset == 0:
        label = f"{strike:,} (ATM)"
    elif offset > 0:
        label = f"{strike:,} (ATM+{offset})"
    else:
        label = f"{strike:,} (ATM{offset})"

    with col_ce_btn:
        if st.button(
            f"BUY {label} CE",
            key=f"ce_{strike}",
            type="primary" if is_atm else "secondary",
            use_container_width=True
        ):
            st.session_state['selected_strike'] = strike
            st.session_state['selected_type'] = 'CE'
            st.session_state['show_order_dialog'] = True

    with col_ce_ltp:
        if ce_ltp > 0:
            st.markdown(f"**₹{ce_ltp:.2f}**")
        else:
            st.markdown("—")

    with col_pe_ltp:
        if pe_ltp > 0:
            st.markdown(f"**₹{pe_ltp:.2f}**")
        else:
            st.markdown("—")

    with col_pe_btn:
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

    # Get LTP for selected strike
    selected_ltp = ltp_map[selected_type].get(selected_strike, 0)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        st.info(f"**Strike:** {selected_strike:,} {selected_type}")

    with col2:
        if selected_ltp > 0:
            st.metric("LTP", f"₹{selected_ltp:.2f}")
        else:
            st.warning("LTP: N/A")

    with col3:
        lots = st.number_input(
            "Lots",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="order_lots"
        )

    with col4:
        order_type = st.selectbox(
            "Order Type",
            ["MARKET", "LIMIT"],
            key="order_type"
        )

    lot_size_dialog = LOT_SIZES_TRADING.get(trade_index, 50)
    total_quantity = lots * lot_size_dialog

    # Calculate estimated cost
    if selected_ltp > 0:
        estimated_cost = total_quantity * selected_ltp
    else:
        estimated_cost = 0

    if order_type == "LIMIT":
        limit_price = st.number_input(
            "Limit Price",
            min_value=0.05,
            value=float(selected_ltp) if selected_ltp > 0 else 50.0,
            step=0.05,
            key="limit_price"
        )
        # Recalculate with limit price
        estimated_cost = total_quantity * limit_price

    # Display calculation breakdown
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Calculation:**")
        st.markdown(f"• Lots: **{lots}**")
        st.markdown(f"• Lot Size: **{lot_size_dialog}**")
        st.markdown(f"• Total Quantity: **{total_quantity}** ({lots} × {lot_size_dialog})")
        if order_type == "LIMIT":
            st.markdown(f"• Limit Price: **₹{limit_price:.2f}**")
        else:
            st.markdown(f"• LTP: **₹{selected_ltp:.2f}**")

    with col2:
        st.markdown("**Cost Breakdown:**")
        if order_type == "LIMIT":
            st.markdown(f"• {total_quantity} × ₹{limit_price:.2f}")
        else:
            st.markdown(f"• {total_quantity} × ₹{selected_ltp:.2f}")
        st.markdown(f"### **Total: ₹{estimated_cost:,.2f}**")

        if selected_ltp > 0:
            st.caption("💡 Estimated cost based on current LTP" if order_type == "MARKET" else "💡 Total cost at limit price")

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
                    # Find security ID from 'oc' structure
                    option_data = option_chain_result.get('data', {})
                    oc_data = option_data.get('oc', {})

                    security_id = None

                    # Convert selected strike to string key
                    strike_key = str(selected_strike)

                    if strike_key in oc_data:
                        strike_data = oc_data[strike_key]

                        # Get CE or PE data based on selection
                        option_type_key = 'ce' if selected_type == 'CE' else 'pe'
                        option_info = strike_data.get(option_type_key)

                        if option_info:
                            # Dhan API uses 'SEM_EXM_EXCH_ID' for security ID
                            security_id = option_info.get('SEM_EXM_EXCH_ID') or option_info.get('security_id')

                    if not security_id:
                        st.error(f"❌ Security ID not found for {trade_index} {selected_strike} {selected_type}")
                        with st.expander("🔍 Debug - Option Chain Structure"):
                            st.json({
                                'strike_key': strike_key,
                                'oc_keys': list(oc_data.keys())[:10] if oc_data else [],
                                'strike_data_exists': strike_key in oc_data,
                                'option_data_sample': str(oc_data.get(strike_key, {}))[:300] if strike_key in oc_data else 'Strike not found'
                            })
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
                    for idx, pos in enumerate(positions):
                        symbol = pos.get('tradingSymbol', 'N/A')
                        qty = pos.get('quantity', 0)
                        avg_price = pos.get('avgPrice', 0)
                        ltp = pos.get('ltp', 0)
                        unrealized_pnl = pos.get('unrealizedProfit', 0)
                        realized_pnl = pos.get('realizedProfit', 0)

                        # Calculate P&L
                        if ltp and avg_price and qty:
                            pnl = (ltp - avg_price) * qty
                            pnl_pct = ((ltp - avg_price) / avg_price * 100) if avg_price > 0 else 0
                        else:
                            pnl = unrealized_pnl or 0
                            pnl_pct = 0

                        # Color based on P&L
                        pnl_color = "🟢" if pnl >= 0 else "🔴"

                        with st.container():
                            st.markdown(f"**{symbol}** {pnl_color}")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Qty", f"{qty}")
                                st.metric("Avg", f"₹{avg_price:.2f}")
                            with col2:
                                st.metric("LTP", f"₹{ltp:.2f}")
                                st.metric("P&L", f"₹{pnl:.2f}", f"{pnl_pct:+.2f}%")

                            # Exit button
                            if st.button(
                                f"🚪 Exit {symbol}",
                                key=f"exit_{idx}_{symbol}",
                                use_container_width=True,
                                type="secondary"
                            ):
                                try:
                                    # Place exit order (opposite side)
                                    exit_transaction = "SELL" if qty > 0 else "BUY"
                                    security_id = pos.get('securityId')
                                    exchange_segment = pos.get('exchangeSegment')

                                    exit_order_data = {
                                        "dhanClientId": creds['client_id'],
                                        "transactionType": exit_transaction,
                                        "exchangeSegment": exchange_segment,
                                        "productType": pos.get('productType', 'INTRADAY'),
                                        "orderType": "MARKET",
                                        "validity": "DAY",
                                        "securityId": str(security_id),
                                        "quantity": abs(qty),
                                        "price": 0.0,
                                        "triggerPrice": 0.0,
                                        "afterMarketOrder": False
                                    }

                                    exit_response = requests.post(
                                        "https://api.dhan.co/v2/orders",
                                        json=exit_order_data,
                                        headers=headers,
                                        timeout=10
                                    )

                                    if exit_response.status_code == 200:
                                        result = exit_response.json()
                                        st.success(f"✅ Exit order placed! Order ID: {result.get('orderId')}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Exit failed: {exit_response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")

                            st.divider()
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
