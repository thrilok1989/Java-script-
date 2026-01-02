"""
Option Chain Table Module
========================
Displays option chain data in a tabulated format using native Streamlit components
- ATM ±5 strikes (11 total strikes)
- Calls on left, Strike in middle, Puts on right
- Uses st.dataframe for display
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Constants
LOT_SIZE = 50  # NIFTY lot size


def safe_float(val, default=0.0):
    """Safe float conversion"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except:
        return default


def safe_int(val, default=0):
    """Safe int conversion"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(val)
    except:
        return default


def format_oi_lakhs(value):
    """Format OI in lakhs"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        return f"{value/100000:.2f}"
    except:
        return "-"


def format_change_oi(value):
    """Format Change in OI"""
    try:
        if pd.isna(value) or value == 0:
            return "0"
        sign = "+" if value > 0 else ""
        if abs(value) >= 100000:
            return f"{sign}{value/100000:.1f}L"
        return f"{sign}{value/1000:.1f}K"
    except:
        return "0"


def format_volume(value):
    """Format volume"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        if value >= 10000000:
            return f"{value/10000000:.1f}Cr"
        elif value >= 100000:
            return f"{value/100000:.1f}L"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        return str(int(value))
    except:
        return "-"


def format_gex(value):
    """Format GEX"""
    try:
        if pd.isna(value) or value == 0:
            return "-"
        abs_val = abs(value)
        if abs_val >= 10000000:
            return f"{value/10000000:.1f}Cr"
        elif abs_val >= 100000:
            return f"{value/100000:.1f}L"
        elif abs_val >= 1000:
            return f"{value/1000:.1f}K"
        return f"{value:.1f}"
    except:
        return "-"


def calculate_max_pain_from_df(df):
    """Calculate Max Pain strike from dataframe"""
    try:
        strikes = df["strikePrice"].values
        max_pain_strike = strikes[0]
        min_loss = float('inf')

        for strike in strikes:
            strike_row = df[df["strikePrice"] == strike]
            if strike_row.empty:
                continue

            call_loss = sum(
                max(0, s - strike) * safe_int(df[df["strikePrice"] == s].iloc[0].get("OI_CE", 0))
                for s in strikes if not df[df["strikePrice"] == s].empty
            )
            put_loss = sum(
                max(0, strike - s) * safe_int(df[df["strikePrice"] == s].iloc[0].get("OI_PE", 0))
                for s in strikes if not df[df["strikePrice"] == s].empty
            )

            total_loss = call_loss + put_loss
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = strike

        return int(max_pain_strike)
    except:
        return 0


def render_option_chain_table_tab(merged_df, spot, atm_strike, strike_gap, expiry, days_to_expiry, tau):
    """
    Render the option chain table using native Streamlit components
    """
    st.subheader("Option Chain Table")
    st.caption("ATM ±5 strikes | Calls on Left, Puts on Right | Click to Trade")

    # Controls row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        num_strikes = st.selectbox(
            "Strikes ±ATM",
            [3, 5, 7],
            index=1,
            key="oc_table_strikes_selector"
        )

    # Filter to selected strikes around ATM
    lower_bound = atm_strike - (num_strikes * strike_gap)
    upper_bound = atm_strike + (num_strikes * strike_gap)

    df_filtered = merged_df[(merged_df["strikePrice"] >= lower_bound) &
                            (merged_df["strikePrice"] <= upper_bound)].copy()
    # Sort by strike price ascending (lower strikes at top, ATM in middle, higher at bottom)
    df_filtered = df_filtered.sort_values("strikePrice", ascending=True).reset_index(drop=True)

    if df_filtered.empty:
        st.error("No data available for option chain table")
        return

    with col2:
        st.metric("ATM", f"{atm_strike:,}")

    with col3:
        total_ce_oi = df_filtered["OI_CE"].sum()
        total_pe_oi = df_filtered["OI_PE"].sum()
        pcr = total_pe_oi / max(total_ce_oi, 1)
        pcr_emoji = "🟢" if pcr > 1.0 else "🔴" if pcr < 0.8 else "🟡"
        st.metric("PCR", f"{pcr:.2f} {pcr_emoji}")

    with col4:
        max_pain = calculate_max_pain_from_df(df_filtered)
        st.metric("Max Pain", f"{max_pain:,}")

    with col5:
        st.metric("NIFTY Spot", f"₹{spot:,.2f}", delta=f"Exp: {expiry} ({days_to_expiry:.1f}d)")

    st.divider()

    # Build display dataframe for CALLS
    calls_data = []
    puts_data = []
    strike_data = []

    for _, row in df_filtered.iterrows():
        strike = int(row["strikePrice"])
        is_atm = strike == atm_strike

        # CE values
        ltp_ce = safe_float(row.get("LTP_CE", 0))
        oi_ce = safe_int(row.get("OI_CE", 0))
        chg_oi_ce = safe_int(row.get("Chg_OI_CE", 0))
        vol_ce = safe_int(row.get("Vol_CE", 0))
        iv_ce = safe_float(row.get("IV_CE", 0))
        delta_ce = safe_float(row.get("Delta_CE", 0))
        gex_ce = safe_float(row.get("GEX_CE", 0))

        # PE values
        ltp_pe = safe_float(row.get("LTP_PE", 0))
        oi_pe = safe_int(row.get("OI_PE", 0))
        chg_oi_pe = safe_int(row.get("Chg_OI_PE", 0))
        vol_pe = safe_int(row.get("Vol_PE", 0))
        iv_pe = safe_float(row.get("IV_PE", 0))
        delta_pe = safe_float(row.get("Delta_PE", 0))
        gex_pe = safe_float(row.get("GEX_PE", 0))

        # Strike PCR
        strike_pcr = oi_pe / max(oi_ce, 1)

        # ATM marker
        atm_marker = " ⭐" if is_atm else ""

        calls_data.append({
            "OI (L)": format_oi_lakhs(oi_ce),
            "Chg OI": format_change_oi(chg_oi_ce),
            "Volume": format_volume(vol_ce),
            "IV %": f"{iv_ce:.1f}" if iv_ce > 0 else "-",
            "LTP": f"₹{ltp_ce:.2f}",
            "Delta": f"{delta_ce:.2f}",
        })

        strike_data.append({
            "Strike": f"{strike:,}{atm_marker}",
            "PCR": f"{strike_pcr:.2f}",
        })

        puts_data.append({
            "Delta": f"{delta_pe:.2f}",
            "LTP": f"₹{ltp_pe:.2f}",
            "IV %": f"{iv_pe:.1f}" if iv_pe > 0 else "-",
            "Volume": format_volume(vol_pe),
            "Chg OI": format_change_oi(chg_oi_pe),
            "OI (L)": format_oi_lakhs(oi_pe),
        })

    calls_df = pd.DataFrame(calls_data)
    strike_df = pd.DataFrame(strike_data)
    puts_df = pd.DataFrame(puts_data)

    # Display tables side by side
    st.write("**CALLS (CE)** | **Strike** | **PUTS (PE)**")

    col_ce, col_strike, col_pe = st.columns([4, 2, 4])

    with col_ce:
        st.dataframe(
            calls_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, (len(calls_df) + 1) * 35)
        )

    with col_strike:
        st.dataframe(
            strike_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, (len(strike_df) + 1) * 35)
        )

    with col_pe:
        st.dataframe(
            puts_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, (len(puts_df) + 1) * 35)
        )

    # Legend
    st.divider()
    st.caption(
        "Legend: ⭐ = ATM Strike | OI in Lakhs | "
        "PCR = Put-Call Ratio | Chg OI = Change in Open Interest"
    )

    st.divider()

    # Buy section
    render_buy_section(df_filtered, spot, expiry, atm_strike)


def render_buy_section(df, spot, expiry, atm_strike):
    """Render the buy option section using native Streamlit components"""
    st.subheader("Quick Buy Option")

    col1, col2, col3 = st.columns([2, 2, 1])

    strikes = sorted(df["strikePrice"].unique())

    with col1:
        selected_strike = st.selectbox(
            "Strike",
            strikes,
            index=strikes.index(atm_strike) if atm_strike in strikes else 0,
            key="buy_strike_selector",
            format_func=lambda x: f"{int(x):,} {'(ATM)' if x == atm_strike else ''}"
        )

    with col2:
        option_type = st.selectbox(
            "Type",
            ["CE", "PE"],
            key="buy_option_type_selector"
        )

    with col3:
        lots = st.number_input(
            "Lots",
            min_value=1,
            max_value=100,
            value=1,
            key="buy_lots_input"
        )

    # Get current data
    strike_row = df[df["strikePrice"] == selected_strike]
    if not strike_row.empty:
        ltp_col = f"LTP_{option_type}"
        current_ltp = safe_float(strike_row.iloc[0].get(ltp_col, 0))
        iv_col = f"IV_{option_type}"
        current_iv = safe_float(strike_row.iloc[0].get(iv_col, 0))
        delta_col = f"Delta_{option_type}"
        current_delta = safe_float(strike_row.iloc[0].get(delta_col, 0))
    else:
        current_ltp = 0
        current_iv = 0
        current_delta = 0

    quantity = lots * LOT_SIZE
    est_cost = current_ltp * quantity

    # Order summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LTP", f"₹{current_ltp:.2f}")
    with col2:
        st.metric("Quantity", f"{quantity}")
    with col3:
        st.metric("Est. Cost", f"₹{est_cost:,.0f}")
    with col4:
        st.metric("IV / Delta", f"{current_iv:.1f}% / {current_delta:.2f}")

    # Buy buttons
    col1, col2, col3 = st.columns([2, 2, 4])

    with col1:
        if st.button(
            f"BUY {option_type} @ Market",
            type="primary",
            use_container_width=True,
            key="buy_market_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "MARKET", spot, expiry)

    with col2:
        limit_price = st.number_input(
            "Limit Price",
            min_value=0.05,
            value=float(current_ltp) if current_ltp > 0 else 10.0,
            step=0.05,
            key="limit_price_field"
        )
        if st.button(
            f"BUY @ ₹{limit_price:.2f}",
            use_container_width=True,
            key="buy_limit_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "LIMIT", spot, expiry, limit_price)

    with col3:
        st.warning("Orders via Dhan API. Ensure sufficient margin.")


def place_buy_order(strike, option_type, lots, order_type, spot, expiry, limit_price=None):
    """Place buy order using Dhan API"""
    try:
        from dhan_api import DhanAPI
        from config import LOT_SIZES

        dhan = DhanAPI()
        quantity = lots * LOT_SIZES.get("NIFTY", 50)

        # Basic SL and target
        sl_offset = 30 if option_type == "CE" else -30
        target_offset = 50 if option_type == "CE" else -50

        sl_price = spot + sl_offset if option_type == "CE" else spot - abs(sl_offset)
        target_price = spot + target_offset if option_type == "CE" else spot - abs(target_offset)

        st.info(
            f"**Order Preview:** NIFTY {int(strike)} {option_type} ({expiry}) | "
            f"Type: {order_type} | Qty: {quantity} ({lots} lots) | Direction: BUY"
            + (f" | Limit: ₹{limit_price:.2f}" if order_type == "LIMIT" and limit_price else "")
        )

        confirm_key = f"confirm_order_{strike}_{option_type}_{order_type}"
        if st.button("Confirm Order", type="primary", key=confirm_key):
            with st.spinner("Placing order..."):
                result = dhan.place_super_order(
                    index="NIFTY",
                    strike=int(strike),
                    option_type=option_type,
                    direction="BUY",
                    quantity=quantity,
                    sl_price=sl_price,
                    target_price=target_price
                )

                if result.get('success'):
                    st.success(
                        f"Order Placed! Order ID: {result.get('order_id', 'N/A')} | "
                        f"Status: {result.get('status', 'PENDING')}"
                    )
                    st.balloons()
                else:
                    st.error(f"Order Failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error placing order: {str(e)}")
