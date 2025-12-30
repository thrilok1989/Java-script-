"""
Option Chain Table Module
========================
Displays option chain data in a tabulated format similar to trading apps
- ATM ±5 strikes (11 total strikes)
- Calls on left, Strike in middle, Puts on right
- Columns: LTP, OI, Change OI, Volume, IV, GEX, IV Skew, Delta Exposure
- Click on LTP to place buy order via Dhan API
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


def calculate_iv_skew_for_strike(iv_ce, iv_pe, iv_atm_ce, iv_atm_pe):
    """
    Calculate IV Skew relative to ATM
    """
    try:
        if pd.isna(iv_ce) or pd.isna(iv_pe) or pd.isna(iv_atm_ce) or pd.isna(iv_atm_pe):
            return 0.0

        atm_avg_iv = (iv_atm_ce + iv_atm_pe) / 2
        current_avg_iv = (iv_ce + iv_pe) / 2

        skew = current_avg_iv - atm_avg_iv
        return round(skew, 2)
    except:
        return 0.0


def calculate_delta_exposure(delta, oi, lot_size, spot):
    """
    Calculate Delta Exposure (DEX)
    DEX = Delta × OI × Lot Size × Spot Price / 10^7 (in Cr)
    """
    try:
        if pd.isna(delta) or pd.isna(oi):
            return 0.0
        dex = abs(delta) * oi * lot_size * spot / 10000000
        return round(dex, 2)
    except:
        return 0.0


def calculate_gex_for_display(gamma, oi, lot_size, spot):
    """
    Calculate Gamma Exposure (GEX) for display
    Scaled for readability
    """
    try:
        if pd.isna(gamma) or pd.isna(oi):
            return 0.0
        gex = gamma * oi * lot_size * (spot ** 2) / 100000000
        return round(gex, 2)
    except:
        return 0.0


def format_number(value, decimals=2):
    """Format number with K/L/Cr suffix"""
    try:
        if pd.isna(value) or value == 0:
            return "-"

        abs_val = abs(value)
        if abs_val >= 10000000:
            return f"{value/10000000:.{decimals}f}Cr"
        elif abs_val >= 100000:
            return f"{value/100000:.{decimals}f}L"
        elif abs_val >= 1000:
            return f"{value/1000:.{decimals}f}K"
        else:
            return f"{value:.{decimals}f}"
    except:
        return "-"


def format_oi(value):
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
        return f"{sign}{value/1000:.1f}K"
    except:
        return "0"


def calculate_max_pain_from_df(df):
    """Calculate Max Pain strike from dataframe"""
    try:
        strikes = df["strikePrice"].values
        max_pain_strike = strikes[0]
        min_loss = float('inf')

        for strike in strikes:
            ce_oi = df[df["strikePrice"] == strike]["OI_CE"].values[0]
            pe_oi = df[df["strikePrice"] == strike]["OI_PE"].values[0]

            call_loss = sum(max(0, s - strike) * df[df["strikePrice"] == s]["OI_CE"].values[0]
                          for s in strikes)
            put_loss = sum(max(0, strike - s) * df[df["strikePrice"] == s]["OI_PE"].values[0]
                         for s in strikes)

            total_loss = call_loss + put_loss

            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = strike

        return int(max_pain_strike)
    except:
        return 0


def render_option_chain_table_tab(merged_df, spot, atm_strike, strike_gap, expiry, days_to_expiry, tau):
    """
    Render the option chain table using data from the parent screener

    Args:
        merged_df: Merged CE/PE dataframe from NiftyOptionScreener
        spot: Current spot price
        atm_strike: ATM strike price
        strike_gap: Gap between strikes
        expiry: Expiry date string
        days_to_expiry: Days to expiry
        tau: Time to expiry in years
    """
    st.markdown("## 📊 Option Chain Table")
    st.caption("ATM ±5 strikes | Calls on Left, Puts on Right | Click to Trade")

    # Filter to ATM ±5 strikes
    num_strikes = 5
    lower_bound = atm_strike - (num_strikes * strike_gap)
    upper_bound = atm_strike + (num_strikes * strike_gap)

    df_filtered = merged_df[(merged_df["strikePrice"] >= lower_bound) &
                            (merged_df["strikePrice"] <= upper_bound)].copy()
    df_filtered = df_filtered.sort_values("strikePrice").reset_index(drop=True)

    if df_filtered.empty:
        st.error("No data available for option chain table")
        return

    # Get ATM IV for skew calculation
    atm_row = df_filtered[df_filtered["strikePrice"] == atm_strike]
    if not atm_row.empty:
        iv_atm_ce = safe_float(atm_row.iloc[0].get("IV_CE", 15.0))
        iv_atm_pe = safe_float(atm_row.iloc[0].get("IV_PE", 15.0))
    else:
        iv_atm_ce = iv_atm_pe = 15.0

    # Calculate additional metrics for each row
    for i, row in df_filtered.iterrows():
        iv_ce = safe_float(row.get("IV_CE", np.nan))
        iv_pe = safe_float(row.get("IV_PE", np.nan))

        # IV Skew
        df_filtered.at[i, "IV_Skew_Calc"] = calculate_iv_skew_for_strike(iv_ce, iv_pe, iv_atm_ce, iv_atm_pe)

        # Delta Exposure
        delta_ce = safe_float(row.get("Delta_CE", 0))
        delta_pe = safe_float(row.get("Delta_PE", 0))
        oi_ce = safe_int(row.get("OI_CE", 0))
        oi_pe = safe_int(row.get("OI_PE", 0))

        df_filtered.at[i, "DEX_CE"] = calculate_delta_exposure(delta_ce, oi_ce, LOT_SIZE, spot)
        df_filtered.at[i, "DEX_PE"] = calculate_delta_exposure(abs(delta_pe), oi_pe, LOT_SIZE, spot)

        # GEX for display (might already be calculated, but recalculate for consistency)
        gamma_ce = safe_float(row.get("Gamma_CE", 0))
        gamma_pe = safe_float(row.get("Gamma_PE", 0))
        df_filtered.at[i, "GEX_Display_CE"] = calculate_gex_for_display(gamma_ce, oi_ce, LOT_SIZE, spot)
        df_filtered.at[i, "GEX_Display_PE"] = calculate_gex_for_display(gamma_pe, oi_pe, LOT_SIZE, spot)

    # Controls row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        num_strikes_display = st.selectbox(
            "Strikes ±ATM",
            [3, 5, 7],
            index=1,
            key="oc_table_strikes_selector"
        )
        # Re-filter if changed
        if num_strikes_display != num_strikes:
            lower_bound = atm_strike - (num_strikes_display * strike_gap)
            upper_bound = atm_strike + (num_strikes_display * strike_gap)
            df_filtered = merged_df[(merged_df["strikePrice"] >= lower_bound) &
                                    (merged_df["strikePrice"] <= upper_bound)].copy()
            df_filtered = df_filtered.sort_values("strikePrice").reset_index(drop=True)

    with col2:
        st.metric("ATM", f"{atm_strike:,}")

    with col3:
        total_ce_oi = df_filtered["OI_CE"].sum()
        total_pe_oi = df_filtered["OI_PE"].sum()
        pcr = total_pe_oi / max(total_ce_oi, 1)
        pcr_color = "🟢" if pcr > 1.0 else "🔴" if pcr < 0.8 else "🟡"
        st.metric("PCR", f"{pcr:.2f} {pcr_color}")

    with col4:
        max_pain = calculate_max_pain_from_df(df_filtered)
        st.metric("Max Pain", f"{max_pain:,}")

    with col5:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 8px 12px; border-radius: 8px; text-align: center;
                    border: 1px solid #0f3460;'>
            <span style='color: #888; font-size: 11px;'>NIFTY SPOT</span>
            <span style='color: #00d4ff; font-size: 20px; font-weight: bold; margin-left: 10px;'>₹{spot:,.2f}</span>
            <span style='color: #888; font-size: 11px; margin-left: 10px;'>Exp: {expiry} ({days_to_expiry:.1f}d)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")  # Spacing

    # Render the table
    render_chain_table_html(df_filtered, spot, atm_strike)

    st.divider()

    # Buy section
    render_buy_section(df_filtered, spot, expiry, atm_strike)


def render_chain_table_html(df, spot, atm_strike):
    """
    Render the option chain table with HTML/CSS
    """

    # Custom CSS
    st.markdown("""
    <style>
    .oc-container {
        overflow-x: auto;
        margin: 10px 0;
    }
    .oc-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
    }
    .oc-table th {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #fff;
        padding: 10px 6px;
        text-align: center;
        font-weight: 600;
        border-bottom: 2px solid #0f3460;
        position: sticky;
        top: 0;
        white-space: nowrap;
    }
    .oc-table td {
        padding: 8px 6px;
        text-align: center;
        border-bottom: 1px solid #2d2d3d;
        white-space: nowrap;
    }
    .oc-table tr:hover {
        background-color: rgba(0, 212, 255, 0.1);
    }
    .oc-table .atm-row {
        background: linear-gradient(90deg, rgba(255,215,0,0.15) 0%, rgba(255,215,0,0.25) 50%, rgba(255,215,0,0.15) 100%);
        font-weight: bold;
    }
    .oc-table .atm-row td {
        border-top: 2px solid #ffd700;
        border-bottom: 2px solid #ffd700;
    }
    .oc-table .strike-col {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #00d4ff;
        font-weight: bold;
        font-size: 13px;
        min-width: 90px;
    }
    .oc-table .ce-section {
        background-color: rgba(0, 180, 100, 0.05);
    }
    .oc-table .pe-section {
        background-color: rgba(255, 70, 70, 0.05);
    }
    .ltp-ce-cell {
        color: #00ff88;
        font-weight: bold;
    }
    .ltp-pe-cell {
        color: #ff6b6b;
        font-weight: bold;
    }
    .oi-high {
        color: #ffd700;
        font-weight: bold;
    }
    .chg-positive {
        color: #00ff88;
    }
    .chg-negative {
        color: #ff6b6b;
    }
    .iv-high {
        color: #ff9f43;
    }
    .iv-low {
        color: #74b9ff;
    }
    .gex-positive {
        color: #00ff88;
    }
    .gex-negative {
        color: #ff6b6b;
    }
    .itm-ce {
        background-color: rgba(0, 255, 136, 0.08);
    }
    .itm-pe {
        background-color: rgba(255, 107, 107, 0.08);
    }
    .header-ce {
        background: linear-gradient(135deg, #0d4f3c 0%, #1a5f4d 100%) !important;
        color: #00ff88 !important;
    }
    .header-pe {
        background: linear-gradient(135deg, #4f0d0d 0%, #5f1a1a 100%) !important;
        color: #ff6b6b !important;
    }
    .header-strike {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%) !important;
        color: #00d4ff !important;
        font-size: 13px !important;
    }
    .pcr-cell {
        color: #ffd700;
        font-size: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Build table HTML
    html = """
    <div class="oc-container">
    <table class="oc-table">
    <thead>
        <tr>
            <th colspan="8" class="header-ce">CALLS (CE)</th>
            <th class="header-strike">STRIKE</th>
            <th colspan="8" class="header-pe">PUTS (PE)</th>
        </tr>
        <tr>
            <th class="header-ce">DEX</th>
            <th class="header-ce">IV Skew</th>
            <th class="header-ce">GEX</th>
            <th class="header-ce">IV</th>
            <th class="header-ce">Vol</th>
            <th class="header-ce">Chg OI</th>
            <th class="header-ce">OI (L)</th>
            <th class="header-ce">LTP</th>
            <th class="header-strike">Strike<br><span style="font-size:10px;color:#888">PCR</span></th>
            <th class="header-pe">LTP</th>
            <th class="header-pe">OI (L)</th>
            <th class="header-pe">Chg OI</th>
            <th class="header-pe">Vol</th>
            <th class="header-pe">IV</th>
            <th class="header-pe">GEX</th>
            <th class="header-pe">IV Skew</th>
            <th class="header-pe">DEX</th>
        </tr>
    </thead>
    <tbody>
    """

    # Get max OI for highlighting
    max_ce_oi = df["OI_CE"].max() if not df["OI_CE"].isna().all() else 1
    max_pe_oi = df["OI_PE"].max() if not df["OI_PE"].isna().all() else 1

    for _, row in df.iterrows():
        strike = int(row["strikePrice"])
        is_atm = strike == atm_strike
        is_itm_ce = strike < spot
        is_itm_pe = strike > spot

        # Calculate strike PCR
        oi_ce = safe_int(row.get("OI_CE", 0))
        oi_pe = safe_int(row.get("OI_PE", 0))
        strike_pcr = oi_pe / max(oi_ce, 1)

        # Row class
        row_class = "atm-row" if is_atm else ""
        ce_class = "itm-ce" if is_itm_ce else ""
        pe_class = "itm-pe" if is_itm_pe else ""

        # Extract values
        ltp_ce = safe_float(row.get("LTP_CE", 0))
        ltp_pe = safe_float(row.get("LTP_PE", 0))
        iv_ce = safe_float(row.get("IV_CE", 0))
        iv_pe = safe_float(row.get("IV_PE", 0))
        vol_ce = safe_int(row.get("Vol_CE", 0))
        vol_pe = safe_int(row.get("Vol_PE", 0))
        chg_oi_ce = safe_int(row.get("Chg_OI_CE", 0))
        chg_oi_pe = safe_int(row.get("Chg_OI_PE", 0))

        # Use pre-calculated or get from row
        gex_ce = safe_float(row.get("GEX_Display_CE", row.get("GEX_CE", 0)))
        gex_pe = safe_float(row.get("GEX_Display_PE", row.get("GEX_PE", 0)))
        iv_skew = safe_float(row.get("IV_Skew_Calc", 0))
        dex_ce = safe_float(row.get("DEX_CE", 0))
        dex_pe = safe_float(row.get("DEX_PE", 0))

        # OI highlighting
        ce_oi_class = "oi-high" if oi_ce > max_ce_oi * 0.8 else ""
        pe_oi_class = "oi-high" if oi_pe > max_pe_oi * 0.8 else ""

        # Change OI class
        chg_ce_class = "chg-positive" if chg_oi_ce > 0 else "chg-negative" if chg_oi_ce < 0 else ""
        chg_pe_class = "chg-positive" if chg_oi_pe > 0 else "chg-negative" if chg_oi_pe < 0 else ""

        # GEX class (scale down for display)
        gex_ce_display = gex_ce / 1000000 if abs(gex_ce) > 1000 else gex_ce
        gex_pe_display = gex_pe / 1000000 if abs(gex_pe) > 1000 else gex_pe
        gex_ce_class = "gex-positive" if gex_ce > 0 else "gex-negative"
        gex_pe_class = "gex-positive" if gex_pe > 0 else "gex-negative"

        # IV class
        iv_ce_class = "iv-high" if iv_ce > 20 else "iv-low" if iv_ce < 12 else ""
        iv_pe_class = "iv-high" if iv_pe > 20 else "iv-low" if iv_pe < 12 else ""

        html += f"""
        <tr class="{row_class}">
            <td class="ce-section {ce_class}">{dex_ce:.2f}</td>
            <td class="ce-section {ce_class}">{iv_skew:+.1f}</td>
            <td class="ce-section {ce_class} {gex_ce_class}">{format_number(gex_ce, 1)}</td>
            <td class="ce-section {ce_class} {iv_ce_class}">{iv_ce:.1f}</td>
            <td class="ce-section {ce_class}">{format_number(vol_ce, 0)}</td>
            <td class="ce-section {ce_class} {chg_ce_class}">{format_change_oi(chg_oi_ce)}</td>
            <td class="ce-section {ce_class} {ce_oi_class}">{format_oi(oi_ce)}</td>
            <td class="ce-section {ce_class} ltp-ce-cell">{ltp_ce:.2f}</td>
            <td class="strike-col">
                {strike:,}<br>
                <span class="pcr-cell">{strike_pcr:.2f}</span>
            </td>
            <td class="pe-section {pe_class} ltp-pe-cell">{ltp_pe:.2f}</td>
            <td class="pe-section {pe_class} {pe_oi_class}">{format_oi(oi_pe)}</td>
            <td class="pe-section {pe_class} {chg_pe_class}">{format_change_oi(chg_oi_pe)}</td>
            <td class="pe-section {pe_class}">{format_number(vol_pe, 0)}</td>
            <td class="pe-section {pe_class} {iv_pe_class}">{iv_pe:.1f}</td>
            <td class="pe-section {pe_class} {gex_pe_class}">{format_number(gex_pe, 1)}</td>
            <td class="pe-section {pe_class}">{iv_skew:+.1f}</td>
            <td class="pe-section {pe_class}">{dex_pe:.2f}</td>
        </tr>
        """

    html += """
    </tbody>
    </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style='margin-top: 10px; padding: 8px; background: #1a1a2e; border-radius: 6px; font-size: 10px; color: #888;'>
        <strong>Legend:</strong>
        <span style='color: #00ff88;'>● CE LTP</span> |
        <span style='color: #ff6b6b;'>● PE LTP</span> |
        <span style='color: #ffd700;'>● High OI</span> |
        <span style='background: rgba(255,215,0,0.2); padding: 1px 6px; border-radius: 2px;'>ATM</span> |
        OI in Lakhs | DEX/GEX scaled | IV Skew = diff from ATM avg IV
    </div>
    """, unsafe_allow_html=True)


def render_buy_section(df, spot, expiry, atm_strike):
    """
    Render the buy option section
    """
    st.markdown("### 🛒 Quick Buy Option")

    col1, col2, col3, col4 = st.columns([2, 2, 1, 3])

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

    with col4:
        quantity = lots * LOT_SIZE
        est_cost = current_ltp * quantity

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 12px; border-radius: 8px; border: 1px solid #0f3460;'>
            <div style='display: flex; justify-content: space-around; align-items: center;'>
                <div style='text-align: center;'>
                    <span style='color: #888; font-size: 10px;'>LTP</span><br>
                    <span style='color: {"#00ff88" if option_type == "CE" else "#ff6b6b"}; font-size: 16px; font-weight: bold;'>
                        ₹{current_ltp:.2f}
                    </span>
                </div>
                <div style='text-align: center;'>
                    <span style='color: #888; font-size: 10px;'>Qty</span><br>
                    <span style='color: #00d4ff; font-size: 16px; font-weight: bold;'>{quantity}</span>
                </div>
                <div style='text-align: center;'>
                    <span style='color: #888; font-size: 10px;'>Est. Cost</span><br>
                    <span style='color: #ffd700; font-size: 16px; font-weight: bold;'>₹{est_cost:,.0f}</span>
                </div>
                <div style='text-align: center;'>
                    <span style='color: #888; font-size: 10px;'>IV / Delta</span><br>
                    <span style='color: #aaa; font-size: 12px;'>{current_iv:.1f}% / {current_delta:.3f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Buy buttons
    col1, col2, col3 = st.columns([2, 2, 4])

    with col1:
        if st.button(
            f"🟢 BUY {option_type} @ Market",
            type="primary",
            use_container_width=True,
            key="buy_market_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "MARKET", spot, expiry)

    with col2:
        limit_price = st.number_input(
            "Limit ₹",
            min_value=0.05,
            value=float(current_ltp) if current_ltp > 0 else 10.0,
            step=0.05,
            key="limit_price_field",
            label_visibility="collapsed"
        )
        if st.button(
            f"🟡 BUY @ ₹{limit_price:.2f}",
            use_container_width=True,
            key="buy_limit_button"
        ):
            place_buy_order(selected_strike, option_type, lots, "LIMIT", spot, expiry, limit_price)

    with col3:
        st.markdown("""
        <div style='padding: 8px; background: rgba(255,215,0,0.1);
                    border-radius: 5px; font-size: 10px; color: #ffd700;'>
            ⚠️ Orders via Dhan API. Ensure sufficient margin. Market orders execute immediately.
        </div>
        """, unsafe_allow_html=True)


def place_buy_order(strike, option_type, lots, order_type, spot, expiry, limit_price=None):
    """
    Place buy order using Dhan API
    """
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

        st.info(f"""
        **Order Preview:**
        - Symbol: NIFTY {int(strike)} {option_type} ({expiry})
        - Type: {order_type}
        - Quantity: {quantity} ({lots} lots)
        - Direction: BUY
        {"- Limit Price: ₹" + f"{limit_price:.2f}" if order_type == "LIMIT" and limit_price else ""}
        """)

        confirm_key = f"confirm_order_{strike}_{option_type}_{order_type}"
        if st.button("✅ Confirm Order", type="primary", key=confirm_key):
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
                    st.success(f"""
                    ✅ **Order Placed!**
                    - Order ID: {result.get('order_id', 'N/A')}
                    - Status: {result.get('status', 'PENDING')}
                    """)
                    st.balloons()
                else:
                    st.error(f"""
                    ❌ **Order Failed**
                    - Error: {result.get('error', 'Unknown error')}
                    """)

    except Exception as e:
        st.error(f"Error placing order: {str(e)}")
