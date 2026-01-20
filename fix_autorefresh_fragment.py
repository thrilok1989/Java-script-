"""
Streamlit Fragment-based Auto-Refresh
This makes auto-refresh more reliable by isolating it in a fragment
"""

import streamlit as st
import time
from datetime import datetime

@st.fragment(run_every="10s")
def auto_refresh_fragment():
    """
    This fragment runs every 10 seconds independently of the main app
    More reliable than st_autorefresh
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Trigger data refresh
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = time.time()

    # Check if enough time has passed
    elapsed = time.time() - st.session_state.last_auto_refresh

    # Get market session to determine interval
    from market_hours_scheduler import scheduler
    current_session = scheduler.get_market_session()
    required_interval = scheduler.get_refresh_interval(current_session)

    if elapsed >= required_interval:
        st.session_state.last_auto_refresh = time.time()
        st.session_state.refresh_trigger = time.time()

        # Display refresh indicator
        st.caption(f"🔄 Auto-refreshed: {current_time}")

# Usage in app.py:
# Add this near the top of your main app, after imports:
# auto_refresh_fragment()
