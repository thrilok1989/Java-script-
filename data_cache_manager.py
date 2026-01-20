"""
Data Cache Manager with Background Loading
===========================================

This module provides:
- Thread-safe caching for all data sources
- Background data loading and auto-refresh
- Pre-loading of all tab data on startup
- Optimized refresh cycles to prevent API rate limiting
- Smart cache invalidation and updates

Cache Strategy (Optimized for Rate Limiting):
- NIFTY/SENSEX data: 60-second TTL, background refresh every 45 seconds (config-driven)
- Bias Analysis: 60-second TTL, background refresh every 300 seconds (5 minutes)
- Option Chain: 60-second TTL, background refresh
- Advanced Charts: 60-second TTL, background refresh

Note: Refresh intervals increased to prevent HTTP 429 (rate limit) errors
Previous 10-second interval caused overlapping cycles and exceeded API limits
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
import streamlit as st
import pandas as pd
from functools import wraps
from market_hours_scheduler import scheduler, is_within_trading_hours
import config


class DataCacheManager:
    """
    Thread-safe data cache manager with background loading and auto-refresh
    """

    def __init__(self):
        """Initialize cache manager"""
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_locks = {}
        self._background_threads = {}
        self._stop_threads = threading.Event()
        self._main_lock = threading.Lock()

        # Cache TTLs (in seconds)
        self.ttl_config = {
            'nifty_data': 10,       # 10 seconds for spot price real-time updates
            'sensex_data': 10,      # 10 seconds for spot price real-time updates
            'bias_analysis': 60,
            'option_chain': 60,
            'advanced_chart': 60,
        }

        # Background refresh intervals (in seconds)
        # Note: These are dynamically adjusted based on market session
        self.refresh_intervals = {
            'market_data': 10,      # NIFTY/SENSEX spot price (10 seconds for real-time)
            'analysis_data': 60,    # All analysis (adjusted by market session)
        }

        # Market hours awareness
        self.market_hours_enabled = getattr(config, 'MARKET_HOURS_ENABLED', True)

    def _get_lock(self, cache_key: str) -> threading.Lock:
        """Get or create a lock for a cache key"""
        with self._main_lock:
            if cache_key not in self._cache_locks:
                self._cache_locks[cache_key] = threading.Lock()
            return self._cache_locks[cache_key]

    def get(self, cache_key: str, default=None) -> Any:
        """
        Get value from cache

        Args:
            cache_key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key in self._cache:
                # Check if cache is still valid
                if cache_key in self.ttl_config:
                    ttl = self.ttl_config[cache_key]
                    timestamp = self._cache_timestamps.get(cache_key, 0)
                    if time.time() - timestamp < ttl:
                        return self._cache[cache_key]
                else:
                    # No TTL configured, return cached value
                    return self._cache[cache_key]

            return default

    def set(self, cache_key: str, value: Any):
        """
        Set value in cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        lock = self._get_lock(cache_key)
        with lock:
            self._cache[cache_key] = value
            self._cache_timestamps[cache_key] = time.time()

    def invalidate(self, cache_key: str):
        """
        Invalidate cache entry

        Args:
            cache_key: Cache key to invalidate
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
            if cache_key in self._cache_timestamps:
                del self._cache_timestamps[cache_key]

    def is_valid(self, cache_key: str) -> bool:
        """
        Check if cache entry is valid

        Args:
            cache_key: Cache key

        Returns:
            True if cache is valid, False otherwise
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key not in self._cache:
                return False

            if cache_key in self.ttl_config:
                ttl = self.ttl_config[cache_key]
                timestamp = self._cache_timestamps.get(cache_key, 0)
                return time.time() - timestamp < ttl

            return True

    def get_or_load(self, cache_key: str, loader_func: Callable, *args, **kwargs) -> Any:
        """
        Get from cache or load using loader function

        Args:
            cache_key: Cache key
            loader_func: Function to load data if not cached
            *args: Arguments for loader function
            **kwargs: Keyword arguments for loader function

        Returns:
            Cached or loaded value
        """
        # Try to get from cache first
        cached_value = self.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Load data
        try:
            value = loader_func(*args, **kwargs)
            self.set(cache_key, value)
            return value
        except Exception as e:
            # Return cached value even if expired, better than nothing
            lock = self._get_lock(cache_key)
            with lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
            raise e

    def start_background_refresh(self, cache_key: str, loader_func: Callable,
                                 interval: int = 60, *args, **kwargs):
        """
        Start background refresh for a cache key

        Args:
            cache_key: Cache key
            loader_func: Function to load data
            interval: Refresh interval in seconds
            *args: Arguments for loader function
            **kwargs: Keyword arguments for loader function
        """
        if cache_key in self._background_threads:
            return  # Already running

        def refresh_loop():
            """Background refresh loop with market hours awareness"""
            while not self._stop_threads.is_set():
                try:
                    # Check if market hours validation is enabled
                    if self.market_hours_enabled:
                        # Only fetch data during trading hours
                        if is_within_trading_hours():
                            # Load data
                            value = loader_func(*args, **kwargs)
                            self.set(cache_key, value)

                            # Use session-based refresh interval
                            session = scheduler.get_market_session()
                            actual_interval = scheduler.get_refresh_interval(session)
                        else:
                            # Market closed - use minimal refresh interval
                            actual_interval = config.REFRESH_INTERVALS.get('closed', 300)
                            # Optionally update cache with "market closed" status
                            # (only if needed by the application)
                    else:
                        # Market hours checking disabled - always refresh
                        value = loader_func(*args, **kwargs)
                        self.set(cache_key, value)
                        actual_interval = interval

                except Exception as e:
                    print(f"Background refresh error for {cache_key}: {e}")
                    actual_interval = interval  # Use default on error

                # Wait for interval or stop event
                self._stop_threads.wait(actual_interval)

        # Start thread
        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()
        self._background_threads[cache_key] = thread

    def stop_all_background_refresh(self):
        """Stop all background refresh threads"""
        self._stop_threads.set()

        # Wait for all threads to finish
        for thread in self._background_threads.values():
            thread.join(timeout=5)

        self._background_threads.clear()
        self._stop_threads.clear()

    def clear_all(self):
        """Clear all cache entries"""
        with self._main_lock:
            self._cache.clear()
            self._cache_timestamps.clear()


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> DataCacheManager:
    """
    Get global cache manager instance

    Returns:
        DataCacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DataCacheManager()
    return _cache_manager


def cache_with_ttl(cache_key: str, ttl: int = 60):
    """
    Decorator to cache function results with TTL

    Args:
        cache_key: Cache key
        ttl: Time to live in seconds

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()

            # Try to get from cache
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Load data
            value = func(*args, **kwargs)
            cache_manager.set(cache_key, value)
            return value

        return wrapper
    return decorator


def preload_all_data():
    """
    Pre-load all data for all tabs in background

    This function should be called on app startup to pre-load:
    - NIFTY/SENSEX data
    - Bias Analysis data
    - Option Chain data (for main instruments)
    - Advanced Chart data (for main symbols)
    """
    cache_manager = get_cache_manager()

    # Import here to avoid circular imports
    from market_data import fetch_nifty_data, fetch_sensex_data
    from bias_analysis import BiasAnalysisPro

    def load_market_data():
        """Load market data in background"""
        try:
            # Load NIFTY data
            nifty_data = fetch_nifty_data()
            cache_manager.set('nifty_data', nifty_data)

            # Load SENSEX data
            sensex_data = fetch_sensex_data()
            cache_manager.set('sensex_data', sensex_data)
        except Exception as e:
            print(f"Error loading market data: {e}")

    def load_bias_analysis_data():
        """Load bias analysis data in background"""
        try:
            if 'bias_analyzer' in st.session_state:
                analyzer = st.session_state.bias_analyzer
            else:
                analyzer = BiasAnalysisPro()

            # Default to NIFTY analysis
            results = analyzer.analyze_all_bias_indicators("^NSEI")
            cache_manager.set('bias_analysis', results)
        except Exception as e:
            print(f"Error loading bias analysis data: {e}")

    # Start background threads for continuous refresh

    # Market data: refresh using config interval to prevent rate limiting
    # Uses regular session interval from config (45 seconds) to avoid overlapping cycles
    cache_manager.start_background_refresh(
        'market_data_refresh',
        load_market_data,
        interval=config.REFRESH_INTERVALS['regular']
    )

    # Bias analysis: refresh every 300 seconds (5 minutes) to reduce API load
    # Previous 60-second interval was contributing to rate limiting
    cache_manager.start_background_refresh(
        'bias_analysis_refresh',
        load_bias_analysis_data,
        interval=300
    )

    # Initial load (immediate)
    initial_load_thread = threading.Thread(target=load_market_data, daemon=True)
    initial_load_thread.start()


def get_cached_nifty_data():
    """
    Get cached NIFTY data (non-blocking)

    Returns cached data or triggers background load if not available.
    Never blocks - returns None if data not ready yet.

    Returns:
        NIFTY data dict or None if not yet loaded
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get('nifty_data')

    if cached_data is not None:
        return cached_data

    # If not cached, trigger background load (non-blocking)
    # Don't block the UI - let background thread handle it
    def load_in_background():
        try:
            from market_data import fetch_nifty_data
            data = fetch_nifty_data()
            cache_manager.set('nifty_data', data)
        except Exception as e:
            print(f"Error loading NIFTY data in background: {e}")

    # Start background thread
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()

    # Return None immediately (non-blocking)
    return None


def get_cached_sensex_data():
    """
    Get cached SENSEX data (non-blocking)

    Returns cached data or triggers background load if not available.
    Never blocks - returns None if data not ready yet.

    Returns:
        SENSEX data dict or None if not yet loaded
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get('sensex_data')

    if cached_data is not None:
        return cached_data

    # If not cached, trigger background load (non-blocking)
    # Don't block the UI - let background thread handle it
    def load_in_background():
        try:
            from market_data import fetch_sensex_data
            data = fetch_sensex_data()
            cache_manager.set('sensex_data', data)
        except Exception as e:
            print(f"Error loading SENSEX data in background: {e}")

    # Start background thread
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()

    # Return None immediately (non-blocking)
    return None


def get_cached_bias_analysis_results():
    """
    Get cached Bias Analysis results

    Returns:
        Bias analysis results dict or None
    """
    cache_manager = get_cache_manager()
    return cache_manager.get('bias_analysis')


def invalidate_all_caches():
    """Invalidate all caches (useful for manual refresh)"""
    cache_manager = get_cache_manager()
    cache_manager.clear_all()


def start_background_signal_monitor():
    """
    Start 24/7 background signal monitoring with telegram alerts

    This function runs continuously on the server (not in browser) and:
    - Fetches option chain data every 60 seconds
    - Runs signal detection logic
    - Sends telegram messages when strong signals detected (confidence >= 70%)
    - Tracks sent signals to prevent duplicates
    - Runs even when browser is closed!
    """
    cache_manager = get_cache_manager()

    # Track sent signals to avoid duplicates
    sent_signals = {}

    def monitor_signals():
        """Background thread that monitors for trading signals"""
        print("🚀 Starting 24/7 background signal monitor...")

        while not cache_manager._stop_threads.is_set():
            try:
                # Only run during market hours (if enabled)
                if cache_manager.market_hours_enabled:
                    if not is_within_trading_hours():
                        print("⏰ Market closed - signal monitoring paused")
                        cache_manager._stop_threads.wait(300)  # Wait 5 minutes
                        continue

                print("🔍 Checking for trading signals...")

                # Import here to avoid circular imports
                from NiftyOptionScreener import (
                    fetch_dhan_option_chain,
                    calculate_seller_bias,
                    calculate_max_pain_smart,
                    find_support_resistance_with_buildup,
                    calculate_entry_signal_extended,
                    calculate_moment_detector_metrics,
                    calculate_atm_bias,
                    check_and_send_signal
                )
                from dhan_api import get_spot_nifty
                from config import get_telegram_credentials
                import os

                # Get telegram credentials
                telegram_creds = get_telegram_credentials()
                if not telegram_creds['enabled']:
                    print("⚠️ Telegram disabled - skipping signal check")
                    cache_manager._stop_threads.wait(60)
                    continue

                # Get NIFTY spot price
                spot = get_spot_nifty()
                if spot == 0.0:
                    print("❌ Unable to fetch NIFTY spot price")
                    cache_manager._stop_threads.wait(60)
                    continue

                print(f"📊 NIFTY Spot: ₹{spot:,.2f}")

                # Get current expiry (use nearest Thursday)
                from datetime import datetime, timedelta
                today = datetime.now()
                days_until_thursday = (3 - today.weekday()) % 7
                if days_until_thursday == 0 and today.hour >= 15:
                    days_until_thursday = 7
                expiry_date = today + timedelta(days=days_until_thursday)
                expiry = expiry_date.strftime("%d-%b-%Y")

                print(f"📅 Using expiry: {expiry}")

                # Fetch option chain
                chain = fetch_dhan_option_chain(expiry)
                if chain is None or chain.empty:
                    print("❌ Failed to fetch option chain")
                    cache_manager._stop_threads.wait(60)
                    continue

                print(f"✅ Fetched option chain with {len(chain)} strikes")

                # Calculate all signal components
                seller_bias_result = calculate_seller_bias(chain, spot)
                seller_max_pain = calculate_max_pain_smart(chain, spot)
                seller_supports_df, seller_resists_df = find_support_resistance_with_buildup(chain, spot)

                # Find nearest support/resistance
                nearest_sup = seller_supports_df.iloc[0].to_dict() if not seller_supports_df.empty else None
                nearest_res = seller_resists_df.iloc[0].to_dict() if not seller_resists_df.empty else None

                # Calculate moment metrics
                moment_metrics = calculate_moment_detector_metrics(chain, spot)

                # Calculate ATM bias
                atm_strike = round(spot / 50) * 50
                strike_gap = 50
                atm_bias = calculate_atm_bias(chain, atm_strike, spot, strike_gap)

                # Calculate support/resistance bias
                support_bias = None
                resistance_bias = None
                if nearest_sup:
                    support_bias = calculate_atm_bias(chain, nearest_sup["strike"], spot, strike_gap)
                if nearest_res:
                    resistance_bias = calculate_atm_bias(chain, nearest_res["strike"], spot, strike_gap)

                # Calculate breakout index
                seller_breakout_index = 0
                if seller_bias_result.get("polarity"):
                    seller_breakout_index = abs(seller_bias_result["polarity"]) * 10

                # Calculate entry signal
                entry_signal = calculate_entry_signal_extended(
                    spot, chain, atm_strike,
                    seller_bias_result, seller_max_pain,
                    seller_supports_df, seller_resists_df,
                    nearest_sup, nearest_res,
                    seller_breakout_index, moment_metrics,
                    atm_bias, support_bias, resistance_bias
                )

                print(f"📈 Signal: {entry_signal['position_type']} | Confidence: {entry_signal['confidence']}%")

                # Check if signal meets threshold (70%+) and is new
                if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 70:
                    signal_key = f"{entry_signal['position_type']}_{entry_signal['optimal_entry_price']:.0f}"

                    # Check if we already sent this signal recently (within 1 hour)
                    now = time.time()
                    if signal_key in sent_signals:
                        last_sent_time = sent_signals[signal_key]
                        if now - last_sent_time < 3600:  # 1 hour = 3600 seconds
                            print(f"⏭️ Signal already sent recently: {signal_key}")
                            cache_manager._stop_threads.wait(60)
                            continue

                    # New signal - send telegram alert
                    print(f"🚨 NEW SIGNAL DETECTED: {signal_key}")

                    telegram_msg = check_and_send_signal(
                        entry_signal, spot, seller_bias_result,
                        seller_max_pain, nearest_sup, nearest_res,
                        moment_metrics, seller_breakout_index, expiry, {},
                        atm_bias, support_bias, resistance_bias
                    )

                    if telegram_msg:
                        # Send to telegram
                        from NiftyOptionScreener import send_telegram_message
                        success, message = send_telegram_message(
                            os.getenv("TELEGRAM_BOT_TOKEN"),
                            os.getenv("TELEGRAM_CHAT_ID"),
                            telegram_msg
                        )

                        if success:
                            print(f"✅ Telegram message sent successfully!")
                            # Mark signal as sent
                            sent_signals[signal_key] = now

                            # Clean up old signals (older than 2 hours)
                            sent_signals_copy = sent_signals.copy()
                            for key, timestamp in sent_signals_copy.items():
                                if now - timestamp > 7200:  # 2 hours
                                    del sent_signals[key]
                        else:
                            print(f"❌ Failed to send telegram message: {message}")
                    else:
                        print("⚠️ No telegram message generated")
                else:
                    print(f"⏭️ No signal or confidence too low")

            except Exception as e:
                print(f"❌ Error in background signal monitor: {e}")
                import traceback
                traceback.print_exc()

            # Wait 60 seconds before next check
            print("⏰ Waiting 60 seconds before next check...")
            cache_manager._stop_threads.wait(60)

    # Start background thread
    thread = threading.Thread(target=monitor_signals, daemon=True)
    thread.start()
    cache_manager._background_threads['signal_monitor'] = thread
    print("✅ Background signal monitor started!")

    return thread
