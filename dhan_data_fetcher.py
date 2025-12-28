"""
Dhan API Data Fetcher with Rate Limiting
==========================================

This module handles all data fetching from Dhan API with proper rate limiting:
- 10-second intervals between different data types
- Respects Dhan API rate limits (1 req/sec for quotes, 5 req/sec for data, 1 req/3sec for option chain)
- Fetches data in a structured sequence within 60 seconds
- Auto-refresh every 1 minute

Rate Limits (from Dhan API docs):
- Quote APIs: 1 request/second, unlimited/day
- Data APIs: 5 requests/second, 100,000/day
- Option Chain: 1 request/3 seconds

Fetch Sequence (60-second cycle):
0-10s: Historical chart data (Nifty, Sensex)
10-20s: Real-time OHLC data (Nifty, Sensex)
20-30s: Option chain data (Nifty)
30-40s: Sensex option chain data
40-50s: Market quote data (other instruments)
50-60s: Buffer time for next cycle
"""

import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st
import pytz
from config import get_dhan_credentials, IST, get_current_time_ist
from api_request_limiter import global_rate_limiter

# Dhan Security IDs (from instrument master)
SECURITY_IDS = {
    # Major Indices
    'NIFTY': '13',      # IDX_I segment
    'SENSEX': '51',     # IDX_I segment
    'BANKNIFTY': '25',  # IDX_I segment
    'FINNIFTY': '27',   # IDX_I segment
    'MIDCPNIFTY': '28', # IDX_I segment

    # Volatility Index
    'INDIAVIX': '99',   # India VIX - Critical for sentiment

    # Sector Indices
    'NIFTY_IT': '65',      # Nifty IT
    'NIFTY_AUTO': '66',    # Nifty Auto
    'NIFTY_PHARMA': '67',  # Nifty Pharma
    'NIFTY_METAL': '68',   # Nifty Metal
    'NIFTY_REALTY': '69',  # Nifty Realty
    'NIFTY_ENERGY': '70',  # Nifty Energy
    'NIFTY_FMCG': '71',    # Nifty FMCG
}

# Exchange Segments
EXCHANGE_SEGMENTS = {
    # Major Indices
    'NIFTY': 'IDX_I',
    'SENSEX': 'IDX_I',
    'BANKNIFTY': 'IDX_I',
    'FINNIFTY': 'IDX_I',
    'MIDCPNIFTY': 'IDX_I',

    # Volatility Index
    'INDIAVIX': 'IDX_I',

    # Sector Indices
    'NIFTY_IT': 'IDX_I',
    'NIFTY_AUTO': 'IDX_I',
    'NIFTY_PHARMA': 'IDX_I',
    'NIFTY_METAL': 'IDX_I',
    'NIFTY_REALTY': 'IDX_I',
    'NIFTY_ENERGY': 'IDX_I',
    'NIFTY_FMCG': 'IDX_I',
}

class DhanDataFetcher:
    """
    Dhan API Data Fetcher with Rate Limiting
    """

    def __init__(self):
        """Initialize Dhan Data Fetcher"""
        creds = get_dhan_credentials()
        if not creds:
            raise Exception("DhanHQ credentials not found")

        self.client_id = creds['client_id']
        self.access_token = creds['access_token']
        self.base_url = "https://api.dhan.co/v2"

        self.headers = {
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }

        # Rate limiting tracking
        self.last_request_time = {}
        self.request_count = {}

        # Data cache
        self.cache = {}
        self.cache_timestamps = {}

    def _rate_limit_wait(self, api_type: str):
        """
        Wait for rate limit compliance using global rate limiter

        Args:
            api_type: 'quote' (1/sec), 'data' (5/sec), or 'option_chain' (1/3sec)

        Note:
            Now uses global rate limiter for thread-safe, cross-instance rate limiting
        """
        # Use global rate limiter instead of per-instance rate limiting
        if not global_rate_limiter.wait_for_slot(api_type):
            raise Exception(f"Circuit breaker active for {api_type}. Please wait and try again.")

    def fetch_ohlc_data(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Fetch OHLC + LTP data for multiple instruments

        Rate Limit: 1 request/second (Quote API)
        Can fetch up to 1000 instruments in single request

        Args:
            instruments: List of instrument names (e.g., ['NIFTY', 'SENSEX'])

        Returns:
            Dict with OHLC data for each instrument
        """
        self._rate_limit_wait('quote')

        # Build request payload
        payload = {}
        for instrument in instruments:
            segment = EXCHANGE_SEGMENTS.get(instrument)
            security_id = SECURITY_IDS.get(instrument)

            if segment and security_id:
                if segment not in payload:
                    payload[segment] = []
                payload[segment].append(int(security_id))

        try:
            url = f"{self.base_url}/marketfeed/ohlc"
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)

            if response.status_code == 200:
                # Track success for rate limiter
                global_rate_limiter.handle_success('quote')

                data = response.json()

                # Parse response
                result = {}
                for instrument in instruments:
                    segment = EXCHANGE_SEGMENTS.get(instrument)
                    security_id = SECURITY_IDS.get(instrument)

                    if segment and security_id:
                        instrument_data = data.get('data', {}).get(segment, {}).get(security_id)

                        if instrument_data:
                            result[instrument] = {
                                'success': True,
                                'last_price': instrument_data.get('last_price'),
                                'open': instrument_data.get('ohlc', {}).get('open'),
                                'high': instrument_data.get('ohlc', {}).get('high'),
                                'low': instrument_data.get('ohlc', {}).get('low'),
                                'close': instrument_data.get('ohlc', {}).get('close'),
                                'timestamp': get_current_time_ist()
                            }
                        else:
                            result[instrument] = {'success': False, 'error': 'No data found'}

                return result
            elif response.status_code == 429:
                # Handle rate limit error with exponential backoff
                global_rate_limiter.handle_rate_limit_error('quote')
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': 'Rate limit exceeded. Will retry with exponential backoff.'}
            else:
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': response.text}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def fetch_intraday_data(self, instrument: str, interval: str = "1",
                           from_date: Optional[str] = None,
                           to_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch intraday historical data

        Rate Limit: 5 requests/second (Data API)

        Args:
            instrument: Instrument name (e.g., 'NIFTY')
            interval: Minute interval - "1", "5", "15", "25", "60"
            from_date: Start date (YYYY-MM-DD HH:MM:SS) - defaults to today 9:15 AM
            to_date: End date (YYYY-MM-DD HH:MM:SS) - defaults to current time

        Returns:
            Dict with intraday OHLC data
        """
        self._rate_limit_wait('data')

        security_id = SECURITY_IDS.get(instrument)
        exchange_segment = EXCHANGE_SEGMENTS.get(instrument)

        if not security_id or not exchange_segment:
            return {'success': False, 'error': f'Unknown instrument: {instrument}'}

        # Default dates
        if not from_date:
            today = get_current_time_ist()
            from_date = today.replace(hour=9, minute=15, second=0).strftime('%Y-%m-%d %H:%M:%S')

        if not to_date:
            to_date = get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S')

        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": "INDEX",
            "interval": interval,
            "fromDate": from_date,
            "toDate": to_date
        }

        try:
            url = f"{self.base_url}/charts/intraday"
            response = requests.post(url, json=payload, headers=self.headers, timeout=15)

            if response.status_code == 200:
                # Track success for rate limiter
                global_rate_limiter.handle_success('data')

                data = response.json()

                # Convert to DataFrame
                df = pd.DataFrame({
                    'open': data.get('open', []),
                    'high': data.get('high', []),
                    'low': data.get('low', []),
                    'close': data.get('close', []),
                    'volume': data.get('volume', []),
                    'timestamp': [datetime.fromtimestamp(ts) for ts in data.get('timestamp', [])]
                })

                return {
                    'success': True,
                    'data': df,
                    'instrument': instrument,
                    'interval': interval,
                    'timestamp': get_current_time_ist()
                }
            elif response.status_code == 429:
                # Handle rate limit error with exponential backoff
                global_rate_limiter.handle_rate_limit_error('data')
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': 'Rate limit exceeded. Will retry with exponential backoff.'}
            else:
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': response.text}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def fetch_option_chain(self, instrument: str, expiry: str) -> Dict[str, Any]:
        """
        Fetch option chain data

        Rate Limit: 1 request per 3 seconds (Option Chain API)

        Args:
            instrument: Instrument name (e.g., 'NIFTY')
            expiry: Expiry date (YYYY-MM-DD)

        Returns:
            Dict with option chain data
        """
        self._rate_limit_wait('option_chain')

        security_id = SECURITY_IDS.get(instrument)
        exchange_segment = EXCHANGE_SEGMENTS.get(instrument)

        if not security_id or not exchange_segment:
            return {'success': False, 'error': f'Unknown instrument: {instrument}'}

        payload = {
            "UnderlyingScrip": int(security_id),
            "UnderlyingSeg": exchange_segment,
            "Expiry": expiry
        }

        try:
            url = f"{self.base_url}/optionchain"
            response = requests.post(url, json=payload, headers=self.headers, timeout=15)

            if response.status_code == 200:
                # Track success for rate limiter
                global_rate_limiter.handle_success('option_chain')

                data = response.json()

                return {
                    'success': True,
                    'data': data.get('data', {}),
                    'instrument': instrument,
                    'expiry': expiry,
                    'timestamp': get_current_time_ist()
                }
            elif response.status_code == 429:
                # Handle rate limit error with exponential backoff
                global_rate_limiter.handle_rate_limit_error('option_chain')
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': 'Rate limit exceeded. Will retry with exponential backoff.'}
            else:
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': response.text}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def fetch_expiry_list(self, instrument: str) -> Dict[str, Any]:
        """
        Fetch expiry list for an instrument

        Args:
            instrument: Instrument name (e.g., 'NIFTY')

        Returns:
            Dict with expiry dates list
        """
        self._rate_limit_wait('option_chain')

        security_id = SECURITY_IDS.get(instrument)
        exchange_segment = EXCHANGE_SEGMENTS.get(instrument)

        if not security_id or not exchange_segment:
            return {'success': False, 'error': f'Unknown instrument: {instrument}'}

        payload = {
            "UnderlyingScrip": int(security_id),
            "UnderlyingSeg": exchange_segment
        }

        try:
            url = f"{self.base_url}/optionchain/expirylist"
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)

            if response.status_code == 200:
                # Track success for rate limiter
                global_rate_limiter.handle_success('option_chain')

                data = response.json()

                return {
                    'success': True,
                    'expiry_dates': data.get('data', []),
                    'instrument': instrument,
                    'timestamp': get_current_time_ist()
                }
            elif response.status_code == 429:
                # Handle rate limit error with exponential backoff
                global_rate_limiter.handle_rate_limit_error('option_chain')
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': 'Rate limit exceeded. Will retry with exponential backoff.'}
            else:
                return {'success': False, 'error': f'API returned {response.status_code}', 'message': response.text}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def fetch_all_data_sequential(self) -> Dict[str, Any]:
        """
        Fetch all required data sequentially with 10-second intervals

        This method orchestrates the complete data fetch cycle:
        - 0-10s: Intraday chart data
        - 10-20s: Real-time OHLC data
        - 20-30s: NIFTY expiry list
        - 30-40s: Option chain data
        - Total: ~40 seconds

        Returns:
            Dict containing all fetched data
        """
        result = {
            'success': True,
            'fetch_start': get_current_time_ist(),
            'nifty': {},
            'sensex': {},
            'option_chain': {},
            'errors': []
        }

        try:
            # STEP 1 (0-10s): Fetch intraday chart data for NIFTY & SENSEX
            print("⏱️  [0-10s] Fetching intraday chart data...")

            nifty_chart = self.fetch_intraday_data('NIFTY', interval="1")
            time.sleep(2)  # Small delay between instruments
            sensex_chart = self.fetch_intraday_data('SENSEX', interval="1")

            if nifty_chart.get('success'):
                result['nifty']['chart'] = nifty_chart
            else:
                result['errors'].append(f"Nifty chart: {nifty_chart.get('error')}")

            if sensex_chart.get('success'):
                result['sensex']['chart'] = sensex_chart
            else:
                result['errors'].append(f"Sensex chart: {sensex_chart.get('error')}")

            # Wait to reach 10-second mark
            elapsed = (get_current_time_ist() - result['fetch_start']).total_seconds()
            if elapsed < 10:
                time.sleep(10 - elapsed)

            # STEP 2 (10-20s): Fetch real-time OHLC data
            print("⏱️  [10-20s] Fetching real-time OHLC data...")

            ohlc_data = self.fetch_ohlc_data(['NIFTY', 'SENSEX'])

            if ohlc_data.get('NIFTY', {}).get('success'):
                result['nifty']['ohlc'] = ohlc_data['NIFTY']
            else:
                result['errors'].append(f"Nifty OHLC: {ohlc_data.get('NIFTY', {}).get('error')}")

            if ohlc_data.get('SENSEX', {}).get('success'):
                result['sensex']['ohlc'] = ohlc_data['SENSEX']
            else:
                result['errors'].append(f"Sensex OHLC: {ohlc_data.get('SENSEX', {}).get('error')}")

            # Wait to reach 20-second mark
            elapsed = (get_current_time_ist() - result['fetch_start']).total_seconds()
            if elapsed < 20:
                time.sleep(20 - elapsed)

            # STEP 3 (20-30s): Fetch NIFTY expiry list
            print("⏱️  [20-30s] Fetching NIFTY expiry list...")

            nifty_expiry = self.fetch_expiry_list('NIFTY')

            if nifty_expiry.get('success'):
                result['nifty']['expiry_list'] = nifty_expiry

                # Wait to reach 30-second mark
                elapsed = (get_current_time_ist() - result['fetch_start']).total_seconds()
                if elapsed < 30:
                    time.sleep(30 - elapsed)

                # STEP 4 (30-40s): Fetch option chain for current expiry
                print("⏱️  [30-40s] Fetching option chain data...")

                if nifty_expiry['expiry_dates']:
                    current_expiry = nifty_expiry['expiry_dates'][0]
                    option_chain = self.fetch_option_chain('NIFTY', current_expiry)

                    if option_chain.get('success'):
                        result['option_chain'] = option_chain
                    else:
                        result['errors'].append(f"Option chain: {option_chain.get('error')}")
            else:
                result['errors'].append(f"Nifty expiry: {nifty_expiry.get('error')}")

            # Calculate ATM strikes
            if result['nifty'].get('ohlc', {}).get('last_price'):
                nifty_ltp = result['nifty']['ohlc']['last_price']
                result['nifty']['atm_strike'] = round(nifty_ltp / 50) * 50

            if result['sensex'].get('ohlc', {}).get('last_price'):
                sensex_ltp = result['sensex']['ohlc']['last_price']
                result['sensex']['atm_strike'] = round(sensex_ltp / 100) * 100

            result['fetch_end'] = get_current_time_ist()
            result['total_time'] = (result['fetch_end'] - result['fetch_start']).total_seconds()
            result['success'] = len(result['errors']) == 0

            print(f"✅ Data fetch completed in {result['total_time']:.1f} seconds")
            if result['errors']:
                print(f"⚠️  {len(result['errors'])} errors occurred")

            return result

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['errors'].append(str(e))
            return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_nifty_data() -> Dict[str, Any]:
    """
    Get comprehensive NIFTY data (OHLC, chart, option chain)

    Returns:
        Dict with NIFTY data
    """
    try:
        fetcher = DhanDataFetcher()
        all_data = fetcher.fetch_all_data_sequential()

        if all_data.get('success') or all_data.get('nifty'):
            nifty = all_data.get('nifty', {})
            ohlc = nifty.get('ohlc', {})
            expiry_list = nifty.get('expiry_list', {})

            # Check if we have valid price data
            spot_price = ohlc.get('last_price', 0)
            atm_strike = nifty.get('atm_strike', 0)

            # If spot price is 0 or None, it means API failed
            if not spot_price or spot_price == 0:
                error_msg = ', '.join(all_data.get('errors', [])) if all_data.get('errors') else 'API returned no data'
                return {
                    'success': False,
                    'error': f'Unable to fetch NIFTY price data: {error_msg}',
                    'spot_price': None,
                    'atm_strike': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'expiry_dates': [],
                    'current_expiry': 'N/A',
                    'chart_data': None,
                    'timestamp': get_current_time_ist()
                }

            return {
                'success': True,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'open': ohlc.get('open', 0),
                'high': ohlc.get('high', 0),
                'low': ohlc.get('low', 0),
                'close': ohlc.get('close', 0),
                'expiry_dates': expiry_list.get('expiry_dates', []),
                'current_expiry': expiry_list.get('expiry_dates', [''])[0] if expiry_list.get('expiry_dates') else '',
                'chart_data': nifty.get('chart', {}).get('data'),
                'timestamp': get_current_time_ist()
            }
        else:
            return {
                'success': False,
                'error': ', '.join(all_data.get('errors', ['Unknown error']))
            }

    except Exception as e:
        error_msg = str(e)
        # Check if it's a credentials error
        if 'credentials' in error_msg.lower() or 'access_token' in error_msg.lower():
            error_msg = 'DhanHQ credentials not configured. Please set up .streamlit/secrets.toml'
        return {
            'success': False,
            'error': error_msg
        }


def get_sensex_data() -> Dict[str, Any]:
    """
    Get comprehensive SENSEX data (OHLC, chart)

    Returns:
        Dict with SENSEX data
    """
    try:
        fetcher = DhanDataFetcher()
        all_data = fetcher.fetch_all_data_sequential()

        if all_data.get('success') or all_data.get('sensex'):
            sensex = all_data.get('sensex', {})
            ohlc = sensex.get('ohlc', {})

            # Check if we have valid price data
            spot_price = ohlc.get('last_price', 0)
            atm_strike = sensex.get('atm_strike', 0)

            # If spot price is 0 or None, it means API failed
            if not spot_price or spot_price == 0:
                error_msg = ', '.join(all_data.get('errors', [])) if all_data.get('errors') else 'API returned no data'
                return {
                    'success': False,
                    'error': f'Unable to fetch SENSEX price data: {error_msg}',
                    'spot_price': None,
                    'atm_strike': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'chart_data': None,
                    'timestamp': get_current_time_ist()
                }

            return {
                'success': True,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'open': ohlc.get('open', 0),
                'high': ohlc.get('high', 0),
                'low': ohlc.get('low', 0),
                'close': ohlc.get('close', 0),
                'chart_data': sensex.get('chart', {}).get('data'),
                'timestamp': get_current_time_ist()
            }
        else:
            return {
                'success': False,
                'error': ', '.join(all_data.get('errors', ['Unknown error']))
            }

    except Exception as e:
        error_msg = str(e)
        # Check if it's a credentials error
        if 'credentials' in error_msg.lower() or 'access_token' in error_msg.lower():
            error_msg = 'DhanHQ credentials not configured. Please set up .streamlit/secrets.toml'
        return {
            'success': False,
            'error': error_msg
        }


def get_nifty_futures_data() -> Dict[str, Any]:
    """
    Get NIFTY Futures data (current month and next month contracts)

    Uses multiple data sources:
    1. Dhan API (preferred) - For real-time futures data
    2. yfinance (fallback) - When Dhan is unavailable

    Returns:
        Dict with futures data including:
        - current_month: Current month futures contract data
        - next_month: Next month futures contract data
        - spot_price: NIFTY spot price for reference
    """
    try:
        # First try to get spot price from Dhan
        nifty_data = get_nifty_data()
        spot_price = nifty_data.get('spot_price') if nifty_data.get('success') else None

        # Try Dhan API for futures data
        try:
            fetcher = DhanDataFetcher()

            # Note: Dhan futures require specific security IDs from instrument master
            # For now, we'll use yfinance as it's more straightforward for futures
            # To use Dhan, download instrument master and map security IDs
            raise Exception("Dhan futures not yet implemented - falling back to yfinance")

        except Exception as dhan_error:
            # Fall back to yfinance
            import yfinance as yf
            from datetime import datetime

            # If Dhan spot price is not available, get it from yfinance
            if spot_price is None:
                try:
                    nifty_ticker = yf.Ticker("^NSEI")
                    nifty_hist = nifty_ticker.history(period="1d", interval="1d")
                    if not nifty_hist.empty:
                        spot_price = nifty_hist['Close'].iloc[-1]
                except Exception as e:
                    print(f"Could not fetch NIFTY spot from yfinance: {e}")
                    spot_price = None

            # Get current month and next month expiry dates
            current_month, next_month = _get_futures_expiry_months()

            # NIFTY futures ticker format: NIFTY{YY}{MMM}FUT.NS
            # Example: NIFTY25JANFUT.NS for January 2025
            current_ticker = f"NIFTY{current_month}FUT.NS"
            next_ticker = f"NIFTY{next_month}FUT.NS"

            result = {
                'success': False,
                'current_month': {},
                'next_month': {},
                'spot_price': spot_price,
                'data_source': 'yfinance',
                'timestamp': get_current_time_ist()
            }

            # Fetch current month futures
            try:
                current_fut = yf.Ticker(current_ticker)
                current_hist = current_fut.history(period="2d", interval="1d")

                if not current_hist.empty:
                    latest = current_hist.iloc[-1]
                    previous = current_hist.iloc[-2] if len(current_hist) > 1 else latest

                    ltp = latest['Close']
                    change_pct = ((ltp - previous['Close']) / previous['Close'] * 100) if len(current_hist) > 1 else 0

                    # Get expiry date
                    expiry_date = _parse_futures_expiry(current_month)
                    days_to_expiry = (expiry_date - get_current_time_ist()).days if expiry_date else 0

                    result['current_month'] = {
                        'ltp': ltp,
                        'open': latest['Open'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'close': latest['Close'],
                        'volume': int(latest['Volume']) if 'Volume' in latest else 0,
                        'expiry': expiry_date.strftime('%Y-%m-%d') if expiry_date else 'N/A',
                        'days_to_expiry': days_to_expiry,
                        'price_change_pct': change_pct,
                        'ticker': current_ticker,
                        # Fields not available from yfinance - set to None
                        'oi': None,
                        'oi_change_pct': None,
                        'volume_change_pct': None,
                        'spot_change_pct': None,
                        'price_change_5d_pct': None,
                        'rsi': None,
                        'macd': None,
                        'bb_position': None,
                        'trend': None,
                        'buildup_pattern': None
                    }
                    result['success'] = True

            except Exception as e:
                result['current_month'] = {'error': f'Failed to fetch {current_ticker}: {str(e)}'}

            # Fetch next month futures
            try:
                next_fut = yf.Ticker(next_ticker)
                next_hist = next_fut.history(period="2d", interval="1d")

                if not next_hist.empty:
                    latest = next_hist.iloc[-1]
                    previous = next_hist.iloc[-2] if len(next_hist) > 1 else latest

                    ltp = latest['Close']
                    change_pct = ((ltp - previous['Close']) / previous['Close'] * 100) if len(next_hist) > 1 else 0

                    # Get expiry date
                    expiry_date = _parse_futures_expiry(next_month)
                    days_to_expiry = (expiry_date - get_current_time_ist()).days if expiry_date else 0

                    result['next_month'] = {
                        'ltp': ltp,
                        'open': latest['Open'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'close': latest['Close'],
                        'volume': int(latest['Volume']) if 'Volume' in latest else 0,
                        'expiry': expiry_date.strftime('%Y-%m-%d') if expiry_date else 'N/A',
                        'days_to_expiry': days_to_expiry,
                        'price_change_pct': change_pct,
                        'ticker': next_ticker,
                        # Fields not available from yfinance - set to None
                        'oi': None,
                        'oi_change_pct': None,
                        'volume_change_pct': None
                    }

            except Exception as e:
                result['next_month'] = {'error': f'Failed to fetch {next_ticker}: {str(e)}'}

            return result

    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to fetch futures data: {str(e)}',
            'current_month': {},
            'next_month': {},
            'spot_price': None,
            'timestamp': get_current_time_ist()
        }


def _get_futures_expiry_months() -> tuple:
    """
    Get current and next month expiry codes for NIFTY futures

    Returns:
        Tuple of (current_month_code, next_month_code)
        Format: "25JAN", "25FEB", etc.
    """
    now = get_current_time_ist()
    current_year = now.year % 100  # Last 2 digits
    current_month = now.month

    # NIFTY futures expire on last Thursday of the month
    # After 3rd week, we consider next month as current
    last_thursday = _get_last_thursday(now.year, now.month)

    if now.date() > last_thursday.date():
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    next_month = current_month + 1
    next_year = current_year
    if next_month > 12:
        next_month = 1
        next_year += 1

    month_codes = ['', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    current_code = f"{current_year:02d}{month_codes[current_month]}"
    next_code = f"{next_year:02d}{month_codes[next_month]}"

    return (current_code, next_code)


def _get_last_thursday(year: int, month: int) -> datetime:
    """
    Get last Thursday of a month (NIFTY futures expiry date)

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        datetime of last Thursday
    """
    from calendar import monthrange

    # Get last day of month
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)

    # Find last Thursday
    # Thursday is weekday 3 (Monday=0)
    days_back = (last_date.weekday() - 3) % 7
    last_thursday = last_date - timedelta(days=days_back)

    return last_thursday


def _parse_futures_expiry(month_code: str) -> Optional[datetime]:
    """
    Parse futures expiry date from month code

    Args:
        month_code: Format "25JAN" (year + month)

    Returns:
        datetime of expiry (last Thursday)
    """
    try:
        year_str = month_code[:2]
        month_str = month_code[2:]

        year = 2000 + int(year_str)

        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        month = month_map.get(month_str.upper())
        if not month:
            return None

        return _get_last_thursday(year, month)

    except Exception:
        return None


# ============================================================================
# TEST CONNECTION
# ============================================================================

def test_dhan_connection() -> bool:
    """
    Test Dhan API connection

    Returns:
        True if connection successful, False otherwise
    """
    try:
        fetcher = DhanDataFetcher()
        # Simple OHLC fetch to test connection
        result = fetcher.fetch_ohlc_data(['NIFTY'])
        return result.get('NIFTY', {}).get('success', False)
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
