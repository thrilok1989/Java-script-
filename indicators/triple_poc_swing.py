"""
Triple POC + Future Swing Indicator
Converted from Pine Script v6 by BigBeluga
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-nc-sa/4.0/

This indicator combines Triple Point of Control (POC) analysis with Future Swing projection.
- Triple POC: Short-term (25), Medium-term (40), Long-term (100) volume profiles
- Future Swing: Projects swing targets based on historical swing percentage moves
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import statistics


class TriplePOCSwingIndicator:
    """
    Triple POC + Future Swing Indicator

    Combines three POC (Point of Control) calculations at different timeframes
    with swing detection and future swing projection.

    POC Calculations:
    - POC 1 (Short-term): 25-period volume profile
    - POC 2 (Medium-term): 40-period volume profile
    - POC 3 (Long-term): 100-period volume profile

    Swing Projection:
    - Detects swing highs and lows
    - Calculates historical swing percentages
    - Projects future swing targets using Average/Median/Mode
    - Provides buy/sell volume analysis for swing legs
    """

    def __init__(
        self,
        poc1_period: int = 25,
        poc2_period: int = 40,
        poc3_period: int = 100,
        swing_length: int = 30,
        projection_offset: int = 10,
        historical_samples: int = 5,
        projection_method: str = "Average",
        bins: int = 25,
    ):
        self.poc1_period = max(10, min(400, poc1_period))
        self.poc2_period = max(10, min(400, poc2_period))
        self.poc3_period = max(10, min(400, poc3_period))
        self.swing_length = max(10, swing_length)
        self.projection_offset = projection_offset
        self.historical_samples = max(3, min(20, historical_samples))
        self.projection_method = projection_method
        self.bins = bins

    def _calculate_poc(self, df: pd.DataFrame, period: int) -> Dict:
        """
        Calculate Point of Control for a given lookback period.

        Mirrors Pine Script pocCalculate() function:
        - Divides price range into 25 bins
        - Accumulates volume at each price level
        - Identifies the bin with maximum volume as POC
        - Returns POC price, upper/lower bands, and volume

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
        period : int
            Lookback period for POC calculation

        Returns:
        --------
        dict with poc, upper_poc, lower_poc, volume, high, low values
        """
        if len(df) < period:
            return {
                'poc': None,
                'upper_poc': None,
                'lower_poc': None,
                'volume': None,
                'high': None,
                'low': None,
                'poc_series': [],
                'upper_series': [],
                'lower_series': [],
            }

        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

        poc_series = []
        upper_series = []
        lower_series = []

        for idx in range(period, len(df)):
            window = df.iloc[idx - period:idx + 1]

            H = window[high_col].max()
            L = window[low_col].min()

            if H == L:
                poc_series.append(H)
                upper_series.append(H)
                lower_series.append(L)
                continue

            step = (H - L) / self.bins

            # Initialize bins
            vol = [0.0] * self.bins
            lvl = [0.0] * self.bins

            for k in range(self.bins):
                l_val = L + k * step
                mid = l_val + step / 2
                lvl[k] = mid

            # Accumulate volume per bin
            for i in range(len(window)):
                c = window[close_col].iloc[i]
                v = window[volume_col].iloc[i] if volume_col in window.columns else 1.0

                for k in range(self.bins):
                    if abs(c - lvl[k]) <= step:
                        vol[k] += v

            # Find POC (bin with max volume)
            max_vol_idx = vol.index(max(vol))
            poc = lvl[max_vol_idx]
            max_volume = vol[max_vol_idx]

            upper_poc = poc + step * 2
            lower_poc = poc - step * 2

            poc_series.append(poc)
            upper_series.append(upper_poc)
            lower_series.append(lower_poc)

        # Current values (last calculated)
        current_poc = poc_series[-1] if poc_series else None
        current_upper = upper_series[-1] if upper_series else None
        current_lower = lower_series[-1] if lower_series else None

        return {
            'poc': current_poc,
            'upper_poc': current_upper,
            'lower_poc': current_lower,
            'volume': max_volume if poc_series else None,
            'high': H if poc_series else None,
            'low': L if poc_series else None,
            'poc_series': poc_series,
            'upper_series': upper_series,
            'lower_series': lower_series,
        }

    def _detect_swings(self, df: pd.DataFrame) -> Dict:
        """
        Detect swing highs and lows, calculate swing percentages,
        and project future swing targets.

        Mirrors Pine Script swing detection logic:
        - Uses ta.highest/ta.lowest for swing detection
        - Tracks direction changes
        - Calculates swing percentage moves
        - Projects future swing using Average/Median/Mode

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data

        Returns:
        --------
        dict with swing data, projections, volume analysis
        """
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

        length = self.swing_length

        if len(df) < length + 2:
            return {
                'swing_high': None,
                'swing_low': None,
                'direction': None,
                'swing_percentages': [],
                'projected_swing': None,
                'projected_price': None,
                'buy_volume': 0,
                'sell_volume': 0,
                'delta_volume': 0,
                'total_volume': 0,
            }

        highs = df[high_col].values
        lows = df[low_col].values
        closes = df[close_col].values
        opens = df[open_col].values
        volumes = df[volume_col].values if volume_col in df.columns else np.ones(len(df))

        # Calculate rolling highest and lowest
        h_swing = pd.Series(highs).rolling(window=length).max().values
        l_swing = pd.Series(lows).rolling(window=length).min().values

        # Track direction and swings
        direc = False  # True = down move, False = up move
        h_swing_val = None
        h_swing_idx = None
        l_swing_val = None
        l_swing_idx = None

        swing_percentages = []

        for i in range(length + 1, len(df)):
            # Detect direction changes
            if highs[i] == h_swing[i]:
                direc = True
            if lows[i] == l_swing[i]:
                direc = False

            # Store completed swing high
            if (i >= 1 and
                highs[i - 1] == h_swing[i - 1] and
                highs[i] < h_swing[i]):
                h_swing_idx = i - 1
                h_swing_val = highs[i - 1]

            # Store completed swing low
            if (i >= 1 and
                lows[i - 1] == l_swing[i - 1] and
                lows[i] > l_swing[i]):
                l_swing_idx = i - 1
                l_swing_val = lows[i - 1]

            # On direction change, calculate swing percentage
            prev_direc = direc
            if i >= length + 2:
                prev_h = h_swing[i - 1]
                prev_l = l_swing[i - 1]
                curr_h_dir = highs[i] == h_swing[i]
                curr_l_dir = lows[i] == l_swing[i]

        # Recalculate swing percentages with a cleaner approach
        swing_percentages = []
        completed_swings = []
        current_direction = None

        for i in range(length + 1, len(df)):
            if highs[i] == h_swing[i]:
                new_dir = True  # Down move
            elif lows[i] == l_swing[i]:
                new_dir = False  # Up move
            else:
                new_dir = current_direction

            if current_direction is not None and new_dir != current_direction:
                # Direction changed - record swing
                if h_swing_val is not None and l_swing_val is not None:
                    if not new_dir:
                        # Was going down, now up: bearish swing completed
                        pc = (l_swing_val - h_swing_val) / h_swing_val * 100
                    else:
                        # Was going up, now down: bullish swing completed
                        pc = (h_swing_val - l_swing_val) / l_swing_val * 100
                    swing_percentages.append(pc)

            current_direction = new_dir

            # Update swing points
            if (i >= 1 and
                highs[i - 1] == h_swing[i - 1] and
                highs[i] < h_swing[i]):
                h_swing_idx = i - 1
                h_swing_val = highs[i - 1]

            if (i >= 1 and
                lows[i - 1] == l_swing[i - 1] and
                lows[i] > l_swing[i]):
                l_swing_idx = i - 1
                l_swing_val = lows[i - 1]

        # Trim to historical samples
        if len(swing_percentages) > self.historical_samples:
            swing_percentages = swing_percentages[-self.historical_samples:]

        # Calculate projected swing value
        abs_swings = [abs(s) for s in swing_percentages]
        if abs_swings:
            if self.projection_method == "Average":
                swing_val = np.mean(abs_swings)
            elif self.projection_method == "Median":
                swing_val = np.median(abs_swings)
            elif self.projection_method == "Mode":
                try:
                    swing_val = statistics.mode([round(s, 1) for s in abs_swings])
                except statistics.StatisticsError:
                    swing_val = np.mean(abs_swings)
            else:
                swing_val = np.mean(abs_swings)
        else:
            swing_val = 0

        # Calculate buy/sell volume for current swing leg
        buy_vol = 0
        sell_vol = 0
        if h_swing_idx is not None and l_swing_idx is not None:
            lower_idx = min(h_swing_idx, l_swing_idx)
            for i in range(lower_idx, len(df)):
                c = closes[i]
                o = opens[i]
                v = volumes[i]
                if c > o:
                    buy_vol += v
                else:
                    sell_vol += v

        # Project future swing price
        projected_price = None
        if h_swing_val is not None and l_swing_val is not None and swing_val > 0:
            if not current_direction:
                # Bearish projection (going down from swing high)
                projected_price = h_swing_val - (h_swing_val * (swing_val / 100))
            else:
                # Bullish projection (going up from swing low)
                projected_price = l_swing_val + (l_swing_val * (swing_val / 100))

        return {
            'swing_high': {'value': h_swing_val, 'index': h_swing_idx},
            'swing_low': {'value': l_swing_val, 'index': l_swing_idx},
            'direction': current_direction,  # True = bullish (going up), False = bearish (going down)
            'swing_percentages': swing_percentages,
            'projected_swing_pct': swing_val,
            'projected_price': projected_price,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'delta_volume': buy_vol - sell_vol,
            'total_volume': buy_vol + sell_vol,
            'projection_method': self.projection_method,
        }

    def _determine_candle_bias(self, close: float, upper_poc: float, lower_poc: float) -> str:
        """
        Determine candle bias based on POC 1 channel position.

        Mirrors Pine Script candle coloring logic:
        - Above upper POC = BULLISH (blue)
        - Between upper/lower POC = NEUTRAL (gray)
        - Below lower POC = BEARISH (orange)
        """
        if close > upper_poc:
            return "BULLISH"
        elif close < lower_poc:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run full Triple POC + Future Swing analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with at least max(poc3_period, swing_length) bars

        Returns:
        --------
        dict with complete analysis results including:
        - Three POC calculations with series data
        - Swing detection and projection
        - Volume analysis
        - Overall bias determination
        - Dashboard-ready summary
        """
        close_col = 'Close' if 'Close' in df.columns else 'close'

        if df is None or len(df) < max(self.poc1_period, self.poc2_period,
                                         self.poc3_period, self.swing_length) + 2:
            return {
                'success': False,
                'error': 'Insufficient data for Triple POC + Swing analysis',
                'poc1': None, 'poc2': None, 'poc3': None,
                'swing': None, 'bias': 'NEUTRAL',
                'bias_results': [],
            }

        # Calculate all three POCs
        poc1 = self._calculate_poc(df, self.poc1_period)
        poc2 = self._calculate_poc(df, self.poc2_period)
        poc3 = self._calculate_poc(df, self.poc3_period)

        # Detect swings and project
        swing = self._detect_swings(df)

        # Current price
        current_price = df[close_col].iloc[-1]

        # Determine POC-based bias (using POC 1 channel)
        poc1_bias = "NEUTRAL"
        if poc1['poc'] is not None and poc1['upper_poc'] is not None:
            poc1_bias = self._determine_candle_bias(
                current_price, poc1['upper_poc'], poc1['lower_poc']
            )

        # Determine POC alignment bias
        poc_alignment_bias = self._determine_poc_alignment(
            current_price, poc1, poc2, poc3
        )

        # Determine swing projection bias
        swing_bias = "NEUTRAL"
        if swing['projected_price'] is not None:
            if swing['direction']:
                # Currently in bullish swing
                swing_bias = "BULLISH"
            else:
                swing_bias = "BEARISH"

        # Volume delta bias from swing leg
        volume_bias = "NEUTRAL"
        if swing['delta_volume'] > 0:
            volume_bias = "BULLISH"
        elif swing['delta_volume'] < 0:
            volume_bias = "BEARISH"

        # Calculate overall bias score
        bias_results = self._compile_bias_results(
            current_price, poc1, poc2, poc3, swing, poc1_bias,
            poc_alignment_bias, swing_bias, volume_bias
        )

        # Calculate overall score and bias
        total_score = sum(b['score'] * b['weight'] for b in bias_results)
        total_weight = sum(b['weight'] for b in bias_results)
        overall_score = total_score / total_weight if total_weight > 0 else 0

        if overall_score > 20:
            overall_bias = "BULLISH"
        elif overall_score < -20:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        # Confidence based on agreement
        bullish_count = sum(1 for b in bias_results if 'BULLISH' in b['bias'])
        bearish_count = sum(1 for b in bias_results if 'BEARISH' in b['bias'])
        neutral_count = sum(1 for b in bias_results if b['bias'] == 'NEUTRAL')
        total = len(bias_results)

        if overall_bias == "BULLISH":
            confidence = (bullish_count / total * 100) if total > 0 else 0
        elif overall_bias == "BEARISH":
            confidence = (bearish_count / total * 100) if total > 0 else 0
        else:
            confidence = (neutral_count / total * 100) if total > 0 else 0

        return {
            'success': True,
            'current_price': current_price,
            'poc1': poc1,
            'poc2': poc2,
            'poc3': poc3,
            'swing': swing,
            'poc1_bias': poc1_bias,
            'poc_alignment_bias': poc_alignment_bias,
            'swing_bias': swing_bias,
            'volume_bias': volume_bias,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'confidence': confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': total,
            'bias_results': bias_results,
            'settings': {
                'poc1_period': self.poc1_period,
                'poc2_period': self.poc2_period,
                'poc3_period': self.poc3_period,
                'swing_length': self.swing_length,
                'projection_method': self.projection_method,
                'historical_samples': self.historical_samples,
            }
        }

    def _determine_poc_alignment(self, price: float, poc1: Dict, poc2: Dict, poc3: Dict) -> str:
        """
        Determine bias from POC alignment.

        If all three POCs are below price = Strong Bullish
        If all three POCs are above price = Strong Bearish
        Mixed = depends on majority
        """
        if poc1['poc'] is None or poc2['poc'] is None or poc3['poc'] is None:
            return "NEUTRAL"

        above_count = 0
        below_count = 0

        for poc in [poc1['poc'], poc2['poc'], poc3['poc']]:
            if price > poc:
                above_count += 1
            else:
                below_count += 1

        if above_count == 3:
            return "BULLISH"
        elif below_count == 3:
            return "BEARISH"
        elif above_count > below_count:
            return "BULLISH"
        elif below_count > above_count:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _compile_bias_results(
        self, price: float, poc1: Dict, poc2: Dict, poc3: Dict,
        swing: Dict, poc1_bias: str, poc_alignment: str,
        swing_bias: str, volume_bias: str
    ) -> List[Dict]:
        """
        Compile all bias indicators into a structured list.
        Matches the format used by BiasAnalysisPro for UI consistency.
        """
        results = []

        # 1. POC 1 Channel Bias (Fast)
        poc1_score = 50 if poc1_bias == "BULLISH" else (-50 if poc1_bias == "BEARISH" else 0)
        results.append({
            'indicator': f'POC 1 Channel ({self.poc1_period})',
            'value': f"POC: {poc1['poc']:.2f}" if poc1['poc'] else 'N/A',
            'bias': poc1_bias,
            'score': poc1_score,
            'weight': 2.0,
            'category': 'fast',
            'details': f"Price {'above' if poc1_bias == 'BULLISH' else 'below' if poc1_bias == 'BEARISH' else 'within'} POC 1 channel"
        })

        # 2. POC 2 Position (Medium)
        poc2_bias = "NEUTRAL"
        poc2_score = 0
        if poc2['poc'] is not None:
            if price > poc2['poc']:
                poc2_bias = "BULLISH"
                poc2_score = 40
            elif price < poc2['poc']:
                poc2_bias = "BEARISH"
                poc2_score = -40
        results.append({
            'indicator': f'POC 2 Position ({self.poc2_period})',
            'value': f"POC: {poc2['poc']:.2f}" if poc2['poc'] else 'N/A',
            'bias': poc2_bias,
            'score': poc2_score,
            'weight': 3.0,
            'category': 'medium',
            'details': f"Price {'above' if poc2_bias == 'BULLISH' else 'below' if poc2_bias == 'BEARISH' else 'at'} POC 2"
        })

        # 3. POC 3 Position (Slow)
        poc3_bias = "NEUTRAL"
        poc3_score = 0
        if poc3['poc'] is not None:
            if price > poc3['poc']:
                poc3_bias = "BULLISH"
                poc3_score = 30
            elif price < poc3['poc']:
                poc3_bias = "BEARISH"
                poc3_score = -30
        results.append({
            'indicator': f'POC 3 Position ({self.poc3_period})',
            'value': f"POC: {poc3['poc']:.2f}" if poc3['poc'] else 'N/A',
            'bias': poc3_bias,
            'score': poc3_score,
            'weight': 5.0,
            'category': 'slow',
            'details': f"Price {'above' if poc3_bias == 'BULLISH' else 'below' if poc3_bias == 'BEARISH' else 'at'} POC 3"
        })

        # 4. Triple POC Alignment
        align_score = 60 if poc_alignment == "BULLISH" else (-60 if poc_alignment == "BEARISH" else 0)
        results.append({
            'indicator': 'Triple POC Alignment',
            'value': poc_alignment,
            'bias': poc_alignment,
            'score': align_score,
            'weight': 4.0,
            'category': 'medium',
            'details': f"All 3 POCs {'below' if poc_alignment == 'BULLISH' else 'above' if poc_alignment == 'BEARISH' else 'mixed relative to'} price"
        })

        # 5. Swing Direction
        swing_score = 50 if swing_bias == "BULLISH" else (-50 if swing_bias == "BEARISH" else 0)
        swing_dir_text = "Up" if swing_bias == "BULLISH" else "Down" if swing_bias == "BEARISH" else "Flat"
        results.append({
            'indicator': 'Swing Direction',
            'value': swing_dir_text,
            'bias': swing_bias,
            'score': swing_score,
            'weight': 3.0,
            'category': 'fast',
            'details': f"Current swing leg: {swing_dir_text}"
        })

        # 6. Swing Projection Target
        proj_bias = "NEUTRAL"
        proj_score = 0
        proj_text = "N/A"
        if swing['projected_price'] is not None and price > 0:
            pct_to_target = ((swing['projected_price'] - price) / price) * 100
            if pct_to_target > 0.5:
                proj_bias = "BULLISH"
                proj_score = min(70, pct_to_target * 20)
            elif pct_to_target < -0.5:
                proj_bias = "BEARISH"
                proj_score = max(-70, pct_to_target * 20)
            proj_text = f"{swing['projected_price']:.2f} ({pct_to_target:+.2f}%)"
        results.append({
            'indicator': 'Swing Projection Target',
            'value': proj_text,
            'bias': proj_bias,
            'score': proj_score,
            'weight': 3.5,
            'category': 'medium',
            'details': f"Projected: {proj_text} ({self.projection_method})"
        })

        # 7. Swing Volume Delta
        vol_score = 40 if volume_bias == "BULLISH" else (-40 if volume_bias == "BEARISH" else 0)
        delta_text = f"{swing['delta_volume']:,.0f}" if swing['delta_volume'] != 0 else "0"
        results.append({
            'indicator': 'Swing Volume Delta',
            'value': delta_text,
            'bias': volume_bias,
            'score': vol_score,
            'weight': 2.5,
            'category': 'fast',
            'details': f"Buy: {swing['buy_volume']:,.0f} | Sell: {swing['sell_volume']:,.0f}"
        })

        # 8. Swing Average (Historical)
        hist_bias = "NEUTRAL"
        hist_score = 0
        if swing['swing_percentages']:
            avg_pct = swing['projected_swing_pct']
            if swing['direction']:
                hist_bias = "BULLISH"
                hist_score = min(50, avg_pct * 10)
            else:
                hist_bias = "BEARISH"
                hist_score = -min(50, avg_pct * 10)
        results.append({
            'indicator': f'Swing AVG ({self.projection_method})',
            'value': f"{swing['projected_swing_pct']:.2f}%",
            'bias': hist_bias,
            'score': hist_score,
            'weight': 2.0,
            'category': 'fast',
            'details': f"{len(swing['swing_percentages'])} historical swings analyzed"
        })

        return results


def analyze_triple_poc_swing(
    df: pd.DataFrame,
    poc1_period: int = 25,
    poc2_period: int = 40,
    poc3_period: int = 100,
    swing_length: int = 30,
    projection_offset: int = 10,
    historical_samples: int = 5,
    projection_method: str = "Average",
) -> Dict:
    """
    Convenience function to run Triple POC + Swing analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    poc1_period : int
        Short-term POC period (default 25)
    poc2_period : int
        Medium-term POC period (default 40)
    poc3_period : int
        Long-term POC period (default 100)
    swing_length : int
        Swing detection length (default 30)
    projection_offset : int
        Future projection offset in bars (default 10)
    historical_samples : int
        Number of historical swings to sample (default 5)
    projection_method : str
        Aggregation method: "Average", "Median", or "Mode" (default "Average")

    Returns:
    --------
    dict with complete analysis results
    """
    indicator = TriplePOCSwingIndicator(
        poc1_period=poc1_period,
        poc2_period=poc2_period,
        poc3_period=poc3_period,
        swing_length=swing_length,
        projection_offset=projection_offset,
        historical_samples=historical_samples,
        projection_method=projection_method,
    )
    return indicator.analyze(df)
