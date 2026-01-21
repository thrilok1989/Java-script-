"""
NIFTY Futures Analyzer
Complete futures analysis with chart analysis, options correlation, and trading signals

Features:
- Futures vs Spot analysis (Basis, Premium/Discount)
- Full technical chart analysis for futures
- Futures-Options correlation
- Cost of carry analysis
- Rollover analysis
- Futures positioning (FII/DII/Pro/Client)
- Trading signals based on futures data
- Risk metrics and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FuturesData:
    """Futures market data"""
    symbol: str
    expiry_date: datetime
    spot_price: float
    futures_price: float
    basis: float  # Futures - Spot
    basis_percentage: float
    premium_discount: str  # "PREMIUM" or "DISCOUNT"
    days_to_expiry: int
    volume: float
    open_interest: float
    change_in_oi: float
    oi_percentage_change: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    spread: float
    ltp: float
    change: float
    change_percentage: float


@dataclass
class CostOfCarryResult:
    """Cost of carry analysis result"""
    theoretical_futures_price: float
    actual_futures_price: float
    cost_of_carry_rate: float  # Annualized %
    mispricing: float
    mispricing_percentage: float
    arbitrage_opportunity: bool
    arbitrage_type: str  # "CASH_FUTURES" or "NONE"
    expected_return: float


@dataclass
class RolloverAnalysis:
    """Rollover analysis between near and next month"""
    current_month_expiry: datetime
    next_month_expiry: datetime
    current_month_oi: float
    next_month_oi: float
    rollover_percentage: float
    rollover_strength: str  # "STRONG", "MODERATE", "WEAK"
    rollover_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    current_month_price: float
    next_month_price: float
    calendar_spread: float
    calendar_spread_percentage: float


@dataclass
class FuturesPositioning:
    """Participant positioning in futures"""
    fii_long: float
    fii_short: float
    fii_net: float
    dii_long: float
    dii_short: float
    dii_net: float
    pro_long: float
    pro_short: float
    pro_net: float
    client_long: float
    client_short: float
    client_net: float
    total_oi: float
    fii_percentage: float
    dominant_participant: str
    positioning_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"


@dataclass
class FuturesSignal:
    """Futures trading signal"""
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward_ratio: float
    rationale: List[str]
    validity: str  # "INTRADAY", "SWING", "POSITIONAL"
    conviction_level: str  # "LOW", "MEDIUM", "HIGH", "VERY HIGH"


class NiftyFuturesAnalyzer:
    """
    Comprehensive NIFTY Futures Analysis

    Analyzes futures market with:
    - Spot vs Futures relationship
    - Cost of carry
    - Rollover dynamics
    - Participant positioning
    - Technical analysis
    - Options correlation
    - Trading signals
    """

    def __init__(self):
        """Initialize NIFTY Futures Analyzer"""
        self.risk_free_rate = 6.5  # Current RBI repo rate (%)

    def analyze_futures_spot_basis(
        self,
        spot_price: float,
        futures_price: float,
        days_to_expiry: int,
        futures_data: Optional[Dict] = None
    ) -> FuturesData:
        """
        Analyze futures vs spot relationship

        Args:
            spot_price: NIFTY spot price
            futures_price: NIFTY futures price
            days_to_expiry: Days until futures expiry
            futures_data: Additional futures market data

        Returns:
            FuturesData with complete analysis
        """
        # Calculate basis
        basis = futures_price - spot_price
        basis_percentage = (basis / spot_price) * 100

        # Determine premium/discount
        premium_discount = "PREMIUM" if basis > 0 else "DISCOUNT"

        # Extract additional data if provided
        volume = futures_data.get('volume', 0) if futures_data else 0
        oi = futures_data.get('openInterest', 0) if futures_data else 0
        change_in_oi = futures_data.get('changeinOpenInterest', 0) if futures_data else 0

        oi_pct_change = (change_in_oi / oi * 100) if oi > 0 else 0

        bid_price = futures_data.get('bidprice', futures_price - 0.5) if futures_data else futures_price - 0.5
        ask_price = futures_data.get('askprice', futures_price + 0.5) if futures_data else futures_price + 0.5
        bid_qty = futures_data.get('bidQty', 0) if futures_data else 0
        ask_qty = futures_data.get('askQty', 0) if futures_data else 0
        spread = ask_price - bid_price

        ltp = futures_data.get('lastPrice', futures_price) if futures_data else futures_price
        change = futures_data.get('change', 0) if futures_data else 0
        change_pct = futures_data.get('pChange', 0) if futures_data else 0

        expiry_date = datetime.now() + timedelta(days=days_to_expiry)

        return FuturesData(
            symbol="NIFTY",
            expiry_date=expiry_date,
            spot_price=spot_price,
            futures_price=futures_price,
            basis=basis,
            basis_percentage=basis_percentage,
            premium_discount=premium_discount,
            days_to_expiry=days_to_expiry,
            volume=volume,
            open_interest=oi,
            change_in_oi=change_in_oi,
            oi_percentage_change=oi_pct_change,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            spread=spread,
            ltp=ltp,
            change=change,
            change_percentage=change_pct
        )

    def calculate_cost_of_carry(
        self,
        spot_price: float,
        futures_price: float,
        days_to_expiry: int,
        dividend_yield: float = 1.5  # Average NIFTY dividend yield %
    ) -> CostOfCarryResult:
        """
        Calculate cost of carry and detect arbitrage opportunities

        Cost of Carry Formula:
        F = S * e^((r - d) * t)

        Where:
        F = Theoretical futures price
        S = Spot price
        r = Risk-free rate
        d = Dividend yield
        t = Time to expiry (in years)
        """
        # Time to expiry in years
        t = days_to_expiry / 365.0

        # Cost of carry rate (annualized)
        cost_of_carry_rate = self.risk_free_rate - dividend_yield

        # Theoretical futures price
        theoretical_futures = spot_price * np.exp((cost_of_carry_rate / 100) * t)

        # Mispricing
        mispricing = futures_price - theoretical_futures
        mispricing_pct = (mispricing / theoretical_futures) * 100

        # Arbitrage opportunity detection
        # Typically, mispricing > 0.5% indicates arbitrage opportunity
        arbitrage_threshold = 0.5
        arbitrage = abs(mispricing_pct) > arbitrage_threshold

        if arbitrage:
            if mispricing > 0:
                arb_type = "CASH_FUTURES_ARBITRAGE"  # Futures overpriced
                # Action: Sell futures, Buy spot
            else:
                arb_type = "REVERSE_CASH_FUTURES"  # Futures underpriced
                # Action: Buy futures, Sell spot
        else:
            arb_type = "NONE"

        # Expected arbitrage return (annualized)
        expected_return = abs(mispricing_pct) * (365 / days_to_expiry) if days_to_expiry > 0 else 0

        return CostOfCarryResult(
            theoretical_futures_price=theoretical_futures,
            actual_futures_price=futures_price,
            cost_of_carry_rate=cost_of_carry_rate,
            mispricing=mispricing,
            mispricing_percentage=mispricing_pct,
            arbitrage_opportunity=arbitrage,
            arbitrage_type=arb_type,
            expected_return=expected_return
        )

    def analyze_rollover(
        self,
        current_month_data: Dict,
        next_month_data: Dict,
        days_to_current_expiry: int
    ) -> RolloverAnalysis:
        """
        Analyze rollover from current month to next month

        Args:
            current_month_data: Current month futures data
            next_month_data: Next month futures data
            days_to_current_expiry: Days until current month expiry

        Returns:
            RolloverAnalysis with rollover metrics
        """
        current_oi = current_month_data.get('openInterest', 0)
        next_oi = next_month_data.get('openInterest', 0)

        current_price = current_month_data.get('lastPrice', 0)
        next_price = next_month_data.get('lastPrice', 0)

        # Rollover percentage (how much OI has moved to next month)
        total_oi = current_oi + next_oi
        rollover_pct = (next_oi / total_oi * 100) if total_oi > 0 else 0

        # Rollover strength
        if rollover_pct > 60:
            rollover_strength = "STRONG"
        elif rollover_pct > 40:
            rollover_strength = "MODERATE"
        else:
            rollover_strength = "WEAK"

        # Calendar spread (Next month - Current month)
        calendar_spread = next_price - current_price
        calendar_spread_pct = (calendar_spread / current_price * 100) if current_price > 0 else 0

        # Rollover bias
        # Positive calendar spread + strong rollover = Bullish
        # Negative calendar spread + strong rollover = Bearish
        if rollover_pct > 50:
            if calendar_spread_pct > 0.2:
                rollover_bias = "BULLISH"
            elif calendar_spread_pct < -0.2:
                rollover_bias = "BEARISH"
            else:
                rollover_bias = "NEUTRAL"
        else:
            rollover_bias = "NEUTRAL"

        current_expiry = datetime.now() + timedelta(days=days_to_current_expiry)
        next_expiry = current_expiry + timedelta(days=28)  # Approximate next expiry

        return RolloverAnalysis(
            current_month_expiry=current_expiry,
            next_month_expiry=next_expiry,
            current_month_oi=current_oi,
            next_month_oi=next_oi,
            rollover_percentage=rollover_pct,
            rollover_strength=rollover_strength,
            rollover_bias=rollover_bias,
            current_month_price=current_price,
            next_month_price=next_price,
            calendar_spread=calendar_spread,
            calendar_spread_percentage=calendar_spread_pct
        )

    def analyze_participant_positioning(
        self,
        participant_data: Dict
    ) -> FuturesPositioning:
        """
        Analyze participant-wise positioning in futures

        Args:
            participant_data: Participant-wise OI data

        Returns:
            FuturesPositioning with detailed positioning
        """
        # Extract participant data
        fii_long = participant_data.get('fii', {}).get('long', 0)
        fii_short = participant_data.get('fii', {}).get('short', 0)
        fii_net = fii_long - fii_short

        dii_long = participant_data.get('dii', {}).get('long', 0)
        dii_short = participant_data.get('dii', {}).get('short', 0)
        dii_net = dii_long - dii_short

        pro_long = participant_data.get('pro', {}).get('long', 0)
        pro_short = participant_data.get('pro', {}).get('short', 0)
        pro_net = pro_long - pro_short

        client_long = participant_data.get('client', {}).get('long', 0)
        client_short = participant_data.get('client', {}).get('short', 0)
        client_net = client_long - client_short

        total_oi = fii_long + dii_long + pro_long + client_long

        # FII percentage of total OI
        fii_pct = (abs(fii_net) / total_oi * 100) if total_oi > 0 else 0

        # Dominant participant (highest net position)
        participants = {
            'FII': abs(fii_net),
            'DII': abs(dii_net),
            'PRO': abs(pro_net),
            'CLIENT': abs(client_net)
        }
        dominant = max(participants.items(), key=lambda x: x[1])[0]

        # Positioning bias (based on FII + DII net)
        institutional_net = fii_net + dii_net
        if institutional_net > total_oi * 0.05:  # Net long > 5% of total OI
            positioning_bias = "BULLISH"
        elif institutional_net < -total_oi * 0.05:
            positioning_bias = "BEARISH"
        else:
            positioning_bias = "NEUTRAL"

        return FuturesPositioning(
            fii_long=fii_long,
            fii_short=fii_short,
            fii_net=fii_net,
            dii_long=dii_long,
            dii_short=dii_short,
            dii_net=dii_net,
            pro_long=pro_long,
            pro_short=pro_short,
            pro_net=pro_net,
            client_long=client_long,
            client_short=client_short,
            client_net=client_net,
            total_oi=total_oi,
            fii_percentage=fii_pct,
            dominant_participant=dominant,
            positioning_bias=positioning_bias
        )

    def generate_futures_signal(
        self,
        futures_data: FuturesData,
        cost_of_carry: CostOfCarryResult,
        rollover: RolloverAnalysis,
        positioning: FuturesPositioning,
        technical_indicators: Dict,
        option_chain_bias: str = "NEUTRAL"
    ) -> FuturesSignal:
        """
        Generate comprehensive futures trading signal

        Args:
            futures_data: Futures market data
            cost_of_carry: Cost of carry analysis
            rollover: Rollover analysis
            positioning: Participant positioning
            technical_indicators: Technical analysis indicators
            option_chain_bias: Bias from option chain analysis

        Returns:
            FuturesSignal with trading recommendation
        """
        rationale = []
        bullish_factors = 0
        bearish_factors = 0

        # Factor 1: Basis analysis
        if futures_data.premium_discount == "PREMIUM":
            if futures_data.basis_percentage > 0.3:
                bearish_factors += 1
                rationale.append(f"High premium ({futures_data.basis_percentage:.2f}%) suggests overbought")
            elif futures_data.basis_percentage > 0:
                bullish_factors += 0.5
                rationale.append(f"Moderate premium ({futures_data.basis_percentage:.2f}%) - normal bullishness")
        else:
            if futures_data.basis_percentage < -0.3:
                bullish_factors += 1
                rationale.append(f"Discount ({futures_data.basis_percentage:.2f}%) suggests oversold")

        # Factor 2: OI change
        if futures_data.oi_percentage_change > 10:
            if futures_data.change_percentage > 0:
                bullish_factors += 1
                rationale.append(f"Long buildup: Price ↑ {futures_data.change_percentage:.2f}%, OI ↑ {futures_data.oi_percentage_change:.2f}%")
            else:
                bearish_factors += 1
                rationale.append(f"Short buildup: Price ↓ {futures_data.change_percentage:.2f}%, OI ↑ {futures_data.oi_percentage_change:.2f}%")
        elif futures_data.oi_percentage_change < -10:
            if futures_data.change_percentage > 0:
                bearish_factors += 0.5
                rationale.append(f"Short covering: Price ↑, OI ↓ - may reverse")
            else:
                bullish_factors += 0.5
                rationale.append(f"Long unwinding: Price ↓, OI ↓ - may reverse")

        # Factor 3: Cost of carry arbitrage
        if cost_of_carry.arbitrage_opportunity:
            if cost_of_carry.mispricing > 0:
                bearish_factors += 0.5
                rationale.append(f"Futures overpriced by {cost_of_carry.mispricing_percentage:.2f}% - arbitrage sell pressure")
            else:
                bullish_factors += 0.5
                rationale.append(f"Futures underpriced by {abs(cost_of_carry.mispricing_percentage):.2f}% - arbitrage buy pressure")

        # Factor 4: Rollover bias
        if rollover.rollover_strength in ["STRONG", "MODERATE"]:
            if rollover.rollover_bias == "BULLISH":
                bullish_factors += 1
                rationale.append(f"Bullish rollover: {rollover.rollover_percentage:.1f}% rolled, +ve calendar spread")
            elif rollover.rollover_bias == "BEARISH":
                bearish_factors += 1
                rationale.append(f"Bearish rollover: {rollover.rollover_percentage:.1f}% rolled, -ve calendar spread")

        # Factor 5: Participant positioning
        if positioning.positioning_bias == "BULLISH":
            bullish_factors += 1
            rationale.append(f"Institutional net long: FII {positioning.fii_net:,.0f}, DII {positioning.dii_net:,.0f}")
        elif positioning.positioning_bias == "BEARISH":
            bearish_factors += 1
            rationale.append(f"Institutional net short: FII {positioning.fii_net:,.0f}, DII {positioning.dii_net:,.0f}")

        # Factor 6: Technical indicators
        if technical_indicators:
            rsi = technical_indicators.get('rsi', 50)
            if rsi > 70:
                bearish_factors += 0.5
                rationale.append(f"RSI overbought: {rsi:.1f}")
            elif rsi < 30:
                bullish_factors += 0.5
                rationale.append(f"RSI oversold: {rsi:.1f}")

            trend = technical_indicators.get('trend', 'NEUTRAL')
            if trend == 'BULLISH':
                bullish_factors += 1
                rationale.append("Technical trend: Bullish")
            elif trend == 'BEARISH':
                bearish_factors += 1
                rationale.append("Technical trend: Bearish")

        # Factor 7: Option chain correlation
        if option_chain_bias == "BULLISH":
            bullish_factors += 0.5
            rationale.append("Option chain shows bullish bias")
        elif option_chain_bias == "BEARISH":
            bearish_factors += 0.5
            rationale.append("Option chain shows bearish bias")

        # Generate signal
        net_score = bullish_factors - bearish_factors

        if net_score >= 2:
            signal_type = "BUY"
            conviction = "VERY HIGH" if net_score >= 3 else "HIGH"
        elif net_score >= 1:
            signal_type = "BUY"
            conviction = "MEDIUM"
        elif net_score <= -2:
            signal_type = "SELL"
            conviction = "VERY HIGH" if net_score <= -3 else "HIGH"
        elif net_score <= -1:
            signal_type = "SELL"
            conviction = "MEDIUM"
        else:
            signal_type = "HOLD"
            conviction = "LOW"

        # Calculate confidence (0-100)
        max_factors = 7
        confidence = min(100, abs(net_score) / max_factors * 100)

        # Calculate entry, SL, targets based on signal
        entry = futures_data.futures_price
        atr = technical_indicators.get('atr', entry * 0.01)  # Default 1% ATR

        if signal_type == "BUY":
            stop_loss = entry - (1.5 * atr)
            target_1 = entry + (1.5 * atr)
            target_2 = entry + (2.5 * atr)
            target_3 = entry + (4 * atr)
        elif signal_type == "SELL":
            stop_loss = entry + (1.5 * atr)
            target_1 = entry - (1.5 * atr)
            target_2 = entry - (2.5 * atr)
            target_3 = entry - (4 * atr)
        else:  # HOLD
            stop_loss = entry
            target_1 = entry
            target_2 = entry
            target_3 = entry

        risk = abs(entry - stop_loss)
        reward = abs(target_2 - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        # Determine validity
        if futures_data.days_to_expiry <= 5:
            validity = "INTRADAY"
        elif futures_data.days_to_expiry <= 15:
            validity = "SWING"
        else:
            validity = "POSITIONAL"

        return FuturesSignal(
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            risk_reward_ratio=rr_ratio,
            rationale=rationale,
            validity=validity,
            conviction_level=conviction
        )

    def calculate_futures_greeks_proxy(
        self,
        futures_price: float,
        days_to_expiry: int,
        implied_volatility: float = 15.0
    ) -> Dict[str, float]:
        """
        Calculate proxy Greeks for futures positions

        Note: Futures don't have Greeks like options, but we can estimate
        equivalent metrics for risk management
        """
        # Delta: Futures always have delta of 1 (or -1 for short)
        delta = 1.0

        # Gamma: Rate of change of delta (0 for futures)
        gamma = 0.0

        # Theta: Time decay (futures have carrying cost, not time decay)
        # Approximate as cost of carry per day
        theta = -(self.risk_free_rate / 365) * futures_price / 100

        # Vega: Sensitivity to volatility (0 for futures)
        vega = 0.0

        # DV01: Dollar value of 1 point move
        dv01 = 75  # NIFTY lot size = 75 (updated as of recent)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'dv01': dv01,
            'lot_size': 75
        }
