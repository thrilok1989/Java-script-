"""
ML Backtester - Validate ML Strategy on Historical Data
========================================================
Tests ML signals against historical price data to measure real performance

Features:
1. Simulates trades on historical data
2. Calculates win rate, profit factor, drawdown
3. Validates ML model before live trading
4. Generates performance reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    entry_time: datetime
    exit_time: datetime
    signal_type: str  # BUY or SELL
    entry_price: float
    exit_price: float
    pnl_points: float
    pnl_percent: float
    outcome: str  # WIN, LOSS, BREAKEVEN
    confidence: float
    holding_period_minutes: int


@dataclass
class BacktestResult:
    """Complete backtest result"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl_points: float
    total_pnl_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float
    avg_holding_minutes: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade]
    equity_curve: List[float]
    summary: str


class MLBacktester:
    """
    Backtests ML trading signals on historical data
    """

    def __init__(self):
        self.min_confidence = 60  # Minimum confidence to take trade
        self.target_percent = 0.5  # 0.5% target
        self.stoploss_percent = 0.3  # 0.3% stop loss
        self.max_holding_minutes = 60  # Max 1 hour hold

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        price_df: pd.DataFrame,
        initial_capital: float = 100000
    ) -> BacktestResult:
        """
        Run backtest on historical signals

        Args:
            signals_df: DataFrame with signals (timestamp, signal_type, confidence, entry_price)
            price_df: DataFrame with OHLCV data
            initial_capital: Starting capital

        Returns:
            BacktestResult with all metrics
        """
        trades = []
        equity = initial_capital
        equity_curve = [equity]
        peak_equity = equity

        for idx, signal in signals_df.iterrows():
            try:
                signal_time = pd.to_datetime(signal.get('timestamp'))
                signal_type = signal.get('signal_type', 'HOLD')
                confidence = signal.get('confidence', 0)
                entry_price = signal.get('entry_price', 0)

                # Skip low confidence or HOLD signals
                if confidence < self.min_confidence or signal_type == 'HOLD':
                    continue

                if entry_price <= 0:
                    continue

                # Find exit price (simulate trade)
                exit_price, exit_time, holding_minutes = self._simulate_trade(
                    price_df, signal_time, entry_price, signal_type
                )

                if exit_price <= 0:
                    continue

                # Calculate P&L
                if signal_type == 'BUY':
                    pnl_points = exit_price - entry_price
                else:  # SELL
                    pnl_points = entry_price - exit_price

                pnl_percent = pnl_points / entry_price * 100

                # Determine outcome
                if pnl_percent > 0.1:
                    outcome = 'WIN'
                elif pnl_percent < -0.1:
                    outcome = 'LOSS'
                else:
                    outcome = 'BREAKEVEN'

                trade = BacktestTrade(
                    entry_time=signal_time,
                    exit_time=exit_time,
                    signal_type=signal_type,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_points=pnl_points,
                    pnl_percent=pnl_percent,
                    outcome=outcome,
                    confidence=confidence,
                    holding_period_minutes=holding_minutes
                )
                trades.append(trade)

                # Update equity
                equity += pnl_points * 25  # NIFTY lot size
                equity_curve.append(equity)

                # Track peak for drawdown
                if equity > peak_equity:
                    peak_equity = equity

            except Exception as e:
                logger.warning(f"Error processing signal: {e}")
                continue

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, initial_capital, peak_equity)

    def _simulate_trade(
        self,
        price_df: pd.DataFrame,
        entry_time: datetime,
        entry_price: float,
        signal_type: str
    ) -> Tuple[float, datetime, int]:
        """
        Simulate trade execution and find exit

        Returns: (exit_price, exit_time, holding_minutes)
        """
        try:
            # Find price data after entry
            if price_df.index.tz is not None:
                entry_time = entry_time.tz_localize(price_df.index.tz) if entry_time.tzinfo is None else entry_time

            future_prices = price_df[price_df.index >= entry_time]

            if len(future_prices) == 0:
                return 0, entry_time, 0

            target = entry_price * (1 + self.target_percent / 100)
            stoploss = entry_price * (1 - self.stoploss_percent / 100)

            if signal_type == 'SELL':
                target = entry_price * (1 - self.target_percent / 100)
                stoploss = entry_price * (1 + self.stoploss_percent / 100)

            # Simulate bar by bar
            for i, (time, row) in enumerate(future_prices.iterrows()):
                high = row.get('High', row.get('high', entry_price))
                low = row.get('Low', row.get('low', entry_price))
                close = row.get('Close', row.get('close', entry_price))

                holding_minutes = int((time - entry_time).total_seconds() / 60)

                if signal_type == 'BUY':
                    # Check if target hit
                    if high >= target:
                        return target, time, holding_minutes
                    # Check if stoploss hit
                    if low <= stoploss:
                        return stoploss, time, holding_minutes
                else:  # SELL
                    # Check if target hit (price goes down)
                    if low <= target:
                        return target, time, holding_minutes
                    # Check if stoploss hit (price goes up)
                    if high >= stoploss:
                        return stoploss, time, holding_minutes

                # Max holding time reached
                if holding_minutes >= self.max_holding_minutes:
                    return close, time, holding_minutes

            # End of data - exit at last close
            last_time = future_prices.index[-1]
            last_close = future_prices.iloc[-1].get('Close', future_prices.iloc[-1].get('close', entry_price))
            holding = int((last_time - entry_time).total_seconds() / 60)
            return last_close, last_time, holding

        except Exception as e:
            logger.warning(f"Trade simulation error: {e}")
            return 0, entry_time, 0

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        initial_capital: float,
        peak_equity: float
    ) -> BacktestResult:
        """Calculate backtest metrics"""

        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                breakeven_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                total_pnl_points=0,
                total_pnl_percent=0,
                max_drawdown_percent=0,
                sharpe_ratio=0,
                avg_holding_minutes=0,
                best_trade=0,
                worst_trade=0,
                trades=[],
                equity_curve=equity_curve,
                summary="No trades executed"
            )

        # Count trades
        wins = [t for t in trades if t.outcome == 'WIN']
        losses = [t for t in trades if t.outcome == 'LOSS']
        breakevens = [t for t in trades if t.outcome == 'BREAKEVEN']

        winning_trades = len(wins)
        losing_trades = len(losses)
        breakeven_trades = len(breakevens)
        total_trades = len(trades)

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # P&L metrics
        avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_percent for t in losses]) if losses else 0

        total_win = sum(t.pnl_percent for t in wins)
        total_loss = abs(sum(t.pnl_percent for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

        total_pnl_points = sum(t.pnl_points for t in trades)
        total_pnl_percent = sum(t.pnl_percent for t in trades)

        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0

        # Holding time
        avg_holding = np.mean([t.holding_period_minutes for t in trades])

        # Best/Worst
        best_trade = max(t.pnl_percent for t in trades)
        worst_trade = min(t.pnl_percent for t in trades)

        # Summary
        summary = f"""
BACKTEST SUMMARY
================
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
Total P&L: {total_pnl_percent:.2f}%
Max Drawdown: {max_drawdown:.2f}%
Sharpe Ratio: {sharpe:.2f}

Winners: {winning_trades} (Avg: {avg_win:.2f}%)
Losers: {losing_trades} (Avg: {avg_loss:.2f}%)
Best Trade: {best_trade:.2f}%
Worst Trade: {worst_trade:.2f}%
Avg Holding: {avg_holding:.0f} minutes
"""

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_pnl_points=total_pnl_points,
            total_pnl_percent=total_pnl_percent,
            max_drawdown_percent=max_drawdown,
            sharpe_ratio=sharpe,
            avg_holding_minutes=avg_holding,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=trades,
            equity_curve=equity_curve,
            summary=summary
        )

    def backtest_ml_signals(self, price_df: pd.DataFrame) -> BacktestResult:
        """
        Backtest using recorded ML signals

        Args:
            price_df: Historical OHLCV data

        Returns:
            BacktestResult
        """
        from src.ml_data_collector import get_data_collector

        collector = get_data_collector()
        signals_df = pd.DataFrame(collector.recorded_signals)

        if len(signals_df) == 0:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                total_pnl_points=0, total_pnl_percent=0, max_drawdown_percent=0,
                sharpe_ratio=0, avg_holding_minutes=0, best_trade=0, worst_trade=0,
                trades=[], equity_curve=[100000],
                summary="No signals recorded yet"
            )

        return self.run_backtest(signals_df, price_df)


# Singleton
_backtester = None

def get_backtester() -> MLBacktester:
    """Get singleton backtester"""
    global _backtester
    if _backtester is None:
        _backtester = MLBacktester()
    return _backtester


def run_ml_backtest(price_df: pd.DataFrame = None) -> BacktestResult:
    """
    Run backtest on ML signals

    Usage:
        from src.ml_backtester import run_ml_backtest
        import streamlit as st

        result = run_ml_backtest(st.session_state.get('chart_data'))
        print(result.summary)
    """
    backtester = get_backtester()

    if price_df is None:
        import streamlit as st
        price_df = st.session_state.get('chart_data')

    if price_df is None or len(price_df) == 0:
        return BacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            total_pnl_points=0, total_pnl_percent=0, max_drawdown_percent=0,
            sharpe_ratio=0, avg_holding_minutes=0, best_trade=0, worst_trade=0,
            trades=[], equity_curve=[100000],
            summary="No price data available for backtesting"
        )

    return backtester.backtest_ml_signals(price_df)
