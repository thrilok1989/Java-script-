#!/usr/bin/env python3
"""
Generate PDF Documentation for NIFTY Option Screener App
Complete list of Data Sources and ML Analysis Features
"""

from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'NIFTY Option Screener - Complete Documentation', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text, indent=10):
        self.set_font('Helvetica', '', 10)
        self.set_x(indent)
        self.multi_cell(0, 5, f"  * {text}")

    def table_header(self, headers, widths):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(widths[i], 7, header, 1, 0, 'C', True)
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, data, widths, fill=False):
        self.set_font('Helvetica', '', 8)
        if fill:
            self.set_fill_color(236, 240, 241)
        for i, item in enumerate(data):
            self.cell(widths[i], 6, str(item)[:40], 1, 0, 'L', fill)
        self.ln()

def generate_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(40)
    pdf.cell(0, 15, 'NIFTY Option Screener', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 10, 'Complete Documentation', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Data Sources & ML Analysis Features', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(30)

    # Summary Box
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_fill_color(46, 204, 113)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, 'SUMMARY', 0, 1, 'C', True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_fill_color(236, 240, 241)
    pdf.cell(0, 7, '7 Data Sources | 25+ ML Features | 23 Technical Indicators | 8 Alert Types', 0, 1, 'C', True)

    # Page 2: Data Sources
    pdf.add_page()
    pdf.chapter_title('PART 1: DATA SOURCES (7 Total)')

    # Data Sources Table
    pdf.ln(3)
    headers = ['#', 'Source', 'Data Provided', 'Method']
    widths = [10, 35, 90, 55]
    pdf.table_header(headers, widths)

    data_sources = [
        ('1', 'Dhan API', 'OHLC, Spot Price, Option Chain, Market Depth, Expiry List', 'REST API'),
        ('2', 'Yahoo Finance', 'Fallback OHLC, Global indices data', 'yfinance library'),
        ('3', 'NSE Participant', 'FII/DII/PRO/CLIENT positions, Net OI changes', 'Web scraping'),
        ('4', 'Option Chain', 'OI, Volume, Greeks, IV, Premiums per strike', 'Dhan API'),
        ('5', 'Market Depth', '5-level bid/ask, Order quantities, Liquidity', 'Dhan API'),
        ('6', 'Perplexity AI', 'Real-time market insights, Web search', 'AI API'),
        ('7', 'NewsData', 'Market news and events', 'NewsData API'),
    ]

    for i, row in enumerate(data_sources):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(8)
    pdf.section_title('1.1 Dhan API Data Fetcher')
    pdf.body_text('File: dhan_data_fetcher.py')
    pdf.bullet_point('OHLC Data: Real-time quotes for Nifty, Sensex, BankNifty, FinnIfty, MidcpNifty')
    pdf.bullet_point('Intraday Charts: 1, 5, 15, 25, 60-minute candle data')
    pdf.bullet_point('Option Chain: Full chain with Greeks (Delta, Gamma, Theta, Vega)')
    pdf.bullet_point('Rate Limits: Quote=1/sec, Data=5/sec, Option Chain=1/3sec')

    pdf.ln(3)
    pdf.section_title('1.2 Option Chain Analysis Data')
    pdf.body_text('File: NiftyOptionScreener.py')
    pdf.bullet_point('Strike prices, OI (Call & Put), Change in OI')
    pdf.bullet_point('Volume, Premium (LTP), Greeks calculated via Black-Scholes')
    pdf.bullet_point('IV (Implied Volatility), PCR, Max Pain, OI Shift, IV Skew')

    pdf.ln(3)
    pdf.section_title('1.3 FII/DII Participant Data')
    pdf.body_text('File: src/fii_dii_fetcher.py')
    pdf.bullet_point('FII (Foreign Institutional Investors) positions')
    pdf.bullet_point('DII (Domestic Institutional Investors) positions')
    pdf.bullet_point('PRO (Proprietary Traders) and CLIENT (Retail) positions')
    pdf.bullet_point('Index Futures/Options Long/Short, Stock Futures, Net OI changes')

    # Page 3: ML Features Part 1
    pdf.add_page()
    pdf.chapter_title('PART 2: ML/ANALYSIS FEATURES (25+ Systems)')

    pdf.section_title('A. Market Regime Detection')
    pdf.ln(2)
    headers = ['Feature', 'Algorithm', 'Output']
    widths = [50, 70, 70]
    pdf.table_header(headers, widths)
    regime_data = [
        ('ML Market Regime', 'XGBoost-style feature engineering', 'Trending Up/Down, Range Bound, Volatile'),
        ('Volatility Regime', 'VIX percentile + ATR analysis', 'Low/Normal/High/Extreme volatility'),
    ]
    for i, row in enumerate(regime_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(5)
    pdf.section_title('B. Institutional Detection (Diamond Level)')
    pdf.ln(2)
    headers = ['Feature', 'Detection Method', 'Output']
    widths = [45, 75, 70]
    pdf.table_header(headers, widths)
    inst_data = [
        ('SL Hunt Detector', '4-layer: OI, Volume, Clusters, Time', 'Hunt probability 0-100%'),
        ('Block Trade Detector', 'Volume spike, OI jump, Sweep patterns', 'Accumulating/Distributing/Neutral'),
        ('Institutional vs Retail', 'Volume signatures, OI patterns', 'Smart money detected Y/N'),
        ('Black Order Detector', 'Hidden/Iceberg order detection', 'Institutional activity level'),
    ]
    for i, row in enumerate(inst_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(5)
    pdf.section_title('C. Option Greeks Analysis')
    pdf.ln(2)
    headers = ['Feature', 'Algorithm', 'Output']
    widths = [45, 75, 70]
    pdf.table_header(headers, widths)
    greeks_data = [
        ('Gamma Wall Flip', 'Black-Scholes GEX calculation', 'Positive/Negative regime, flip signals'),
        ('OI Trap Detection', 'OI change pattern analysis', 'Trap type, probability, direction'),
        ('ATM Bias Analyzer', '12-dimension composite analysis', 'Bullish/Bearish/Neutral bias score'),
    ]
    for i, row in enumerate(greeks_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(5)
    pdf.section_title('D. Order Flow Analysis')
    pdf.ln(2)
    headers = ['Feature', 'Method', 'Output']
    widths = [50, 70, 70]
    pdf.table_header(headers, widths)
    flow_data = [
        ('Cumulative Delta (CVD)', 'Delta accumulation + divergence', 'Bullish/Bearish, divergence alerts'),
        ('Market Depth Analysis', 'Bid/Ask imbalance, spoof detection', 'Liquidity profile, hidden orders'),
    ]
    for i, row in enumerate(flow_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(5)
    pdf.section_title('E. False Signal Filters')
    pdf.ln(2)
    headers = ['Feature', 'Purpose', 'Output']
    widths = [50, 70, 70]
    pdf.table_header(headers, widths)
    filter_data = [
        ('Expiry Day Killer', 'Filter false breakouts on expiry', 'Risk level, recommendation'),
    ]
    for i, row in enumerate(filter_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    # Page 4: XGBoost and Technical Indicators
    pdf.add_page()
    pdf.section_title('F. XGBoost ML System')
    pdf.body_text('File: src/xgboost_ml_analyzer.py')
    pdf.ln(2)
    headers = ['Component', 'Function', 'Details']
    widths = [50, 70, 70]
    pdf.table_header(headers, widths)
    xgb_data = [
        ('XGBoost Analyzer', '70+ features input', 'BUY/SELL/HOLD with probability'),
        ('ML Data Collector', 'Records trading signals', 'Tracks outcomes at 5/15/30/60min'),
        ('ML Real Trainer', 'Trains on actual results', 'Cross-validation, feature importance'),
        ('ML Backtester', 'Validates strategy', 'Win rate, profit factor, drawdown'),
    ]
    for i, row in enumerate(xgb_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(8)
    pdf.chapter_title('PART 3: TECHNICAL INDICATORS (23 Total)')

    pdf.section_title('Bias Analysis Pro (13 Indicators)')
    pdf.body_text('File: bias_analysis.py')
    indicators_13 = [
        'RSI (Relative Strength Index)',
        'MACD (Moving Average Convergence Divergence)',
        'Stochastic Oscillator',
        'Bollinger Bands',
        'ADX (Average Directional Index)',
        'CCI (Commodity Channel Index)',
        'Williams %R',
        'Volume Rate of Change',
        'Price Rate of Change',
        'Moving Average Crossovers',
        'Fibonacci Retracement Levels',
        'Pivot Points',
        'ATR (Average True Range)'
    ]
    for ind in indicators_13:
        pdf.bullet_point(ind)

    pdf.ln(5)
    pdf.section_title('Advanced Indicators (10 Systems)')
    pdf.body_text('Folder: indicators/')
    adv_indicators = [
        'Volume Order Blocks (VOB) - High-volume price levels',
        'HTF Support/Resistance - Multi-timeframe S/R',
        'HTF Volume Footprint - Volume distribution at price levels',
        'Ultimate RSI - Enhanced multi-timeframe RSI',
        'OM Indicator - Comprehensive order flow',
        'Liquidity Sentiment Profile - Fair value gaps (FVG)',
        'Money Flow Profile - Volume-weighted price levels',
        'DeltaFlow Volume Profile - Delta per price level',
        'Reversal Probability Zones - Swing pattern analysis',
        'Advanced Price Action - BOS, CHOCH, Fibonacci'
    ]
    for ind in adv_indicators:
        pdf.bullet_point(ind)

    # Page 5: ATM Bias and Alerts
    pdf.add_page()
    pdf.section_title('ATM Bias Analyzer (12 Metrics)')
    pdf.body_text('File: NiftyOptionScreener.py')
    atm_metrics = [
        '1. OI Bias (Call vs Put OI)',
        '2. Change in OI Bias',
        '3. Volume Bias (Call vs Put Volume)',
        '4. Delta Bias (Net Delta Position)',
        '5. Gamma Bias (Net Gamma Position)',
        '6. Premium Bias',
        '7. IV Bias (Call IV vs Put IV)',
        '8. Delta Exposure Bias',
        '9. Gamma Exposure Bias',
        '10. IV Skew Bias',
        '11. OI Change Acceleration Bias',
        '12. Volume x IV x OI Combined Bias'
    ]
    for m in atm_metrics:
        pdf.bullet_point(m)

    pdf.ln(8)
    pdf.chapter_title('PART 4: SMART ALERT SYSTEM (8 Alert Types)')
    pdf.body_text('File: src/smart_alert_system.py')
    pdf.ln(2)
    headers = ['Alert Type', 'Trigger Condition', 'Priority']
    widths = [50, 100, 40]
    pdf.table_header(headers, widths)
    alert_data = [
        ('Price Alerts', 'Spot crosses upper/lower bounds, S/R proximity', 'HIGH'),
        ('OI Alerts', 'Large OI buildups or breakdowns detected', 'MEDIUM'),
        ('Gamma Flip', 'GEX flips from positive to negative or vice versa', 'HIGH'),
        ('CVD Divergence', 'Order flow diverges from price action', 'MEDIUM'),
        ('Block Trade', 'Large institutional trade detected', 'HIGH'),
        ('SL Hunt', 'Stop-loss hunt trap identified', 'HIGH'),
        ('Expiry Day', 'Special conditions on expiry day', 'MEDIUM'),
        ('FII/DII Change', 'Significant institutional position changes', 'LOW'),
    ]
    for i, row in enumerate(alert_data):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.body_text('Features: Telegram integration, 5-minute cooldown, duplicate prevention, priority-based notifications')

    # Page 6: Unified Signal and Session State
    pdf.add_page()
    pdf.chapter_title('PART 5: UNIFIED ML TRADING SIGNAL')
    pdf.body_text('File: src/unified_ml_signal.py')
    pdf.body_text('Combines ALL 12 modules into a single trading signal:')
    pdf.ln(2)

    unified_modules = [
        '1. ML Market Regime Detection',
        '2. XGBoost Prediction (BUY/SELL/HOLD)',
        '3. Volatility Regime Analysis',
        '4. OI Trap Detection',
        '5. CVD Delta Imbalance',
        '6. Liquidity Gravity Analysis',
        '7. Institutional vs Retail Detection',
        '8. Expiry Day Killer (False Breakout Filter)',
        '9. SL Hunt Detector',
        '10. Block Trade Detector',
        '11. Gamma Wall Flip',
        '12. FII/DII Detection'
    ]
    for m in unified_modules:
        pdf.bullet_point(m)

    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(46, 204, 113)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, 'OUTPUT: Single BUY/SELL/HOLD signal with confidence percentage', 0, 1, 'C', True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(8)
    pdf.chapter_title('PART 6: SESSION STATE VARIABLES')
    pdf.body_text('Key data stored in Streamlit session_state for real-time access:')
    pdf.ln(2)

    pdf.section_title('Market Data')
    market_vars = ['merged_df', 'chart_data', 'market_depth_data', 'option_chain_data', 'nifty_data', 'sensex_data']
    for v in market_vars:
        pdf.bullet_point(v)

    pdf.ln(3)
    pdf.section_title('Analysis Results')
    analysis_vars = ['bias_analysis_results', 'ml_market_regime', 'volatility_regime', 'oi_trap_result',
                     'cvd_analysis', 'sl_hunt_result', 'block_trade_analysis', 'gamma_analysis', 'fii_dii_data']
    for v in analysis_vars:
        pdf.bullet_point(v)

    pdf.ln(3)
    pdf.section_title('Signal Management')
    signal_vars = ['unified_ml_signal', 'active_vob_signals', 'active_htf_sr_signals', 'signal_manager']
    for v in signal_vars:
        pdf.bullet_point(v)

    # Final Summary Page
    pdf.add_page()
    pdf.chapter_title('SUMMARY')
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)

    summary_items = [
        ('Data Sources', '7', 'Dhan API, Yahoo Finance, NSE, Perplexity AI, NewsData'),
        ('ML/Analysis Features', '25+', 'Regime detection, institutional tracking, order flow'),
        ('Technical Indicators', '23', 'Bias Analysis Pro (13) + Advanced Indicators (10)'),
        ('Alert Types', '8', 'Price, OI, Gamma, CVD, Block, SL Hunt, Expiry, FII/DII'),
        ('ATM Bias Metrics', '12', 'OI, Volume, Delta, Gamma, IV, Premium, Skew'),
        ('XGBoost Features', '70+', 'All modules combined for ML prediction'),
    ]

    headers = ['Category', 'Count', 'Details']
    widths = [50, 20, 120]
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(41, 128, 185)
    for i, header in enumerate(headers):
        pdf.cell(widths[i], 8, header, 1, 0, 'C', True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, row in enumerate(summary_items):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(39, 174, 96)
    pdf.cell(0, 10, 'All components work together as a unified trading signal system', 0, 1, 'C')
    pdf.cell(0, 10, 'for NIFTY 50 options trading with institutional-level analysis!', 0, 1, 'C')

    # Save PDF
    output_path = '/home/user/Java-script-/NIFTY_Option_Screener_Documentation.pdf'
    pdf.output(output_path)
    return output_path

if __name__ == '__main__':
    path = generate_pdf()
    print(f"PDF generated: {path}")
