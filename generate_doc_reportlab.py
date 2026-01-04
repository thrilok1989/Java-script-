#!/usr/bin/env python3
"""
Generate PDF Documentation using ReportLab
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

def generate_pdf():
    output_path = '/home/user/Java-script-/NIFTY_Option_Screener_Documentation.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=20)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=16, alignment=TA_CENTER, spaceAfter=10)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2980b9'), spaceBefore=15, spaceAfter=10)
    subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'], fontSize=11, textColor=colors.HexColor('#27ae60'), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=5)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=9, leftIndent=20, spaceAfter=3)

    story = []

    # Title Page
    story.append(Spacer(1, 100))
    story.append(Paragraph("NIFTY Option Screener", title_style))
    story.append(Paragraph("Complete Documentation", subtitle_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Data Sources & ML Analysis Features", styles['Heading3']))
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    story.append(Spacer(1, 30))

    # Summary Box
    summary_data = [['SUMMARY'],
                    ['7 Data Sources | 25+ ML Features | 23 Technical Indicators | 8 Alert Types']]
    summary_table = Table(summary_data, colWidths=[450])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # PART 1: DATA SOURCES
    story.append(Paragraph("PART 1: DATA SOURCES (7 Total)", heading_style))

    data_sources = [
        ['#', 'Source', 'Data Provided', 'Method'],
        ['1', 'Dhan API', 'OHLC, Spot Price, Option Chain, Market Depth', 'REST API'],
        ['2', 'Yahoo Finance', 'Fallback OHLC, Global indices data', 'yfinance'],
        ['3', 'NSE Participant', 'FII/DII/PRO/CLIENT positions', 'Web scraping'],
        ['4', 'Option Chain', 'OI, Volume, Greeks, IV, Premiums', 'Dhan API'],
        ['5', 'Market Depth', '5-level bid/ask, Order quantities', 'Dhan API'],
        ['6', 'Perplexity AI', 'Real-time market insights', 'AI API'],
        ['7', 'NewsData', 'Market news and events', 'NewsData API'],
    ]

    t = Table(data_sources, colWidths=[25, 80, 200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("1.1 Dhan API Data Fetcher", subheading_style))
    story.append(Paragraph("File: dhan_data_fetcher.py", body_style))
    for item in ['OHLC Data: Real-time quotes for Nifty, Sensex, BankNifty, FinnIfty, MidcpNifty',
                 'Intraday Charts: 1, 5, 15, 25, 60-minute candle data',
                 'Option Chain: Full chain with Greeks (Delta, Gamma, Theta, Vega)',
                 'Rate Limits: Quote=1/sec, Data=5/sec, Option Chain=1/3sec']:
        story.append(Paragraph(f"* {item}", bullet_style))

    story.append(Paragraph("1.2 FII/DII Participant Data", subheading_style))
    story.append(Paragraph("File: src/fii_dii_fetcher.py", body_style))
    for item in ['FII (Foreign Institutional Investors) positions',
                 'DII (Domestic Institutional Investors) positions',
                 'PRO (Proprietary Traders) and CLIENT (Retail) positions',
                 'Index Futures/Options Long/Short, Net OI changes']:
        story.append(Paragraph(f"* {item}", bullet_style))

    story.append(PageBreak())

    # PART 2: ML/ANALYSIS FEATURES
    story.append(Paragraph("PART 2: ML/ANALYSIS FEATURES (25+ Systems)", heading_style))

    story.append(Paragraph("A. Market Regime Detection", subheading_style))
    regime_data = [
        ['Feature', 'Algorithm', 'Output'],
        ['ML Market Regime', 'XGBoost-style feature engineering', 'Trending Up/Down, Range Bound, Volatile'],
        ['Volatility Regime', 'VIX percentile + ATR analysis', 'Low/Normal/High/Extreme volatility'],
    ]
    t = Table(regime_data, colWidths=[120, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("B. Institutional Detection (Diamond Level)", subheading_style))
    inst_data = [
        ['Feature', 'Detection Method', 'Output'],
        ['SL Hunt Detector', '4-layer: OI, Volume, Clusters, Time', 'Hunt probability 0-100%'],
        ['Block Trade Detector', 'Volume spike, OI jump, Sweep patterns', 'Accumulating/Distributing'],
        ['Institutional vs Retail', 'Volume signatures, OI patterns', 'Smart money detected Y/N'],
        ['Black Order Detector', 'Hidden/Iceberg order detection', 'Activity level'],
    ]
    t = Table(inst_data, colWidths=[110, 160, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("C. Option Greeks Analysis", subheading_style))
    greeks_data = [
        ['Feature', 'Algorithm', 'Output'],
        ['Gamma Wall Flip', 'Black-Scholes GEX calculation', 'Positive/Negative regime, flip signals'],
        ['OI Trap Detection', 'OI change pattern analysis', 'Trap type, probability, direction'],
        ['ATM Bias Analyzer', '12-dimension composite analysis', 'Bullish/Bearish/Neutral bias'],
    ]
    t = Table(greeks_data, colWidths=[120, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("D. Order Flow Analysis", subheading_style))
    flow_data = [
        ['Feature', 'Method', 'Output'],
        ['Cumulative Delta (CVD)', 'Delta accumulation + divergence', 'Bullish/Bearish, divergence alerts'],
        ['Market Depth Analysis', 'Bid/Ask imbalance, spoof detection', 'Liquidity profile, hidden orders'],
    ]
    t = Table(flow_data, colWidths=[130, 160, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("E. XGBoost ML System", subheading_style))
    story.append(Paragraph("File: src/xgboost_ml_analyzer.py", body_style))
    xgb_data = [
        ['Component', 'Function', 'Details'],
        ['XGBoost Analyzer', '70+ features input', 'BUY/SELL/HOLD with probability'],
        ['ML Data Collector', 'Records trading signals', 'Tracks outcomes at 5/15/30/60min'],
        ['ML Real Trainer', 'Trains on actual results', 'Cross-validation, feature importance'],
        ['ML Backtester', 'Validates strategy', 'Win rate, profit factor, drawdown'],
    ]
    t = Table(xgb_data, colWidths=[110, 150, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)

    story.append(PageBreak())

    # PART 3: TECHNICAL INDICATORS
    story.append(Paragraph("PART 3: TECHNICAL INDICATORS (23 Total)", heading_style))

    story.append(Paragraph("Bias Analysis Pro (13 Indicators)", subheading_style))
    story.append(Paragraph("File: bias_analysis.py", body_style))
    indicators = ['RSI (Relative Strength Index)', 'MACD (Moving Average Convergence Divergence)',
                  'Stochastic Oscillator', 'Bollinger Bands', 'ADX (Average Directional Index)',
                  'CCI (Commodity Channel Index)', 'Williams %R', 'Volume Rate of Change',
                  'Price Rate of Change', 'Moving Average Crossovers', 'Fibonacci Retracement Levels',
                  'Pivot Points', 'ATR (Average True Range)']
    for ind in indicators:
        story.append(Paragraph(f"* {ind}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Advanced Indicators (10 Systems)", subheading_style))
    story.append(Paragraph("Folder: indicators/", body_style))
    adv_indicators = ['Volume Order Blocks (VOB) - High-volume price levels',
                      'HTF Support/Resistance - Multi-timeframe S/R',
                      'HTF Volume Footprint - Volume distribution at price levels',
                      'Ultimate RSI - Enhanced multi-timeframe RSI',
                      'OM Indicator - Comprehensive order flow',
                      'Liquidity Sentiment Profile - Fair value gaps (FVG)',
                      'Money Flow Profile - Volume-weighted price levels',
                      'DeltaFlow Volume Profile - Delta per price level',
                      'Reversal Probability Zones - Swing pattern analysis',
                      'Advanced Price Action - BOS, CHOCH, Fibonacci']
    for ind in adv_indicators:
        story.append(Paragraph(f"* {ind}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("ATM Bias Analyzer (12 Metrics)", subheading_style))
    atm_metrics = ['1. OI Bias (Call vs Put OI)', '2. Change in OI Bias',
                   '3. Volume Bias (Call vs Put Volume)', '4. Delta Bias (Net Delta Position)',
                   '5. Gamma Bias (Net Gamma Position)', '6. Premium Bias',
                   '7. IV Bias (Call IV vs Put IV)', '8. Delta Exposure Bias',
                   '9. Gamma Exposure Bias', '10. IV Skew Bias',
                   '11. OI Change Acceleration Bias', '12. Volume x IV x OI Combined Bias']
    for m in atm_metrics:
        story.append(Paragraph(f"* {m}", bullet_style))

    story.append(PageBreak())

    # PART 4: SMART ALERT SYSTEM
    story.append(Paragraph("PART 4: SMART ALERT SYSTEM (8 Alert Types)", heading_style))
    story.append(Paragraph("File: src/smart_alert_system.py", body_style))

    alert_data = [
        ['Alert Type', 'Trigger Condition', 'Priority'],
        ['Price Alerts', 'Spot crosses upper/lower bounds, S/R proximity', 'HIGH'],
        ['OI Alerts', 'Large OI buildups or breakdowns detected', 'MEDIUM'],
        ['Gamma Flip', 'GEX flips positive to negative or vice versa', 'HIGH'],
        ['CVD Divergence', 'Order flow diverges from price action', 'MEDIUM'],
        ['Block Trade', 'Large institutional trade detected', 'HIGH'],
        ['SL Hunt', 'Stop-loss hunt trap identified', 'HIGH'],
        ['Expiry Day', 'Special conditions on expiry day', 'MEDIUM'],
        ['FII/DII Change', 'Significant institutional position changes', 'LOW'],
    ]
    t = Table(alert_data, colWidths=[100, 230, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Features: Telegram integration, 5-minute cooldown, duplicate prevention, priority notifications", body_style))

    story.append(Spacer(1, 20))

    # PART 5: UNIFIED ML SIGNAL
    story.append(Paragraph("PART 5: UNIFIED ML TRADING SIGNAL", heading_style))
    story.append(Paragraph("File: src/unified_ml_signal.py", body_style))
    story.append(Paragraph("Combines ALL 12 modules into a single trading signal:", body_style))

    unified = ['1. ML Market Regime Detection', '2. XGBoost Prediction (BUY/SELL/HOLD)',
               '3. Volatility Regime Analysis', '4. OI Trap Detection',
               '5. CVD Delta Imbalance', '6. Liquidity Gravity Analysis',
               '7. Institutional vs Retail Detection', '8. Expiry Day Killer (False Breakout Filter)',
               '9. SL Hunt Detector', '10. Block Trade Detector',
               '11. Gamma Wall Flip', '12. FII/DII Detection']
    for item in unified:
        story.append(Paragraph(f"* {item}", bullet_style))

    story.append(Spacer(1, 15))
    output_box = [['OUTPUT: Single BUY/SELL/HOLD signal with confidence percentage']]
    t = Table(output_box, colWidths=[400])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t)

    story.append(PageBreak())

    # FINAL SUMMARY
    story.append(Paragraph("SUMMARY", heading_style))

    summary_data = [
        ['Category', 'Count', 'Details'],
        ['Data Sources', '7', 'Dhan API, Yahoo Finance, NSE, Perplexity AI, NewsData'],
        ['ML/Analysis Features', '25+', 'Regime detection, institutional tracking, order flow'],
        ['Technical Indicators', '23', 'Bias Analysis Pro (13) + Advanced Indicators (10)'],
        ['Alert Types', '8', 'Price, OI, Gamma, CVD, Block, SL Hunt, Expiry, FII/DII'],
        ['ATM Bias Metrics', '12', 'OI, Volume, Delta, Gamma, IV, Premium, Skew'],
        ['XGBoost Features', '70+', 'All modules combined for ML prediction'],
    ]
    t = Table(summary_data, colWidths=[120, 50, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(t)

    story.append(Spacer(1, 30))
    story.append(Paragraph("All components work together as a unified trading signal system",
                          ParagraphStyle('Final', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor('#27ae60'))))
    story.append(Paragraph("for NIFTY 50 options trading with institutional-level analysis!",
                          ParagraphStyle('Final', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor('#27ae60'))))

    doc.build(story)
    return output_path

if __name__ == '__main__':
    path = generate_pdf()
    print(f"PDF generated successfully: {path}")
