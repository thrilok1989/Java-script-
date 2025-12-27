# 📱 Complete App Structure: All Tabs, Sub-Tabs & Data

## 🎯 Main Tab Structure

Your app has **9 main tabs** with extensive sub-tabs and data points:

```
Tab 1: 🌟 Overall Market Sentiment
Tab 2: 🎯 Trade Setup
Tab 3: 📊 Active Signals
Tab 4: 📈 Positions
Tab 5: 🎲 Bias Analysis Pro
Tab 6: 📉 Advanced Chart Analysis
Tab 7: 🎯 NIFTY Option Screener v7.0
Tab 8: 🌐 Enhanced Market Data
Tab 9: 🔍 NSE Stock Screener
```

---

## 📊 TAB 1: Overall Market Sentiment

### Main Function: `render_overall_market_sentiment()`

### Data Displayed:

#### 1. **Overall Market Direction**
```
Data Points:
- Overall Sentiment (BULLISH/BEARISH/NEUTRAL)
- Confidence Score (0-100%)
- Color-coded sentiment display
```

#### 2. **Multiple Data Sources Analysis**

**Source 1: NIFTY 50 PCR Analysis**
```
- PCR Value (Put-Call Ratio)
- Bias (BULLISH/BEARISH/NEUTRAL)
- Score (0-100)
```

**Source 2: NIFTY ATM Zone**
```
- ATM Strike analysis
- Zone verdict (Bullish/Bearish/Neutral)
- Strike level details
```

**Source 3: Option Chain Analysis**
```
- CE OI (Call Open Interest)
- PE OI (Put Open Interest)
- OI Imbalance
- Bias determination
- Score (0-100)
```

**Source 4: BankNIFTY PCR**
```
- BankNIFTY PCR value
- Bias
- Score
```

**Source 5: FINNIFTY PCR**
```
- FINNIFTY PCR value
- Bias
- Score
```

**Source 6: India VIX**
```
- Current VIX level
- Change %
- Bias interpretation
- Score
```

**Source 7: FII/DII Data**
```
- FII Net (Index Futures)
- FII Net (Stock Futures)
- DII Net
- Combined bias
- Score
```

**Source 8: NIFTY Futures Premium/Discount**
```
- Spot price
- Futures price
- Premium/Discount value
- Bias
- Score
```

**Source 9: Technical Indicators (Your Bias Analysis)**
```
- 13 Bias Indicators combined
- Overall technical bias
- Score
```

**Source 10: Market Breadth**
```
- Advances vs Declines
- Breadth ratio
- Bias
- Score
```

**Source 11: Sector Rotation**
```
- Leading sectors
- Lagging sectors
- Rotation bias
- Score
```

#### 3. **Confluence Analysis**
```
- Total sources analyzed
- Bullish sources count
- Bearish sources count
- Neutral sources count
- Agreement percentage
- Strongest indicators
```

#### 4. **Risk Assessment**
```
- Risk level (Low/Medium/High)
- Volatility state
- Market uncertainty score
```

#### 5. **Trading Recommendation**
```
- Action (BUY/SELL/HOLD/WAIT)
- Conviction level
- Suggested position size
- Risk-reward assessment
```

---

## 🎯 TAB 2: Trade Setup

### Purpose: Create new trade setups

### Data Input:
```
1. Index Selection
   - NIFTY
   - SENSEX

2. Direction Selection
   - CALL (Bullish)
   - PUT (Bearish)

3. Entry Type
   - Market Price
   - Limit Order

4. Position Sizing
   - Number of lots
   - Capital allocation

5. Risk Parameters
   - Stop Loss %
   - Target %
   - Trailing Stop settings
```

### Data Calculated & Displayed:
```
1. Entry Price
2. Stop Loss Level
3. Target Levels (T1, T2, T3)
4. Risk Amount (₹)
5. Reward Amount (₹)
6. Risk-Reward Ratio
7. Breakeven Level
8. Max Loss Potential
9. Max Profit Potential
10. Position Greeks (if options)
11. Margin Required
```

---

## 📊 TAB 3: Active Signals

### Purpose: Display active trading signals

### Signal Types:

#### 1. **HTF S/R Signals (Higher Timeframe Support/Resistance)**
```
For Each Signal:
- Signal ID
- Timestamp
- Direction (BULLISH/BEARISH)
- Entry Price
- Current Price
- P&L %
- Status (ACTIVE/CLOSED)
- Timeframe (5m/15m/1h/4h/1d)
- S/R Level
- Distance from S/R
- Confidence Score
- Volume confirmation
```

#### 2. **VOB Signals (Volume Order Blocks)**
```
For Each Signal:
- Signal ID
- Timestamp
- Type (BULLISH VOB/BEARISH VOB)
- Entry Price
- Current Price
- P&L %
- Volume strength
- Block validity
- Support/Resistance level
```

#### 3. **Bias Analysis Signals**
```
For Each Signal:
- Combined bias score
- Active bias indicators (from 13)
- Conviction level
- Entry recommendation
```

#### 4. **Signal Management**
```
- Total active signals
- Winning signals
- Losing signals
- Overall P&L
- Win rate %
- Best performing signal
- Worst performing signal
```

---

## 📈 TAB 4: Positions

### Purpose: Track active and closed positions

### Data Displayed:

#### 1. **Active Positions**
```
For Each Position:
- Position ID
- Symbol (NIFTY/SENSEX)
- Direction (CALL/PUT/LONG/SHORT)
- Entry Price
- Current Price
- Quantity/Lots
- Entry Time
- Duration
- Unrealized P&L (₹)
- Unrealized P&L (%)
- Stop Loss Level
- Target Level
- Distance to SL
- Distance to Target
- Greeks (Delta, Gamma, Theta, Vega)
- Break-even price
- Trailing stop status
```

#### 2. **Position Analytics**
```
- Total positions
- Total capital deployed
- Total unrealized P&L
- Largest position
- Best performing position
- Worst performing position
- Average P&L per position
```

#### 3. **Closed Positions History**
```
For Each Closed Position:
- Position ID
- Symbol
- Direction
- Entry Price
- Exit Price
- Entry Time
- Exit Time
- Duration
- Realized P&L (₹)
- Realized P&L (%)
- Exit Reason (Target hit/SL hit/Manual/Trailing stop)
- Max Favorable Excursion
- Max Adverse Excursion
```

#### 4. **Performance Statistics**
```
- Total trades
- Winning trades
- Losing trades
- Win rate %
- Total P&L
- Average win
- Average loss
- Profit factor
- Largest win
- Largest loss
- Average holding time
- Best trading day
- Worst trading day
```

---

## 🎲 TAB 5: Bias Analysis Pro

### Purpose: Analyze 13 proprietary bias indicators

### Sub-Structure:
```
Main Dashboard → Individual Indicator Details
```

### 13 Bias Indicators:

#### **1. OI Bias (Open Interest)**
```
Data:
- Total CE OI
- Total PE OI
- OI Ratio
- Bias Score (-100 to +100)
- Direction (BULLISH/BEARISH/NEUTRAL)
- Strength (WEAK/MODERATE/STRONG)
- Chart: OI distribution
```

#### **2. ChgOI Bias (Change in Open Interest)**
```
Data:
- CE OI Change
- PE OI Change
- ChgOI Ratio
- Bias Score
- Direction
- Fresh positions vs unwinding
- Chart: OI change over time
```

#### **3. Volume Bias**
```
Data:
- CE Volume
- PE Volume
- Volume Ratio
- Bias Score
- Direction
- Volume concentration
- Chart: Volume comparison
```

#### **4. Delta Bias**
```
Data:
- Total Delta (calls - puts)
- Delta Imbalance
- Bias Score
- Direction
- Delta concentration levels
- Chart: Delta distribution
```

#### **5. IV Bias (Implied Volatility)**
```
Data:
- Average CE IV
- Average PE IV
- IV Skew
- Bias Score
- Direction
- IV percentile
- Chart: IV curve
```

#### **6. ATM IV Bias**
```
Data:
- ATM Call IV
- ATM Put IV
- ATM IV Spread
- Bias Score
- Direction
- IV vs historical
- Chart: ATM IV history
```

#### **7. PCR Bias (Put-Call Ratio)**
```
Data:
- PCR (OI based)
- PCR (Volume based)
- PCR interpretation
- Bias Score
- Direction
- PCR percentile
- Historical comparison
- Chart: PCR trend
```

#### **8. Buildup Bias**
```
Data:
- Long buildup detection
- Short buildup detection
- Buildup strength
- Bias Score
- Direction
- Buildup zones
- Chart: Buildup analysis
```

#### **9. Unwinding Bias**
```
Data:
- Long unwinding detection
- Short unwinding detection
- Unwinding strength
- Bias Score
- Direction
- Unwinding zones
- Chart: Unwinding analysis
```

#### **10. Max Pain Bias**
```
Data:
- Max Pain level
- Current Price
- Distance to Max Pain
- Bias Score
- Direction
- Max Pain history
- Gravitational pull strength
- Chart: Max Pain vs Price
```

#### **11. Gamma Bias**
```
Data:
- Positive Gamma levels
- Negative Gamma levels
- Gamma walls
- Bias Score
- Direction
- Gamma flip level
- Chart: Gamma exposure
```

#### **12. Vanna Bias**
```
Data:
- Total Vanna
- Vanna distribution
- Bias Score
- Direction
- Vanna hedging impact
- Chart: Vanna profile
```

#### **13. Charm Bias**
```
Data:
- Total Charm
- Charm distribution
- Bias Score
- Direction
- Time decay impact
- Chart: Charm profile
```

### **Combined Analysis**
```
- Overall Bias Score (All 13 combined)
- Bullish Indicators Count
- Bearish Indicators Count
- Neutral Indicators Count
- Confluence Strength
- Conviction Level
- Trading Recommendation
- Chart: Bias indicator heatmap
```

---

## 📉 TAB 6: Advanced Chart Analysis

### Purpose: Multi-timeframe technical analysis with 15+ indicators

### Chart Timeframes Available:
```
- 1 minute
- 5 minutes
- 15 minutes
- 1 hour
- 4 hours
- 1 day
```

### Sub-Tabs (Dynamic based on enabled indicators):

#### **Sub-Tab 1: ML Market Regime Analysis**
```
Data Displayed:

1. Trading Sentiment
   - Sentiment (STRONG LONG/LONG/NEUTRAL/SHORT/STRONG SHORT)
   - Confidence %
   - Sentiment Score (-100 to +100)

2. Current Regime
   - Regime Type (Trending Up/Down/Range Bound/Volatile Breakout/Consolidation)
   - Confidence %
   - Volatility State (Low/Normal/High/Extreme)

3. Support & Resistance Levels
   - Near Support (₹, distance %)
   - Major Support (₹, distance %)
   - Near Resistance (₹, distance %)
   - Major Resistance (₹, distance %)
   - S1, S2, S3 levels
   - R1, R2, R3 levels

4. Entry/Exit Signals
   - Action (BUY_ON_PULLBACK/SELL_ON_RALLY/BUY_ON_BREAK/etc.)
   - Direction (LONG/SHORT/NEUTRAL/BOTH)
   - Conviction (LOW/MEDIUM/HIGH/VERY HIGH)
   - Entry Zone (₹ range)
   - Stop Loss Level
   - Target Levels (T1, T2, T3)
   - Risk-Reward Ratio

5. Feature Importance
   - Top 10 features driving prediction
   - Feature scores
   - Chart: Feature importance bar graph

6. Regime Probabilities
   - Trending Up probability %
   - Trending Down probability %
   - Range Bound probability %
   - Volatile Breakout probability %
   - Consolidation probability %
   - Chart: Probability distribution

7. Market Context
   - Trend strength (0-100)
   - Volatility level
   - Momentum state
   - Market phase (Accumulation/Markup/Distribution/Markdown)
   - Optimal timeframe for trading
```

#### **Sub-Tab 2: Volume Bars**
```
Data:
- Volume histogram
- Green bars (bullish candles)
- Red bars (bearish candles)
- Volume MA
- Volume spikes highlighted
- Average volume line
- Volume percentile
```

#### **Sub-Tab 3: Volume Order Blocks (VOB)**
```
Data:
For Each Block:
- Block type (BULLISH/BEARISH)
- Top price
- Bottom price
- Middle price
- Volume strength
- Active/Inactive status
- Formation time
- Test count
- Block age
- Validity score

Chart Overlays:
- Green boxes (Bullish VOBs - Support)
- Red boxes (Bearish VOBs - Resistance)
- Price interaction with blocks
```

#### **Sub-Tab 4: Ultimate RSI**
```
Data:
- RSI value (0-100)
- RSI MA
- Overbought level (>70)
- Oversold level (<30)
- RSI divergence detection
- Divergence type (Bullish/Bearish/Hidden)
- RSI zones color-coded
- RSI signals (BUY/SELL)

Chart:
- RSI line
- Overbought/Oversold bands
- Divergence markers
- Zero line
```

#### **Sub-Tab 5: Money Flow Profile**
```
Data:
- POC (Point of Control) price
- POC volume
- High Volume Nodes (HVN)
- Low Volume Nodes (LVN)
- Value Area High (VAH)
- Value Area Low (VAL)
- Bullish volume %
- Bearish volume %
- Total volume
- Distance from POC %
- Price position (Above/At/Below POC)
- Sentiment (BULLISH/BEARISH/NEUTRAL)

For Each Price Level:
- Price
- Volume
- % of total
- Bullish volume
- Bearish volume
- Net flow

Chart:
- Horizontal volume histogram
- POC line
- VAH/VAL lines
- HVN/LVN zones highlighted
```

#### **Sub-Tab 6: DeltaFlow Profile**
```
Data:
- Overall Delta
- Overall Bull %
- Overall Bear %
- POC price (delta-based)
- Strong Buy levels (Delta > threshold)
- Strong Sell levels (Delta < threshold)
- Absorption zones
- Delta imbalance
- Cumulative delta
- Distance from POC %
- Price position
- Sentiment (STRONG BULLISH to STRONG BEARISH)

For Each Price Level:
- Price
- Delta value
- Bull %
- Bear %
- Absorption detected (Yes/No)

Chart:
- Horizontal delta histogram
- POC line
- Strong buy/sell zones highlighted
- Absorption zones marked
```

#### **Sub-Tab 7: Price Action (BOS, CHOCH, Patterns)**
```
Data:

BOS (Break of Structure):
For Each Event:
- Type (BULLISH BOS/BEARISH BOS)
- Price level
- Break strength
- Volume confirmation
- Time of break
- Previous structure level
- Chart marker

CHOCH (Change of Character):
For Each Event:
- Type (BULLISH CHOCH/BEARISH CHOCH)
- Price level
- Significance
- Time of change
- Chart marker

Fibonacci Levels:
- 0% (Swing Low/High)
- 23.6%
- 38.2%
- 50%
- 61.8%
- 78.6%
- 100% (Swing High/Low)
- Current price proximity to levels

Chart Patterns:
- Head & Shoulders
- Double Top/Bottom
- Triangle (Ascending/Descending/Symmetrical)
- Wedge (Rising/Falling)
- Rectangle
- Flag/Pennant
- Pattern completion %
- Target projection
```

#### **Sub-Tab 8: Reversal Probability Zones**
```
Data:
For Each Zone:
- Zone type (Bullish Reversal/Bearish Reversal)
- Top price
- Bottom price
- Middle price
- Probability score (0-100)
- Confluence factors count
- Volume support
- Price action confirmation
- Active/Inactive status

Chart:
- Green zones (Bullish reversal areas)
- Red zones (Bearish reversal areas)
- Probability labels
- Current price interaction
```

#### **Main Chart Features**
```
On the Chart:
- Candlesticks (OHLC)
- Volume bars
- All active VOBs
- All reversal zones
- BOS/CHOCH markers
- Fibonacci levels
- Support/Resistance lines
- Trend lines
- Moving averages (if enabled)
- All indicators overlaid
```

---

## 🎯 TAB 7: NIFTY Option Screener v7.0

### Purpose: Comprehensive option chain analysis

### Main Data Sections:

#### **1. Option Chain Overview**
```
Current Market Data:
- NIFTY Spot Price
- Futures Price
- Premium/Discount
- ATM Strike
- PCR (Put-Call Ratio)
- Max Pain
- IV Index
- Time to Expiry

Expiry Selection:
- Current week
- Next week
- Monthly expiry
- Quarterly expiry
```

#### **2. ATM Strike Analysis**
```
For ATM and ATM±1, ATM±2 strikes:

Call Side (CE):
- Strike
- LTP (Last Traded Price)
- Change %
- Volume
- Open Interest (OI)
- Change in OI
- IV (Implied Volatility)
- Greeks:
  - Delta
  - Gamma
  - Theta
  - Vega

Put Side (PE):
- (Same data as Call side)

Comparison:
- CE vs PE OI
- CE vs PE Volume
- CE vs PE IV
- Bias determination
```

#### **3. OI Analysis**
```
Data:
- Total CE OI
- Total PE OI
- PCR (OI)
- OI distribution chart
- Highest CE OI strikes (Top 5)
- Highest PE OI strikes (Top 5)
- OI concentration %
- OI based support levels
- OI based resistance levels
```

#### **4. Change in OI Analysis**
```
Data:
- CE OI additions (Top 5 strikes)
- PE OI additions (Top 5 strikes)
- CE OI reductions (Top 5 strikes)
- PE OI reductions (Top 5 strikes)
- Fresh positions vs unwinding
- Long buildup strikes
- Short buildup strikes
- Long unwinding strikes
- Short unwinding strikes
- Chart: ChgOI distribution
```

#### **5. Volume Analysis**
```
Data:
- Total CE Volume
- Total PE Volume
- PCR (Volume)
- Highest CE volume strikes (Top 5)
- Highest PE volume strikes (Top 5)
- Volume distribution chart
- Volume concentration %
- Volume vs OI ratio
- Unusual volume alerts
```

#### **6. IV Analysis**
```
Data:
- IV Surface chart
- ATM IV
- OTM IV (both sides)
- ITM IV (both sides)
- IV Skew
  - 25 Delta skew
  - 10 Delta skew
- IV Percentile
- IV Rank
- High IV strikes
- Low IV strikes
- IV vs Historical IV
- Chart: IV smile/skew
```

#### **7. Greeks Analysis**
```
Aggregate Greeks:
- Total Delta (Net)
- Total Gamma
- Total Theta
- Total Vega
- Gamma walls (Support/Resistance)
- Delta distribution
- Vanna exposure
- Charm exposure

Per Strike Greeks:
- Delta profile chart
- Gamma profile chart
- Theta decay chart
- Vega sensitivity chart
```

#### **8. Max Pain Analysis**
```
Data:
- Max Pain level
- Current price
- Distance to Max Pain
- Max Pain change over time
- Pain calculation for each strike
- Chart: Pain curve
- Probability of spot at max pain
- Max Pain history (last 5 sessions)
```

#### **9. Put-Call Ratio (PCR)**
```
Data:
- PCR (OI based)
- PCR (Volume based)
- PCR (Value based)
- PCR interpretation
- PCR percentile
- PCR vs historical average
- PCR change over day
- Chart: PCR trend
- PCR based signals
```

#### **10. Momentum & Flow**
```
Data:
- Momentum Burst score
- Orderbook Pressure
  - Bid-Ask imbalance
  - Depth analysis
- Gamma Cluster Concentration
- OI Acceleration
- Expiry Spike detection
- Net Vega Exposure
- Skew Ratio
- ATM Vol Premium
```

#### **11. Buildup Analysis**
```
For Each Buildup Type:

Long Call Buildup:
- Strikes
- OI increase
- Price increase
- Strength score

Short Call Buildup:
- Strikes
- OI increase
- Price decrease
- Strength score

Long Put Buildup:
- Strikes
- OI increase
- Price increase
- Strength score

Short Put Buildup:
- Strikes
- OI increase
- Price decrease
- Strength score
```

#### **12. Participant Analysis**
```
Data:
- FII positions (Index Futures)
- DII positions
- Pro positions
- Client positions
- Net positions by participant
- Participant flow over time
- Chart: Participant positioning
```

#### **13. Strike-wise Full Data**
```
For Each Strike (Full option chain):

Call (CE):
- Strike
- LTP
- Change
- % Change
- Bid Price
- Bid Qty
- Ask Price
- Ask Qty
- Volume
- OI
- Change in OI
- IV
- Delta
- Gamma
- Theta
- Vega
- In-the-money amount

Put (PE):
- (Same as Call)

Additional:
- Strike importance score
- Support/Resistance classification
- Recommended actions
```

#### **14. Advanced Analytics**
```
Data:
- Gamma Squeeze probability
- Dealer positioning (Long/Short gamma)
- Volatility term structure
- Skew term structure
- Pin risk analysis
- Roll risk (near expiry)
- Arbitrage opportunities
- Spread analysis (Bull/Bear/Butterfly/Condor)
```

---

## 🌐 TAB 8: Enhanced Market Data

### Purpose: Sector rotation, VIX, gamma squeeze, macro analysis

### Sub-Sections:

#### **1. Sector Rotation Analysis**
```
Data:

For Each Sector (11 sectors):
- Sector name (IT, Banking, Auto, Pharma, FMCG, Energy, Metal, Realty, Media, PSU Bank, Private Bank)
- Current performance (%)
- 1-day change
- 5-day change
- 1-month change
- Relative strength vs NIFTY
- Sector status (Leading/Lagging)
- Money flow (Inflow/Outflow)
- Volume trend

Sector Rankings:
- Top 3 performing sectors
- Bottom 3 performing sectors
- Sector rotation matrix

Breadth Analysis:
- Advance/Decline ratio
- New highs/lows
- % stocks above 50 DMA
- % stocks above 200 DMA
- Market breadth score

Rotation Bias:
- Defensive vs Cyclical
- Growth vs Value
- Large cap vs Mid cap vs Small cap
- Overall market rotation bias
- Chart: Sector performance heatmap
- Chart: Rotation wheel
```

#### **2. India VIX Analysis**
```
Data:
- Current VIX level
- Change
- % Change
- Intraday high/low
- 52-week high/low
- VIX percentile (1-year)
- VIX interpretation
  - Low (<12): Complacency
  - Normal (12-15): Normal volatility
  - Elevated (15-20): Caution
  - High (20-25): Fear
  - Extreme (>25): Panic

VIX Term Structure:
- Current month VIX
- Next month VIX
- 3-month VIX
- Term structure slope
- Contango/Backwardation

Fear & Greed Index:
- Current reading
- State (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
- Components breakdown

Chart:
- VIX trend
- VIX bands
- Historical comparison
- VIX vs NIFTY correlation
```

#### **3. Gamma Squeeze Detection**
```
Data:
- Gamma Exposure (GEX) value
- Positive Gamma ($ value)
- Negative Gamma ($ value)
- Net Gamma
- Dealer Positioning %
  - Long gamma (price dampening)
  - Short gamma (price accelerating)

Gamma Flip Level:
- Price level where gamma flips
- Distance to flip level
- Flip probability

Squeeze Metrics:
- Squeeze intensity score (0-100)
- Squeeze probability %
- Gamma walls (Support/Resistance)
  - Put wall (Support)
  - Call wall (Resistance)
- Distance to walls

Squeeze Detection:
- Squeeze detected (Yes/No)
- Squeeze direction (Upside/Downside)
- Expected price range
- Risk level

Chart:
- GEX curve
- Gamma flip level
- Gamma walls
- Historical squeezes marked
```

#### **4. FII/DII Data**
```
Data:
- Date
- FII Cash (Buy/Sell/Net)
- FII F&O (Buy/Sell/Net)
- FII Index Futures (Long/Short/Net)
- FII Stock Futures (Long/Short/Net)
- FII Index Options (Call/Put/Net)
- DII Cash (Buy/Sell/Net)
- DII F&O (Buy/Sell/Net)

Cumulative Data:
- FII cumulative (MTD, YTD)
- DII cumulative (MTD, YTD)

Interpretation:
- Net flow bias
- Positioning bias
- Flow strength

Chart:
- FII/DII daily flow
- FII/DII cumulative
- FII vs DII comparison
```

#### **5. Global Market Indices**
```
For Each Index:
- Index name
- Current level
- Change
- % Change
- Status (Open/Closed)

Indices:
- US Markets:
  - S&P 500
  - NASDAQ
  - Dow Jones
- Asian Markets:
  - Nikkei
  - Hang Seng
  - Shanghai
- European Markets:
  - FTSE
  - DAX
  - CAC

Correlation:
- NIFTY correlation with each
- Global cues impact
```

#### **6. Market Breadth Details**
```
Data:
- Total stocks
- Advances
- Declines
- Unchanged
- Advance/Decline ratio
- New 52-week highs
- New 52-week lows
- Stocks above 50 DMA
- Stocks above 200 DMA
- Bullish crossovers
- Bearish crossovers
- Breadth thrust (Yes/No)
- Market internals score

Chart:
- A/D line
- Breadth indicators
- High/Low ratio
```

#### **7. Put-Call Ratio (Market-wide)**
```
Data:
- NSE PCR (OI)
- NSE PCR (Volume)
- NIFTY PCR
- BankNIFTY PCR
- FINNIFTY PCR
- Stock PCR
- Combined interpretation
- Historical percentile

Chart:
- PCR trend (all instruments)
- PCR distribution
```

---

## 🔍 TAB 9: NSE Stock Screener

### Purpose: Screen individual NIFTY 50 stocks with technical analysis

### Stock List:
```
All NIFTY 50 stocks with real-time data
```

### For Each Stock:

#### **1. Basic Data**
```
- Stock Symbol
- Company Name
- Current Price (LTP)
- Change
- % Change
- Day High
- Day Low
- Open
- Previous Close
- Volume
- Average Volume
- Volume ratio (Current/Average)
- VWAP
- Market Cap
- 52-week High
- 52-week Low
- % from 52W High
- % from 52W Low
```

#### **2. Technical Indicators**
```
- RSI (14)
- RSI interpretation (Overbought/Oversold/Neutral)
- MACD
- MACD Signal
- MACD Histogram
- MACD Status (Bullish/Bearish)
- Moving Averages:
  - 20 DMA
  - 50 DMA
  - 100 DMA
  - 200 DMA
- Price vs MA status (Above/Below each)
- Golden Cross/Death Cross
- Bollinger Bands (Upper/Middle/Lower)
- BB Width
- BB %B
- ATR (Average True Range)
- Supertrend (Bullish/Bearish)
```

#### **3. Pattern Recognition**
```
- Chart patterns detected
- Support levels (S1, S2, S3)
- Resistance levels (R1, R2, R3)
- Pivot points
- Fibonacci levels
- Trend lines
```

#### **4. Momentum Indicators**
```
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- MFI (Money Flow Index)
- ADX (Trend strength)
- Parabolic SAR
```

#### **5. Volume Analysis**
```
- Volume trend (Increasing/Decreasing)
- Volume spikes
- OBV (On Balance Volume)
- Volume Price Trend
- Accumulation/Distribution
```

#### **6. Screening Filters**
```
Filter by:
- Price range
- % Change range
- Volume criteria
- RSI levels
- Moving average crossovers
- Breakout detection
- Volatility
- Market cap
- Sector
- Custom combinations
```

#### **7. Signals**
```
For Each Stock:
- Overall signal (BUY/SELL/HOLD)
- Signal strength (1-5 stars)
- Bullish indicators count
- Bearish indicators count
- Neutral indicators count
- Conviction level
- Risk level
```

#### **8. Stock Comparison**
```
- Side-by-side comparison (up to 5 stocks)
- Relative strength comparison
- Correlation matrix
- Sector peers comparison
```

---

## 📊 SUMMARY: Complete Data Inventory

### Total Data Points in App:

| Tab | Main Sections | Sub-Tabs | Unique Data Points | Charts |
|-----|---------------|----------|-------------------|--------|
| **Tab 1: Overall Market Sentiment** | 5 | 0 | 50+ | 3 |
| **Tab 2: Trade Setup** | 3 | 0 | 20+ | 1 |
| **Tab 3: Active Signals** | 4 | 0 | 40+ | 2 |
| **Tab 4: Positions** | 4 | 0 | 60+ | 5 |
| **Tab 5: Bias Analysis Pro** | 14 | 13 | 150+ | 26 |
| **Tab 6: Advanced Chart Analysis** | 8 | 8+ | 200+ | 20+ |
| **Tab 7: NIFTY Option Screener** | 14 | 0 | 300+ | 15+ |
| **Tab 8: Enhanced Market Data** | 7 | 0 | 150+ | 10 |
| **Tab 9: NSE Stock Screener** | 8 | 0 | 400+ (per stock) | 5 |

### **GRAND TOTAL:**
- **9 Main Tabs**
- **60+ Major Sections**
- **20+ Sub-Tabs** (in Chart Analysis)
- **1,300+ Unique Data Points**
- **85+ Charts and Visualizations**

---

## 🎯 XGBoost AI Integration Summary

### Data Points AI Uses for Training:

From the **1,300+ total data points**, the AI extracts **~150 key features**:

✅ **Tab 1**: 5 features (Overall sentiment, confluence, etc.)
✅ **Tab 5**: 13 features (All 13 bias indicators)
✅ **Tab 6**: 40+ features (Chart indicators, VOB, RSI, MFP, DFP, etc.)
✅ **Tab 7**: 30+ features (OI, PCR, Greeks, Gamma, Max Pain, etc.)
✅ **Tab 8**: 50+ features (Sector rotation, VIX, Gamma squeeze, FII/DII, etc.)
✅ **Advanced Modules**: 40+ features (Regime, volatility, OI traps, CVD, etc.)

**The AI learns from YOUR complete trading strategy across ALL tabs!**

---

**Generated**: 2025-12-27
**Status**: ✅ Complete App Structure Documented
**Total Data Points**: 1,300+
**AI Training Features**: ~150 from all tabs
