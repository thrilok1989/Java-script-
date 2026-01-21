# 📊 ICT Indicator Data Output Structure

## Complete Data Structure Returned by `calculate()` Method

### Main Dictionary Keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `swing_order_blocks` | List[OrderBlock] | Large timeframe order blocks | 2-10 blocks |
| `internal_order_blocks` | List[OrderBlock] | Short timeframe order blocks | 2-10 blocks |
| `fvgs` | List[FairValueGap] | Fair value gaps | 0-10 gaps |
| `supply_zones` | List[SupplyDemandZone] | Supply (resistance) zones | 0-20 zones |
| `demand_zones` | List[SupplyDemandZone] | Demand (support) zones | 0-20 zones |
| `volume_profile` | List[VolumeProfileRow] | Volume distribution rows | 30 rows |
| `poc_price` | float | Point of Control price | 23456.50 |
| `signals` | Dict | Trading signals summary | See below |

---

## 1. Order Block Data Structure

### Swing Order Blocks (List of OrderBlock objects)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `bar_high` | float | Top of order block | 23500.00 |
| `bar_low` | float | Bottom of order block | 23450.00 |
| `bar_index` | int | Bar index in dataframe | 145 |
| `bar_time` | pd.Timestamp | Timestamp of the bar | 2024-01-21 10:30:00 |
| `bias` | int | 1=Bullish, -1=Bearish | 1 |
| `is_internal` | bool | False for swing, True for internal | False |
| `is_mitigated` | bool | Has price broken through? | False |

**Example Output:**
```python
{
    'bar_high': 23500.00,
    'bar_low': 23450.00,
    'bar_index': 145,
    'bar_time': Timestamp('2024-01-21 10:30:00'),
    'bias': 1,  # Bullish
    'is_internal': False,
    'is_mitigated': False
}
```

### Internal Order Blocks (Same structure, `is_internal=True`)

---

## 2. Fair Value Gap (FVG) Data Structure

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `top` | float | Upper boundary of gap | 23480.00 |
| `bottom` | float | Lower boundary of gap | 23420.00 |
| `start_index` | int | Starting bar index | 150 |
| `gap_type` | str | 'bullish' or 'bearish' | 'bullish' |
| `is_mitigated` | bool | Has gap been filled? | False |

**Example Output:**
```python
{
    'top': 23480.00,
    'bottom': 23420.00,
    'start_index': 150,
    'gap_type': 'bullish',
    'is_mitigated': False
}
```

---

## 3. Supply/Demand Zone Data Structure

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `top` | float | Top of zone | 23550.00 |
| `bottom` | float | Bottom of zone | 23520.00 |
| `poi` | float | Point of Interest (midpoint) | 23535.00 |
| `left_index` | int | Starting bar index | 100 |
| `right_index` | int | Ending bar index | 200 |
| `zone_type` | str | 'supply' or 'demand' | 'supply' |
| `is_broken` | bool | Has zone been invalidated? | False |

**Example Output:**
```python
# Supply Zone
{
    'top': 23550.00,
    'bottom': 23520.00,
    'poi': 23535.00,
    'left_index': 100,
    'right_index': 200,
    'zone_type': 'supply',
    'is_broken': False
}

# Demand Zone
{
    'top': 23420.00,
    'bottom': 23380.00,
    'poi': 23400.00,
    'left_index': 95,
    'right_index': 200,
    'zone_type': 'demand',
    'is_broken': False
}
```

---

## 4. Volume Profile Data Structure

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `top` | float | Top price of this row | 23500.00 |
| `bottom` | float | Bottom price of this row | 23490.00 |
| `total_volume` | float | Total volume in this range | 125000 |
| `bull_volume` | float | Buying volume | 70000 |
| `bear_volume` | float | Selling volume | 55000 |

**Example Output (30 rows):**
```python
[
    VolumeProfileRow(
        top=23500.00,
        bottom=23490.00,
        total_volume=125000.0,
        bull_volume=70000.0,
        bear_volume=55000.0
    ),
    VolumeProfileRow(
        top=23490.00,
        bottom=23480.00,
        total_volume=98000.0,
        bull_volume=45000.0,
        bear_volume=53000.0
    ),
    # ... 28 more rows
]
```

---

## 5. Signals Data Structure (Most Important)

### Main Signal Dictionary

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `overall_bias` | str | BULLISH / BEARISH / NEUTRAL | 'BULLISH' |
| `bullish_count` | int | Number of bullish signals | 5 |
| `bearish_count` | int | Number of bearish signals | 1 |
| `bullish_signals` | List[Dict] | List of active bullish signals | See below |
| `bearish_signals` | List[Dict] | List of active bearish signals | See below |

### Bullish Signals Array

Each signal in the array:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `type` | str | Signal type | 'Bullish Order Block' |
| `price` | str | Price range or level | '23400.00 - 23450.00' |
| `level` | str | Optional: 'Swing' or 'Internal' | 'Swing' |
| `poi` | str | Optional: Point of Interest | '23425.00' |

**Example Bullish Signals:**
```python
'bullish_signals': [
    {
        'type': 'Bullish Order Block',
        'price': '23400.00 - 23450.00',
        'level': 'Swing'
    },
    {
        'type': 'Bullish Order Block',
        'price': '23420.00 - 23440.00',
        'level': 'Internal'
    },
    {
        'type': 'Bullish FVG',
        'price': '23380.00 - 23420.00'
    },
    {
        'type': 'Demand Zone',
        'price': '23350.00 - 23400.00',
        'poi': '23375.00'
    },
    {
        'type': 'Above POC',
        'price': '23425.00'
    }
]
```

### Bearish Signals Array

Same structure as bullish:

**Example Bearish Signals:**
```python
'bearish_signals': [
    {
        'type': 'Bearish Order Block',
        'price': '23500.00 - 23520.00',
        'level': 'Internal'
    },
    {
        'type': 'Supply Zone',
        'price': '23550.00 - 23580.00',
        'poi': '23565.00'
    }
]
```

---

## 6. Complete Example Output

```python
{
    # Order Blocks (Swing)
    'swing_order_blocks': [
        OrderBlock(
            bar_high=23500.00,
            bar_low=23450.00,
            bar_index=145,
            bar_time=Timestamp('2024-01-21 10:30:00'),
            bias=1,  # Bullish
            is_internal=False,
            is_mitigated=False
        ),
        OrderBlock(
            bar_high=23320.00,
            bar_low=23280.00,
            bar_index=120,
            bar_time=Timestamp('2024-01-21 10:05:00'),
            bias=1,  # Bullish
            is_internal=False,
            is_mitigated=False
        )
    ],

    # Order Blocks (Internal)
    'internal_order_blocks': [
        OrderBlock(
            bar_high=23480.00,
            bar_low=23460.00,
            bar_index=180,
            bar_time=Timestamp('2024-01-21 11:00:00'),
            bias=1,  # Bullish
            is_internal=True,
            is_mitigated=False
        )
    ],

    # Fair Value Gaps
    'fvgs': [
        FairValueGap(
            top=23480.00,
            bottom=23420.00,
            start_index=150,
            gap_type='bullish',
            is_mitigated=False
        )
    ],

    # Supply Zones
    'supply_zones': [
        SupplyDemandZone(
            top=23550.00,
            bottom=23520.00,
            poi=23535.00,
            left_index=100,
            right_index=200,
            zone_type='supply',
            is_broken=False
        )
    ],

    # Demand Zones
    'demand_zones': [
        SupplyDemandZone(
            top=23420.00,
            bottom=23380.00,
            poi=23400.00,
            left_index=95,
            right_index=200,
            zone_type='demand',
            is_broken=False
        ),
        SupplyDemandZone(
            top=23350.00,
            bottom=23320.00,
            poi=23335.00,
            left_index=80,
            right_index=200,
            zone_type='demand',
            is_broken=False
        )
    ],

    # Volume Profile (30 rows)
    'volume_profile': [
        VolumeProfileRow(top=23550.00, bottom=23540.00, total_volume=125000.0, bull_volume=70000.0, bear_volume=55000.0),
        VolumeProfileRow(top=23540.00, bottom=23530.00, total_volume=98000.0, bull_volume=45000.0, bear_volume=53000.0),
        # ... 28 more rows
    ],

    # Point of Control
    'poc_price': 23425.00,

    # Trading Signals
    'signals': {
        'overall_bias': 'BULLISH',
        'bullish_count': 5,
        'bearish_count': 1,
        'bullish_signals': [
            {'type': 'Bullish Order Block', 'price': '23400.00 - 23450.00', 'level': 'Swing'},
            {'type': 'Bullish Order Block', 'price': '23420.00 - 23440.00', 'level': 'Internal'},
            {'type': 'Bullish FVG', 'price': '23380.00 - 23420.00'},
            {'type': 'Demand Zone', 'price': '23350.00 - 23400.00', 'poi': '23375.00'},
            {'type': 'Above POC', 'price': '23425.00'}
        ],
        'bearish_signals': [
            {'type': 'Bearish Order Block', 'price': '23500.00 - 23520.00', 'level': 'Internal'}
        ]
    }
}
```

---

## 7. How to Access Data in Streamlit App

```python
# In app.py after ICT indicator calculation
ict_data = ict_indicator.calculate(df)

# Access individual components
swing_obs = ict_data['swing_order_blocks']
print(f"Found {len(swing_obs)} swing order blocks")

# Access first swing order block
if swing_obs:
    first_ob = swing_obs[0]
    print(f"Bullish OB: {first_ob.bar_low} - {first_ob.bar_high}")

# Access signals
signals = ict_data['signals']
print(f"Bias: {signals['overall_bias']}")
print(f"Bullish signals: {signals['bullish_count']}")
print(f"Bearish signals: {signals['bearish_count']}")

# Access POC
poc = ict_data['poc_price']
print(f"POC at: {poc}")

# Access FVGs
fvgs = ict_data['fvgs']
for fvg in fvgs:
    print(f"{fvg.gap_type} FVG: {fvg.bottom} - {fvg.top}")
```

---

## 8. Signal Counting Logic

### How Bias is Determined

| Condition | Bias Result |
|-----------|-------------|
| `bullish_count > bearish_count + 1` | BULLISH |
| `bearish_count > bullish_count + 1` | BEARISH |
| Otherwise | NEUTRAL |

### Signal Weights

| Signal Type | Weight | Reasoning |
|-------------|--------|-----------|
| Swing Order Block | +2 | Strongest level |
| Internal Order Block | +1 | Short-term level |
| Fair Value Gap | +1 | Price magnet |
| Supply/Demand Zone | +1 | Pressure zone |
| POC Position | +1 | Volume level |

**Example Calculation:**
```
Price in:
- Bullish Swing OB: +2
- Bullish Internal OB: +1
- Bullish FVG: +1
- Demand Zone: +1
- Above POC: +1
Total: 6 bullish points

- Bearish Internal OB: +1
Total: 1 bearish point

Result: bullish_count=6 > bearish_count+1 (2)
→ BULLISH BIAS ✅
```

---

## 9. Data Validation Rules

### Active Order Blocks
- ✅ `is_mitigated = False`
- ✅ Price within range: `bar_low <= current_price <= bar_high`

### Active FVGs
- ✅ `is_mitigated = False`
- ✅ Price within gap: `bottom <= current_price <= top`

### Active Zones
- ✅ `is_broken = False`
- ✅ Price within zone: `bottom <= current_price <= top`

### POC Signal
- ✅ `poc_price > 0`
- ✅ Current price compared to POC

---

## 10. Empty Data Scenarios

| Scenario | Possible Causes | Expected Output |
|----------|-----------------|-----------------|
| No Order Blocks | Low volatility, small dataset | `swing_order_blocks: []` |
| No FVGs | No gap patterns formed | `fvgs: []` |
| No Supply/Demand | Insufficient swing points | `supply_zones: [], demand_zones: []` |
| POC = 0 | No volume data | `poc_price: 0.0` |
| Neutral Bias | Equal bullish/bearish signals | `overall_bias: 'NEUTRAL'` |

This is normal and expected in some market conditions!
