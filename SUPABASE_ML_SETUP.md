# Supabase ML Storage Setup Guide

Complete guide to set up Supabase cloud storage for your AI/ML training data and models.

---

## 🎯 What This Does

Stores all AI/ML data in Supabase cloud database:
- ✅ Training data (predictions & outcomes)
- ✅ Trained model metadata
- ✅ Model files (in Supabase Storage)
- ✅ Training statistics
- ✅ Performance metrics

**Benefits:**
- 📊 Persistent storage across sessions
- ☁️ Cloud-based (no local files)
- 🔄 Sync across multiple instances
- 📈 Better data management and querying
- 🚀 Scalable and fast

---

## 📋 Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up / Log in
3. Click "New Project"
4. Fill in:
   - **Name**: `trading-ai` (or any name)
   - **Database Password**: (save this!)
   - **Region**: Choose closest to you
5. Click "Create new project"
6. Wait for project to initialize (~2 minutes)

---

## 🗄️ Step 2: Create Database Tables

### Table 1: ml_training_data

Go to **SQL Editor** in Supabase and run:

```sql
CREATE TABLE ml_training_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence FLOAT,
    features JSONB,
    actual_outcome TEXT,
    outcome_timestamp TIMESTAMPTZ,
    profit_loss FLOAT,
    notes TEXT
);

-- Create indexes for better query performance
CREATE INDEX idx_training_symbol ON ml_training_data(symbol);
CREATE INDEX idx_training_timestamp ON ml_training_data(timestamp DESC);
CREATE INDEX idx_training_outcome ON ml_training_data(actual_outcome);
```

### Table 2: ml_models

```sql
CREATE TABLE ml_models (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    training_samples INT,
    feature_count INT,
    hyperparameters JSONB,
    model_file_url TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Create indexes
CREATE INDEX idx_models_active ON ml_models(is_active);
CREATE INDEX idx_models_created ON ml_models(created_at DESC);

-- Ensure only one active model
CREATE UNIQUE INDEX idx_one_active_model ON ml_models(is_active)
WHERE is_active = TRUE;
```

### Table 3: ml_statistics

```sql
CREATE TABLE ml_statistics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stats JSONB NOT NULL
);

-- Create index
CREATE INDEX idx_stats_timestamp ON ml_statistics(timestamp DESC);
```

---

## 📦 Step 3: Create Storage Bucket

1. Go to **Storage** in Supabase
2. Click "New bucket"
3. Name: `models`
4. **Public**: ✅ Enable (so we can download models)
5. Click "Create bucket"

---

## 🔑 Step 4: Get API Credentials

1. Go to **Settings** > **API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

---

## ⚙️ Step 5: Add to Streamlit Secrets

### Option A: Streamlit Cloud

1. Go to your app on Streamlit Cloud
2. Click **Settings** > **Secrets**
3. Add:

```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Option B: Local Development

Create `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## ✅ Step 6: Verify Setup

Run this Python code to test:

```python
from src.supabase_ml_storage import SupabaseMLStorage

# Initialize
storage = SupabaseMLStorage()

# Test connection
stats = storage.get_stats_summary()
print(f"✅ Connected! Stats: {stats}")
```

---

## 🎯 Usage Examples

### Save Training Data

```python
from src.supabase_ml_storage import SupabaseMLStorage, TrainingRecord
from datetime import datetime

storage = SupabaseMLStorage()

# Create record
record = TrainingRecord(
    timestamp=datetime.now(),
    symbol="NIFTY",
    prediction="BUY",
    confidence=0.85,
    features={"rsi": 45, "macd": 1.2},
    actual_outcome=None,  # Will update later
    notes="Strong bullish signal"
)

# Save to Supabase
storage.save_training_record(record)
```

### Retrieve Training Data

```python
# Get last 100 records
records = storage.get_training_data(limit=100)

# Get NIFTY records only
nifty_records = storage.get_training_data(symbol="NIFTY", limit=50)

# Get only records with outcomes
completed = storage.get_training_data(outcome_only=True)
```

### Update Outcomes

```python
# Update outcome for a specific record
storage.update_outcome(
    record_id=123,
    actual_outcome="CORRECT",
    profit_loss=150.50
)
```

### Save Model

```python
from src.supabase_ml_storage import ModelMetadata
import xgboost as xgb

# Train your model
model = xgb.XGBClassifier()
# ... training code ...

# Create metadata
metadata = ModelMetadata(
    model_name="nifty_predictor",
    version=1,
    created_at=datetime.now(),
    accuracy=0.87,
    precision=0.85,
    recall=0.89,
    f1_score=0.87,
    training_samples=1000,
    feature_count=150,
    hyperparameters={"max_depth": 5, "learning_rate": 0.1},
    is_active=True
)

# Save metadata
model_id = storage.save_model_metadata(metadata)

# Upload model file to Supabase Storage
url = storage.upload_model_file(model_id, model, "nifty_predictor")
print(f"Model uploaded: {url}")
```

### Load Active Model

```python
# Get currently active model
model_id, model = storage.get_active_model()

if model:
    # Use model for predictions
    predictions = model.predict(X)
```

---

## 📊 Dashboard Integration

The AI Training UI (Tab 11) will automatically use Supabase when:

1. Supabase credentials are set in secrets
2. Tables are created in Supabase
3. Storage bucket `models` exists

**If Supabase is not configured**, it falls back to local CSV files.

---

## 🔒 Security Notes

1. **Never commit secrets** - Always use `.streamlit/secrets.toml` (in .gitignore)
2. **Use Row Level Security** - Enable RLS in Supabase for production
3. **API Keys** - The anon key is safe for client-side use
4. **Sensitive Data** - Consider encryption for sensitive trading data

---

## 🐛 Troubleshooting

### "Module not found: supabase"
```bash
pip install supabase
```

### "Connection error"
- Check SUPABASE_URL and SUPABASE_KEY are correct
- Verify project is not paused in Supabase dashboard
- Check internet connection

### "Table does not exist"
- Run all SQL commands from Step 2
- Check table names match exactly (case-sensitive)

### "Storage bucket not found"
- Create `models` bucket in Supabase Storage
- Make it public

---

## 📈 Data Migration

To migrate existing local data to Supabase:

```python
import pandas as pd
from src.supabase_ml_storage import SupabaseMLStorage, TrainingRecord

storage = SupabaseMLStorage()

# Read local CSV
df = pd.read_csv('data/training_data.csv')

# Upload to Supabase
for _, row in df.iterrows():
    record = TrainingRecord(
        timestamp=row['timestamp'],
        symbol=row['symbol'],
        prediction=row['prediction'],
        # ... map other fields
    )
    storage.save_training_record(record)

print(f"✅ Migrated {len(df)} records to Supabase!")
```

---

## 🎉 You're Done!

Your AI training data is now stored in the cloud!

- ☁️ Access from anywhere
- 🔄 Persistent across sessions
- 📊 Easy to query and analyze
- 🚀 Scalable and fast

Check **Tab 11 (AI Training & Models)** to see it in action!
